"""Endpoints REST de Hugo.

POST /verify           ← Paco/Luis preguntan: "¿este candidato es duplicado?"
POST /audit            ← dispara auditoría completa on-demand
GET  /products/{id}/check  ← chequea un producto específico (precio + duplicado)
GET  /audit-log        ← últimas N acciones (para dashboard)
GET  /health           ← liveness probe
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, func, select

from app import runtime
from app.db.models import AuditLog, PriceHistory
from app.db.session import engine, get_session
from app.dedup.orchestrator import CandidateInput, find_duplicate_in
from app.integrations import paco as paco_integration
from app.pricing.source_check import fetch_source_price
from app.security import verify_api_key
from app.vendure.client import VendureClient

log = logging.getLogger(__name__)

router = APIRouter()


# ─── Traducción de acciones técnicas a texto humano ─────────────

_ACTION_LABELS: dict[str, dict[str, str]] = {
    "duplicate_disabled": {
        "icon": "duplicate",
        "title": "Duplicado deshabilitado en Vendure",
        "tone": "warning",
    },
    "duplicate_flagged": {
        "icon": "duplicate",
        "title": "Ya existe en el catálogo (no se reenvió a Paco)",
        "tone": "warning",
    },
    "verify_passed_to_paco": {
        "icon": "send",
        "title": "Producto nuevo · enviado a Paco para enriquecer",
        "tone": "info",
    },
    "paco_failed": {
        "icon": "alert",
        "title": "Producto nuevo · pero Paco no respondió",
        "tone": "danger",
    },
    "verify_no_match": {
        "icon": "info",
        "title": "Producto nuevo · sin imagen para mandar a Paco",
        "tone": "muted",
    },
    "price_updated": {
        "icon": "price",
        "title": "Precio actualizado",
        "tone": "info",
    },
    "price_flagged": {
        "icon": "price",
        "title": "Cambio de precio detectado",
        "tone": "warning",
    },
    "no_change": {
        "icon": "check",
        "title": "Sin cambios",
        "tone": "muted",
    },
    "error": {
        "icon": "alert",
        "title": "Error",
        "tone": "danger",
    },
    "quality_issue_found": {
        "icon": "alert",
        "title": "Producto con problemas (revisar/eliminar)",
        "tone": "warning",
    },
}


def _humanize(entry: AuditLog) -> dict[str, Any]:
    meta = _ACTION_LABELS.get(entry.action, {"icon": "info", "title": entry.action, "tone": "muted"})
    before = json.loads(entry.before) if entry.before else None
    after = json.loads(entry.after) if entry.after else None
    return {
        "id": entry.id,
        "action": entry.action,
        "source": entry.source,
        "title": meta["title"],
        "icon": meta["icon"],
        "tone": meta["tone"],
        "dismissed": entry.dismissed,
        "product": {
            "id": entry.product_id,
            "name": entry.product_name,
            "code": entry.product_code,
            "image_url": entry.product_image_url,
            "source_url": entry.product_source_url,
        },
        "related_product": (
            {
                "id": entry.related_product_id,
                "name": entry.related_product_name,
                "code": entry.related_product_code,
            }
            if entry.related_product_id
            else None
        ),
        "detail": entry.detail,
        "before": before,
        "after": after,
        "confidence": entry.confidence,
        # ISO con "Z" para que el frontend lo interprete inequívocamente como UTC
        "created_at": (entry.created_at.isoformat() + "Z") if entry.created_at else None,
    }


# ─── Definición de las "secciones"/tabs del dashboard ──────────────
# Cada sección filtra el AuditLog por (source, actions). Usado por
# /api/sections y /audit-log.

SECTIONS: dict[str, dict[str, Any]] = {
    "inbox_luis": {
        "label": "Llegan de Luis",
        "source": "luis",
        "actions": None,
    },
    "inbox_orders": {
        "label": "Llegan de Orders",
        "source": "orders",
        "actions": None,
    },
    "duplicates": {
        "label": "Duplicados",
        "source": None,
        "actions": ["duplicate_disabled", "duplicate_flagged"],
    },
    "price_changes": {
        "label": "Cambios de precio",
        "source": None,
        "actions": ["price_flagged"],
    },
    "sent_to_paco": {
        "label": "Enviados a Paco",
        "source": None,
        "actions": ["verify_passed_to_paco"],
    },
    "paco_errors": {
        "label": "Errores con Paco",
        "source": None,
        "actions": ["paco_failed"],
    },
    "quality_issues": {
        "label": "Problemas de calidad",
        "source": None,
        "actions": ["quality_issue_found"],
    },
    "all": {
        "label": "Todo",
        "source": None,
        "actions": None,
    },
}


def _apply_section_filter(stmt, section_key: str | None):
    """Aplica filtros source/actions de una sección a un statement select(AuditLog)."""
    if not section_key or section_key not in SECTIONS:
        return stmt
    s = SECTIONS[section_key]
    if s["source"]:
        stmt = stmt.where(AuditLog.source == s["source"])
    if s["actions"]:
        stmt = stmt.where(AuditLog.action.in_(s["actions"]))  # type: ignore[attr-defined]
    return stmt


# ─── DTOs ────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    name: str = ""
    description: str = ""
    source_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    # De qué sistema viene este verify. Se usa en el dashboard para tabs.
    # Valores típicos: "luis" (default), "orders", "manual"
    source: str = "luis"


class VerifyResponse(BaseModel):
    is_duplicate: bool
    confidence: float
    matched_by: list[str]
    per_strategy_scores: dict[str, float]
    candidate_id: str | None
    # Si no era duplicado y Hugo le pasó el job a Paco, devolvemos el search_id
    paco_search_id: str | None = None
    paco_status: str | None = None
    paco_error: str | None = None


# ─── Endpoints ───────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "hugo"}


@router.post(
    "/verify",
    response_model=VerifyResponse,
    dependencies=[Depends(verify_api_key)],
)
async def verify(payload: VerifyRequest) -> VerifyResponse:
    """Llamado por Luis cuando alguien pone 👍 en un producto viral.

    Hugo:
      1. Compara contra Vendure (URL + imagen + texto).
      2. Si NO es duplicado → reenvía la imagen a Paco (similarity search).
      3. Devuelve a Luis el veredicto + el search_id de Paco (si aplica).

    Auditoría: cada verify queda registrado en AuditLog para trazabilidad.
    """
    client = VendureClient()
    existing = await client.list_products(skip=0, take=500)
    verdict = await find_duplicate_in(
        CandidateInput(
            name=payload.name,
            description=payload.description,
            source_url=payload.source_url,
            image_urls=payload.image_urls,
        ),
        existing,
    )

    response = VerifyResponse(
        is_duplicate=verdict.is_duplicate,
        confidence=verdict.confidence,
        matched_by=list(verdict.matched_by),
        per_strategy_scores={k: v for k, v in verdict.per_strategy_scores.items()},
        candidate_id=verdict.candidate_id,
    )

    # Si no es duplicado y tenemos imagen, le pasamos a Paco automáticamente
    if not verdict.is_duplicate and payload.image_urls:
        try:
            result = await paco_integration.submit(payload.image_urls[0])
            response.paco_search_id = result.search_id
            response.paco_status = result.status
        except paco_integration.PacoError as exc:
            response.paco_error = str(exc)
            log.warning("Paco submit falló: %s", exc)
        except Exception as exc:  # noqa: BLE001
            response.paco_error = f"{type(exc).__name__}: {exc}"
            log.exception("Paco submit error inesperado")

    # Registrar la verificación en AuditLog
    try:
        with Session(engine) as session:
            short_name = payload.name[:60] if payload.name else "(sin nombre)"
            if verdict.is_duplicate:
                action = "duplicate_flagged"
                detail = (
                    f"'{short_name}' ya existe en Vendure como producto "
                    f"#{verdict.candidate_id}. Match por {','.join(verdict.matched_by)} "
                    f"(confianza {verdict.confidence:.0%})"
                )
            elif response.paco_search_id:
                action = "verify_passed_to_paco"
                detail = (
                    f"'{short_name}' es nuevo. Hugo lo envió a Paco "
                    f"(search_id={response.paco_search_id})"
                )
            elif response.paco_error:
                action = "paco_failed"
                detail = (
                    f"'{short_name}' es nuevo, pero Paco no respondió. "
                    f"Error: {response.paco_error[:120]}"
                )
            else:
                action = "verify_no_match"
                detail = (
                    f"'{short_name}' es nuevo, pero no llegó imagen para mandarle a Paco"
                )
            # Filtrar imagenes vacías que pueda mandar Luis (image_urls=[""] etc.)
            valid_imgs = [u for u in (payload.image_urls or []) if u and u.strip()]
            session.add(AuditLog(
                action=action,
                source=payload.source or "luis",
                product_id=verdict.candidate_id or "(nuevo)",
                detail=detail[:500],
                confidence=verdict.confidence,
                product_name=payload.name[:200] if payload.name else None,
                product_image_url=valid_imgs[0] if valid_imgs else None,
                product_source_url=payload.source_url,
            ))
            session.commit()
    except Exception:  # noqa: BLE001
        log.exception("No se pudo registrar AuditLog del verify")

    return response


@router.post("/audit")
async def audit_now(
    target: str = "all",
) -> dict[str, str]:
    """Dispara una auditoría on-demand.

    target: "all" (default) | "duplicates" | "prices" | "quality"
    """
    from app.scheduler.jobs import (
        audit_catalog_quality,
        audit_dupes_lock,
        audit_duplicates,
        audit_prices_lock,
        audit_quality_lock,
        audit_source_prices,
    )
    import asyncio

    wants_prices = target in ("prices", "all")
    wants_dupes = target in ("duplicates", "all")
    wants_quality = target in ("quality", "all")

    if wants_prices and audit_prices_lock.locked():
        raise HTTPException(409, "Ya hay una auditoría de precios en curso.")
    if wants_dupes and audit_dupes_lock.locked():
        raise HTTPException(409, "Ya hay una auditoría de duplicados en curso.")
    if wants_quality and audit_quality_lock.locked():
        raise HTTPException(409, "Ya hay una auditoría de calidad en curso.")

    if wants_dupes:
        asyncio.create_task(audit_duplicates())
    if wants_prices:
        asyncio.create_task(audit_source_prices())
    if wants_quality:
        asyncio.create_task(audit_catalog_quality())
    return {"status": "scheduled", "target": target}


@router.get("/products/{product_id}/check")
async def check_product(product_id: str) -> dict[str, Any]:
    """Chequea un producto puntual: trae sus datos y evalúa si su precio está alineado."""
    client = VendureClient()
    prod = await client.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Producto no existe en Vendure")

    payload: dict[str, Any] = {
        "product": {
            "id": prod.id,
            "name": prod.name,
            "source_url": prod.source_url,
            "enabled": prod.enabled,
        },
    }

    if prod.source_url:
        quote = await fetch_source_price(prod.source_url)
        if quote:
            payload["source_price"] = {
                "price_cents": quote.price_cents,
                "currency": quote.currency,
                "usd_equivalent_cents": quote.usd_price_cents,
                "fetched_from": quote.source,
            }
        else:
            payload["source_price"] = {"status": "source_unreachable"}
    else:
        payload["source_price"] = {"status": "skipped", "reason": "sin supplierLink"}

    return payload


@router.get("/audit-log")
async def list_audit_log(
    skip: int = 0,
    limit: int = 25,
    section: str | None = None,
    action: str | None = None,
    include_dismissed: bool = False,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Devuelve eventos paginados.

    Params:
      skip               — offset para paginación
      limit              — page size (max 100)
      section            — filtra por tab
      action             — filtra por action puntual
      include_dismissed  — por defecto False, ocultar eventos descartados
    """
    limit = max(1, min(100, limit))
    base = select(AuditLog)
    count_q = select(func.count(AuditLog.id))  # type: ignore[arg-type]

    if not include_dismissed:
        # is_not(True) matchea False y NULL (filas previas a la migración)
        base = base.where(AuditLog.dismissed.is_not(True))  # type: ignore[union-attr]
        count_q = count_q.where(AuditLog.dismissed.is_not(True))  # type: ignore[union-attr]
    if section:
        base = _apply_section_filter(base, section)
        count_q = _apply_section_filter(count_q, section)
    if action:
        base = base.where(AuditLog.action == action)
        count_q = count_q.where(AuditLog.action == action)

    total = session.exec(count_q).one() or 0
    stmt = base.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)
    items = [_humanize(e) for e in session.exec(stmt)]
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total,
    }


# ─── Settings runtime (configurables desde el dashboard) ──────────


class SettingUpdate(BaseModel):
    value: float | int | str


@router.get("/api/settings")
async def list_settings() -> list[dict[str, Any]]:
    """Devuelve todos los settings runtime con su valor actual + metadata."""
    return runtime.get_all_with_meta()


@router.put("/api/settings/{key}")
async def update_setting(key: str, payload: SettingUpdate) -> dict[str, Any]:
    """Actualiza un setting runtime. Persiste en DB y aplica en la próxima lectura."""
    try:
        new_value = runtime.set_value(key, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"key": key, "value": new_value, "ok": True}


@router.delete("/api/settings/{key}")
async def reset_setting(key: str) -> dict[str, Any]:
    """Borra el override y vuelve al default del .env."""
    try:
        new_value = runtime.reset_to_default(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"key": key, "value": new_value, "ok": True, "reset": True}


@router.post("/api/audit-log/{event_id}/dismiss")
async def dismiss_event(
    event_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Marca un evento como descartado. No vuelve a aparecer en el listado por defecto."""
    entry = session.get(AuditLog, event_id)
    if not entry:
        raise HTTPException(404, "Evento no encontrado")
    entry.dismissed = True
    entry.dismissed_at = datetime.utcnow()
    session.add(entry)
    session.commit()
    return {"ok": True, "id": event_id}


@router.post("/api/audit-log/{event_id}/retry-paco")
async def retry_paco(
    event_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Reintenta enviar un evento a Paco (solo si tiene image_url guardada).

    Crea un nuevo AuditLog con el resultado y descarta el original.
    """
    entry = session.get(AuditLog, event_id)
    if not entry:
        raise HTTPException(404, "Evento no encontrado")
    if not entry.product_image_url:
        raise HTTPException(400, "Este evento no tiene image_url para reintentar")

    # Llamar a Paco
    try:
        result = await paco_integration.submit(entry.product_image_url)
        new_action = "verify_passed_to_paco"
        new_detail = (
            f"Reintento manual exitoso. '{(entry.product_name or '?')[:60]}' "
            f"enviado a Paco (search_id={result.search_id})"
        )
        new_after = json.dumps({"paco_search_id": result.search_id, "retry_of": event_id})
        session.add(AuditLog(
            action=new_action,
            source="manual",
            product_id=entry.product_id,
            detail=new_detail,
            after=new_after,
            product_name=entry.product_name,
            product_code=entry.product_code,
            product_image_url=entry.product_image_url,
            product_source_url=entry.product_source_url,
        ))
        # Marcar el evento original como dismissed (ya se resolvió)
        entry.dismissed = True
        entry.dismissed_at = datetime.utcnow()
        session.add(entry)
        session.commit()
        return {"ok": True, "paco_search_id": result.search_id, "paco_status": result.status}
    except paco_integration.PacoError as exc:
        session.add(AuditLog(
            action="paco_failed",
            source="manual",
            product_id=entry.product_id,
            detail=f"Reintento manual falló: {str(exc)[:200]}",
            product_name=entry.product_name,
            product_code=entry.product_code,
            product_image_url=entry.product_image_url,
            product_source_url=entry.product_source_url,
        ))
        session.commit()
        raise HTTPException(502, f"Paco rechazó el reintento: {exc}")


@router.get("/api/duplicates-stats")
async def duplicates_stats(
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Resumen agregado de los duplicados detectados:
    - cuántos hay
    - cuántos disparó cada estrategia (url / image / text)
    - score promedio por estrategia
    - distribución del score (>=0.99, 0.95-0.99, 0.90-0.95, <0.90)
    """
    rows = list(session.exec(
        select(AuditLog).where(
            AuditLog.action.in_(["duplicate_flagged", "duplicate_disabled"])  # type: ignore[attr-defined]
        )
    ))

    total = len(rows)
    by_strategy = {"url": 0, "image": 0, "text": 0}
    score_buckets = {"99-100": 0, "95-99": 0, "90-95": 0, "<90": 0}
    score_sums = {"url": 0.0, "image": 0.0, "text": 0.0}
    score_counts = {"url": 0, "image": 0, "text": 0}

    for r in rows:
        # Bucket por confidence
        c = r.confidence or 0.0
        if c >= 0.99:
            score_buckets["99-100"] += 1
        elif c >= 0.95:
            score_buckets["95-99"] += 1
        elif c >= 0.90:
            score_buckets["90-95"] += 1
        else:
            score_buckets["<90"] += 1

        # Si tiene `after` con scores detallados, los usamos
        if r.after:
            try:
                data = json.loads(r.after)
                for s in (data.get("matched_by") or []):
                    if s in by_strategy:
                        by_strategy[s] += 1
                for k, v in (data.get("per_strategy_scores") or {}).items():
                    if k in score_sums and v is not None:
                        score_sums[k] += float(v)
                        score_counts[k] += 1
                continue
            except (ValueError, TypeError):
                pass
        # Fallback: parsear el detail (eventos viejos)
        detail = (r.detail or "").lower()
        for s in by_strategy:
            if s in detail:
                by_strategy[s] += 1

    avg_score = {
        k: (score_sums[k] / score_counts[k]) if score_counts[k] else None
        for k in score_sums
    }

    return {
        "total_flagged": total,
        "by_strategy": by_strategy,
        "score_buckets": score_buckets,
        "average_score_per_strategy": avg_score,
        "hint": (
            "Si la mayoría tiene score 0.90-0.95, podés subir el threshold a 0.97 "
            "para reducir falsos positivos. Andá a Configuración."
        ),
    }


@router.post("/api/restore-duplicates")
async def restore_duplicates(
    confirm: bool = False,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Re-habilita en Vendure todos los productos que Hugo deshabilitó (action='duplicate_disabled').

    Pasale ?confirm=true para ejecutar de verdad. Sin confirm, solo te dice cuántos restauraría.
    """
    stmt = select(AuditLog).where(AuditLog.action == "duplicate_disabled")
    rows = list(session.exec(stmt))
    total = len(rows)
    unique_ids = sorted({r.product_id for r in rows if r.product_id and r.product_id != "(nuevo)"})

    if not confirm:
        return {
            "would_restore": len(unique_ids),
            "total_disable_events": total,
            "preview_ids": unique_ids[:20],
            "hint": "Llamá de nuevo con ?confirm=true para ejecutar",
        }

    client = VendureClient()
    restored: list[str] = []
    failed: list[dict[str, str]] = []
    for pid in unique_ids:
        try:
            await client.enable_product(pid)
            restored.append(pid)
        except Exception as exc:  # noqa: BLE001
            failed.append({"product_id": pid, "error": f"{type(exc).__name__}: {exc}"[:200]})

    # Loguear que se hizo restauración masiva
    session.add(AuditLog(
        action="duplicates_restored_bulk",
        source="manual",
        product_id="(bulk)",
        detail=f"Restauración masiva: {len(restored)} productos re-habilitados, {len(failed)} fallaron.",
    ))
    session.commit()

    return {
        "restored": len(restored),
        "failed": len(failed),
        "failed_details": failed[:10],
    }


@router.get("/api/debug-config")
async def debug_config() -> dict[str, Any]:
    """Diagnóstico: dice qué env vars están seteadas (sin exponer los valores).

    Útil para verificar desde fuera del container que la config está completa
    sin necesidad de entrar a Coolify ni mirar logs.
    """
    from app.config import get_settings
    s = get_settings()

    def _mask(v: str | None) -> dict[str, Any]:
        if not v:
            return {"set": False, "preview": None, "length": 0}
        # mostramos solo los primeros 4 chars y los últimos 2 (suficiente para identificar)
        if len(v) <= 8:
            preview = v[:1] + "***"
        else:
            preview = f"{v[:4]}…{v[-2:]}"
        return {"set": True, "preview": preview, "length": len(v)}

    return {
        "vendure": {
            "api_url": s.vendure_api_url,
            "bearer": _mask(s.vendure_bearer),
            "channel_token": s.vendure_channel_token,
            "user": _mask(s.vendure_user),
            "pass": _mask(s.vendure_pass),
            "source_url_field": s.vendure_source_url_field,
        },
        "rapidapi": {
            "key": _mask(s.rapidapi_key),
            "host": s.otapi_1688_host,
        },
        "hugo_auth": {
            "api_key": _mask(s.hugo_api_key),
        },
        "paco": {
            "url": s.paco_url,
            "api_key": _mask(s.paco_api_key),
            "submit_path": s.paco_submit_path,
            "cf_client_id": _mask(s.paco_cf_client_id),
            "cf_client_secret": _mask(s.paco_cf_client_secret),
        },
        "alerts": {
            "smtp_host": s.alert_smtp_host,
            "smtp_user": _mask(s.alert_smtp_user),
            "smtp_pass": _mask(s.alert_smtp_pass),
            "email_to": s.alert_email_to,
            "webhook_url": s.alert_webhook_url or None,
        },
        "database": {
            "url_starts_with": (s.database_url[:30] + "…") if s.database_url else None,
            "is_postgres": s.database_url.startswith("postgresql"),
        },
    }


@router.get("/api/sections")
async def section_counts(
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Devuelve los conteos de cada sección/tab del dashboard. Excluye eventos descartados."""
    out: dict[str, Any] = {}
    for key, s in SECTIONS.items():
        stmt = (
            select(func.count(AuditLog.id))  # type: ignore[arg-type]
            .where(AuditLog.dismissed.is_not(True))  # type: ignore[union-attr]
        )
        stmt = _apply_section_filter(stmt, key)
        out[key] = {
            "label": s["label"],
            "count": session.exec(stmt).one() or 0,
        }
    return out


@router.get("/api/status")
async def dashboard_status(
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Resumen agregado para el dashboard: conteos, último audit y eventos recientes."""
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)

    # Conteos
    products_tracked = session.exec(
        select(func.count(func.distinct(PriceHistory.product_id)))  # type: ignore[arg-type]
    ).one() or 0
    snapshots_total = session.exec(select(func.count(PriceHistory.id))).one() or 0  # type: ignore[arg-type]
    alerts_24h = session.exec(
        select(func.count(AuditLog.id)).where(  # type: ignore[arg-type]
            AuditLog.action == "price_flagged",
            AuditLog.created_at >= last_24h,
        )
    ).one() or 0
    duplicates_7d = session.exec(
        select(func.count(AuditLog.id)).where(  # type: ignore[arg-type]
            AuditLog.action == "duplicate_disabled",
            AuditLog.created_at >= last_7d,
        )
    ).one() or 0
    last_audit = session.exec(
        select(PriceHistory.captured_at).order_by(PriceHistory.captured_at.desc()).limit(1)
    ).first()

    # Estado actual de los locks para que el dashboard muestre "trabajando"
    from app.scheduler.jobs import audit_dupes_lock, audit_prices_lock
    in_progress = {
        "prices": audit_prices_lock.locked(),
        "duplicates": audit_dupes_lock.locked(),
    }

    # Últimos 15 eventos para mostrar
    recent = session.exec(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(15)
    )
    events = [_humanize(e) for e in recent]

    return {
        "agent": "Hugo",
        "status": "healthy",
        "now": now.isoformat() + "Z",
        "metrics": {
            "products_tracked": products_tracked,
            "snapshots_total": snapshots_total,
            "alerts_last_24h": alerts_24h,
            "duplicates_last_7d": duplicates_7d,
            "audit_in_progress": in_progress,
        },
        "last_audit": (last_audit.isoformat() + "Z") if last_audit else None,
        "recent_events": events,
    }
