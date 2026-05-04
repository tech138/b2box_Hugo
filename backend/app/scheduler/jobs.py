"""Tareas programadas de Hugo.

Tres jobs:
  1. audit_duplicates    → recorre Vendure y deshabilita duplicados.
  2. audit_source_prices → snapshot de precios fuente + alerta cuando cambian.
  3. daily_digest        → email/webhook con el resumen de las últimas 24h.

Optimizaciones clave:
  · Streaming  — procesa cada página de Vendure apenas llega.
  · Paralelo   — asyncio.Semaphore(N) para consultar OTAPI a N productos a la vez.
  · Lock       — un asyncio.Lock por job evita que dos auditorías corran a la vez.

Hugo NO modifica el precio de venta de Vendure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import AsyncIterator

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import Session, select

from app.config import get_settings
from app.db.models import AuditLog, PriceHistory
from app.db.session import engine
from app.dedup.orchestrator import CandidateInput, find_duplicate_in
from app.notifier.dispatcher import notify, notify_digest
from app.pricing.diff import compare_source_snapshots
from app.pricing.source_check import fetch_source_price
from app.vendure.client import VendureClient, VendureProduct

log = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()

# Locks que evitan dos auditorías concurrentes del mismo tipo
audit_prices_lock = asyncio.Lock()
audit_dupes_lock = asyncio.Lock()
audit_quality_lock = asyncio.Lock()

# Cuántos productos consultar a OTAPI a la vez (1688 / RapidAPI tolera bien esto)
PARALLEL_OTAPI = 10


# ─── Helpers ───────────────────────────────────────────────────────


async def _iter_product_pages(
    client: VendureClient, page_size: int = 25,
) -> AsyncIterator[list[VendureProduct]]:
    """Itera por páginas de productos. Hace streaming: yield apenas llega cada página."""
    skip = 0
    while True:
        page = await client.list_products(skip=skip, take=page_size)
        if not page:
            return
        yield page
        if len(page) < page_size:
            return
        skip += page_size


async def _flatten_products(client: VendureClient) -> list[VendureProduct]:
    """Para casos donde necesitamos TODO el catálogo en memoria (ej. duplicados)."""
    out: list[VendureProduct] = []
    async for page in _iter_product_pages(client):
        out.extend(page)
    return out


# ─── Job 1: duplicados ────────────────────────────────────────────


async def audit_duplicates() -> None:
    if audit_dupes_lock.locked():
        log.warning("audit_duplicates ya está corriendo, ignoro la nueva invocación")
        return
    async with audit_dupes_lock:
        log.info("Iniciando audit_duplicates")
        client = VendureClient()
        products = await _flatten_products(client)
        log.info("Catálogo: %d productos a comparar", len(products))

        with Session(engine) as session:
            for i, prod in enumerate(products):
                if not prod.enabled:
                    continue
                try:
                    others = [p for j, p in enumerate(products) if j != i and p.enabled]
                    verdict = await find_duplicate_in(
                        CandidateInput(
                            name=prod.name,
                            description=prod.description,
                            source_url=prod.source_url,
                            image_urls=prod.image_urls,
                        ),
                        others,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("dedup falló para %s: %s", prod.id, exc)
                    continue
                if not (verdict.is_duplicate and verdict.candidate_id):
                    continue
                keep, drop = sorted([prod.id, verdict.candidate_id])
                if drop != prod.id:
                    continue  # el otro producto será visto en su propia iteración
                # Buscar el "kept" en la lista para denormalizar sus datos también
                keep_prod = next((p for p in products if p.id == keep), None)
                # MODO SOLO-FLAGEAR: NO deshabilitamos en Vendure, solo logueamos.
                # El usuario revisa la lista y decide manualmente cuáles deshabilitar.
                # Guardamos los scores detallados para que pueda analizar por qué disparó.
                session.add(AuditLog(
                    action="duplicate_flagged",
                    source="audit",
                    product_id=drop,
                    related_product_id=keep,
                    confidence=verdict.confidence,
                    detail=(
                        f"Posible duplicado de #{keep} por {','.join(verdict.matched_by)} "
                        f"(confianza {verdict.confidence:.0%}). Revisalo manualmente."
                    ),
                    after=json.dumps({
                        "per_strategy_scores": verdict.per_strategy_scores,
                        "matched_by": verdict.matched_by,
                    }),
                    product_name=prod.name,
                    product_code=prod.product_code,
                    product_image_url=prod.featured_image_url,
                    product_source_url=prod.source_url,
                    related_product_name=keep_prod.name if keep_prod else None,
                    related_product_code=keep_prod.product_code if keep_prod else None,
                ))
                session.commit()
        log.info("audit_duplicates terminado")


# ─── Job 2: precios fuente (streaming + paralelo) ─────────────────


async def _process_pricing_for_one(
    prod: VendureProduct, semaphore: asyncio.Semaphore,
) -> tuple[str, str | None]:
    """Procesa un producto. Devuelve (status, alert_text|None).

    status ∈ {"processed", "skipped", "failed"}
    """
    async with semaphore:
        if not prod.source_url:
            return ("skipped", None)
        try:
            quote = await fetch_source_price(prod.source_url)
        except Exception as exc:  # noqa: BLE001
            with Session(engine) as session:
                session.add(AuditLog(
                    action="error",
                    source="audit",
                    product_id=prod.id,
                    detail=f"No se pudo consultar la fuente: {type(exc).__name__}: {exc}"[:300],
                    product_name=prod.name,
                    product_code=prod.product_code,
                    product_image_url=prod.featured_image_url,
                ))
                session.commit()
            log.warning("fetch_source_price falló para %s: %s", prod.id, exc)
            return ("failed", None)
        if not quote:
            return ("skipped", None)

        try:
            with Session(engine) as session:
                # 1) snapshot
                session.add(PriceHistory(
                    product_id=prod.id,
                    source=quote.source,
                    price_cents=quote.price_cents,
                    currency=quote.currency,
                    extra=json.dumps({"usd_price_cents": quote.usd_price_cents}),
                ))
                session.commit()

                # 2) snapshot anterior
                stmt = (
                    select(PriceHistory)
                    .where(
                        PriceHistory.product_id == prod.id,
                        PriceHistory.source == quote.source,
                    )
                    .order_by(PriceHistory.captured_at.desc())
                    .offset(1)
                    .limit(1)
                )
                previous = session.exec(stmt).first()

                # 3) decisión
                decision = compare_source_snapshots(
                    current_price_cents=quote.price_cents,
                    current_currency=quote.currency,
                    previous_price_cents=previous.price_cents if previous else None,
                    previous_currency=previous.currency if previous else None,
                )
                if decision.action in ("first_observation", "ok"):
                    return ("processed", None)

                action_label = "price_flagged" if decision.action != "skip_currency" else "error"
                session.add(AuditLog(
                    action=action_label,
                    source="audit",
                    product_id=prod.id,
                    detail=decision.reason,
                    before=json.dumps({"price_cents": decision.previous_price_cents,
                                       "currency": decision.currency}),
                    after=json.dumps({"price_cents": decision.current_price_cents,
                                      "currency": decision.currency}),
                    product_name=prod.name,
                    product_code=prod.product_code,
                    product_image_url=prod.featured_image_url,
                ))
                session.commit()

                if decision.action in ("alert", "alert_critical"):
                    marker = "[!]" if decision.action == "alert" else "[!!]"
                    return ("processed", (
                        f"{marker} [{prod.id}] {prod.name[:50]} "
                        f"{decision.previous_price_cents/100:.2f} → "
                        f"{decision.current_price_cents/100:.2f} {decision.currency} "
                        f"({decision.drift_pct:+.1%})"
                    ))
                return ("processed", None)
        except Exception as exc:  # noqa: BLE001
            log.exception("Error guardando snapshot para %s: %s", prod.id, exc)
            return ("failed", None)


async def audit_source_prices() -> None:
    if audit_prices_lock.locked():
        log.warning("audit_source_prices ya está corriendo, ignoro la nueva invocación")
        return
    async with audit_prices_lock:
        log.info("Iniciando audit_source_prices (streaming + paralelo)")
        client = VendureClient()
        sem = asyncio.Semaphore(PARALLEL_OTAPI)
        tasks: list[asyncio.Task[tuple[str, str | None]]] = []

        # Streaming: arrancamos a procesar cada página apenas llega
        async for page in _iter_product_pages(client):
            for prod in page:
                tasks.append(asyncio.create_task(_process_pricing_for_one(prod, sem)))

        # Esperamos a que todos terminen
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = sum(1 for r in results if isinstance(r, tuple) and r[0] == "processed")
        skipped = sum(1 for r in results if isinstance(r, tuple) and r[0] == "skipped")
        failed = sum(
            1 for r in results
            if isinstance(r, Exception) or (isinstance(r, tuple) and r[0] == "failed")
        )
        alerts = [r[1] for r in results if isinstance(r, tuple) and r[1]]

        if alerts:
            body = "Cambios de costo detectados en proveedores:\n\n" + "\n".join(alerts)
            await notify("Cambios de precio en 1688", body)

        log.info(
            "audit_source_prices terminado: %d procesados, %d sin fuente, %d errores, %d alertas",
            processed, skipped, failed, len(alerts),
        )


# ─── Job 3: calidad del catálogo (precio 0, sin imagen, etc.) ─────


def _detect_quality_issues(prod: VendureProduct) -> list[str]:
    """Devuelve la lista de problemas detectados en un producto, o lista vacía."""
    issues: list[str] = []
    if not prod.featured_image_url:
        issues.append("sin imagen")
    if prod.first_variant_price_cents == 0:
        issues.append("precio = 0")
    if prod.first_variant_price_cents is None and prod.variant_count == 0:
        issues.append("sin variantes")
    if not prod.name or len(prod.name.strip()) < 3:
        issues.append("nombre vacío o muy corto")
    if not prod.source_url:
        issues.append("sin link de proveedor")
    return issues


async def audit_catalog_quality() -> None:
    if audit_quality_lock.locked():
        log.warning("audit_catalog_quality ya está corriendo, ignoro")
        return
    async with audit_quality_lock:
        log.info("Iniciando audit_catalog_quality")
        client = VendureClient()
        products = await _flatten_products(client)
        log.info("Catálogo: %d productos a revisar", len(products))

        flagged = 0
        with Session(engine) as session:
            for prod in products:
                if not prod.enabled:
                    continue
                issues = _detect_quality_issues(prod)
                if not issues:
                    continue
                price_str = (
                    f"{prod.first_variant_price_cents/100:.2f}"
                    if prod.first_variant_price_cents is not None
                    else "—"
                )
                detail = (
                    f"Producto con problemas: {', '.join(issues)}. "
                    f"Precio actual: {price_str}, imágenes: "
                    f"{'sí' if prod.featured_image_url else 'NO'}"
                )
                session.add(AuditLog(
                    action="quality_issue_found",
                    source="audit",
                    product_id=prod.id,
                    detail=detail[:500],
                    product_name=prod.name,
                    product_code=prod.product_code,
                    product_image_url=prod.featured_image_url,
                    product_source_url=prod.source_url,
                ))
                session.commit()
                flagged += 1

        if flagged:
            await notify(
                f"{flagged} productos con problemas de calidad",
                f"Hugo revisó {len(products)} productos en Vendure y encontró {flagged} con problemas "
                "(sin imagen, precio 0, sin link de proveedor, etc.). "
                "Revisalos en el dashboard → tab 'Problemas de calidad'.",
            )

        log.info(
            "audit_catalog_quality terminado: %d productos revisados, %d flagged",
            len(products), flagged,
        )


# ─── Job 4: digest diario ─────────────────────────────────────────


async def daily_digest() -> None:
    cutoff = datetime.utcnow() - timedelta(hours=24)
    with Session(engine) as session:
        stmt = (
            select(AuditLog)
            .where(AuditLog.created_at >= cutoff, AuditLog.notified == False)  # noqa: E712
        )
        rows = list(session.exec(stmt))
        if not rows:
            return
        await notify_digest(rows)
        for r in rows:
            r.notified = True
            session.add(r)
        session.commit()


# ─── Registro ─────────────────────────────────────────────────────


def register_jobs() -> None:
    s = get_settings()
    scheduler.add_job(
        audit_duplicates,
        IntervalTrigger(hours=s.audit_interval_hours),
        id="audit_duplicates",
        replace_existing=True, coalesce=True, max_instances=1,
    )
    scheduler.add_job(
        audit_source_prices,
        IntervalTrigger(hours=s.audit_interval_hours),
        id="audit_source_prices",
        replace_existing=True, coalesce=True, max_instances=1,
    )
    scheduler.add_job(
        audit_catalog_quality,
        IntervalTrigger(hours=s.audit_interval_hours),
        id="audit_catalog_quality",
        replace_existing=True, coalesce=True, max_instances=1,
    )
    scheduler.add_job(
        daily_digest,
        CronTrigger(hour=9, minute=0),
        id="daily_digest",
        replace_existing=True,
    )
