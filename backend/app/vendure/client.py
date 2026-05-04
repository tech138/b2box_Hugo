"""Cliente GraphQL para la Admin API de Vendure.

Hugo necesita:
  - leer productos (con custom fields e imagen principal)
  - desactivar duplicados (updateProduct → enabled: false)

El bearer de Vendure expira (típicamente cada 12h). Cuando recibimos un error
de auth, automáticamente hacemos login con VENDURE_USER/VENDURE_PASS, obtenemos
un bearer nuevo del header `vendure-auth-token`, actualizamos el transport, y
reintentamos la request original. El bearer renovado vive en memoria (si Hugo
restartea, hace login al primer call de nuevo).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from gql import Client, gql
from gql.transport.exceptions import TransportError, TransportQueryError
from gql.transport.httpx import HTTPXAsyncTransport

from app.config import get_settings

log = logging.getLogger(__name__)


# ─── DTOs ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class VendureProduct:
    """Vista mínima de un producto Vendure que Hugo necesita."""

    id: str
    name: str
    slug: str
    description: str
    enabled: bool
    source_url: str | None
    image_urls: list[str]
    product_code: str | None  # b2boxProductCode (BX)
    featured_image_url: str | None  # primera imagen (preview/source)
    first_variant_price_cents: int | None  # precio de la 1ra variante (centavos)
    variant_count: int  # cuántas variantes tiene


# ─── Cliente ───────────────────────────────────────────────────────


# Mensajes de error de auth devueltos por Vendure cuando el bearer expiró
_AUTH_ERROR_HINTS = (
    "FORBIDDEN",
    "UNAUTHORIZED",
    "NOT_VERIFIED",
    "no token",
    "session has expired",
    "invalid token",
)


class VendureClient:
    """Wrapper async sobre la Admin API de Vendure con auto-renovación del bearer."""

    DEFAULT_PAGE_SIZE = 25

    def __init__(self) -> None:
        s = get_settings()
        self._url = s.vendure_api_url
        self._channel_token = s.vendure_channel_token
        self._user = s.vendure_user
        self._pass = s.vendure_pass
        self._source_field = s.vendure_source_url_field
        # Bearer actual: arranca con el del .env, se reemplaza cuando se renueva
        self._bearer: str = s.vendure_bearer or ""
        self._login_lock = asyncio.Lock()  # evita re-logins concurrentes
        if not self._bearer:
            log.info(
                "VENDURE_BEARER vacío — Hugo se va a loguear con user/pass al primer call"
            )
        self._build_client()

    def _build_client(self) -> None:
        """(Re)crea el gql.Client con el bearer actual."""
        headers = {"Authorization": f"Bearer {self._bearer}"}
        if self._channel_token:
            headers["vendure-token"] = self._channel_token
        transport = HTTPXAsyncTransport(
            url=self._url,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._client = Client(transport=transport, fetch_schema_from_transport=False)

    async def _login(self) -> bool:
        """Hace login con VENDURE_USER/PASS y guarda el nuevo bearer.

        Devuelve True si el login fue exitoso, False si falla (típicamente
        porque no hay credenciales configuradas).
        """
        if not self._user or not self._pass:
            log.error("VENDURE_USER/PASS no configurados — no puedo renovar el bearer")
            return False

        async with self._login_lock:
            log.info("Renovando bearer de Vendure (login con usuario %s)", self._user)
            mutation = (
                "mutation Login($u: String!, $p: String!) { "
                "  login(username: $u, password: $p, rememberMe: true) { "
                "    __typename "
                "    ... on CurrentUser { id identifier } "
                "    ... on InvalidCredentialsError { message } "
                "    ... on NativeAuthStrategyError { message } "
                "  } "
                "}"
            )
            headers = {"Content-Type": "application/json"}
            if self._channel_token:
                headers["vendure-token"] = self._channel_token

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        self._url,
                        json={
                            "query": mutation,
                            "variables": {"u": self._user, "p": self._pass},
                        },
                        headers=headers,
                    )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                log.error("Login a Vendure falló: %s", exc)
                return False

            new_bearer = resp.headers.get("vendure-auth-token")
            if not new_bearer:
                log.error(
                    "Login OK pero sin header vendure-auth-token. "
                    "Vendure tiene que estar configurado con bearer auth (no cookie). Body: %s",
                    resp.text[:200],
                )
                return False

            data = resp.json().get("data", {}).get("login", {})
            if data.get("__typename") != "CurrentUser":
                log.error("Login devolvió error: %s", data.get("message") or data)
                return False

            self._bearer = new_bearer
            self._build_client()
            log.info("Bearer de Vendure renovado OK")
            return True

    @staticmethod
    def _is_auth_error(exc: Exception) -> bool:
        """Detecta si la excepción es por auth/token expirado."""
        msg = str(exc).upper()
        return any(hint.upper() in msg for hint in _AUTH_ERROR_HINTS) or "401" in msg

    # ── Lectura ────────────────────────────────────────────────

    async def list_products(
        self,
        skip: int = 0,
        take: int = DEFAULT_PAGE_SIZE,
    ) -> list[VendureProduct]:
        query = gql(
            f"""
            query Products($skip: Int!, $take: Int!) {{
              products(options: {{ skip: $skip, take: $take }}) {{
                items {{
                  id
                  name
                  slug
                  description
                  enabled
                  customFields {{ {self._source_field} b2boxProductCode }}
                  featuredAsset {{ source preview }}
                  variantList(options: {{ take: 1 }}) {{
                    items {{ priceWithTax }}
                    totalItems
                  }}
                }}
                totalItems
              }}
            }}
            """
        )
        data = await self._execute_with_retry(
            query, {"skip": skip, "take": take}, what=f"list_products(skip={skip})"
        )
        return [self._map_product(p) for p in (data.get("products", {}).get("items") or [])]

    async def get_product(self, product_id: str) -> VendureProduct | None:
        query = gql(
            f"""
            query Product($id: ID!) {{
              product(id: $id) {{
                id
                name
                slug
                description
                enabled
                customFields {{ {self._source_field} b2boxProductCode }}
                featuredAsset {{ source preview }}
              }}
            }}
            """
        )
        data = await self._execute_with_retry(
            query, {"id": product_id}, what=f"get_product({product_id})"
        )
        return self._map_product(data["product"]) if data.get("product") else None

    # ── Escritura ──────────────────────────────────────────────

    async def disable_product(self, product_id: str) -> None:
        await self._set_enabled(product_id, False)

    async def enable_product(self, product_id: str) -> None:
        await self._set_enabled(product_id, True)

    async def _set_enabled(self, product_id: str, enabled: bool) -> None:
        mutation = gql(
            """
            mutation SetEnabled($input: UpdateProductInput!) {
              updateProduct(input: $input) { id enabled }
            }
            """
        )
        await self._execute_with_retry(
            mutation,
            {"input": {"id": product_id, "enabled": enabled}},
            what=f"set_enabled({product_id}, {enabled})",
        )

    # ── Helpers ────────────────────────────────────────────────

    async def _execute_with_retry(
        self,
        query,
        variables: dict[str, Any],
        what: str,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Ejecuta la query con retry exponencial + auto-renovación del bearer."""
        # Si arrancamos sin bearer, hacemos login proactivo antes del primer call
        if not self._bearer:
            await self._login()

        last_exc: Exception | None = None
        relogged_in = False
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._client as session:
                    return await session.execute(query, variable_values=variables)
            except (TransportError, TransportQueryError, httpx.HTTPError) as exc:
                last_exc = exc
                # Si el error parece ser de auth y todavía no intentamos renovar,
                # hacemos login y retry SIN consumir attempts del backoff
                if self._is_auth_error(exc) and not relogged_in:
                    log.warning(
                        "%s tiró error de auth, intento renovar el bearer", what
                    )
                    relogged_in = True
                    if await self._login():
                        continue  # retry inmediato con bearer nuevo
                if attempt < max_attempts:
                    backoff = 2 ** (attempt - 1)
                    log.warning(
                        "%s falló (intento %d/%d): %s — reintento en %ds",
                        what, attempt, max_attempts, type(exc).__name__, backoff,
                    )
                    await asyncio.sleep(backoff)
        log.error("%s falló definitivamente tras %d intentos", what, max_attempts)
        raise last_exc  # type: ignore[misc]

    def _map_product(self, raw: dict[str, Any]) -> VendureProduct:
        custom = raw.get("customFields") or {}
        featured = raw.get("featuredAsset") or {}
        featured_preview = featured.get("preview") or featured.get("source")
        image_urls: list[str] = []
        if featured.get("source"):
            image_urls.append(featured["source"])
        # Precio + cantidad de variantes (puede no venir en queries antiguas)
        variant_list = raw.get("variantList") or {}
        variant_items = variant_list.get("items") or []
        first_price = None
        if variant_items:
            try:
                first_price = int(variant_items[0].get("priceWithTax") or 0)
            except (TypeError, ValueError):
                first_price = None
        return VendureProduct(
            id=str(raw["id"]),
            name=raw.get("name", ""),
            slug=raw.get("slug", ""),
            description=raw.get("description", "") or "",
            enabled=bool(raw.get("enabled", True)),
            source_url=custom.get(self._source_field),
            image_urls=image_urls,
            product_code=custom.get("b2boxProductCode"),
            featured_image_url=featured_preview,
            first_variant_price_cents=first_price,
            variant_count=int(variant_list.get("totalItems") or len(variant_items)),
        )
