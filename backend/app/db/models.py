"""Modelos SQLModel para Hugo. Tabla local con SQLite por defecto.

Dos tablas core:
  - PriceHistory: cada vez que vemos un precio (en Vendure, en fuente o en
    competidor), lo logueamos. Sirve para análisis y para detectar tendencias.
  - AuditLog: cada acción que toma Hugo (auto-update, disable duplicado,
    flag para revisión, etc.). Sirve para trazabilidad y reportes diarios.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlmodel import Field, SQLModel

PriceSource = Literal["vendure", "source", "competitor"]
ActionType = Literal[
    "price_updated",
    "duplicate_disabled",
    "duplicate_flagged",
    "price_flagged",
    "no_change",
    "error",
]


class PriceHistory(SQLModel, table=True):
    __tablename__ = "price_history"

    id: int | None = Field(default=None, primary_key=True)
    product_id: str = Field(index=True)
    variant_id: str | None = Field(default=None, index=True)
    source: str = Field(description="vendure | source | competitor name")
    price_cents: int
    currency: str = Field(default="USD", max_length=8)
    captured_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    extra: str | None = Field(default=None, description="JSON con info adicional")


class Setting(SQLModel, table=True):
    """Settings runtime editables desde el dashboard.

    Si una key no existe acá, se usa el default del .env (ver app/runtime.py).
    """
    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str  # se guarda siempre como string; el módulo runtime parsea según tipo
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: int | None = Field(default=None, primary_key=True)
    action: str = Field(index=True, description=str(ActionType))
    # De dónde viene el evento — sirve para los tabs del dashboard
    # Valores: "luis" | "audit" | "orders" | "manual"
    source: str | None = Field(default=None, index=True)
    product_id: str = Field(index=True)
    related_product_id: str | None = Field(default=None)
    detail: str = Field(default="", description="Descripción humana de qué pasó")
    before: str | None = Field(default=None, description="JSON del estado previo")
    after: str | None = Field(default=None, description="JSON del estado nuevo")
    confidence: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    notified: bool = Field(default=False, description="Si ya se incluyó en algún email")

    # Datos denormalizados del producto (snapshot al momento del evento, para
    # que el dashboard pueda mostrar miniatura/nombre/código sin repreguntar a Vendure)
    product_name: str | None = Field(default=None)
    product_code: str | None = Field(default=None)        # b2boxProductCode (BX)
    product_image_url: str | None = Field(default=None)
    product_source_url: str | None = Field(default=None)  # link al proveedor
    related_product_name: str | None = Field(default=None)
    related_product_code: str | None = Field(default=None)
    # Acciones manuales del usuario sobre el evento
    dismissed: bool = Field(default=False, index=True, description="Descartado por el usuario")
    dismissed_at: datetime | None = Field(default=None)
