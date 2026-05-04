"""Engine + session factory + migración liviana de columnas para SQLite."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

log = logging.getLogger(__name__)

_settings = get_settings()
_is_sqlite = _settings.database_url.startswith("sqlite")
_connect_args: dict = {"check_same_thread": False} if _is_sqlite else {}

# Para Postgres: pool sano sin desconexiones colgadas en hosts efímeros (Coolify).
_engine_kwargs: dict = {"echo": False, "connect_args": _connect_args}
if not _is_sqlite:
    _engine_kwargs.update(
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=300,
    )

engine = create_engine(_settings.database_url, **_engine_kwargs)


# ─── Migración liviana de columnas ──────────────────────────────


def _add_missing_columns() -> None:
    """Para cada tabla del modelo, detecta columnas faltantes y las agrega.

    Funciona tanto en SQLite como en Postgres. Si el modelo declara un default
    Python (ej. False, 0, ""), también seteamos ese valor en las filas existentes
    para que no queden con NULL (que rompe los WHERE x = False posteriores).
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    dialect = engine.dialect

    with engine.begin() as conn:
        for table_name, table in SQLModel.metadata.tables.items():
            if table_name not in existing_tables:
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name in existing_cols:
                    continue
                col_type = col.type.compile(dialect=dialect)
                ddl = f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {col_type}'
                log.warning("Migración: agregando columna %s.%s (%s)",
                            table_name, col.name, col_type)
                conn.execute(text(ddl))
                # Si la columna tiene default escalar, llenar filas existentes
                default_val = getattr(col.default, "arg", None)
                if default_val is not None and not callable(default_val):
                    placeholder_val = default_val
                    conn.execute(
                        text(f'UPDATE "{table_name}" SET "{col.name}" = :v WHERE "{col.name}" IS NULL'),
                        {"v": placeholder_val},
                    )
                    log.warning("Migración: backfill %s.%s = %r en filas existentes",
                                table_name, col.name, placeholder_val)


def init_db() -> None:
    """Crea tablas si no existen, y agrega columnas faltantes a tablas existentes."""
    # Asegurar que los modelos están registrados
    from app.db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def get_session() -> Iterator[Session]:
    """Dependency-injection style para FastAPI."""
    with Session(engine) as session:
        yield session
