"""Database layer (SQLAlchemy Core). SQLite locally, PostgreSQL in production.

Set DATABASE_URL to a managed Postgres (Supabase, Neon, Azure) for Streamlit Community
Cloud: its container filesystem is ephemeral, so a local SQLite file would be lost on restart.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(64), nullable=False, unique=True),
    Column("full_name", String(128), nullable=False),
    Column("role", String(16), nullable=False),
    Column("password_hash", String(128), nullable=False),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("last_login", DateTime(timezone=True)),
)

login_attempts = Table(
    "login_attempts", metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String(64), nullable=False),
    Column("success", Boolean, nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),
    Index("ix_login_attempts_user_ts", "username", "ts"),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("model_name", String(64), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("target", String(64), nullable=False),  # what is predicted, e.g. PINE4.SA close
    Column("scenario", String(64), nullable=False, default="base"),
    Column("as_of", DateTime(timezone=False), nullable=False),  # data date the model saw
    Column("target_date", DateTime(timezone=False), nullable=False),
    Column("value", Float, nullable=False),  # point forecast (median)
    Column("lower", Float),  # P10
    Column("upper", Float),  # P90
    Column("unit", String(16)),
    Column("meta", JSON),
    Column("created_by", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    UniqueConstraint("model_name", "target", "scenario", "as_of", "target_date", name="uq_pred"),
    Index("ix_predictions_model_target", "model_name", "target"),
)

model_runs = Table(
    "model_runs", metadata,
    Column("id", Integer, primary_key=True),
    Column("model_name", String(64), nullable=False),
    Column("model_version", String(32), nullable=False),
    Column("metrics", JSON, nullable=False),
    Column("run_at", DateTime(timezone=True), server_default=func.now()),
    Column("created_by", String(64), nullable=False),
)


@lru_cache(maxsize=4)
def get_engine(url: str) -> Engine:
    """Create (once) the engine and the schema."""
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **kwargs)
    metadata.create_all(engine)
    return engine
