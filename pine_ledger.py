"""Prediction ledger: persist every forecast the models produce and evaluate it later."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.engine import Engine

from pine_db import model_runs, predictions

log = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """One forecast point."""

    model_name: str
    model_version: str
    target: str
    as_of: datetime
    target_date: datetime
    value: float
    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    scenario: str = "base"
    meta: dict = field(default_factory=dict)


def records_from_frame(
    df: pd.DataFrame, *, model_name: str, model_version: str, target: str, as_of: datetime,
    unit: str, scenario: str = "base", meta: dict | None = None,
) -> list[PredictionRecord]:
    """Convert a forecast frame (index=target date, columns p50 [p10, p90]) to records."""
    recs = []
    for d, row in df.iterrows():
        recs.append(PredictionRecord(
            model_name=model_name, model_version=model_version, target=target,
            as_of=pd.Timestamp(as_of).to_pydatetime().replace(tzinfo=None),
            target_date=pd.Timestamp(d).to_pydatetime().replace(tzinfo=None),
            value=float(row["p50"]),
            lower=_opt(row.get("p10")), upper=_opt(row.get("p90")),
            unit=unit, scenario=scenario, meta=meta or {},
        ))
    return recs


def _opt(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def save_predictions(
    engine: Engine, recs: list[PredictionRecord], user: str, overwrite: bool = False
) -> int:
    """Insert records; existing keys are skipped (or updated when `overwrite`).

    Returns:
        Number of rows inserted or updated.
    """
    n = 0
    with engine.begin() as conn:
        for r in recs:
            key = and_(
                predictions.c.model_name == r.model_name, predictions.c.target == r.target,
                predictions.c.scenario == r.scenario, predictions.c.as_of == r.as_of,
                predictions.c.target_date == r.target_date,
            )
            existing = conn.execute(select(predictions.c.id).where(key)).first()
            payload = {**asdict(r), "created_by": user}
            if existing is None:
                conn.execute(predictions.insert().values(**payload))
                n += 1
            elif overwrite:
                conn.execute(predictions.update().where(predictions.c.id == existing.id).values(**payload))
                n += 1
    return n


def log_model_run(engine: Engine, model_name: str, version: str, metrics: dict, user: str) -> None:
    """Store validation metrics for a model fit (one row per day per model)."""
    today = pd.Timestamp.today().normalize()
    with engine.begin() as conn:
        rows = conn.execute(
            select(model_runs.c.run_at).where(model_runs.c.model_name == model_name)
            .order_by(model_runs.c.run_at.desc()).limit(1)
        ).first()
        if rows and pd.Timestamp(rows.run_at).tz_localize(None).normalize() >= today:
            return
        clean = {k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)}
        conn.execute(model_runs.insert().values(
            model_name=model_name, model_version=version, metrics=clean, created_by=user
        ))


def load_predictions(engine: Engine, model_name: str | None = None) -> pd.DataFrame:
    """All stored predictions, optionally for one model."""
    q = select(predictions)
    if model_name:
        q = q.where(predictions.c.model_name == model_name)
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(q.order_by(predictions.c.created_at.desc())).mappings().all())
    if df.empty:
        return pd.DataFrame(columns=[c.name for c in predictions.columns])
    for c in ("as_of", "target_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def load_model_runs(engine: Engine) -> pd.DataFrame:
    """Model validation history."""
    with engine.connect() as conn:
        rows = conn.execute(select(model_runs).order_by(model_runs.c.run_at.desc())).mappings().all()
    return pd.DataFrame(rows)


def evaluate_predictions(preds: pd.DataFrame, actuals: dict[str, pd.Series]) -> pd.DataFrame:
    """Join matured predictions with realised values (pure).

    Args:
        preds: frame from `load_predictions`.
        actuals: target name -> realised series indexed by date.

    Returns:
        Matured predictions with `actual`, `abs_pct_error` and `in_band`.
    """
    if preds.empty:
        return preds.assign(actual=pd.Series(dtype=float))
    today = pd.Timestamp.today().normalize()
    mat = preds[preds["target_date"] <= today].copy()
    vals = []
    for _, r in mat.iterrows():
        s = actuals.get(r["target"])
        vals.append(float(s.asof(r["target_date"])) if s is not None and len(s) else np.nan)
    mat["actual"] = vals
    mat = mat.dropna(subset=["actual"])
    mat["abs_pct_error"] = (mat["value"] - mat["actual"]).abs() / mat["actual"].abs()
    lo = mat["lower"].fillna(mat["value"])
    hi = mat["upper"].fillna(mat["value"])
    mat["in_band"] = (mat["actual"] >= lo) & (mat["actual"] <= hi)
    return mat
