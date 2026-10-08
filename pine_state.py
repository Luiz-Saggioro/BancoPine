"""Cross-filter state shared by every visual (the glue that makes charts talk to each other).

A click on one chart writes to the filter state; every other chart reads from it. Pure
helpers here; the Streamlit session is passed in so they stay testable.
"""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from typing import Any

import pandas as pd

from pine_config import RATINGS, REGIONS, SEGMENTS, TICKER, TICKER_LABELS

DEFAULTS: dict[str, Any] = {
    "segment": None,
    "region": None,
    "rating": None,
    "ticker": TICKER,
    "window_start": None,
    "window_end": None,
    "selic_shock_bps": 0,
    "scenario": "Base",
    "cohort": None,
}
_ALLOWED = {
    "segment": set(SEGMENTS),
    "region": set(REGIONS),
    "rating": set(RATINGS),
    "ticker": set(TICKER_LABELS),
}
_STATE_KEY = "filters"
_EPOCH_KEY = "filters_epoch"


def filters(session: MutableMapping) -> dict[str, Any]:
    """Current filter dict (initialised with defaults)."""
    if _STATE_KEY not in session:
        session[_STATE_KEY] = dict(DEFAULTS)
        session[_EPOCH_KEY] = 0
    return session[_STATE_KEY]


def set_filter(session: MutableMapping, key: str, value: Any) -> None:
    """Set a filter after whitelisting categorical values (ignores unknown values)."""
    if key not in DEFAULTS:
        raise KeyError(key)
    if value is not None and key in _ALLOWED and value not in _ALLOWED[key]:
        return
    filters(session)[key] = value


def clear_filters(session: MutableMapping, keep: tuple[str, ...] = ("selic_shock_bps", "scenario")) -> None:
    """Reset filters and bump the epoch so chart selections are visually reset too."""
    cur = filters(session)
    session[_STATE_KEY] = {k: (cur[k] if k in keep else v) for k, v in DEFAULTS.items()}
    session[_EPOCH_KEY] = session.get(_EPOCH_KEY, 0) + 1


def chart_key(session: MutableMapping, name: str) -> str:
    """Widget key that changes when filters are cleared (drops stale selections)."""
    filters(session)
    return f"{name}_e{session[_EPOCH_KEY]}"


def selection_changed(session: MutableMapping, name: str, points: list[dict]) -> bool:
    """True only the first time a given selection is seen for a chart."""
    sig = json.dumps(points, sort_keys=True, default=str)
    store = session.setdefault("_sel_sigs", {})
    if store.get(name) == sig:
        return False
    store[name] = sig
    return bool(points)


def apply_book_filters(book: pd.DataFrame, flt: dict[str, Any]) -> pd.DataFrame:
    """Filter the loan book by segment / region / rating (pure)."""
    m = pd.Series(True, index=book.index)
    for key in ("segment", "region", "rating"):
        if flt.get(key):
            m &= book[key] == flt[key]
    return book[m]


def active_filter_labels(flt: dict[str, Any]) -> list[str]:
    """Human-readable list of non-default filters."""
    out = []
    for k in ("segment", "region", "rating", "cohort"):
        if flt.get(k):
            out.append(f"{k.title()}: {flt[k]}")
    if flt.get("ticker") and flt["ticker"] != TICKER:
        out.append(f"Ticker: {TICKER_LABELS.get(flt['ticker'], flt['ticker'])}")
    if flt.get("window_start"):
        out.append(f"Window: {flt['window_start']:%d %b %Y} - {flt['window_end']:%d %b %Y}")
    if flt.get("selic_shock_bps"):
        out.append(f"Selic shock: {flt['selic_shock_bps']:+d} bp")
    return out
