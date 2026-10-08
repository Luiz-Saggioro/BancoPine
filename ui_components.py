"""Reusable Streamlit building blocks: headers, provenance tags, selectable charts, ledger hooks."""

from __future__ import annotations

import html
import logging
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pine_state as state
from pine_auth import User
from pine_datasets import MODEL_VERSION, engine
from pine_ledger import PredictionRecord, log_model_run, save_predictions

log = logging.getLogger(__name__)


def current_user() -> User:
    """The signed-in user (pages are only reachable after login)."""
    return st.session_state["user"]


def _md_safe(text: str) -> str:
    """Escape '$' so currency (R$) is never parsed as LaTeX by Streamlit markdown."""
    return text.replace("$", "\\$")


def caption(text: str) -> None:
    """st.caption with currency-safe escaping."""
    st.caption(_md_safe(text))


def visual_header(number: int, title: str, caption: str) -> None:
    """Numbered section title with a one-line explanation of how to use the visual."""
    st.markdown(f"### {number:02d}  {_md_safe(html.escape(title))}")
    safe = html.escape(caption).replace("$", "&#36;")
    st.markdown(f'<p class="pine-caption">{safe}</p>', unsafe_allow_html=True)


def source_tags(sources: dict[str, bool]) -> None:
    """Render provenance tags (live vs fallback) for the data on the page."""
    tags = []
    for name, is_live in sources.items():
        cls, txt = ("live", "live") if is_live else ("fallback", "fallback")
        tags.append(f'<span class="pine-tag {cls}">{html.escape(name)}: {txt}</span>')
    st.markdown("".join(tags), unsafe_allow_html=True)


def selectable_chart(fig: go.Figure, name: str, mode: tuple[str, ...] = ("points",)) -> list[dict]:
    """Render a chart that reports clicks/selections. Returns *new* selected points only."""
    key = state.chart_key(st.session_state, name)
    event = st.plotly_chart(fig, key=key, on_select="rerun", selection_mode=mode,
                            theme=None, config={"displaylogo": False})
    sel = (event or {}).get("selection", {}) if isinstance(event, dict) else getattr(event, "selection", {})
    points = list(sel.get("points", []) or [])
    box = list(sel.get("box", []) or [])
    payload = points or [{"box": b} for b in box]
    return payload if state.selection_changed(st.session_state, name, payload) else []


def point_value(point: dict, field: str = "customdata", index: int | None = None):
    """Extract a value from a Plotly selection point (customdata can be scalar or list)."""
    v = point.get(field)
    if isinstance(v, (list, tuple)) and index is not None:
        return v[index] if len(v) > index else None
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def empty_state(message: str) -> None:
    """Friendly message instead of an empty chart."""
    st.info(message)


def autosave(records: list[PredictionRecord], model_name: str, metrics: dict | None = None) -> None:
    """Idempotently store today's forecasts (and model metrics) in the prediction ledger."""
    user = current_user()
    tag = f"_saved_{model_name}_{records[0].as_of:%Y%m%d}_{records[0].scenario}" if records else None
    if not records or st.session_state.get(tag):
        return
    try:
        save_predictions(engine(), records, user.username, overwrite=False)
        if metrics:
            log_model_run(engine(), model_name, MODEL_VERSION, metrics, user.username)
        st.session_state[tag] = True
    except Exception:  # noqa: BLE001 - ledger failure must not break the page
        log.exception("Prediction ledger write failed for %s", model_name)


def save_button(records: list[PredictionRecord], label: str, key: str) -> None:
    """Explicit 'save scenario' action for analysts (overwrites same key)."""
    user = current_user()
    if not user.can_write:
        return
    if st.button(label, key=key, type="secondary"):
        n = save_predictions(engine(), records, user.username, overwrite=True)
        st.toast(f"Saved {n} prediction rows to the ledger.")


def sidebar_controls() -> None:
    """Global scenario controls and the active cross-filters, shown on every page."""
    flt = state.filters(st.session_state)
    with st.sidebar:
        st.markdown("**Scenario**")
        shock = st.slider("Selic shock vs Focus path (bp)", -300, 300,
                          int(flt["selic_shock_bps"]), step=25,
                          help="Feeds the delinquency, ECL, NII and pricing models.")
        state.set_filter(st.session_state, "selic_shock_bps", int(shock))
        scen = st.radio("Credit scenario", ["Base", "Adverse", "Severe"], horizontal=True,
                        index=["Base", "Adverse", "Severe"].index(flt["scenario"]))
        state.set_filter(st.session_state, "scenario", scen)
        st.markdown("**Active filters**")
        labels = state.active_filter_labels(flt)
        if labels:
            st.markdown("".join(f'<span class="pine-tag">{html.escape(t)}</span>' for t in labels),
                        unsafe_allow_html=True)
        else:
            st.caption("None. Click any bar, bubble or line to filter the other visuals.")
        if st.button("Clear filters", width="stretch"):
            state.clear_filters(st.session_state)
            st.rerun()


def as_of_today() -> datetime:
    """Normalised timestamp for 'today' used as the prediction as-of date."""
    return pd.Timestamp.today().normalize().to_pydatetime()
