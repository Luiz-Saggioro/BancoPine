"""Page: Prediction ledger - every stored forecast, and how the live ones performed."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pine_datasets as ds
from pine_config import TICKER_LABELS
from pine_ledger import evaluate_predictions, load_model_runs, load_predictions
from ui_theme import SERIES, SURFACE


def _actuals() -> dict[str, pd.Series]:
    px = ds.prices().data
    m = ds.macro()
    out = {t: px[t].dropna() for t in px.columns if t in TICKER_LABELS}
    out.update({"IPCA_12M": m.ipca12, "NPL_PF": m.npl_pf, "NPL_PJ": m.npl_pj})
    return out


def render() -> None:
    """Render the page."""
    st.title("Prediction ledger")
    st.caption("Every forecast is stored once per day per model and scenario (and on demand by analysts). "
               "Forecasts on live targets are scored automatically once their date arrives.")
    preds = load_predictions(ds.engine())
    if preds.empty:
        st.info("No predictions stored yet. Open the analysis pages to generate today's forecasts.")
        return
    models = sorted(preds["model_name"].unique())
    c = st.columns([2, 2, 1])
    pick = c[0].multiselect("Models", models, default=models)
    targets = sorted(preds.loc[preds["model_name"].isin(pick), "target"].unique())
    tpick = c[1].multiselect("Targets", targets, default=[])
    view = preds[preds["model_name"].isin(pick)]
    if tpick:
        view = view[view["target"].isin(tpick)]

    ev = evaluate_predictions(view, _actuals())
    k = st.columns(4)
    k[0].metric("Stored predictions", f"{len(view):,}")
    k[1].metric("Matured and scored", f"{len(ev):,}")
    k[2].metric("Median absolute % error", f"{ev['abs_pct_error'].median():.2%}" if len(ev) else "n/a")
    k[3].metric("Inside P10-P90 band", f"{ev['in_band'].mean():.0%}" if len(ev) else "n/a",
                help="Well-calibrated 80% bands should contain about 80% of outcomes.")

    if len(ev):
        st.markdown("### Accuracy of matured forecasts")
        fig = go.Figure()
        for i, (name, g) in enumerate(ev.groupby("model_name")):
            g = g.sort_values("target_date")
            fig.add_trace(go.Scatter(x=g["target_date"], y=g["abs_pct_error"], mode="markers",
                                     name=name, marker=dict(size=8, color=SERIES[i % 3])))
        fig.update_yaxes(title="Absolute % error", tickformat=".1%")
        fig.update_layout(height=320, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
        st.plotly_chart(fig, theme=None, config={"displaylogo": False})

    st.markdown("### Stored forecasts")
    cols = ["model_name", "model_version", "target", "scenario", "as_of", "target_date", "value",
            "lower", "upper", "unit", "created_by", "created_at"]
    st.dataframe(view[cols], hide_index=True, height=360)
    st.download_button("Download ledger (CSV)", view[cols].to_csv(index=False).encode("utf-8"),
                       file_name="pine_prediction_ledger.csv", mime="text/csv")

    runs = load_model_runs(ds.engine())
    if not runs.empty:
        st.markdown("### Model validation history")
        flat = pd.concat([runs.drop(columns=["metrics"]), pd.json_normalize(runs["metrics"])], axis=1)
        st.dataframe(flat, hide_index=True, height=240)
