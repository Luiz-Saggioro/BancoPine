"""Chart builders: data in, Plotly figure out. No I/O, no model fitting here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pine_config import BASEL_MINIMUM, RATINGS, REGIONS, SEGMENTS
from ui_theme import (
    INK_2,
    MUTED,
    NEUTRAL,
    SCENARIO_COLORS,
    SEGMENT_COLORS,
    SEQ_BLUE,
    SERIES,
    STATUS,
    SURFACE,
    rgba,
)

PCT = ".1%"


def _band(fig: go.Figure, x, lo, hi, color: str, name: str, row=None, col=None) -> None:
    """Shaded P10-P90 band (two traces, second fills to first)."""
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, 0.16), name=name, hoverinfo="skip"), **kw)


def _layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    """Apply size and explicit surfaces; subplot figures get the legend below the plot."""
    has_titles = bool(fig.layout.annotations) and any(
        a.yref == "paper" and (a.y or 0) >= 1 for a in fig.layout.annotations
    )
    base = dict(height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE)
    if has_titles and "legend" not in kw:
        base["legend"] = dict(orientation="h", yanchor="top", y=-0.1, x=0)
        base["margin"] = dict(l=64, r=16, t=40, b=80)
    fig.update_layout(**base, **kw)
    return fig


def _same(a: pd.Series, b: pd.Series) -> bool:
    """True when two paths are numerically identical (skip redundant traces)."""
    return len(a) == len(b) and np.allclose(np.asarray(a, float), np.asarray(b, float))


# 1 -------------------------------------------------------------- price forecast
def price_forecast_fig(close: pd.Series, fc: pd.DataFrame, name: str) -> go.Figure:
    """History plus P10/P50/P90 fan for the next 20 business days."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name=f"{name} close",
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate="%{x|%d %b %Y}<br>R$ %{y:.2f}<extra></extra>"))
    _band(fig, fc.index, fc["p10"], fc["p90"], SERIES[0], "P10-P90 forecast")
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p50"], mode="lines", name="Median forecast",
                             line=dict(color=SERIES[0], width=2, dash="dot"),
                             hovertemplate="%{x|%d %b %Y}<br>Median R$ %{y:.2f}<extra></extra>"))
    last = fc.iloc[-1]
    fig.add_annotation(x=fc.index[-1], y=last["p50"], text=f"R$ {last['p50']:.2f}",
                       showarrow=False, xanchor="left", xshift=6, font=dict(color=INK_2))
    fig.update_yaxes(title="Price (R$)", tickformat=".2f")
    fig.update_xaxes(title=None)
    return _layout(fig, 400, hovermode="x unified", dragmode="select",
                   selectdirection="h")


# 2 --------------------------------------------------------------------- peers
def peer_fig(df: pd.DataFrame, selected: str) -> go.Figure:
    """Realised window return (bars) next to the model's 20-day outlook (dot + P10-P90)."""
    df = df.sort_values("window_ret")
    colors = [SERIES[0] if t == selected else NEUTRAL for t in df["ticker"]]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("Return over selected window", "Model 20-day outlook (P10-P90)"))
    fig.add_trace(go.Bar(x=df["window_ret"], y=df["label"], orientation="h", marker_color=colors,
                         customdata=df["ticker"], name="Window return",
                         hovertemplate="%{y}: %{x:.1%}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(
        x=df["p50"], y=df["label"], mode="markers", name="Median 20d",
        marker=dict(size=9, color=colors, line=dict(color="#ffffff", width=2)),
        error_x=dict(type="data", symmetric=False, array=df["p90"] - df["p50"],
                     arrayminus=df["p50"] - df["p10"], color=MUTED, thickness=1.5, width=0),
        customdata=df["ticker"],
        hovertemplate="%{y}<br>Median %{x:.1%}<extra></extra>"), 1, 2)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1), row=1, col=2)
    fig.update_xaxes(tickformat=".0%")
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 360, showlegend=False, clickmode="event+select")


# 3 -------------------------------------------------------------------- regimes
REGIME_COLORS = {"Calm": SERIES[0], "Normal": SERIES[2], "Stressed": SERIES[1]}


def regime_fig(close: pd.Series, reg: pd.DataFrame, name: str) -> go.Figure:
    """Price colored by detected regime, with the stressed-regime probability below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    joined = reg.join(close.rename("close"), how="inner")
    for r, color in REGIME_COLORS.items():
        part = joined[joined["regime"] == r]
        fig.add_trace(go.Scatter(x=part.index, y=part["close"], mode="markers", name=r,
                                 marker=dict(size=5, color=color),
                                 hovertemplate=f"{r}<br>%{{x|%d %b %Y}}<br>R$ %{{y:.2f}}<extra></extra>"),
                      1, 1)
    fig.add_trace(go.Scatter(x=joined.index, y=joined["p_stressed"], mode="lines",
                             line=dict(color=SERIES[1], width=1.5), fill="tozeroy",
                             fillcolor=rgba(SERIES[1], 0.12), name="P(stressed)",
                             showlegend=False,
                             hovertemplate="%{x|%d %b %Y}<br>P(stressed) %{y:.0%}<extra></extra>"),
                  2, 1)
    fig.update_yaxes(title=f"{name} (R$)", row=1, col=1)
    fig.update_yaxes(title="P(stressed)", tickformat=".0%", range=[0, 1], row=2, col=1)
    return _layout(fig, 420, hovermode="closest")


# 4 ---------------------------------------------------------------------- macro
def macro_fig(selic: pd.Series, paths: dict[str, pd.Series], ipca12: pd.Series,
              ipca_fc: pd.DataFrame, focus_ipca: pd.DataFrame) -> go.Figure:
    """Selic history + forward paths, and IPCA 12m history + SARIMA fan vs Focus."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("Selic target (% p.a.)", "IPCA, 12-month (%)"))
    s = selic[selic.index >= selic.index[-1] - pd.DateOffset(years=5)]
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name="Selic (actual)",
                             line=dict(color=SERIES[0], width=2, shape="hv")), 1, 1)
    styles = [("dot", SERIES[0]), ("dash", SERIES[1]), ("dot", MUTED)]
    first = next(iter(paths.values()))
    for k, ((name, p), (dash, c)) in enumerate(zip(paths.items(), styles, strict=False)):
        if k and _same(p, first):
            continue
        fig.add_trace(go.Scatter(x=p.index, y=p, mode="lines", name=name,
                                 line=dict(color=c, width=2, dash=dash)), 1, 1)
    i = ipca12[ipca12.index >= ipca12.index[-1] - pd.DateOffset(years=5)]
    fig.add_trace(go.Scatter(x=i.index, y=i, mode="lines", name="IPCA 12m (actual)",
                             line=dict(color=SERIES[2], width=2)), 1, 2)
    _band(fig, ipca_fc.index, ipca_fc["p10"], ipca_fc["p90"], SERIES[2], "SARIMA P10-P90", 1, 2)
    fig.add_trace(go.Scatter(x=ipca_fc.index, y=ipca_fc["p50"], mode="lines", name="SARIMA median",
                             line=dict(color=SERIES[2], width=2, dash="dot")), 1, 2)
    fx = pd.to_datetime(focus_ipca["year"].astype(int).astype(str) + "-12-01")
    fig.add_trace(go.Scatter(x=fx, y=focus_ipca["median"], mode="markers", name="Focus year-end",
                             marker=dict(size=9, symbol="diamond", color=SERIES[6],
                                         line=dict(color="#fff", width=2))), 1, 2)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 400, hovermode="x unified")


# 5 ------------------------------------------------------------------------ NPL
def npl_fig(npl_pf: pd.Series, npl_pj: pd.Series, fc_pf: pd.DataFrame, fc_pj: pd.DataFrame) -> go.Figure:
    """System NPL 90+ for households and companies with scenario forecasts."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, shared_yaxes=False,
                        subplot_titles=("Households (PF), NPL 90+ %", "Companies (PJ), NPL 90+ %"))
    for col, (hist, fc) in enumerate(((npl_pf, fc_pf), (npl_pj, fc_pj)), start=1):
        h = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=4)]
        fig.add_trace(go.Scatter(x=h.index, y=h, mode="lines", name="Actual",
                                 line=dict(color=INK_2, width=2), showlegend=col == 1), 1, col)
        for j, name in enumerate(fc.columns):
            if j and _same(fc[name], fc.iloc[:, 0]):
                continue
            x = [h.index[-1], *fc.index]
            y = [h.iloc[-1], *fc[name]]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, showlegend=col == 1,
                                     line=dict(color=[SERIES[0], SERIES[1], MUTED][j % 3],
                                               width=2, dash="dot")), 1, col)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 360, hovermode="x unified")


# 6 ------------------------------------------------------------- exposure matrix
def exposure_matrix_fig(agg: pd.DataFrame, sel_segment: str | None, sel_region: str | None) -> go.Figure:
    """Bubble matrix: segment x region, size = exposure, color = model PD."""
    sizes = np.sqrt(agg["balance"] / agg["balance"].max()) * 52 + 6
    selected = (
        (agg["segment"] == sel_segment if sel_segment else True)
        & (agg["region"] == sel_region if sel_region else True)
    )
    opacity = np.where(selected, 1.0, 0.25) if (sel_segment or sel_region) else 1.0
    fig = go.Figure(go.Scatter(
        x=agg["region"], y=agg["segment"], mode="markers",
        marker=dict(size=sizes, color=agg["pd"], colorscale=SEQ_BLUE, opacity=opacity,
                    cmin=0, cmax=float(agg["pd"].quantile(0.95)),
                    line=dict(color="#ffffff", width=2),
                    colorbar=dict(title=dict(text="Model PD", side="right"), tickformat=".0%",
                                  thickness=10, outlinewidth=0)),
        customdata=np.stack([agg["segment"], agg["region"], agg["balance"] / 1e9, agg["pd"]], axis=1),
        hovertemplate="%{customdata[0]} / %{customdata[1]}<br>Exposure R$ %{customdata[2]:.2f} bn"
                      "<br>Model PD %{customdata[3]:.2%}<extra></extra>",
    ))
    fig.update_xaxes(categoryorder="array", categoryarray=REGIONS, title=None, showgrid=False)
    fig.update_yaxes(categoryorder="array", categoryarray=SEGMENTS[::-1], title=None, showgrid=False)
    return _layout(fig, 400, clickmode="event+select")


# 7 ---------------------------------------------------------------------- growth
def growth_fig(hist: pd.DataFrame, fc: pd.DataFrame, segment: str | None) -> go.Figure:
    """Stacked quarterly balances by segment and a damped-trend forecast of the total."""
    fig = go.Figure()
    segs = [segment] if segment else SEGMENTS
    for s in segs:
        part = hist[hist["segment"] == s]
        fig.add_trace(go.Bar(x=part["quarter"], y=part["balance_bn"], name=s,
                             marker_color=SEGMENT_COLORS[s], customdata=[s] * len(part),
                             hovertemplate=f"{s}<br>%{{x|%b %Y}}: R$ %{{y:.2f}} bn<extra></extra>"))
    total = hist[hist["segment"].isin(segs)].groupby("quarter")["balance_bn"].sum()
    _band(fig, fc.index, fc["p10"], fc["p90"], INK_2, "Forecast P10-P90")
    fig.add_trace(go.Scatter(x=[total.index[-1], *fc.index], y=[total.iloc[-1], *fc["p50"]],
                             mode="lines+markers", name="Forecast median",
                             line=dict(color=INK_2, width=2, dash="dot"), marker=dict(size=8),
                             hovertemplate="%{x|%b %Y}: R$ %{y:.2f} bn<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_yaxes(title="Balance (R$ bn)")
    return _layout(fig, 400, hovermode="x unified", clickmode="event+select")


# 8 --------------------------------------------------------------------- PD model
def pd_model_fig(calib: pd.DataFrame, importance: pd.DataFrame) -> go.Figure:
    """Calibration (predicted vs observed default) and permutation importance."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.26,
                        subplot_titles=("Calibration: predicted vs observed", "What drives default risk"))
    hi = float(max(calib["predicted"].max(), calib["observed"].max(), 0.01)) * 1.1
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                             name="Perfect calibration", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=calib["predicted"], y=calib["observed"], mode="lines+markers",
                             line=dict(color=SERIES[0], width=2), marker=dict(size=8),
                             name="Model", hovertemplate="Predicted %{x:.2%}<br>Observed %{y:.2%}<extra></extra>"),
                  1, 1)
    imp = importance.head(8).iloc[::-1]
    fig.add_trace(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color=SERIES[0],
                         name="AUC drop", hovertemplate="%{y}: AUC drop %{x:.3f}<extra></extra>"), 1, 2)
    fig.update_xaxes(tickformat=".0%", title="Predicted PD", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", title="Observed default rate", row=1, col=1)
    fig.update_xaxes(title="AUC drop when shuffled", row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 380, showlegend=False)


# 9 -------------------------------------------------------------------- migration
def migration_fig(matrix: pd.DataFrame, dist: pd.DataFrame, selected: str | None) -> go.Figure:
    """12-month transition heatmap and projected balance distribution by rating."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.52, 0.48], horizontal_spacing=0.1,
                        subplot_titles=("12-month migration (from row to column)",
                                        "Balance by rating: today vs projected"))
    z = matrix.to_numpy()
    scale = [[0.0, SURFACE], [0.004, SEQ_BLUE[0]]] + [
        [(i + 1) / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE[1:])
    ]
    scale[-1][0] = 1.0
    fig.add_trace(go.Heatmap(z=z, x=RATINGS, y=RATINGS, colorscale=scale, showscale=False,
                             zmin=0, zmax=1,
                             text=np.vectorize(lambda v: f"{v:.0%}" if v >= 0.005 else "")(z),
                             texttemplate="%{text}", textfont=dict(size=10),
                             hovertemplate="From %{y} to %{x}: %{z:.1%}<extra></extra>",
                             xgap=2, ygap=2), 1, 1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    for j, col in enumerate(dist.columns):
        fig.add_trace(go.Bar(x=dist.index, y=dist[col], name=col, customdata=dist.index,
                             marker_color=[SERIES[0], SERIES[2], SERIES[1]][j % 3],
                             marker_opacity=[1.0 if (selected is None or r == selected) else 0.3
                                             for r in dist.index],
                             hovertemplate=f"{col}<br>%{{x}}: %{{y:.1%}}<extra></extra>"), 1, 2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 400, barmode="group", clickmode="event+select")


# 10 ---------------------------------------------------------------------- ECL
def ecl_fig(ecl: pd.DataFrame, selected: str | None) -> go.Figure:
    """ECL by segment for each macro scenario (R$ mn), grouped bars."""
    fig = go.Figure()
    for scen, color in SCENARIO_COLORS.items():
        part = ecl[ecl["scenario"] == scen]
        fig.add_trace(go.Bar(
            x=part["segment"], y=part["ecl_mn"], name=scen, marker_color=color,
            customdata=np.stack([part["segment"], part["ecl_rate"]], axis=1),
            marker_opacity=[1.0 if (selected is None or s == selected) else 0.3 for s in part["segment"]],
            hovertemplate=f"{scen}<br>%{{x}}: R$ %{{y:,.0f}} mn (%{{customdata[1]:.2%}} of EAD)<extra></extra>",
        ))
    fig.update_yaxes(title="12-month ECL (R$ mn)")
    fig.update_xaxes(categoryorder="array", categoryarray=SEGMENTS)
    return _layout(fig, 380, barmode="group", clickmode="event+select")


# 11 ------------------------------------------------------------------ vintages
def vintage_fig(v: pd.DataFrame, selected: str | None) -> go.Figure:
    """Cumulative over-90 by months on book; newer cohorts darker; dashed = model."""
    cohorts = sorted(v["cohort"].unique())
    ramp = np.linspace(1, len(SEQ_BLUE) - 1, len(cohorts)).round().astype(int)
    fig = go.Figure()
    for c, k in zip(cohorts, ramp, strict=True):
        g = v[v["cohort"] == c]
        color = SEQ_BLUE[k]
        width = 3 if c == selected else 1.6
        alpha_ok = selected is None or c == selected
        obs = g[~g["is_forecast"]]
        fc = g[g["mob"] >= obs["mob"].max()] if len(obs) else g
        fig.add_trace(go.Scatter(x=obs["mob"], y=obs["over90"], mode="lines", name=c,
                                 line=dict(color=color if alpha_ok else NEUTRAL, width=width),
                                 customdata=[c] * len(obs), legendgroup=c,
                                 hovertemplate=f"{c}<br>MOB %{{x}}: %{{y:.2%}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["mob"], y=fc["predicted"], mode="lines", showlegend=False,
                                 line=dict(color=color if alpha_ok else NEUTRAL, width=width, dash="dot"),
                                 customdata=[c] * len(fc), legendgroup=c,
                                 hovertemplate=f"{c} (model)<br>MOB %{{x}}: %{{y:.2%}}<extra></extra>"))
    if selected:
        g = v[(v["cohort"] == selected) & v["is_forecast"]]
        _band(fig, g["mob"], g["lo"], g["hi"], SEQ_BLUE[-2], f"{selected} P10-P90")
    fig.update_xaxes(title="Months on book")
    fig.update_yaxes(title="Cumulative over-90", tickformat=".1%")
    return _layout(fig, 420, hovermode="closest", clickmode="event+select",
                   legend=dict(orientation="v", x=1.01, y=1, yanchor="top"))


# 12 --------------------------------------------------------------------- churn
ACTION_COLORS = {
    "Counter-offer: rate match": SERIES[1],
    "Refinance with term extension": SERIES[0],
    "Proactive relationship call": SERIES[2],
    "Monitor": NEUTRAL,
}


def churn_fig(sample: pd.DataFrame) -> go.Figure:
    """Balance vs predicted portability probability, colored by recommended action."""
    fig = go.Figure()
    for action, color in ACTION_COLORS.items():
        part = sample[sample["action"] == action]
        fig.add_trace(go.Scattergl(
            x=part["balance"], y=part["p_churn"], mode="markers", name=action,
            marker=dict(size=np.clip(np.sqrt(part["margin_at_risk"]) / 25, 5, 18), color=color,
                        opacity=0.75 if action != "Monitor" else 0.45, line=dict(width=0)),
            customdata=part[["client_id", "segment", "nbo"]].to_numpy(),
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Balance R$ %{x:,.0f}"
                          "<br>P(portability) %{y:.0%}<br>Next best offer: %{customdata[2]}<extra></extra>",
        ))
    fig.update_xaxes(type="log", title="Outstanding balance (R$, log scale)")
    fig.update_yaxes(title="P(portability in 12m)", tickformat=".0%")
    return _layout(fig, 440, dragmode="lasso", hovermode="closest")


# 13 ------------------------------------------------------------------- pricing
def pricing_fig(weekly: pd.DataFrame, fc: pd.DataFrame, curve: pd.DataFrame, chosen: float,
                cap: float) -> go.Figure:
    """Origination forecast (left) and rate -> volume / contribution trade-off (right)."""
    fig = make_subplots(rows=2, cols=2, column_widths=[0.5, 0.5], specs=[[{"rowspan": 2}, {}], [None, {}]],
                        shared_xaxes=False, horizontal_spacing=0.09, vertical_spacing=0.14,
                        subplot_titles=("Weekly originations (R$ mn) and 12-week forecast",
                                        "Expected lifetime contribution (R$ mn / month of offers)",
                                        "Expected monthly volume (R$ mn)"))
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["originations_mn"], mode="lines", name="Actual",
                             line=dict(color=SERIES[0], width=2)), 1, 1)
    _band(fig, fc.index, fc["p10"], fc["p90"], SERIES[0], "Forecast P10-P90", 1, 1)
    fig.add_trace(go.Scatter(x=fc.index, y=fc["p50"], mode="lines", name="Forecast median",
                             line=dict(color=SERIES[0], width=2, dash="dot")), 1, 1)
    ok = curve[curve["rate"] <= cap + 1e-9]
    best = ok.loc[ok["contribution_mn"].idxmax()] if not ok.empty else curve.iloc[0]
    fig.add_trace(go.Scatter(x=curve["rate"], y=curve["contribution_mn"], mode="lines",
                             name="Contribution", line=dict(color=SERIES[2], width=2),
                             hovertemplate="Rate %{x:.2%}/month<br>R$ %{y:,.1f} mn<extra></extra>"), 1, 2)
    fig.add_trace(go.Scatter(x=[best["rate"]], y=[best["contribution_mn"]], mode="markers",
                             name="Optimal under cap", marker=dict(size=11, color=SERIES[2],
                                                                   line=dict(color="#fff", width=2))), 1, 2)
    fig.add_trace(go.Scatter(x=curve["rate"], y=curve["volume_mn"], mode="lines", name="Volume",
                             line=dict(color=SERIES[6], width=2),
                             hovertemplate="Rate %{x:.2%}/month<br>R$ %{y:,.0f} mn<extra></extra>"), 2, 2)
    for r in (1, 2):
        fig.add_vline(x=cap, line=dict(color=STATUS["critical"], width=1.5, dash="dash"), row=r, col=2)
        fig.add_vline(x=chosen, line=dict(color=INK_2, width=1.5), row=r, col=2)
    fig.add_annotation(x=cap, y=1, yref="y2 domain", xref="x2", text="Legal cap", showarrow=False,
                       xanchor="left", xshift=4, font=dict(color=STATUS["critical"], size=11))
    fig.update_xaxes(tickformat=".1%", row=1, col=2)
    fig.update_xaxes(tickformat=".1%", title="Offered rate (% per month)", row=2, col=2)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 480, showlegend=False, hovermode="x unified")


# 14 ---------------------------------------------------------------------- ALM
def alm_fig(gap: pd.DataFrame, nii: dict[str, pd.Series]) -> go.Figure:
    """Repricing gap by bucket and monthly NII under Selic paths."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.1,
                        subplot_titles=("Repricing gap (R$ bn)", "Projected monthly NII (R$ mn)"))
    for side, color in (("asset", SERIES[0]), ("liability", SERIES[1])):
        part = gap[gap["side"] == side]
        fig.add_trace(go.Bar(x=part["bucket"].astype(str), y=part["amount_bn"], name=side.title() + "s",
                             marker_color=color,
                             hovertemplate=f"{side.title()}s %{{x}}: R$ %{{y:.1f}} bn<extra></extra>"), 1, 1)
    net = gap.groupby("bucket", observed=True)["amount_bn"].sum()
    fig.add_trace(go.Scatter(x=net.index.astype(str), y=net.cumsum(), mode="lines+markers",
                             name="Cumulative net gap", line=dict(color=INK_2, width=2),
                             marker=dict(size=8)), 1, 1)
    styles = [(SERIES[0], "solid"), (SERIES[1], "dash"), (MUTED, "dot")]
    first = next(iter(nii.values()))
    for k, ((name, s), (c, d)) in enumerate(zip(nii.items(), styles, strict=False)):
        if k and _same(s, first):
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=name,
                                 line=dict(color=c, width=2, dash=d),
                                 hovertemplate=f"{name}<br>%{{x|%b %Y}}: R$ %{{y:,.0f}} mn<extra></extra>"),
                      1, 2)
    fig.update_layout(barmode="relative")
    fig.update_annotations(font=dict(size=12, color=INK_2))
    return _layout(fig, 400)


# 15 -------------------------------------------------------------------- capital
def basel_fig(hist: pd.DataFrame, fan: pd.DataFrame) -> go.Figure:
    """Reported Basel ratio and Monte Carlo projection vs the regulatory minimum."""
    fig = go.Figure()
    h = hist.dropna(subset=["basel"])
    fig.add_trace(go.Scatter(x=h["period"], y=h["basel"], mode="lines+markers", name="Reported",
                             line=dict(color=INK_2, width=2), marker=dict(size=8)))
    _band(fig, fan.index, fan["p10"], fan["p90"], SERIES[0], "Projection P10-P90")
    x = [h["period"].iloc[-1], *fan.index]
    fig.add_trace(go.Scatter(x=x, y=[h["basel"].iloc[-1], *fan["p50"]], mode="lines+markers",
                             name="Projection median", line=dict(color=SERIES[0], width=2, dash="dot"),
                             marker=dict(size=7)))
    fig.add_hline(y=BASEL_MINIMUM, line=dict(color=STATUS["critical"], width=1.5, dash="dash"),
                  annotation_text="Minimum incl. buffer 10.5%", annotation_position="bottom left",
                  annotation_font=dict(color=STATUS["critical"], size=11))
    fig.update_yaxes(title="Total capital ratio", tickformat=".1%")
    return _layout(fig, 380, hovermode="x unified")
