"""Design tokens and the Plotly template. One place for every color and font decision.

Palette: validated categorical order (CVD-checked), one-hue sequential ramp, reserved status
colors. Colors follow the entity (a segment is always the same color on every chart).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from pine_config import SEGMENTS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
NEUTRAL = "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a8a", "#c23a3a"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

SEGMENT_COLORS = {s: SERIES[i] for i, s in enumerate(SEGMENTS)}
SCENARIO_COLORS = {"Base": SERIES[0], "Adverse": SERIES[1], "Severe": SERIES[6]}


def rgba(hex_color: str, alpha: float) -> str:
    """Hex to rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def register_template() -> None:
    """Register and activate the 'pine' Plotly template."""
    axis = dict(
        gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, zerolinewidth=1,
        tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=INK_2, size=12)),
        showline=True, ticks="", automargin=True,
    )
    tpl = go.layout.Template(
        layout=dict(
            font=dict(family=FONT, color=INK, size=12),
            paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
            xaxis=axis, yaxis={**axis, "showline": False},
            margin=dict(l=64, r=16, t=48, b=48),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=INK_2),
                        bgcolor="rgba(0,0,0,0)", title=None),
            hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(color=INK, family=FONT)),
            title=dict(font=dict(size=13, color=INK_2), x=0, xanchor="left"),
            bargap=0.35, barcornerradius=4,
        )
    )
    pio.templates["pine"] = tpl
    pio.templates.default = "pine"


APP_CSS = f"""
<style>
  .block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}}
  h1, h2, h3 {{letter-spacing: -0.01em; color: {INK};}}
  h1 {{font-weight: 650; font-size: 1.65rem;}}
  h3 {{font-weight: 600; font-size: 1.02rem; margin-bottom: 0.1rem;}}
  [data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
     border-radius: 8px; padding: 12px 14px;}}
  [data-testid="stMetricLabel"] p {{color: {INK_2}; font-size: 0.78rem; text-transform: uppercase;
     letter-spacing: 0.04em;}}
  [data-testid="stMetricValue"] {{font-size: 1.45rem; font-weight: 600;}}
  .pine-caption {{color: {INK_2}; font-size: 0.84rem; margin: -0.2rem 0 0.4rem 0;}}
  .pine-tag {{display:inline-block; border:1px solid {GRID}; border-radius: 4px; padding: 1px 7px;
     margin: 0 6px 4px 0; font-size: 0.75rem; color: {INK_2}; background: {SURFACE};}}
  .pine-tag.live {{border-color: {STATUS['good']}; color: #006300;}}
  .pine-tag.fallback {{border-color: {STATUS['serious']}; color: #8a3b17;}}
  .pine-num {{font-variant-numeric: tabular-nums;}}
</style>
"""
