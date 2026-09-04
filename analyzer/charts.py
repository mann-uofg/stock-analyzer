"""Interactive Plotly charts for the dashboard.

Colours are fixed here rather than inherited from a template so the charts read
identically regardless of the Streamlit theme in use.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Charts sit on a card, so their ground is always transparent; only the ink
# changes between modes. ``set_mode`` is called once per render from app.py.
# These values mirror views/theme.py - a chart drawn in a different green from
# the figure beside it reads as two unrelated measurements.
BG = "rgba(0,0,0,0)"
FONT = 'InterVar, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
GRID = "rgba(255,255,255,.06)"
TEXT = "#f4f4f7"
UP, DOWN = "#34d399", "#f87171"
ACCENT = "#5b9cff"

_MODES = {
    "dark": {"grid": "rgba(255,255,255,.06)", "text": "#f4f4f7",
             "up": "#34d399", "down": "#f87171", "accent": "#5b9cff"},
    "light": {"grid": "rgba(0,0,0,.07)", "text": "#111114",
              "up": "#0f9d58", "down": "#d92d20", "accent": "#2563eb"},
}

_LAYOUT: dict[str, Any] = {}


def set_mode(mode: str = "dark") -> None:
    """Point the chart palette at the active appearance."""
    global GRID, TEXT, UP, DOWN, ACCENT, _LAYOUT
    palette = _MODES.get(mode, _MODES["dark"])
    GRID, TEXT = palette["grid"], palette["text"]
    UP, DOWN, ACCENT = palette["up"], palette["down"], palette["accent"]
    # Plotly draws in the browser, so it can use the same bundled face as the
    # rest of the page; leaving it on the default would make every axis label
    # visibly different from its surrounding text.
    _LAYOUT = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12, family=FONT),
        margin=dict(l=8, r=8, t=32, b=8),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor=BG),
    )


set_mode("dark")


def _axes(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True,
                     spikemode="across", spikethickness=1, spikecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)


def price_chart(
    df: pd.DataFrame,
    tech: dict[str, Any],
    setup: dict[str, Any] | None = None,
    months: int = 12,
    show_levels: bool = True,
) -> go.Figure:
    """Candlesticks with moving averages, Bollinger band, volume, RSI and MACD.

    The trade plan is drawn directly onto the price panel so the entry, stop
    and targets can be read against actual structure rather than a table.
    """
    bars = min(len(df), max(months, 1) * 21)
    view = df.iloc[-bars:]

    # No secondary_y on the price row: an axis declared but never plotted on has
    # no range, and hline/hrect annotations resolve against it to -Infinity,
    # which Plotly then emits as an invalid SVG text coordinate.
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.62, 0.16, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], name="Price",
            increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
        ),
        row=1, col=1,
    )

    # --- Moving averages ---
    for period, colour in ((20, "#f5c542"), (50, "#4c8dff"), (200, "#c77dff")):
        if len(df) >= period:
            ma = df["Close"].rolling(period).mean().iloc[-bars:]
            fig.add_trace(
                go.Scatter(x=view.index, y=ma, name=f"SMA {period}",
                           line=dict(color=colour, width=1.4)),
                row=1, col=1,
            )

    # --- Bollinger envelope ---
    if len(df) >= 20:
        mid = df["Close"].rolling(20).mean()
        sd = df["Close"].rolling(20).std()
        upper, lower = (mid + 2 * sd).iloc[-bars:], (mid - 2 * sd).iloc[-bars:]
        fig.add_trace(go.Scatter(x=view.index, y=upper, name="Bollinger",
                                 line=dict(color="rgba(255,255,255,0.18)", width=1),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=view.index, y=lower, name="Bollinger",
                                 line=dict(color="rgba(255,255,255,0.18)", width=1),
                                 fill="tonexty", fillcolor="rgba(76,141,255,0.05)",
                                 showlegend=False), row=1, col=1)

    # --- Support / resistance ---
    if show_levels:
        levels = tech.get("levels", {}) or {}
        for value in (levels.get("support") or []):
            fig.add_hline(y=value, line=dict(color=UP, width=1, dash="dot"),
                          opacity=0.45, row=1, col=1)
        for value in (levels.get("resistance") or []):
            fig.add_hline(y=value, line=dict(color=DOWN, width=1, dash="dot"),
                          opacity=0.45, row=1, col=1)

    # --- Trade plan ---
    if setup and setup.get("valid"):
        fig.add_hrect(y0=setup["entry_low"], y1=setup["entry_high"],
                      fillcolor=ACCENT, opacity=0.16, line_width=0, row=1, col=1)
        for value, colour, label in (
            (setup["stop_loss"], DOWN, "Stop"),
            (setup["target_1"], UP, "Target 1"),
            (setup["target_2"], UP, "Target 2"),
        ):
            fig.add_hline(
                y=value, line=dict(color=colour, width=1.6, dash="dash"),
                annotation_text=f"{label} {value:,.2f}",
                annotation_position="right",
                annotation_font=dict(color=colour, size=11),
                row=1, col=1,
            )

    # --- Volume ---
    colours = [
        UP if c >= o else DOWN
        for c, o in zip(view["Close"], view["Open"])
    ]
    fig.add_trace(
        go.Bar(x=view.index, y=view["Volume"], name="Volume",
               marker_color=colours, opacity=0.55, showlegend=False),
        row=2, col=1,
    )

    # --- RSI & MACD ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = (100 - 100 / (1 + rs)).replace([float("inf"), float("-inf")], float("nan"))
    rsi = rsi.iloc[-bars:]

    fig.add_trace(
        go.Scatter(x=view.index, y=rsi, name="RSI(14)",
                   line=dict(color="#f5c542", width=1.4)),
        row=3, col=1,
    )
    for level, colour in ((70, DOWN), (30, UP)):
        fig.add_hline(y=level, line=dict(color=colour, width=1, dash="dot"),
                      opacity=0.5, row=3, col=1)

    fig.update_layout(
        **_LAYOUT, height=760, xaxis_rangeslider_visible=False,
        bargap=0.05,
    )
    _axes(fig)
    fig.update_yaxes(title_text="Price", row=1, col=1, side="right")
    fig.update_yaxes(title_text="Vol", row=2, col=1, side="right", showticklabels=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, side="right", range=[0, 100])
    return fig


def portfolio_value(
    series: pd.Series,
    comparisons: dict[str, pd.Series] | None = None,
    currency: str = "",
    intraday: bool = False,
) -> go.Figure:
    """Portfolio value over time, or percentage change when comparing.

    With a comparison the axis switches to percent: a dollar line and an index
    line share no scale, and plotting them together would imply one.
    """
    fig = go.Figure()
    comparing = bool(comparisons)

    if comparing:
        plotted = (series / float(series.iloc[0]) - 1) * 100 if len(series) else series
        hover = "%{y:+.2f}%<extra>Portfolio</extra>"
    else:
        plotted = series
        hover = f"{currency} %{{y:,.2f}}<extra></extra>"

    # Green when the window is up, red when down - the same semantics as every
    # other figure in the app.
    rising = len(plotted) > 1 and float(plotted.iloc[-1]) >= float(plotted.iloc[0])
    line_colour = UP if rising else DOWN
    fill_colour = "rgba(69,193,122,.10)" if rising else "rgba(242,100,90,.10)"

    fig.add_trace(go.Scatter(
        x=plotted.index, y=plotted, name="Portfolio", mode="lines",
        line=dict(color=line_colour, width=2),
        fill="tozeroy" if not comparing else None,
        fillcolor=fill_colour if not comparing else None,
        hovertemplate=hover,
    ))

    for i, (label, other) in enumerate((comparisons or {}).items()):
        if other is None or other.empty:
            continue
        rebased = (other / float(other.iloc[0]) - 1) * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased, name=label, mode="lines",
            line=dict(color=["#8b95a8", ACCENT, "#c77dff"][i % 3], width=1.4,
                      dash="dot"),
            hovertemplate=f"%{{y:+.2f}}%<extra>{label}</extra>",
        ))

    if comparing:
        fig.add_hline(y=0, line=dict(color=GRID, width=1))

    fig.update_layout(**_LAYOUT, height=330, showlegend=comparing)
    _axes(fig)

    # A portfolio's intraday range is a fraction of a percent. Anchoring the
    # axis at zero would flatten every real move into a straight line, so the
    # range hugs the data and the fill simply runs off the bottom.
    y_axis: dict[str, Any] = {
        "title_text": "Change %" if comparing else f"Value {currency}".strip(),
        "side": "right",
        "tickformat": "+.1f" if comparing else ",.0f",
    }
    if not comparing and len(plotted):
        low, high = float(plotted.min()), float(plotted.max())
        span = high - low
        pad = span * 0.12 if span > 0 else max(abs(high) * 0.01, 1.0)
        y_axis["range"] = [low - pad, high + pad]
    fig.update_yaxes(**y_axis)
    # Intraday gaps between sessions would otherwise render as flat stretches.
    if intraday:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


def gauge(score: float, verdict: str) -> go.Figure:
    """Conviction dial coloured by verdict."""
    colour = {
        "STRONG BUY": "#00c853", "BUY": "#66bb6a", "HOLD": "#f5c542",
        "SELL": "#ef5350", "STRONG SELL": "#c62828",
    }.get(verdict, ACCENT)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 34, "color": TEXT, "family": FONT}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT,
                         "tickfont": {"size": 10}},
                "bar": {"color": colour, "thickness": 0.72},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(239,83,80,0.10)"},
                    {"range": [35, 65], "color": "rgba(245,197,66,0.10)"},
                    {"range": [65, 100], "color": "rgba(38,166,154,0.10)"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT, family=FONT),
                      height=200, margin=dict(l=16, r=16, t=8, b=8))
    return fig


SEGMENT_COLOURS = (
    "#6aa5ff", "#45c17a", "#e0a83c", "#c77dff", "#4dd0c1",
    "#f2645a", "#8b95a8", "#f0a5c0", "#9ad36b", "#ffb27a",
)


def donut(items: dict[str, float], centre_label: str = "") -> go.Figure:
    """Allocation as a ring. A pie's centre is dead space; a label earns it."""
    labels = list(items.keys())
    values = [items[k] for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False,
        marker=dict(colors=[SEGMENT_COLOURS[i % len(SEGMENT_COLOURS)]
                            for i in range(len(labels))],
                    line=dict(color="#0a0c10", width=2)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
    ))
    if centre_label:
        fig.add_annotation(
            text=centre_label, showarrow=False,
            font=dict(size=13, color=TEXT), x=0.5, y=0.5,
        )
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, size=11, family=FONT),
        height=250, margin=dict(l=6, r=6, t=6, b=6),
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle",
                    bgcolor=BG, font=dict(size=10.5)),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure | None:
    """Correlation as colour. A grid of numbers hides the clusters."""
    if corr is None or corr.empty:
        return None
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        # Diverging around zero: negative correlation is a different thing
        # from weak correlation, and a single-hue ramp would hide that.
        colorscale=[[0.0, "#4d7fd6"], [0.5, "#141922"], [1.0, "#f2645a"]],
        zmid=0, zmin=-1, zmax=1,
        xgap=2, ygap=2,
        hovertemplate="%{y} vs %{x}<br>%{z:.2f}<extra></extra>",
        colorbar=dict(thickness=9, len=0.85, tickfont=dict(size=9.5),
                      outlinewidth=0),
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, size=10.5, family=FONT),
        height=max(250, 26 * len(corr) + 90),
        margin=dict(l=6, r=6, t=6, b=6),
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    fig.update_yaxes(autorange="reversed")
    return fig


def bucket_chart(buckets: dict[str, Any]) -> go.Figure:
    """Signal contribution by bucket, signed and sorted."""
    names = [n.title() for n in buckets]
    values = [b["contribution"] for b in buckets.values()]
    order = sorted(range(len(values)), key=lambda i: values[i])
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in values],
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT, size=11),
            hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=GRID, width=1))
    fig.update_layout(**_LAYOUT, height=280, showlegend=False)
    _axes(fig)
    fig.update_xaxes(title_text="Weighted contribution to composite score")
    return fig


def iv_smile(options_panel: dict[str, Any]) -> go.Figure | None:
    """Implied-volatility smile across near-the-money strikes."""
    if not options_panel.get("available"):
        return None
    ntm = options_panel.get("near_the_money", {}) or {}
    fig = go.Figure()
    plotted = False

    for kind, colour in (("calls", UP), ("puts", DOWN)):
        rows = [r for r in (ntm.get(kind) or []) if r.get("iv_pct")]
        if not rows:
            continue
        plotted = True
        fig.add_trace(
            go.Scatter(
                x=[r["strike"] for r in rows],
                y=[r["iv_pct"] for r in rows],
                mode="lines+markers", name=kind.title(),
                line=dict(color=colour, width=2),
                marker=dict(size=7),
                hovertemplate="Strike %{x}<br>IV %{y:.1f}%<extra></extra>",
            )
        )

    if not plotted:
        return None

    spot = options_panel.get("spot_used")
    if spot:
        fig.add_vline(x=spot, line=dict(color=ACCENT, width=1.5, dash="dash"),
                      annotation_text="spot", annotation_font=dict(color=ACCENT))

    fig.update_layout(**_LAYOUT, height=320)
    _axes(fig)
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Implied volatility (%)")
    return fig


def earnings_chart(earnings: dict[str, Any]) -> go.Figure | None:
    """EPS surprise by quarter, paired with the next-day price reaction."""
    history = list(reversed(earnings.get("history") or []))
    if not history:
        return None

    moves = {m["date"]: m["move_pct"] for m in (earnings.get("post_earnings_moves") or [])}
    dates = [h["date"] for h in history]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=dates, y=[h.get("eps_surprise_pct") for h in history],
            name="EPS surprise %",
            marker_color=[
                UP if (h.get("eps_surprise_pct") or 0) >= 0 else DOWN for h in history
            ],
            hovertemplate="%{x}<br>surprise %{y:+.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates, y=[moves.get(d) for d in dates],
            name="Next-day move %", mode="lines+markers",
            line=dict(color=ACCENT, width=2), marker=dict(size=9),
            hovertemplate="%{x}<br>move %{y:+.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(**_LAYOUT, height=300)
    _axes(fig)
    fig.update_yaxes(title_text="EPS surprise %", secondary_y=False)
    fig.update_yaxes(title_text="Price reaction %", secondary_y=True, showgrid=False)
    return fig
