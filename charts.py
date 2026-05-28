"""
charts.py — All chart builders for the Stock Analyzer.
Clean legends, proper date axes, currency-aware, mobile-friendly.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st


# ─── Shared helpers ───────────────────────────────────────────────────────────

DARK_BG   = "#0e1117"
PLOT_BG   = "#1a1d2e"
GRID_COL  = "rgba(255,255,255,0.06)"
HOVER_CFG = dict(bgcolor="#1a1d2e", font_size=12, bordercolor="#3a3f5c")
TICK_FONT = dict(size=11, color="#a0aec0")
DATE_FMT  = "%b '%y"          # e.g. "Jan '24"
DATE_FMT_YEAR = "%Y"

def _date_axis(dtickval="M3"):
    """Standard date x-axis — no decimal years."""
    return dict(
        showgrid=False,
        tickfont=TICK_FONT,
        tickformat=DATE_FMT,
        dtick=dtickval,
        tickangle=-30,
        type="date",
    )

def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

def _ccy_label(info: dict, display_ccy: str) -> str:
    """Build a currency label e.g. '(USD)' or '(EUR, converted from USD)'"""
    native = (info.get("currency") or "USD").upper()
    if not display_ccy or display_ccy == native:
        return f"({native})"
    return f"({display_ccy}, converted from {native})"


# ─── 1. Price history chart ───────────────────────────────────────────────────

def build_price_chart(hist_5y: pd.DataFrame, ticker: str, info: dict,
                       fmt_currency_fn, display_ccy: str, rates: dict) -> go.Figure:
    """
    Full price history area chart — uses 5Y (or max available) data.
    Shows price in display currency with high/low annotations.
    """
    if hist_5y is None or hist_5y.empty:
        return go.Figure()

    native  = (info.get("currency") or "USD").upper()
    close   = hist_5y["Close"].astype(float)
    dates   = hist_5y.index

    # Convert prices to display currency
    def conv(v):
        return fmt_currency_fn(v, native, display_ccy, rates)

    cp   = float(close.iloc[-1])
    high = float(close.max())
    low  = float(close.min())
    chg_1y = None
    if len(close) >= 252:
        chg_1y = (cp - float(close.iloc[-252])) / float(close.iloc[-252])
    chg_all = (cp - float(close.iloc[0])) / float(close.iloc[0])
    yrs = len(close) / 252
    color = "#26a69a" if chg_all >= 0 else "#ef5350"
    fill  = _hex_rgba(color, 0.07)

    # Subtitle with change info
    chg_label = f"{'▲' if chg_all>=0 else '▼'} {abs(chg_all):.1%} ({yrs:.0f}Y)"
    ccy_label = _ccy_label(info, display_ccy)

    fig = go.Figure()

    # 52-week range band
    if len(close) >= 252:
        y_52h = float(close[-252:].max())
        y_52l = float(close[-252:].min())
        fig.add_hrect(y0=y_52l, y1=y_52h,
                      fillcolor="rgba(255,255,255,0.03)",
                      line_width=0,
                      annotation_text="52-week range",
                      annotation_font_size=9,
                      annotation_font_color="#718096",
                      annotation_position="top right")

    # Main price line
    fig.add_trace(go.Scatter(
        x=dates, y=close,
        name=f"{ticker} Price",
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=fill,
        hovertemplate=f"%{{x|{DATE_FMT}}}  %{{y:,.2f}} {native}<extra></extra>",
    ))

    # Current price line
    fig.add_hline(
        y=cp,
        line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot"),
        annotation_text=f"  {conv(cp)} {ccy_label}",
        annotation_font_size=11,
        annotation_font_color="#e2e8f0",
    )

    # High annotation
    hi_idx = close.idxmax()
    fig.add_annotation(
        x=hi_idx, y=high,
        text=f"↑ {conv(high)}",
        showarrow=True, arrowhead=2, arrowcolor="#FFD700",
        font=dict(color="#FFD700", size=10),
        bgcolor=DARK_BG, bordercolor="#FFD700", borderwidth=1,
        ay=-36,
    )
    # Low annotation
    lo_idx = close.idxmin()
    fig.add_annotation(
        x=lo_idx, y=low,
        text=f"↓ {conv(low)}",
        showarrow=True, arrowhead=2, arrowcolor="#ef5350",
        font=dict(color="#ef5350", size=10),
        bgcolor=DARK_BG, bordercolor="#ef5350", borderwidth=1,
        ay=36,
    )

    # 1Y change badge
    if chg_1y is not None:
        badge = f"1Y: {'▲' if chg_1y>=0 else '▼'} {abs(chg_1y):.1%}"
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.97,
            text=badge,
            showarrow=False,
            font=dict(size=12, color="#26a69a" if chg_1y>=0 else "#ef5350"),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        height=300,
        template="plotly_dark",
        title=dict(
            text=(f"<b>{ticker}</b>  ·  {conv(cp)}  "
                  f"<span style='color:{color}'>{chg_label}</span>  "
                  f"<span style='color:#718096;font-size:11px'>{ccy_label}</span>"),
            font_size=14, x=0.01, xanchor="left",
        ),
        margin=dict(l=60, r=20, t=60, b=50),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        showlegend=False,
        xaxis=_date_axis("M6"),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COL,
            tickfont=TICK_FONT,
            title=dict(text=f"Price ({native})", font_size=11, standoff=5),
        ),
        hoverlabel=HOVER_CFG,
    )
    return fig


# ─── 2. Technical chart (candlestick + indicators) ───────────────────────────

def build_technical_chart(hist: pd.DataFrame, ind: dict, ticker: str,
                           native_ccy: str) -> go.Figure:
    """4-panel: Price+MAs+BB / Volume+OBV / RSI / MACD — 2Y data."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.04,
        subplot_titles=["", "Volume + OBV", "RSI (14)", "MACD"],
    )

    # ── Candles ───────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    # Moving averages
    ma_cfg = [
        ("sma_20",  "#FFD700", "SMA 20"),
        ("sma_50",  "#FF6B35", "SMA 50"),
        ("sma_200", "#00BCD4", "SMA 200"),
        ("ema_12",  "#CE93D8", "EMA 12"),
        ("ema_26",  "#9575CD", "EMA 26"),
    ]
    for k, col, lbl in ma_cfg:
        s = ind.get(k)
        if isinstance(s, pd.Series) and not s.dropna().empty:
            fig.add_trace(go.Scatter(
                x=s.index, y=s, name=lbl,
                line=dict(color=col, width=1.3),
                opacity=0.9, legendgroup="ma",
            ), row=1, col=1)

    # Bollinger bands
    bu, bl = ind.get("bb_upper"), ind.get("bb_lower")
    if isinstance(bu, pd.Series) and isinstance(bl, pd.Series):
        fig.add_trace(go.Scatter(x=bu.index, y=bu,
            line=dict(color="rgba(100,181,246,0.35)", width=1),
            name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bl.index, y=bl,
            line=dict(color="rgba(100,181,246,0.35)", width=1),
            fill="tonexty", fillcolor="rgba(100,181,246,0.05)",
            name="BB Bands", showlegend=False), row=1, col=1)

    # Support / Resistance
    for r in ind.get("sr_resistance", [])[:2]:
        fig.add_hline(y=r, line=dict(color="#ef5350", width=1, dash="dot"),
            annotation_text=f"R {r:.2f}", annotation_font_size=9,
            annotation_font_color="#ef5350", row=1, col=1)
    for s in ind.get("sr_support", [])[:2]:
        fig.add_hline(y=s, line=dict(color="#26a69a", width=1, dash="dot"),
            annotation_text=f"S {s:.2f}", annotation_font_size=9,
            annotation_font_color="#26a69a", row=1, col=1)

    # ── Volume + OBV ──────────────────────────────────────────────────────────
    vc = ["#26a69a" if c >= o else "#ef5350"
          for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
        marker_color=vc, opacity=0.6, showlegend=False), row=2, col=1)
    obv = ind.get("obv")
    if isinstance(obv, pd.Series) and not obv.dropna().empty:
        fig.add_trace(go.Scatter(x=obv.index, y=obv,
            line=dict(color="#FFB74D", width=1.5),
            name="OBV", showlegend=False), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi = ind.get("rsi")
    if isinstance(rsi, pd.Series) and not rsi.dropna().empty:
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi,
            line=dict(color="#CE93D8", width=1.5),
            showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#ef5350", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", width=1, dash="dash"), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.04)", row=3, col=1, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.04)", row=3, col=1, line_width=0)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd, macd_s, macd_h = ind.get("macd"), ind.get("macd_sig"), ind.get("macd_hist")
    if isinstance(macd, pd.Series) and not macd.dropna().empty:
        fig.add_trace(go.Scatter(x=macd.index, y=macd,
            line=dict(color="#42A5F5", width=1.5), showlegend=False), row=4, col=1)
    if isinstance(macd_s, pd.Series) and not macd_s.dropna().empty:
        fig.add_trace(go.Scatter(x=macd_s.index, y=macd_s,
            line=dict(color="#EF5350", width=1.2), showlegend=False), row=4, col=1)
    if isinstance(macd_h, pd.Series) and not macd_h.dropna().empty:
        hc = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h]
        fig.add_trace(go.Bar(x=macd_h.index, y=macd_h,
            marker_color=hc, opacity=0.7, showlegend=False), row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    date_ax = _date_axis("M3")

    fig.update_layout(
        height=800,
        template="plotly_dark",
        title=dict(
            text=f"<b>{ticker}</b> — Technical Analysis  "
                 f"<span style='color:#718096;font-size:11px'>({native_ccy})</span>",
            font_size=14, x=0.01,
        ),
        margin=dict(l=60, r=30, t=70, b=50),
        xaxis_rangeslider_visible=False,
        # Legend for MAs — positioned clearly above the top panel
        legend=dict(
            orientation="h",
            x=0, y=1.04,
            xanchor="left", yanchor="bottom",
            font_size=11,
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor="#3a3f5c",
            borderwidth=1,
            itemwidth=40,
        ),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        hoverlabel=HOVER_CFG,
    )
    # Domain layout for subplots with breathing room
    fig.update_layout(
        yaxis =dict(domain=[0.44,1.00], showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT,
                    title=dict(text=f"Price ({native_ccy})", font_size=10, standoff=5)),
        yaxis2=dict(domain=[0.30,0.42], showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT),
        yaxis3=dict(domain=[0.16,0.28], showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT,
                    range=[0,100]),
        yaxis4=dict(domain=[0.00,0.14], showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT),
    )
    # Date axis on all x axes
    for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig.update_layout(**{ax: date_ax})
    # Subplot title styling
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = "#94a3b8"
    return fig


# ─── 3. Financials chart (revenue/income + margins) ──────────────────────────

def build_financials_chart(fund: dict, ticker: str, native_ccy: str,
                            display_ccy: str, rates: dict,
                            fmt_currency_fn) -> go.Figure:
    """
    Two-panel: Revenue & Net Income bars / Profit Margins %.
    Each panel has its own clearly labelled legend placed just above it.
    X-axis: year labels (integers → no decimal).
    """
    rev = fund.get("revenue_series", {})
    ni  = fund.get("net_income_series", {})
    gm  = fund.get("gross_margin_series", {})
    om  = fund.get("op_margin_series", {})
    nm  = fund.get("net_margin_series", {})

    years = sorted(set(list(rev.keys()) + list(ni.keys())), reverse=True)[:6]
    years.sort()
    year_labels = [str(y) for y in years]

    ccy_note = _ccy_label({"currency": native_ccy}, display_ccy)

    # Convert values to display currency
    def c(v):
        if v is None: return None
        from currency import convert as _conv
        converted = _conv(float(v), native_ccy, display_ccy, rates)
        return converted

    rev_vals = [c(rev.get(y)) for y in years]
    ni_vals  = [c(ni.get(y))  for y in years]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.22,
        row_heights=[0.56, 0.44],
    )

    # ── Panel 1: Revenue & Net Income ─────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=year_labels, y=rev_vals,
        name="Revenue", marker_color="#42A5F5", opacity=0.88,
        legendgroup="income", legendgrouptitle_text="",
        hovertemplate="%{x}: %{y:,.0f}<extra>Revenue</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=year_labels, y=ni_vals,
        name="Net Income", marker_color="#66BB6A", opacity=0.88,
        legendgroup="income",
        hovertemplate="%{x}: %{y:,.0f}<extra>Net Income</extra>",
    ), row=1, col=1)

    # ── Panel 2: Margins ──────────────────────────────────────────────────────
    for d, col, lbl in [
        (gm, "#FFD700", "Gross Margin"),
        (om, "#FF6B35", "Op. Margin"),
        (nm, "#AB47BC", "Net Margin"),
    ]:
        vals = [d.get(y) for y in years]
        if any(v is not None for v in vals):
            pct_vals = [v * 100 if v is not None else None for v in vals]
            fig.add_trace(go.Scatter(
                x=year_labels, y=pct_vals,
                name=lbl,
                mode="lines+markers",
                line=dict(color=col, width=2.5),
                marker=dict(size=8, color=col,
                            line=dict(color=DARK_BG, width=1.5)),
                legendgroup="margins",
                hovertemplate=f"%{{x}}: %{{y:.1f}}%<extra>{lbl}</extra>",
            ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    year_axis = dict(
        showgrid=False,
        tickfont=TICK_FONT,
        type="category",          # ensures clean integer year labels
        tickangle=0,
    )

    fig.update_layout(
        height=560,
        template="plotly_dark",
        barmode="group",
        margin=dict(l=70, r=30, t=80, b=50),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        hoverlabel=HOVER_CFG,
        # Single legend — we'll fake two via annotations instead
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0, y=1.13,
            xanchor="left", yanchor="bottom",
            font_size=11,
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor="#3a3f5c",
            borderwidth=1,
            tracegroupgap=16,
        ),
    )

    fig.update_layout(
        xaxis =year_axis,
        xaxis2=year_axis,
        yaxis =dict(
            showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT,
            title=dict(text=f"Amount {ccy_note}", font_size=10, standoff=5),
            tickformat=",.0f",
        ),
        yaxis2=dict(
            showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT,
            title=dict(text="Margin (%)", font_size=10, standoff=5),
            ticksuffix="%",
        ),
    )

    # Panel titles as annotations (clearer than subplot_titles)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=1.01, xanchor="left", yanchor="bottom",
        text=f"<b>Revenue & Net Income</b>  {ccy_note}",
        font=dict(size=12, color="#e2e8f0"),
        showarrow=False,
    )
    # Second panel title — placed between the two panels
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=0.43, xanchor="left", yanchor="bottom",
        text="<b>Profit Margins</b>",
        font=dict(size=12, color="#e2e8f0"),
        showarrow=False,
    )
    return fig


# ─── 4. Peer bar chart ────────────────────────────────────────────────────────

def build_peer_chart(comparison: list, ticker: str) -> go.Figure:
    keys  = ["pe_trailing","pe_forward","ev_ebitda","price_book",
             "gross_margin","operating_margin","net_margin","roe","revenue_growth"]
    data  = [r for r in comparison
             if r["key"] in keys
             and r.get("target") is not None
             and r.get("peer_median") is not None][:9]
    if not data:
        return go.Figure()

    labels = [r["metric"] for r in data]
    diffs  = [(r["target"] - r["peer_median"]) / abs(r["peer_median"]) * 100
              if r["peer_median"] else 0 for r in data]
    colors = []
    for r, d in zip(data, diffs):
        hb = r.get("higher_better")
        if hb is True:    colors.append("#26a69a" if d >= 0 else "#ef5350")
        elif hb is False: colors.append("#26a69a" if d <= 0 else "#ef5350")
        else:             colors.append("#FFC107")

    fig = go.Figure(go.Bar(
        x=diffs, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{d:+.1f}%" for d in diffs],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{y}: %{x:+.1f}% vs peer median<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_width=1)
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> vs Peer Median (% difference)",
                   font_size=13, x=0.01),
        height=max(320, len(data) * 44 + 100),
        template="plotly_dark",
        margin=dict(l=160, r=80, t=50, b=40),
        paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(title="% vs Peer Median", tickfont=TICK_FONT,
                   showgrid=True, gridcolor=GRID_COL),
        yaxis=dict(tickfont=dict(size=11, color="#e2e8f0")),
        hoverlabel=HOVER_CFG,
        showlegend=False,
    )
    return fig


# ─── 5. Earnings surprise chart ───────────────────────────────────────────────

def build_earnings_chart(earn_rows: list, ticker: str) -> go.Figure:
    past = [r for r in earn_rows
            if not r["is_future"] and r.get("surprise_pct") is not None][:8]
    if not past:
        return go.Figure()
    past = list(reversed(past))
    labels = [r["date"][:7] for r in past]
    surps  = [r["surprise_pct"] for r in past]
    colors = ["#26a69a" if s >= 0 else "#ef5350" for s in surps]
    avg    = np.mean(surps)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=surps,
        marker_color=colors,
        text=[f"{s:+.1f}%" for s in surps],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x}: %{y:+.2f}%<extra>EPS Surprise</extra>",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.add_hline(y=avg, line=dict(color="#FFC107", width=1.5, dash="dash"),
                  annotation_text=f"Avg {avg:+.1f}%",
                  annotation_font_size=10, annotation_font_color="#FFC107")
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — EPS Surprise History",
                   font_size=13, x=0.01),
        height=270,
        template="plotly_dark",
        margin=dict(l=50, r=20, t=50, b=40),
        paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
        yaxis=dict(title="Surprise %", ticksuffix="%",
                   showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT),
        xaxis=dict(tickfont=TICK_FONT),
        hoverlabel=HOVER_CFG,
        showlegend=False,
    )
    return fig


# ─── 6. Analyst donut ─────────────────────────────────────────────────────────

def build_sentiment_donut(sent: dict, ticker: str) -> go.Figure:
    labels = ["Strong Buy","Buy","Hold","Sell","Strong Sell"]
    values = [sent.get(k, 0) for k in
              ["strong_buy","buy","hold","sell","strong_sell"]]
    colors = ["#00C853","#4CAF50","#FFC107","#FF9800","#F44336"]
    if sum(values) == 0:
        return go.Figure()
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker_colors=colors,
        textinfo="label+percent",
        textfont_size=11,
        hovertemplate="%{label}: %{value} analysts (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b> — Analyst Ratings",
                   font_size=13, x=0.01),
        height=310,
        template="plotly_dark",
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor=DARK_BG,
        legend=dict(
            orientation="v", x=1.02, y=0.5,
            font_size=11,
            bgcolor="rgba(14,17,23,0.85)",
            bordercolor="#3a3f5c", borderwidth=1,
        ),
        annotations=[dict(
            text=f"<b>{sent.get('num_analysts',0)}<br>Analysts</b>",
            x=0.5, y=0.5, font_size=13,
            showarrow=False, font_color="white",
        )],
    )
    return fig


# ─── 7. EPS revision trend ────────────────────────────────────────────────────

def build_eps_trend_chart(trend_data: dict, ticker: str) -> go.Figure:
    if not trend_data:
        return go.Figure()
    period_key = next(iter(trend_data), None)
    if not period_key:
        return go.Figure()
    trend = trend_data[period_key]
    periods = ["90daysAgo","60daysAgo","30daysAgo","7daysAgo","current"]
    labels  = ["90d ago","60d ago","30d ago","7d ago","Current"]
    vals = [trend.get(p) for p in periods]
    valid = [(l, v) for l, v in zip(labels, vals) if v is not None]
    if len(valid) < 2:
        return go.Figure()
    ls, vs = zip(*valid)
    up = vs[-1] >= vs[0]
    colors = ["#FFD700"] * (len(vs)-1) + ["#00C853" if up else "#ef5350"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(ls), y=list(vs),
        mode="lines+markers",
        line=dict(color="#42A5F5", width=2),
        marker=dict(color=colors, size=10,
                    line=dict(color=DARK_BG, width=2)),
        hovertemplate="%{x}: $%{y:.2f}<extra>EPS Estimate</extra>",
    ))
    fig.update_layout(
        title=dict(text=f"EPS Estimate Revisions — {period_key}",
                   font_size=12, x=0.01),
        height=240,
        template="plotly_dark",
        margin=dict(l=60, r=20, t=45, b=40),
        paper_bgcolor=DARK_BG, plot_bgcolor=PLOT_BG,
        yaxis=dict(title="EPS ($)", tickprefix="$",
                   showgrid=True, gridcolor=GRID_COL, tickfont=TICK_FONT),
        xaxis=dict(tickfont=TICK_FONT),
        hoverlabel=HOVER_CFG,
        showlegend=False,
    )
    return fig
