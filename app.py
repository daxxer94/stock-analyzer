"""
app.py - Main Streamlit UI for the Stock Analyzer.

Run locally:  streamlit run app.py
Deploy guide: see DEPLOY.md
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import urllib.parse
from PIL import Image

from data                import (fetch_ticker_data, fetch_risk_free_rate,
                                  fetch_peer_metrics, get_current_price,
                                  parse_earnings_dates, parse_calendar,
                                  parse_estimates, parse_eps_trend)
from peers               import get_peers
from fundamental         import run_dcf_models, analyze_fundamentals
from technical           import calculate_indicators, generate_signals
from valuation_sentiment import get_valuation_ratios, compare_with_peers, analyze_sentiment
from scoring             import score_fundamental, score_valuation, score_sentiment, calculate_composite
from tooltips            import TOOLTIP_CSS, tooltip_html, section_header, METRICS
from search              import search_ticker, get_ticker_display_name, TICKER_NAMES
from currency            import (fetch_fx_rates, fmt_currency, get_ticker_currency,
                                  apply_currency_to_info, sidebar_currency_selector,
                                  CURRENCY_SYMBOLS)
from sec_data            import (fetch_news, build_news_search_links, fetch_sec_cik,
                                  fetch_sec_filings, extract_customers_suppliers,
                                  generate_swot, get_regulatory_info)

# ─── Page config ─────────────────────────────────────────────────────────────
try:
    _favicon = Image.open("favicon.png")
except Exception:
    _favicon = "📈"

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── iPhone home screen icon injected into <head> via JS ────────────────────
import streamlit.components.v1 as _components
_components.html("""
<script>
(function() {
  var head = window.parent.document.head;

  function addLink(rel, sizes, href) {
    var el = window.parent.document.createElement('link');
    el.rel  = rel;
    if (sizes) el.sizes = sizes;
    el.href = href;
    head.appendChild(el);
  }
  function addMeta(name, content) {
    var el = window.parent.document.createElement('meta');
    el.name    = name;
    el.content = content;
    head.appendChild(el);
  }

  addLink('apple-touch-icon',       '180x180', '/app/static/apple-touch-icon.png');
  addLink('apple-touch-icon-precomposed', '180x180', '/app/static/apple-touch-icon.png');
  addLink('shortcut icon',          '',        '/app/static/apple-touch-icon.png');
  addMeta('apple-mobile-web-app-capable',          'yes');
  addMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
  addMeta('apple-mobile-web-app-title',            'Stock Analyzer');
  addMeta('mobile-web-app-capable',                'yes');
  addMeta('theme-color',                           '#0e1117');
})();
</script>
""", height=0, scrolling=False)

# ─── Inject CSS ──────────────────────────────────────────────────────────────
st.markdown(TOOLTIP_CSS + """
<style>
.metric-card{background:#1a1d2e;border-radius:10px;padding:16px 20px;margin:4px 0;border:1px solid #2d3748;}
.signal-badge{display:inline-block;padding:5px 15px;border-radius:20px;font-weight:700;font-size:13px;}
.score-bar-bg{background:#2a2d3e;border-radius:6px;height:10px;margin:2px 0 6px 0;}
.score-bar-fill{border-radius:6px;height:10px;}
.section-header{font-size:16px;font-weight:700;margin:14px 0 6px;color:#e2e8f0;}
.earn-card{background:#1a1d2e;border:1px solid #2d3748;border-radius:8px;padding:10px 14px;margin:5px 0;font-size:13px;}
.deploy-box{background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;padding:16px;margin:8px 0;}
@media(max-width:768px){
  .mrow{font-size:11px;}
  .tt-box{width:240px!important;}
  .signal-badge{font-size:11px;padding:3px 10px;}
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt(v, pct=False, dec=2, prefix="", native_ccy="USD"):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if pct:
        return f"{v:+.1%}" if abs(v) < 10 else f"{v:.1%}"
    if prefix == "$":
        # Use display currency from session state if available
        disp_ccy = st.session_state.get("display_ccy_code", "USD")
        rates    = st.session_state.get("fx_rates", {})
        return fmt_currency(v, native_ccy, disp_ccy, rates)
    return f"{v:.{dec}f}"

def signal_color(signal):
    return {"STRONG BUY":"#00C853","BUY":"#4CAF50","WEAK BUY":"#8BC34A",
            "HOLD":"#FFC107","WEAK SELL":"#FF9800","SELL":"#F44336",
            "STRONG SELL":"#B71C1C"}.get(signal, "#888")

def score_bar(label, score, color="#4CAF50"):
    pct = int(max(0, min(10, score)) / 10 * 100)
    st.markdown(f"""
    <div style='margin:4px 0'>
      <div style='display:flex;justify-content:space-between;font-size:12px;color:#94a3b8'>
        <span>{label}</span><span style='color:#e2e8f0;font-weight:600'>{score:.1f}/10</span>
      </div>
      <div class='score-bar-bg'><div class='score-bar-fill' style='width:{pct}%;background:{color}'></div></div>
    </div>""", unsafe_allow_html=True)

def _color_for_score(s):
    if s >= 7: return "#00C853"
    if s >= 5: return "#FFC107"
    return "#F44336"

def forecast_status(actual, estimate, threshold=0.02):
    """Return 'beat'/'miss'/'inline'/'' for a single actual vs estimate comparison."""
    if actual is None or estimate is None or estimate == 0:
        return ""
    diff = (actual - estimate) / abs(estimate)
    if diff > threshold:   return "beat"
    if diff < -threshold:  return "miss"
    return "inline"

def render_mrow(metric_key, label, value_str, fs=""):
    st.markdown(tooltip_html(metric_key, label, value_str, fs), unsafe_allow_html=True)

# ─── Chart Builders ──────────────────────────────────────────────────────────


def build_current_price_chart(hist: pd.DataFrame, ticker: str, info: dict) -> go.Figure:
    """Simple area chart showing 2-year price history."""
    if hist is None or hist.empty:
        return go.Figure()
    close  = hist["Close"].astype(float)
    dates  = hist.index
    color  = "#26a69a" if float(close.iloc[-1]) >= float(close.iloc[0]) else "#ef5350"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=close, name="Price",
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.08)" if color.startswith("#") else color,
    ))
    cp   = float(close.iloc[-1])
    high = float(close.max())
    low  = float(close.min())
    chg  = (cp - float(close.iloc[0])) / float(close.iloc[0])
    disp = st.session_state.get("display_ccy_code", info.get("currency","USD"))
    rates = st.session_state.get("fx_rates", {})
    cp_str = fmt_currency(cp, info.get("currency","USD"), disp, rates)

    fig.add_hline(y=cp, line=dict(color="white", width=1, dash="dot"),
                  annotation_text=f"  {cp_str}", annotation_font_size=11,
                  annotation_font_color="white")
    fig.update_layout(
        height=260,
        template="plotly_dark",
        title=dict(
            text=f"{ticker}  ·  {cp_str}  "
                 f"<span style='color:{'#26a69a' if chg>=0 else '#ef5350'}'>"
                 f"{'▲' if chg>=0 else '▼'} {abs(chg):.1%} (2Y)</span>",
            font_size=14, x=0.01,
        ),
        margin=dict(l=50, r=30, t=50, b=30),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    # Annotate high/low
    hi_idx = close.idxmax(); lo_idx = close.idxmin()
    fig.add_annotation(x=hi_idx, y=high,
        text=f"High {fmt_currency(high, info.get('currency','USD'), disp, rates)}",
        showarrow=True, arrowhead=2, arrowcolor="#FFD700", font=dict(color="#FFD700", size=10),
        bgcolor="#0e1117", bordercolor="#FFD700", borderwidth=1, ay=-30)
    fig.add_annotation(x=lo_idx, y=low,
        text=f"Low {fmt_currency(low, info.get('currency','USD'), disp, rates)}",
        showarrow=True, arrowhead=2, arrowcolor="#ef5350", font=dict(color="#ef5350", size=10),
        bgcolor="#0e1117", bordercolor="#ef5350", borderwidth=1, ay=30)
    return fig


def build_price_chart(hist, ind, ticker):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         row_heights=[0.55, 0.15, 0.15, 0.15], vertical_spacing=0.02,
                         subplot_titles=("", "Volume + OBV", "RSI (14)", "MACD"))
    # Candles
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False), row=1, col=1)
    # MAs
    for k, col, lbl in [("sma_20","#FFD700","SMA20"),("sma_50","#FF6B35","SMA50"),
                          ("sma_200","#00BCD4","SMA200"),("ema_12","#AB47BC","EMA12"),("ema_26","#7E57C2","EMA26")]:
        s = ind.get(k)
        if isinstance(s, pd.Series) and not s.dropna().empty:
            fig.add_trace(go.Scatter(x=s.index, y=s, name=lbl,
                line=dict(color=col, width=1.2), opacity=0.85), row=1, col=1)
    # Bollinger
    bu, bl = ind.get("bb_upper"), ind.get("bb_lower")
    if isinstance(bu, pd.Series) and isinstance(bl, pd.Series):
        fig.add_trace(go.Scatter(x=bu.index, y=bu, line=dict(color="rgba(100,181,246,0.4)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bl.index, y=bl, line=dict(color="rgba(100,181,246,0.4)", width=1),
            fill="tonexty", fillcolor="rgba(100,181,246,0.07)", showlegend=False), row=1, col=1)
    # S/R
    for r in ind.get("sr_resistance", [])[:2]:
        fig.add_hline(y=r, line=dict(color="#ef5350", width=1, dash="dot"),
            annotation_text=f"R {r:.2f}", annotation_font_size=10, row=1, col=1)
    for s in ind.get("sr_support", [])[:2]:
        fig.add_hline(y=s, line=dict(color="#26a69a", width=1, dash="dot"),
            annotation_text=f"S {s:.2f}", annotation_font_size=10, row=1, col=1)
    # Volume
    vc = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=vc, opacity=0.7, showlegend=False), row=2, col=1)
    obv = ind.get("obv")
    if isinstance(obv, pd.Series) and not obv.dropna().empty:
        fig.add_trace(go.Scatter(x=obv.index, y=obv, name="OBV",
            line=dict(color="#FFB74D", width=1.5), showlegend=False), row=2, col=1)
    # RSI
    rsi = ind.get("rsi")
    if isinstance(rsi, pd.Series) and not rsi.dropna().empty:
        fig.add_trace(go.Scatter(x=rsi.index, y=rsi, line=dict(color="#AB47BC", width=1.5), showlegend=False), row=3, col=1)
        fig.add_hline(y=70, line=dict(color="#ef5350", width=1, dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="#26a69a", width=1, dash="dash"), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.05)", row=3, col=1, line_width=0)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.05)", row=3, col=1, line_width=0)
    # MACD
    macd, macd_s, macd_h = ind.get("macd"), ind.get("macd_sig"), ind.get("macd_hist")
    if isinstance(macd, pd.Series) and not macd.dropna().empty:
        fig.add_trace(go.Scatter(x=macd.index, y=macd, line=dict(color="#42A5F5", width=1.5), showlegend=False), row=4, col=1)
    if isinstance(macd_s, pd.Series) and not macd_s.dropna().empty:
        fig.add_trace(go.Scatter(x=macd_s.index, y=macd_s, line=dict(color="#EF5350", width=1.2), showlegend=False), row=4, col=1)
    if isinstance(macd_h, pd.Series) and not macd_h.dropna().empty:
        hc = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_h]
        fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, marker_color=hc, opacity=0.7, showlegend=False), row=4, col=1)
    fig.update_layout(
        height=760, template="plotly_dark",
        title=dict(text=f"{ticker} — Technical Analysis", font_size=15, x=0.01),
        margin=dict(l=50, r=30, t=60, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.07, x=0, font_size=11,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    # Increase spacing between subplots for mobile
    fig.update_layout(
        yaxis1=dict(domain=[0.42, 1.00], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(domain=[0.28, 0.40], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis3=dict(domain=[0.15, 0.26], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis4=dict(domain=[0.00, 0.13], showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    return fig


def build_financials_chart(fund, ticker):
    rev = fund.get("revenue_series", {}); ni = fund.get("net_income_series", {})
    gm  = fund.get("gross_margin_series", {}); om = fund.get("op_margin_series", {}); nm = fund.get("net_margin_series", {})
    years = sorted(set(list(rev.keys()) + list(ni.keys())), reverse=True)[:5]; years.sort()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,          # ← more breathing room between panels
        row_heights=[0.58, 0.42],
        subplot_titles=[
            "<b>Revenue & Net Income</b>  (annual)",
            "<b>Profit Margins</b>  (%)",
        ],
    )

    # ── Panel 1: Revenue & Net Income bars ────────────────────────────────
    fig.add_trace(go.Bar(
        x=years, y=[rev.get(y) for y in years],
        name="Revenue", marker_color="#42A5F5", opacity=0.85,
        legendgroup="income",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=years, y=[ni.get(y) for y in years],
        name="Net Income", marker_color="#66BB6A", opacity=0.85,
        legendgroup="income",
    ), row=1, col=1)

    # ── Panel 2: Margin lines ─────────────────────────────────────────────
    for d, col, lbl in [
        (gm, "#FFD700", "Gross Margin"),
        (om, "#FF6B35", "Op. Margin"),
        (nm, "#AB47BC", "Net Margin"),
    ]:
        vals = [d.get(y) for y in years]
        if any(v is not None for v in vals):
            pct_vals = [v * 100 if v is not None else None for v in vals]
            fig.add_trace(go.Scatter(
                x=years, y=pct_vals, name=lbl,
                mode="lines+markers",
                line=dict(color=col, width=2.5),
                marker=dict(size=7),
                legendgroup="margins",
            ), row=2, col=1)

    fig.update_layout(
        height=520,
        template="plotly_dark",
        barmode="group",
        margin=dict(l=60, r=30, t=60, b=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
        # Two separate legend groups, positioned clearly
        legend=dict(
            orientation="h",
            x=0, y=1.14,
            font_size=11,
            bgcolor="rgba(0,0,0,0)",
            groupclick="toggleitem",
        ),
    )
    # Format y-axis 2 as percentage
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11))
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_xaxes(tickfont=dict(size=11))

    # Bold subplot titles
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = "#e2e8f0"
    return fig


def build_price_overview_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """Clean price + volume chart for the Overview tab."""
    if hist is None or hist.empty:
        return go.Figure()

    close  = hist["Close"]
    volume = hist["Volume"]
    dates  = hist.index

    # Colour area under price green/red vs first price
    start_price = float(close.iloc[0])
    end_price   = float(close.iloc[-1])
    area_color  = "rgba(38,166,154,0.15)" if end_price >= start_price else "rgba(239,83,80,0.12)"
    line_color  = "#26a69a"               if end_price >= start_price else "#ef5350"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.78, 0.22], vertical_spacing=0.03)

    # Price area
    fig.add_trace(go.Scatter(
        x=dates, y=close, name="Price",
        line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor=area_color,
        hovertemplate="%{x|%b %d %Y}<br><b>%{y:.2f}</b><extra></extra>",
    ), row=1, col=1)

    # 50-day and 200-day SMA
    if len(close) >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(x=dates, y=sma50, name="SMA 50",
            line=dict(color="#FF6B35", width=1.2, dash="dot"), opacity=0.8,
            hovertemplate="SMA50: %{y:.2f}<extra></extra>"), row=1, col=1)
    if len(close) >= 200:
        sma200 = close.rolling(200).mean()
        fig.add_trace(go.Scatter(x=dates, y=sma200, name="SMA 200",
            line=dict(color="#00BCD4", width=1.2, dash="dot"), opacity=0.8,
            hovertemplate="SMA200: %{y:.2f}<extra></extra>"), row=1, col=1)

    # Volume bars
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=dates, y=volume, name="Volume",
        marker_color=vol_colors, opacity=0.6, showlegend=False,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>"), row=2, col=1)

    # 52-week high/low lines
    high_52 = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
    low_52  = float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
    fig.add_hline(y=high_52, line=dict(color="#FFD700", width=1, dash="dot"),
                   annotation_text=f"52W High {high_52:.2f}",
                   annotation_font=dict(size=10, color="#FFD700"), row=1, col=1)
    fig.add_hline(y=low_52,  line=dict(color="#94a3b8", width=1, dash="dot"),
                   annotation_text=f"52W Low {low_52:.2f}",
                   annotation_font=dict(size=10, color="#94a3b8"), row=1, col=1)

    pct_chg = (end_price - start_price) / start_price * 100
    fig.update_layout(
        height=400,
        template="plotly_dark",
        title=dict(
            text=f"{ticker} — 2-Year Price  "
                 f"<span style='color:{line_color}'>{pct_chg:+.1f}% (2Y)</span>",
            font_size=14, x=0.01,
        ),
        margin=dict(l=50, r=30, t=50, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.07, x=0, font_size=11,
                    bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                      tickfont=dict(size=11))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    return fig


def build_earnings_surprise_chart(earn_rows, ticker):
    past = [r for r in earn_rows if not r["is_future"] and r.get("surprise_pct") is not None][:8]
    if not past: return go.Figure()
    past = list(reversed(past))
    labels = [r["date"][:7] for r in past]
    surps  = [r["surprise_pct"] for r in past]
    colors = ["#26a69a" if s >= 0 else "#ef5350" for s in surps]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=surps, marker_color=colors,
        text=[f"{s:+.1f}%" for s in surps], textposition="outside"))
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(title=f"{ticker} — EPS Surprise History (%)", height=260,
        template="plotly_dark", margin=dict(l=40, r=20, t=40, b=20),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", yaxis_title="Surprise %")
    return fig


def build_eps_trend_chart(trend_data, ticker):
    """Show how current-quarter EPS estimates have been revised over time."""
    if not trend_data: return go.Figure()
    period_key = next(iter(trend_data), None)
    if not period_key: return go.Figure()
    trend = trend_data[period_key]
    periods = ["90daysAgo", "60daysAgo", "30daysAgo", "7daysAgo", "current"]
    labels  = ["90d ago", "60d ago", "30d ago", "7d ago", "Current"]
    vals = [trend.get(p) for p in periods]
    valid_pairs = [(l, v) for l, v in zip(labels, vals) if v is not None]
    if len(valid_pairs) < 2: return go.Figure()
    ls, vs = zip(*valid_pairs)
    colors = ["#FFD700"] * (len(vs) - 1) + ["#00C853" if vs[-1] >= vs[0] else "#ef5350"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ls), y=list(vs), mode="lines+markers",
        line=dict(color="#42A5F5", width=2),
        marker=dict(color=colors, size=10)))
    fig.update_layout(title=f"EPS Estimate Revisions — {period_key}", height=240,
        template="plotly_dark", margin=dict(l=40, r=20, t=40, b=20),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", yaxis_title="EPS Estimate $")
    return fig


def build_peer_chart(comparison, ticker):
    keys  = ["pe_trailing","pe_forward","ev_ebitda","price_book","gross_margin","operating_margin","net_margin","roe","revenue_growth"]
    data  = [r for r in comparison if r["key"] in keys and r.get("target") is not None and r.get("peer_median") is not None][:9]
    if not data: return go.Figure()
    labels  = [r["metric"] for r in data]
    diffs   = [(r["target"] - r["peer_median"]) / abs(r["peer_median"]) * 100 if r["peer_median"] else 0 for r in data]
    colors  = []
    for r, d in zip(data, diffs):
        hb = r.get("higher_better")
        if hb is True:    colors.append("#26a69a" if d >= 0 else "#ef5350")
        elif hb is False: colors.append("#26a69a" if d <= 0 else "#ef5350")
        else:             colors.append("#FFC107")
    fig = go.Figure(go.Bar(x=diffs, y=labels, orientation="h",
        marker_color=colors, text=[f"{d:+.1f}%" for d in diffs], textposition="outside"))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(title=f"{ticker} vs Peer Median (%)", height=max(300, len(data)*42+80),
        template="plotly_dark", margin=dict(l=160, r=80, t=40, b=20),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e", xaxis_title="% vs Peer Median")
    return fig


def build_sentiment_donut(sent, ticker):
    labels = ["Strong Buy","Buy","Hold","Sell","Strong Sell"]
    values = [sent.get("strong_buy",0), sent.get("buy",0), sent.get("hold",0), sent.get("sell",0), sent.get("strong_sell",0)]
    colors = ["#00C853","#4CAF50","#FFC107","#FF9800","#F44336"]
    if sum(values) == 0: return go.Figure()
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker_colors=colors,
        textinfo="label+value", hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.update_layout(title=f"{ticker} — Analyst Ratings", height=300,
        template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor="#0e1117",
        annotations=[dict(text=f"<b>{sent.get('num_analysts',0)}<br>Analysts</b>",
            x=0.5, y=0.5, font_size=13, showarrow=False, font_color="white")])
    return fig


def build_dcf_comparison(dcf, current_price):
    rows = []
    for key, label in [("wacc","WACC DCF"),("capm","CAPM DCF"),("fixed","Fixed Rate"),("two_stage","Two-Stage FCF")]:
        m = dcf.get(key, {})
        if m.get("error"):
            rows.append({"Model": label, "Rate": "—", "g Stage 1": "—", "g Terminal": "—",
                          "Intrinsic Value": m["error"][:50], "Upside": "—", "TV% of EV": "—"})
        else:
            iv     = m.get("intrinsic_value")
            upside = ((iv - current_price) / current_price) if (iv and current_price > 0) else None
            rows.append({
                "Model":       label,
                "Rate":        fmt(m.get("discount_rate"), pct=True),
                "g Stage 1":   fmt(m.get("stage1_growth"), pct=True),
                "g Terminal":  fmt(m.get("terminal_growth", 0.025), pct=True),
                "Intrinsic Value": fmt(iv, prefix="$") if iv else "—",
                "Upside":      fmt(upside, pct=True) if upside is not None else "—",
                "TV% of EV":   f"{m.get('tv_pct',0):.1f}%",
            })
    return pd.DataFrame(rows)


# ─── Earnings Calendar Section ───────────────────────────────────────────────

def render_earnings_section(data, ticker, earn_rows, cal_info, estimates, eps_trend_data):
    """Full earnings tab: calendar, history, estimates, EPS trend."""
    cp = get_current_price(data["info"])

    # ── Upcoming Earnings ─────────────────────────────────────────────────────
    next_date = cal_info.get("next_earnings_date")
    if next_date:
        today = datetime.date.today()
        try:
            ed = datetime.date.fromisoformat(next_date)
            days_away = (ed - today).days
            color = "#26a69a" if days_away > 0 else "#FFB74D"
            note  = f"in {days_away} days" if days_away > 0 else "today / recently passed"
            st.markdown(f"""<div class='earn-card' style='border-color:{color}'>
              <div class='earn-date'>📅 Next Earnings Date</div>
              <div style='font-size:16px;font-weight:700;color:{color}'>{next_date}
                <span style='font-size:12px;color:#94a3b8;font-weight:400;margin-left:8px'>{note}</span>
              </div></div>""", unsafe_allow_html=True)
        except Exception:
            st.markdown(f"📅 **Next Earnings:** {next_date}")

    # EPS range estimates from calendar
    for lbl, key in [("EPS Estimate (Low–High)", ("earnings_low","earnings_high")),
                      ("Revenue Estimate (Low–High)", ("revenue_low","revenue_high"))]:
        lo = cal_info.get(key[0]); hi = cal_info.get(key[1])
        if lo is not None and hi is not None:
            st.markdown(f"<div class='earn-card'><span style='color:#94a3b8'>{lbl}: </span>"
                        f"<span style='color:#e2e8f0;font-weight:600'>{fmt(lo, prefix='$' if 'revenue' in lbl.lower() else '')} "
                        f"— {fmt(hi, prefix='$' if 'revenue' in lbl.lower() else '')}</span></div>",
                        unsafe_allow_html=True)

    # ── Analyst Estimates Table ───────────────────────────────────────────────
    if estimates:
        st.markdown(section_header("📐 Analyst Estimates"), unsafe_allow_html=True)
        period_labels = {"0q": "This Quarter", "1q": "Next Quarter", "0y": "This Year", "1y": "Next Year"}
        for period_key, period_label in period_labels.items():
            eps_k = f"eps_est_{period_key}"; rev_k = f"rev_est_{period_key}"
            eps_e = estimates.get(eps_k); rev_e = estimates.get(rev_k)
            if not eps_e and not rev_e: continue
            with st.expander(f"📊 {period_label}", expanded=(period_key in ["0q", "0y"])):
                c1, c2 = st.columns(2)
                if eps_e:
                    with c1:
                        st.markdown("**EPS Estimates**")
                        st.metric("Consensus", f"${eps_e.get('avg'):.2f}" if eps_e.get("avg") else "—")
                        st.caption(f"Range: ${eps_e.get('low', 0):.2f} – ${eps_e.get('high', 0):.2f} | "
                                   f"Analysts: {int(eps_e.get('count') or 0)}")
                        if eps_e.get("growth"):
                            st.caption(f"Est. growth: {eps_e['growth']:+.1%}")
                if rev_e:
                    with c2:
                        st.markdown("**Revenue Estimates**")
                        avg = rev_e.get("avg")
                        st.metric("Consensus", fmt(avg, prefix="$") if avg else "—")
                        if rev_e.get("low") and rev_e.get("high"):
                            st.caption(f"Range: {fmt(rev_e['low'], prefix='$')} – {fmt(rev_e['high'], prefix='$')}")
                        if rev_e.get("growth"):
                            st.caption(f"Est. growth: {rev_e['growth']:+.1%}")

    # ── EPS Revision Trend ────────────────────────────────────────────────────
    if eps_trend_data:
        st.markdown(section_header("📉 EPS Estimate Revisions"), unsafe_allow_html=True)
        fig_trend = build_eps_trend_chart(eps_trend_data, ticker)
        if fig_trend.data:
            st.plotly_chart(fig_trend, use_container_width=True)

    # ── Historical Earnings Surprise ─────────────────────────────────────────
    past = [r for r in earn_rows if not r["is_future"]]
    st.markdown(section_header("📊 EPS Surprise History"), unsafe_allow_html=True)
    if past:
        fig_sur = build_earnings_surprise_chart(earn_rows, ticker)
        if fig_sur.data:
            st.plotly_chart(fig_sur, use_container_width=True)

        # Summary stats
        surprises = [r["surprise_pct"] for r in past if r.get("surprise_pct") is not None]
        if surprises:
            beats  = sum(1 for s in surprises if s > 0)
            misses = sum(1 for s in surprises if s < 0)
            avg_s  = np.mean(surprises)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beats",  beats)
            c2.metric("Misses", misses)
            c3.metric("Avg Surprise", f"{avg_s:+.1f}%")
            c4.metric("Hit Rate", f"{beats/(beats+misses)*100:.0f}%" if beats+misses > 0 else "—")

        # Detailed table
        with st.expander("📋 Detailed EPS Table", expanded=False):
            tbl = []
            for r in past[:12]:
                s = r.get("surprise_pct")
                badge = ("✅ Beat" if s > 0 else "🔴 Miss" if s < 0 else "🟡 In line") if s is not None else "—"
                tbl.append({
                    "Date":          r["date"],
                    "EPS Estimate":  f"${r['eps_estimate']:.2f}" if r.get("eps_estimate") else "—",
                    "EPS Actual":    f"${r['eps_actual']:.2f}"   if r.get("eps_actual")   else "—",
                    "Surprise %":    f"{s:+.2f}%" if s is not None else "—",
                    "Result":        badge,
                })
            st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

        # ── Upcoming future dates ─────────────────────────────────────────────
        future = [r for r in earn_rows if r["is_future"]]
        if future:
            st.markdown(section_header("🔮 Upcoming Earnings (Estimated Dates)"), unsafe_allow_html=True)
            for r in future[:4]:
                st.markdown(f"""<div class='earn-card'>
                  <span style='color:#94a3b8'>📅 {r['date']}</span>
                  {"<span style='color:#fbd38d;margin-left:12px'>EPS Estimate: " + fmt(r['eps_estimate']) + "</span>" if r.get('eps_estimate') else ""}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No historical earnings data available for this ticker.")



# ─── Intelligence Tab ────────────────────────────────────────────────────────

def render_intelligence_tab(ticker, info, scoring, signals, sentiment, fund, comparison):
    """SWOT analysis, customers/suppliers, regulatory filings."""
    country      = info.get("country", "")
    company_name = info.get("shortName", ticker)
    description  = info.get("longBusinessSummary", "")

    # ── SWOT ─────────────────────────────────────────────────────────────────
    st.markdown(section_header("♟️ Dynamic SWOT Analysis"), unsafe_allow_html=True)
    st.caption("Generated from live financial data, technical signals, peer comparison, and sentiment.")

    swot = generate_swot(info, scoring, signals, sentiment, fund, comparison)

    sw_col, ot_col = st.columns(2)
    with sw_col:
        st.markdown("""
        <div style='background:#0d2b1a;border:1px solid #2d5a3d;border-radius:10px;
                     padding:14px 16px;margin-bottom:12px'>
          <div style='color:#68d391;font-weight:700;font-size:14px;margin-bottom:8px'>
            💪 STRENGTHS
          </div>""", unsafe_allow_html=True)
        for s in swot["strengths"]:
            st.markdown(f"<div style='font-size:12px;color:#c6f6d5;padding:3px 0;border-bottom:1px solid #2d5a3d'>{s}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#2d1b00;border:1px solid #5a3d0d;border-radius:10px;
                     padding:14px 16px;margin-bottom:12px'>
          <div style='color:#fbd38d;font-weight:700;font-size:14px;margin-bottom:8px'>
            🌱 OPPORTUNITIES
          </div>""", unsafe_allow_html=True)
        for o in swot["opportunities"]:
            st.markdown(f"<div style='font-size:12px;color:#fefcbf;padding:3px 0;border-bottom:1px solid #5a3d0d'>{o}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ot_col:
        st.markdown("""
        <div style='background:#2d1a1a;border:1px solid #5a2d2d;border-radius:10px;
                     padding:14px 16px;margin-bottom:12px'>
          <div style='color:#fc8181;font-weight:700;font-size:14px;margin-bottom:8px'>
            ⚠️ WEAKNESSES
          </div>""", unsafe_allow_html=True)
        for w in swot["weaknesses"]:
            st.markdown(f"<div style='font-size:12px;color:#fed7d7;padding:3px 0;border-bottom:1px solid #5a2d2d'>{w}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#1a1a2d;border:1px solid #3d2d5a;border-radius:10px;
                     padding:14px 16px;margin-bottom:12px'>
          <div style='color:#b794f4;font-weight:700;font-size:14px;margin-bottom:8px'>
            ⚡ THREATS
          </div>""", unsafe_allow_html=True)
        for t in swot["threats"]:
            st.markdown(f"<div style='font-size:12px;color:#e9d8fd;padding:3px 0;border-bottom:1px solid #3d2d5a'>{t}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Customers & Suppliers ────────────────────────────────────────────────
    st.markdown(section_header("🤝 Customers & Suppliers"), unsafe_allow_html=True)
    cs = extract_customers_suppliers(description)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**👥 Key Customers / End Markets**")
        if cs["customers"]:
            for name in cs["customers"]:
                st.markdown(f"<div style='font-size:12px;color:#90cdf4;padding:2px 0'>• {name}</div>",
                            unsafe_allow_html=True)
        if cs["customer_notes"]:
            with st.expander("From business description", expanded=False):
                st.markdown(f"<div style='font-size:12px;color:#a0aec0;line-height:1.6'>{cs['customer_notes']}</div>",
                            unsafe_allow_html=True)
        if not cs["customers"] and not cs["customer_notes"]:
            st.caption("Customer data not available in public description. Check the SEC 10-K filing below.")

    with c2:
        st.markdown("**🏭 Key Suppliers / Partners**")
        if cs["suppliers"]:
            for name in cs["suppliers"]:
                st.markdown(f"<div style='font-size:12px;color:#90cdf4;padding:2px 0'>• {name}</div>",
                            unsafe_allow_html=True)
        if cs["supplier_notes"]:
            with st.expander("From business description", expanded=False):
                st.markdown(f"<div style='font-size:12px;color:#a0aec0;line-height:1.6'>{cs['supplier_notes']}</div>",
                            unsafe_allow_html=True)
        if not cs["suppliers"] and not cs["supplier_notes"]:
            st.caption("Supplier data not in public description. Check the 10-K/annual report filing below.")

    # ── Regulatory Filings ───────────────────────────────────────────────────
    st.markdown(section_header("📁 Regulatory Filings & Disclosures"), unsafe_allow_html=True)

    reg = get_regulatory_info(country, company_name)
    if reg:
        st.markdown(
            f"{reg['icon']} **{reg['name']}** — "
            f"[Search filings for {company_name}]({reg['url']})",
            unsafe_allow_html=False
        )

    # SEC EDGAR for US stocks
    exchange = info.get("exchange", "")
    is_us = country == "United States" or exchange in ["NMS","NYQ","NGM","PCX","BTS","ASE"]
    if is_us:
        st.markdown("**🇺🇸 SEC EDGAR (US)**")
        with st.spinner("Looking up SEC filings…"):
            cik = fetch_sec_cik(ticker)
        if cik:
            filings_10k = fetch_sec_filings(cik, "10-K", count=3)
            filings_10q = fetch_sec_filings(cik, "10-Q", count=3)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Annual Reports (10-K)**")
                if filings_10k:
                    for f in filings_10k:
                        st.markdown(f"[📄 10-K — {f['date']}]({f['index']})")
                else:
                    st.markdown(f"[🔍 Search 10-K filings on EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={urllib.parse.quote(company_name)}&CIK=&type=10-K&dateb=&owner=include&count=5)")
            with c2:
                st.markdown("**Quarterly Reports (10-Q)**")
                if filings_10q:
                    for f in filings_10q:
                        st.markdown(f"[📄 10-Q — {f['date']}]({f['index']})")
                else:
                    edgar_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{urllib.parse.quote(ticker)}%22&forms=10-Q"
                    st.markdown(f"[🔍 Search 10-Q filings on EDGAR]({edgar_url})")
            # Direct EDGAR company page
            st.markdown(f"[🏛️ Full EDGAR filing history for CIK {cik}](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=&dateb=&owner=include&count=40)")
        else:
            name_enc = urllib.parse.quote(company_name)
            st.markdown(f"[🔍 Search EDGAR for {company_name}](https://efts.sec.gov/LATEST/search-index?q=%22{name_enc}%22&forms=10-K)")

    # Additional IR link
    ir_website = info.get("irWebsite") or info.get("website")
    if ir_website:
        st.markdown(f"[🌐 Investor Relations Website]({ir_website})")


# ─── News Tab ────────────────────────────────────────────────────────────────

def render_news_tab(ticker, info, news_items):
    """Recent news + links to free news sources."""
    import urllib.parse
    company_name = info.get("shortName", ticker)

    # ── Free news source links ────────────────────────────────────────────────
    st.markdown(section_header("🔗 Free News Sources"), unsafe_allow_html=True)
    links = build_news_search_links(ticker, company_name)
    cols = st.columns(3)
    for i, link in enumerate(links):
        with cols[i % 3]:
            st.markdown(
                f"""<a href="{link['url']}" target="_blank"
                   style="display:block;background:#1a1d2e;border:1px solid #3a3f5c;
                   border-radius:8px;padding:10px 14px;text-decoration:none;
                   color:#90cdf4;font-size:13px;margin-bottom:8px;text-align:center">
                   {link['icon']} {link['source']}</a>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Yahoo Finance news feed ───────────────────────────────────────────────
    st.markdown(section_header("📰 Recent News"), unsafe_allow_html=True)

    if news_items:
        import datetime
        for item in news_items:
            title     = item.get("title", "")
            link      = item.get("link", "")
            publisher = item.get("publisher", "")
            ts        = item.get("providerPublishTime", 0)
            if not title or not link:
                continue
            date_str = ""
            if ts:
                try:
                    date_str = datetime.datetime.fromtimestamp(ts).strftime("%b %d, %Y")
                except Exception:
                    pass
            st.markdown(
                f"""<div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:8px;
                    padding:12px 16px;margin-bottom:8px'>
                  <a href="{link}" target="_blank"
                     style="color:#90cdf4;font-size:13px;font-weight:600;text-decoration:none">
                    {title}
                  </a>
                  <div style="font-size:11px;color:#718096;margin-top:5px">
                    {publisher}{"  ·  " + date_str if date_str else ""}
                  </div>
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.info("No recent news found. Use the source links above to search manually.")

    # ── Earnings calendar link ────────────────────────────────────────────────
    st.markdown(section_header("📅 Earnings Calendar Links"), unsafe_allow_html=True)
    tick_enc = urllib.parse.quote(ticker)
    st.markdown(
        f"[📅 Earnings Whispers](https://www.earningswhispers.com/stocks/{ticker.lower()})  ·  "
        f"[📊 Seeking Alpha Earnings](https://seekingalpha.com/symbol/{tick_enc}/earnings)  ·  "
        f"[🗓️ Yahoo Finance Calendar](https://finance.yahoo.com/quote/{tick_enc}/financials/)"
    )


# ─── Per-stock Deep Dive ──────────────────────────────────────────────────────

def render_stock_tab(ticker, data, dcf, fund, indicators, signals,
                     valuation, comparison, sentiment, scoring, peer_data):
    info = data["info"]
    cp   = get_current_price(info)

    # Parse new data sources
    earn_rows     = parse_earnings_dates(data.get("earnings_dates", pd.DataFrame()))
    cal_info      = parse_calendar(data.get("calendar", {}))
    estimates     = parse_estimates(data.get("earnings_estimate"), data.get("revenue_estimate"))
    eps_trend_data = parse_eps_trend(data.get("eps_trend"))

    # Determine forecast status from EPS beat/miss history
    beat_pct = None
    past_surp = [r["surprise_pct"] for r in earn_rows if not r["is_future"] and r.get("surprise_pct") is not None]
    if past_surp:
        beat_pct = sum(1 for s in past_surp if s > 0) / len(past_surp)
    eps_fs = "beat" if beat_pct and beat_pct > 0.6 else ("miss" if beat_pct and beat_pct < 0.4 else "inline")

    tabs = st.tabs(["📋 Overview", "💰 Financials & DCF", "📊 Valuation & Peers",
                     "📈 Technical Analysis", "🎯 Sentiment", "📅 Earnings & Forecasts",
                     "🏢 Intelligence", "📰 News"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tabs[0]:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            sc = scoring["scores"]; comp = scoring["composite"]
            st.markdown(f"""
            <div class='metric-card' style='text-align:center'>
              <div style='font-size:13px;color:#94a3b8'>{info.get('shortName','')}</div>
              <div style='font-size:40px;font-weight:800;color:#fff;margin:8px 0'>
                {comp:.1f}<span style='font-size:17px;color:#94a3b8'>/10</span>
              </div>
              <div><span class='signal-badge' style='background:{scoring["color"]};color:#fff;font-size:16px'>{scoring["signal"]}</span></div>
              <div style='font-size:11px;color:#94a3b8;margin-top:8px'>1–3 year medium-term outlook</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
            score_bar("Fundamental",  sc["fundamental"],  _color_for_score(sc["fundamental"]))
            score_bar("Valuation",    sc["valuation"],    _color_for_score(sc["valuation"]))
            score_bar("Technical",    sc["technical"],    _color_for_score(sc["technical"]))
            score_bar("Sentiment",    sc["sentiment"],    _color_for_score(sc["sentiment"]))
            # Earnings track record
            if beat_pct is not None:
                color = "#00C853" if beat_pct > 0.6 else ("#F44336" if beat_pct < 0.4 else "#FFC107")
                st.markdown(f"<div style='margin-top:10px;font-size:12px;color:{color}'>"
                            f"{'✅' if beat_pct>0.6 else ('🔴' if beat_pct<0.4 else '🟡')} "
                            f"EPS beat rate: {beat_pct:.0%} ({len(past_surp)} quarters)</div>", unsafe_allow_html=True)

        with col2:
            # Price chart spanning full width above metrics
            hist = data.get("hist_2y", pd.DataFrame())
            if not hist.empty:
                fig_price = build_current_price_chart(hist, ticker, info)
                st.plotly_chart(fig_price, use_container_width=True)
            st.markdown(section_header("Key Metrics"), unsafe_allow_html=True)
            de_raw = info.get("debtToEquity"); de_norm = (de_raw or 0) / 100
            render_mrow("pe_trailing",   "P/E (TTM)",        fmt(info.get("trailingPE"), dec=1))
            render_mrow("pe_forward",    "Forward P/E",      fmt(info.get("forwardPE"), dec=1))
            render_mrow("ev_ebitda",     "EV/EBITDA",        fmt(info.get("enterpriseToEbitda"), dec=1))
            render_mrow("gross_margin",  "Gross Margin",     fmt(info.get("grossMargins"), pct=True))
            render_mrow("net_margin",    "Net Margin",       fmt(info.get("profitMargins"), pct=True), eps_fs)
            render_mrow("roe",           "ROE",              fmt(info.get("returnOnEquity"), pct=True))
            render_mrow("revenue_growth","Revenue Growth",   fmt(info.get("revenueGrowth"), pct=True))
            render_mrow("debt_equity",   "Debt/Equity",      fmt(de_norm, dec=2))
            render_mrow("current_ratio", "Current Ratio",    fmt(info.get("currentRatio"), dec=2))
            render_mrow("beta",          "Beta",             fmt(info.get("beta"), dec=2))
            render_mrow("dividend_yield","Dividend Yield",   fmt(info.get("dividendYield"), pct=True))

        with col3:
            st.markdown(section_header("Company Info"), unsafe_allow_html=True)
            for lbl, key in [("Name","shortName"),("Sector","sector"),("Industry","industry"),
                              ("Country","country"),("Exchange","exchange"),("Currency","currency")]:
                v = info.get(key, "—")
                st.markdown(f"<div class='mrow'><span class='mrow-label'>{lbl}</span>"
                            f"<span class='mrow-val'>{v}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='mrow'><span class='mrow-label'>Mkt Cap</span>"
                        f"<span class='mrow-val'>{fmt(info.get('marketCap'), prefix='$')}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='mrow'><span class='mrow-label'>52W Range</span>"
                        f"<span class='mrow-val'>{fmt(info.get('fiftyTwoWeekLow'), prefix='$')} – "
                        f"{fmt(info.get('fiftyTwoWeekHigh'), prefix='$')}</span></div>", unsafe_allow_html=True)
            desc = info.get("longBusinessSummary", "")
            if desc:
                st.markdown(f"<div style='font-size:12px;color:#94a3b8;line-height:1.5;margin-top:10px'>"
                            f"{desc[:480]}{'…' if len(desc)>480 else ''}</div>", unsafe_allow_html=True)

        # ── Price chart in overview ───────────────────────────────────────────
        hist_ov = data.get("hist_2y", pd.DataFrame())
        if not hist_ov.empty:
            st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
            st.plotly_chart(build_price_overview_chart(hist_ov, ticker),
                            use_container_width=True)

    # ── Financials & DCF ──────────────────────────────────────────────────────
    with tabs[1]:
        st.plotly_chart(build_financials_chart(fund, ticker), use_container_width=True)
        with st.expander("📋 Cash Flow Summary"):
            cfs = fund.get("cashflows", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Operating CF",   fmt(cfs.get("operating_cf",[None])[0], prefix="$") if cfs.get("operating_cf") else "—")
            c2.metric("CapEx",          fmt(cfs.get("capex",[None])[0], prefix="$")        if cfs.get("capex")        else "—")
            c3.metric("Free Cash Flow", fmt(cfs.get("fcf",[None])[0], prefix="$")          if cfs.get("fcf")          else "—")

        st.markdown(section_header("🧮 DCF Intrinsic Value — 4 Models"), unsafe_allow_html=True)
        wacc_c = dcf.get("wacc_components", {})
        if wacc_c:
            with st.expander("⚙️ WACC Components — hover ⓘ for explanations"):
                # Inline formula display
                beta = wacc_c.get("beta", 1.0) or 1.0
                rfr  = wacc_c.get("rfr", 0.045) or 0.045
                ke   = wacc_c.get("cost_of_equity", 0) or 0
                kd   = wacc_c.get("cost_of_debt",   0) or 0
                kd_pre = wacc_c.get("kd_pretax",    0) or 0
                we   = wacc_c.get("w_equity",  1.0) or 1.0
                wd   = wacc_c.get("w_debt",    0.0) or 0.0
                tr   = wacc_c.get("tax_rate",  0.21) or 0.21
                wacc_val = wacc_c.get("wacc", 0) or 0
                mrp  = wacc_c.get("mrp", 0.055) or 0.055

                # Row 1: CAPM components
                st.markdown("**① Cost of Equity (CAPM)**")
                st.markdown(tooltip_html("beta", "Beta (β)",
                    f"{beta:.2f}",
                    position=""), unsafe_allow_html=True)
                st.markdown(tooltip_html("capm", "Risk-Free Rate (Rfr)",
                    f"{rfr:.2%}  ← 10Y US Treasury live",
                    position=""), unsafe_allow_html=True)
                st.markdown(f"""
                <div class='mrow' style='background:#1a2a3a;border-radius:6px;padding:6px 10px;margin:4px 0'>
                  <span class='mrow-label'>Market Risk Premium (MRP)</span>
                  <span class='mrow-val'>{mrp:.1%}  <span style='color:#718096;font-size:11px'>(long-run historical avg)</span></span>
                </div>
                <div class='mrow' style='background:#0d2040;border-radius:6px;padding:6px 10px;margin:4px 0;border:1px solid #2d4a6a'>
                  <span style='color:#94a3b8'>Ke = Rfr + β × MRP = </span>
                  <span style='color:#90cdf4;font-weight:700'>{rfr:.2%} + {beta:.2f} × {mrp:.1%} = <b>{ke:.2%}</b></span>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>**② Cost of Debt**", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='mrow'>
                  <span class='mrow-label'>Pre-tax Cost of Debt (Kd)</span>
                  <span class='mrow-val'>{kd_pre:.2%}  <span style='color:#718096;font-size:11px'>(Interest / Total Debt)</span></span>
                </div>
                <div class='mrow'>
                  <span class='mrow-label'>Corporate Tax Rate</span>
                  <span class='mrow-val'>{tr:.1%}  <span style='color:#718096;font-size:11px'>(from income statement)</span></span>
                </div>
                <div class='mrow' style='background:#0d2040;border-radius:6px;padding:6px 10px;margin:4px 0;border:1px solid #2d4a6a'>
                  <span style='color:#94a3b8'>After-tax Kd = Kd × (1 − Tax) = </span>
                  <span style='color:#90cdf4;font-weight:700'>{kd_pre:.2%} × (1 − {tr:.1%}) = <b>{kd:.2%}</b></span>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>**③ Capital Weights**", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='mrow'>
                  <span class='mrow-label'>Equity Weight (E/V)</span>
                  <span class='mrow-val'>{we:.1%}  <span style='color:#718096;font-size:11px'>(Market Cap / EV)</span></span>
                </div>
                <div class='mrow'>
                  <span class='mrow-label'>Debt Weight (D/V)</span>
                  <span class='mrow-val'>{wd:.1%}  <span style='color:#718096;font-size:11px'>(Total Debt / EV)</span></span>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>**④ Final WACC**", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='mrow' style='background:#0d2a0d;border-radius:8px;padding:10px 14px;margin:6px 0;border:1px solid #2d5a2d'>
                  <span style='color:#94a3b8;font-size:12px'>WACC = (E/V × Ke) + (D/V × Kd × (1−Tax))</span><br>
                  <span style='color:#94a3b8;font-size:12px'>WACC = ({we:.1%} × {ke:.2%}) + ({wd:.1%} × {kd_pre:.2%} × {1-tr:.2f})</span><br>
                  <span style='font-size:18px;font-weight:800;color:#68d391'>= {wacc_val:.2%}</span>
                  <span style='color:#718096;font-size:11px;margin-left:8px'>
                    {'(equity-funded — debt has minimal weight)' if wd < 0.05 else
                     f'({we:.0%} equity / {wd:.0%} debt blend)'}
                  </span>
                </div>
                <div style='font-size:11px;color:#718096;margin-top:4px'>
                  📚 Source: <a href="https://www.investopedia.com/terms/w/wacc.asp" target="_blank"
                  style="color:#42A5F5">Investopedia — WACC</a>  ·
                  <a href="https://www.investopedia.com/terms/c/capm.asp" target="_blank"
                  style="color:#42A5F5">CAPM</a>  ·
                  <a href="https://www.investopedia.com/terms/b/beta.asp" target="_blank"
                  style="color:#42A5F5">Beta</a>
                </div>""", unsafe_allow_html=True)

        df_dcf = build_dcf_comparison(dcf, dcf.get("current_price", 0) or cp)
        st.dataframe(df_dcf, use_container_width=True, hide_index=True)
        cp_ref = dcf.get("current_price", 0) or cp
        st.caption(f"Current price: **{fmt(cp_ref, prefix='$')}** · "
                   f"FCF base: {fmt((dcf.get('fcf_list',[None])[0]), prefix='$') if dcf.get('fcf_list') else '—'}")

        # What the numbers mean
        with st.expander("📊 What do these DCF numbers actually mean?"):
            st.markdown("""
**Intrinsic Value** — What the model estimates each share is worth today, based on projected future cash flows discounted at the chosen rate. Compare this directly to the current stock price.

| Intrinsic Value vs Price | Interpretation |
|---|---|
| IV > Price by > 30% | Potentially significantly undervalued |
| IV > Price by 10–30% | Moderate upside / margin of safety |
| IV ≈ Price (±10%) | Fairly valued |
| IV < Price by 10–30% | Potentially overvalued |
| IV < Price by > 30% | Significantly overvalued at this discount rate |

**Upside / Downside** — The percentage difference between Intrinsic Value and current market price. A positive number means the model thinks the stock has upside.

**Discount Rate** — The annual return you require (WACC/CAPM) or set manually. *Higher rate = lower intrinsic value.* This is the most sensitive input.

**Stage 1 Growth (g)** — The FCF growth rate assumed for the first 5 years, estimated from historical Free Cash Flow CAGR. Capped at 35% to avoid unrealistic projections.

**Terminal Growth (g)** — The perpetual growth rate after year 10, set at 2.5% (roughly in line with long-run nominal GDP growth). Changing this by even 0.5% can significantly move the Intrinsic Value.

**TV as % of EV** — How much of the total DCF value comes from the Terminal Value. Values above 70–80% indicate the valuation is heavily dependent on long-term assumptions — treat with more caution.

> ⚠️ **Important:** DCF models are directional tools, not precise targets. The intrinsic value range across the 4 models gives you an **uncertainty band** — a reasonable fair value range rather than a single number to bet on.
            """)

        with st.expander("📖 In-Depth: How All 4 DCF Models Work"):
            st.markdown("""
### What is a DCF?
A **Discounted Cash Flow (DCF)** model estimates the intrinsic value of a stock by projecting future Free Cash Flows and discounting them back to today's value. The core principle: a dollar received in the future is worth less than a dollar today.

**The formula:**
```
Intrinsic Value = Σ [FCF_t / (1+r)^t]  +  Terminal Value / (1+r)^n
```
Where `r` = discount rate, `t` = year, `n` = total years modelled.

---

### 🔵 Model 1 — WACC DCF
**Best for:** Companies with significant debt (banks, industrials, utilities)

The **Weighted Average Cost of Capital** blends the cost of equity and after-tax cost of debt, weighted by capital structure:
```
WACC = (E/V × Ke) + (D/V × Kd × (1 − Tax Rate))
Ke   = Risk-Free Rate + β × Market Risk Premium
Kd   = Interest Expense / Total Debt
```
This tool uses **5.5% as the Market Risk Premium** (long-run historical average). The risk-free rate is pulled live from the US 10-year Treasury (^TNX).

**Limitation:** WACC requires accurate debt/equity data and is sensitive to beta estimation.

---

### 🟢 Model 2 — CAPM DCF
**Best for:** Asset-light, low-debt companies (tech, software, consumer brands)

Uses only the **cost of equity** as the discount rate — appropriate when the company is primarily equity-funded:
```
Ke = Risk-Free Rate + β × 5.5%
```
For a stock with β=1.2 and Rfr=4.5%: Ke = 4.5% + 1.2×5.5% = **11.1%**

**Limitation:** Ignores the benefit of tax-deductible debt financing; may overstate the discount rate for leveraged companies.

---

### 🟡 Model 3 — Fixed Rate DCF
**Best for:** Sensitivity analysis and personal required-return testing

You set the discount rate manually via the sidebar slider. This lets you answer: *"What is this company worth if I require a 12% annual return?"*

Common choices:
- **8–10%:** Conservative long-term investor
- **10–12%:** Standard value investor hurdle rate
- **12–15%:** Aggressive / small-cap premium

---

### 🔴 Model 4 — Two-Stage FCF
**Best for:** High-growth companies transitioning to mature growth

Projects two distinct growth phases before terminal value:
```
Stage 1 (Years 1–5):  High growth (based on historical FCF CAGR)
Stage 2 (Years 6–10): Fading growth (Stage 1 rate × 0.4, min 2.5%)
Terminal:             Gordon Growth Model at 2.5% perpetuity
```
Example for a company with 18% historical FCF growth:
- Stage 1: 18% for 5 years
- Stage 2: 7.2% fade over 5 years
- Terminal: 2.5% forever

**Limitation:** Most sensitive model — small changes in Stage 1 growth rate cause large IV swings.

---

### ⚠️ Why the 4 models give different values
Each model answers a **different question**:
| | WACC | CAPM | Fixed | Two-Stage |
|---|---|---|---|---|
| Includes debt cost? | ✅ | ❌ | Partially | Partially |
| Uses market beta? | ✅ | ✅ | ❌ | ✅ |
| Models growth phases? | 2 | 2 | 2 | 3 |
| Best anchors to use | Enterprise value | Equity value | Hurdle rate | Growth story |

**A conservative investor** takes the lowest of the 4. **A bull case** takes the highest. The range itself tells you the **uncertainty band** around fair value.
            """)

    # ── Valuation & Peers ─────────────────────────────────────────────────────
    with tabs[2]:
        n_peers = len(peer_data)
        if n_peers == 0:
            st.info("No peer data available. Try entering manual peers in the sidebar.")
        else:
            st.success(f"**{n_peers} peers found:** {', '.join(peer_data.keys())}")

        if comparison:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(section_header("Ratio vs Peers"), unsafe_allow_html=True)
                is_pct_set = {"gross_margin","operating_margin","net_margin","roe","roa","revenue_growth","earnings_growth","dividend_yield"}
                for r in comparison:
                    t = r.get("target"); m = r.get("peer_median")
                    is_pct = r["key"] in is_pct_set
                    t_str = (f"{t*100:.1f}%" if is_pct and t else fmt(t, dec=2)) if t else "—"
                    m_str = (f"{m*100:.1f}%" if is_pct and m else fmt(m, dec=2)) if m else "—"
                    sig = r.get("signal","")
                    fs  = "beat" if "Above" in sig else ("miss" if "Below" in sig else "inline") if "In line" in sig else ""
                    st.markdown(tooltip_html(r["key"], r["metric"], f"{t_str}  (peers: {m_str})", fs),
                                unsafe_allow_html=True)
            with c2:
                fig = build_peer_chart(comparison, ticker)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)

        if peer_data:
            st.markdown(section_header("Full Peer Table"), unsafe_allow_html=True)
            st.caption("Click **▶ Analyze** to add a peer as a new analysis tab.")

            for pt, pd_info in peer_data.items():
                with st.container():
                    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([1.8, 1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2])
                    c1.markdown(f"**{pt}**  ·  <span style='color:#94a3b8;font-size:11px'>{pd_info.get('name','')[:18]}</span>", unsafe_allow_html=True)
                    c2.markdown(f"<span style='font-size:11px;color:#94a3b8'>{pd_info.get('country','')}</span>  {fmt(pd_info.get('market_cap'), prefix='$')}", unsafe_allow_html=True)
                    c3.markdown(fmt(pd_info.get("pe_trailing"), dec=1))
                    c4.markdown(fmt(pd_info.get("pe_forward"), dec=1))
                    c5.markdown(f"{pd_info.get('net_margin',0)*100:.1f}%" if pd_info.get("net_margin") else "—")
                    c6.markdown(f"{pd_info.get('roe',0)*100:.1f}%" if pd_info.get("roe") else "—")
                    c7.markdown(f"{pd_info.get('revenue_growth',0)*100:.1f}%" if pd_info.get("revenue_growth") else "—")
                    if c8.button(f"▶ Analyze", key=f"peer_analyze_{pt}_{ticker}", use_container_width=True):
                        slots = st.session_state.get("ticker_slots", ["","","","",""])
                        if pt not in slots:
                            added = False
                            for si in range(5):
                                if not slots[si]:
                                    slots[si] = pt
                                    st.session_state["ticker_slots"] = slots
                                    st.session_state[f"t{si}"] = pt
                                    st.toast(f"Added {pt} — click Analyze to run", icon="✅")
                                    added = True
                                    break
                            if not added:
                                st.toast("All 5 slots are full — remove one first", icon="⚠️")
                        else:
                            st.toast(f"{pt} is already in your analysis", icon="ℹ️")
                st.divider()

    # ── Technical Analysis ────────────────────────────────────────────────────
    with tabs[3]:
        hist = data.get("hist_2y", pd.DataFrame())
        if hist.empty or not indicators:
            st.warning("Insufficient price history for technical analysis.")
        else:
            st.plotly_chart(build_price_chart(hist, indicators, ticker), use_container_width=True)

        if signals:
            st.markdown(section_header("Signal Breakdown"), unsafe_allow_html=True)
            groups = ["sma_ema","rsi","macd","bollinger","obv","support_resistance"]
            cols   = st.columns(3)
            for i, grp in enumerate(groups):
                s = signals.get(grp, {})
                if not s: continue
                score_g = s.get("score", 0); max_g = s.get("max", 1)
                norm = (score_g + max_g) / (2 * max_g) * 10 if max_g > 0 else 5
                color = _color_for_score(norm)
                # Tooltip key mapping
                tt_key = {"sma_ema":"sma","rsi":"rsi","macd":"macd","bollinger":"bollinger","obv":"obv"}.get(grp,"")
                with cols[i % 3]:
                    lbl = s.get("label", grp)
                    if tt_key:
                        st.markdown(tooltip_html(tt_key, f"**{lbl}**", f"Score: {score_g:+d} / ±{max_g}"), unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{lbl}** — Score: `{score_g:+d}` / ±{max_g}")
                    for emoji, text in s.get("signals", []):
                        st.markdown(f"<div style='display:flex;gap:6px;font-size:12px;margin:2px 0'>"
                                    f"<span>{emoji}</span><span style='color:#cbd5e0'>{text}</span></div>",
                                    unsafe_allow_html=True)
                    st.markdown("")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    with tabs[4]:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_sent = build_sentiment_donut(sentiment, ticker)
            if fig_sent.data:
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.info("No analyst rating data available.")

            st.markdown(section_header("Consensus"), unsafe_allow_html=True)
            smets = [
                ("analyst_consensus", "Recommendation",   sentiment.get("rec_key","—").title()),
                ("analyst_consensus", "# Analysts",       str(sentiment.get("num_analysts","—"))),
                ("pe_forward",        "Target (Mean)",    fmt(sentiment.get("target_mean"), prefix="$")),
                ("pe_forward",        "Target (High)",    fmt(sentiment.get("target_high"), prefix="$")),
                ("pe_forward",        "Target (Low)",     fmt(sentiment.get("target_low"), prefix="$")),
                ("pe_forward",        "Upside to Target", fmt(sentiment.get("target_upside"), pct=True)),
            ]
            for key, lbl, val in smets:
                render_mrow(key, lbl, val)

        with c2:
            fig_earn = build_earnings_surprise_chart(earn_rows, ticker)
            if fig_earn.data:
                st.plotly_chart(fig_earn, use_container_width=True)

            st.markdown(section_header("Ownership & Short Interest"), unsafe_allow_html=True)
            for key, lbl, val in [
                ("short_interest","Institutional Ownership", fmt(sentiment.get("inst_pct"), pct=True)),
                ("short_interest","Insider Ownership",       fmt(sentiment.get("insider_pct"), pct=True)),
                ("short_interest","Short % of Float",        fmt(sentiment.get("short_pct"), pct=True)),
                ("short_interest","Short Ratio",             fmt(sentiment.get("short_ratio"), dec=1)),
                ("beta",          "52W High",                fmt(sentiment.get("52w_high"), prefix="$")),
                ("beta",          "52W Low",                 fmt(sentiment.get("52w_low"), prefix="$")),
                ("beta",          "% from 52W High",         fmt(sentiment.get("pct_from_52h"), pct=True)),
                ("beta",          "% from 52W Low",          fmt(sentiment.get("pct_from_52l"), pct=True)),
            ]:
                render_mrow(key, lbl, val)

    # ── Earnings & Forecasts ──────────────────────────────────────────────────
    with tabs[5]:
        render_earnings_section(data, ticker, earn_rows, cal_info, estimates, eps_trend_data)

    # ── Intelligence (SWOT + Customers/Suppliers + Filings) ───────────────────
    with tabs[6]:
        render_intelligence_tab(ticker, info, scoring, signals, sentiment, fund, comparison)

    # ── News ──────────────────────────────────────────────────────────────────
    with tabs[7]:
        render_news_tab(ticker, info, data.get("news", []))


# ─── Summary Dashboard ───────────────────────────────────────────────────────

def render_summary(results, rfr):
    st.markdown("## 📊 Portfolio Summary")
    valid = [(t, r) for t, r in results.items() if not r.get("error")]
    if not valid: return

    cols = st.columns(len(valid))
    for col, (ticker, r) in zip(cols, valid):
        sc = r["scoring"]; cp = get_current_price(r["data"]["info"])
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:center'>
              <div style='font-size:20px;font-weight:800;color:#fff'>{ticker}</div>
              <div style='font-size:13px;color:#94a3b8;margin-bottom:4px'>{r['data']['info'].get('shortName','')[:22]}</div>
              <div style='font-size:34px;font-weight:800'>{sc['composite']:.1f}</div>
              <span class='signal-badge' style='background:{sc["color"]};color:#fff'>{sc["signal"]}</span>
              <div style='font-size:13px;color:#94a3b8;margin-top:6px'>{fmt(cp, prefix="$")}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics comparison
    st.markdown("### 📐 Metrics Comparison")
    metrics = [
        ("P/E (TTM)",       lambda i: i.get("trailingPE"),                         False),
        ("Forward P/E",     lambda i: i.get("forwardPE"),                           False),
        ("EV/EBITDA",       lambda i: i.get("enterpriseToEbitda"),                  False),
        ("Revenue Growth",  lambda i: i.get("revenueGrowth"),                       True),
        ("Net Margin",      lambda i: i.get("profitMargins"),                       True),
        ("ROE",             lambda i: i.get("returnOnEquity"),                      True),
        ("Gross Margin",    lambda i: i.get("grossMargins"),                        True),
        ("Debt/Equity",     lambda i: (i.get("debtToEquity") or 0) / 100,          False),
        ("Beta",            lambda i: i.get("beta"),                                False),
        ("Market Cap",      lambda i: i.get("marketCap"),                           "usd"),
    ]
    rows = []
    for label, fn, fmt_type in metrics:
        row = {"Metric": label}
        for t, r in valid:
            v = fn(r["data"]["info"])
            if v is None:
                row[t] = "—"
            elif fmt_type is True:
                row[t] = f"{v*100:.1f}%"
            elif fmt_type == "usd":
                row[t] = fmt(v, prefix="$")
            else:
                row[t] = fmt(v, dec=1)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    # Score breakdown
    st.markdown("### 🏆 Score Breakdown")
    sc_rows = []
    for t, r in valid:
        sc = r["scoring"]
        earn_rows = parse_earnings_dates(r["data"].get("earnings_dates", pd.DataFrame()))
        past_surp = [row["surprise_pct"] for row in earn_rows if not row["is_future"] and row.get("surprise_pct") is not None]
        beat_rate = f"{sum(1 for s in past_surp if s>0)/len(past_surp)*100:.0f}%" if past_surp else "—"
        sc_rows.append({
            "Ticker": t, "Signal": sc["signal"], "Composite": f"{sc['composite']:.1f}",
            "Fundamental": f"{sc['scores']['fundamental']:.1f}",
            "Valuation":   f"{sc['scores']['valuation']:.1f}",
            "Technical":   f"{sc['scores']['technical']:.1f}",
            "Sentiment":   f"{sc['scores']['sentiment']:.1f}",
            "EPS Beat Rate": beat_rate,
        })
    st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)


# ─── Mobile / Deploy Guide ───────────────────────────────────────────────────

def render_deploy_guide():
    st.markdown("## 📱 Access on Phone / Web Deployment")
    st.markdown("Your Streamlit app runs in any browser — including on your phone. Here are your options:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='deploy-box'>
          <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>
            🌐 Option 1 — Streamlit Community Cloud (Free)
          </div>
          <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
            <b>Best for:</b> permanent, always-on access from any device<br><br>
            1. Push this folder to a GitHub repository<br>
            2. Go to <a href='https://share.streamlit.io' target='_blank' style='color:#42A5F5'>share.streamlit.io</a><br>
            3. Click "New app" → select your repo → <code>app.py</code><br>
            4. Deploy → get a permanent <code>https://yourapp.streamlit.app</code> URL<br>
            5. Open that URL on your phone — it works!<br><br>
            <b>Cost:</b> Free · <b>Sleep:</b> Inactive apps sleep after 7 days
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='deploy-box'>
          <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>
            🚂 Option 2 — Railway (Free tier)
          </div>
          <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
            1. Create account at <a href='https://railway.app' target='_blank' style='color:#42A5F5'>railway.app</a><br>
            2. New project → Deploy from GitHub repo<br>
            3. Set start command: <code>streamlit run app.py --server.port $PORT --server.address 0.0.0.0</code><br>
            4. Get a permanent HTTPS URL you can open on your phone<br><br>
            <b>Cost:</b> Free $5/mo credit · <b>Sleep:</b> No (always on)
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='deploy-box'>
          <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>
            🐳 Option 3 — Docker (any VPS / NAS)
          </div>
          <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
            A <code>Dockerfile</code> is included in your download.<br><br>
            <code>docker build -t stock-analyzer .</code><br>
            <code>docker run -p 8501:8501 stock-analyzer</code><br><br>
            Access at <code>http://your-server-ip:8501</code><br><br>
            Suitable for a home server, Synology NAS, or VPS (€5/mo DigitalOcean droplet).
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='deploy-box'>
          <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>
            💻 Option 4 — Local + ngrok (quick phone testing)
          </div>
          <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
            Test on your phone while running locally:<br><br>
            1. Run: <code>streamlit run app.py</code><br>
            2. Install ngrok: <a href='https://ngrok.com' target='_blank' style='color:#42A5F5'>ngrok.com</a><br>
            3. Run: <code>ngrok http 8501</code><br>
            4. Open the ngrok HTTPS URL on your phone<br><br>
            <b>Best for:</b> testing · <b>Free tier:</b> limited sessions
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='deploy-box'>
          <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>
            📲 Add to Home Screen (PWA-like)
          </div>
          <div style='font-size:13px;color:#94a3b8;line-height:1.6'>
            Once deployed (Option 1–3), open the URL in your phone browser:<br>
            • <b>iOS Safari:</b> Share → "Add to Home Screen"<br>
            • <b>Android Chrome:</b> Menu → "Add to Home Screen"<br><br>
            The app will open fullscreen like a native app!
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("# 📈 Stock Analyzer")
        st.markdown("---")
        st.markdown("**Search & add stocks**")
        st.caption("Type a name (e.g. Apple) or ticker (AAPL, ASML.AS, 7203.T)")

        # ── Live search box ──────────────────────────────────────────────────
        search_query = st.text_input("🔍 Search by name or ticker", key="search_q",
                                      placeholder="e.g. Microsoft, ASML, Samsung…",
                                      label_visibility="collapsed")
        if search_query and len(search_query) >= 2:
            with st.spinner("Searching…"):
                results = search_ticker(search_query, max_results=8)
            if results:
                st.caption("Click to add:")
                cols_s = st.columns(2)
                for idx, r in enumerate(results[:6]):
                    label = f"{r['symbol']}"
                    sub   = r['name'][:22] + ("…" if len(r['name']) > 22 else "")
                    with cols_s[idx % 2]:
                        if st.button(f"**{label}**\n{sub}", key=f"add_{r['symbol']}_{idx}",
                                     use_container_width=True):
                            slots = st.session_state.get("ticker_slots", ["","","","",""])
                            for si in range(5):
                                if not slots[si]:
                                    slots[si] = r["symbol"]
                                    st.session_state["ticker_slots"] = slots
                                    st.session_state[f"t{si}"] = r["symbol"]
                                    break
                            st.rerun()
            else:
                st.caption("No results — try a different name or enter the ticker directly.")

        st.markdown("**Selected tickers:**")

        # ── Ticker slots ─────────────────────────────────────────────────────
        if "ticker_slots" not in st.session_state:
            st.session_state["ticker_slots"] = ["", "", "", "", ""]
        slots = st.session_state["ticker_slots"]

        # Only seed widget keys that haven't been set yet (first load).
        # After that, the text_input widgets own their own state via their keys.
        # Programmatic adds (search button, peer button) set st.session_state[f"t{i}"]
        # directly and call st.rerun(), which is the correct Streamlit pattern.
        for i in range(5):
            if f"t{i}" not in st.session_state:
                st.session_state[f"t{i}"] = slots[i]

        ticker_inputs = []
        new_slots = []
        for i in range(5):
            col_t, col_x = st.columns([4, 1])
            with col_t:
                val = st.text_input(f"Slot {i+1}", key=f"t{i}",
                                    label_visibility="collapsed",
                                    placeholder=f"Ticker {i+1}")
                # Look up name for display
                if val.strip():
                    disp_name = TICKER_NAMES.get(val.strip().upper(), "")
                    if disp_name:
                        st.caption(f"↳ {disp_name[:30]}")
            with col_x:
                if slots[i] and st.button("✕", key=f"clear_{i}", help="Remove"):
                    slots[i] = ""
                    st.session_state["ticker_slots"] = slots
                    st.session_state[f"t{i}"] = ""
                    st.rerun()
            new_slots.append(val.strip().upper() if val.strip() else "")
            if val.strip():
                ticker_inputs.append(val.strip().upper())
        st.session_state["ticker_slots"] = new_slots

        st.markdown("---")
        fixed_rate = st.slider("Fixed Discount Rate (Model 3)", 4.0, 20.0, 10.0, 0.5, format="%.1f%%") / 100

        st.markdown("---")
        st.markdown("**Peer Override** *(optional)*")
        peer_overrides = {}
        for t in ticker_inputs:
            with st.expander(f"Override peers for {t}", expanded=False):
                ov = st.text_input("Comma-separated tickers", key=f"peer_{t}",
                                    placeholder="e.g. MSFT, SAP.DE, 035420.KS")
                peer_overrides[t] = ov

        st.markdown("---")
        run_btn   = st.button("🔍 Analyze",      type="primary", use_container_width=True)
        clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
        deploy_btn = st.button("📱 Mobile/Deploy Guide", use_container_width=True)

        if clear_btn:
            for k in ["results","rfr","show_deploy"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")
        st.markdown("---")
        st.markdown("**🌍 Display Currency**")
        selected_ccy, fx_rates = sidebar_currency_selector()
        st.session_state["display_ccy_code"] = selected_ccy
        st.session_state["fx_rates"]         = fx_rates

        st.markdown("---")
        st.caption("Data: Yahoo Finance · Cache: 1h\n"
                   "Peers: 60+ industries · 30+ exchanges\n"
                   "Scores: F=40% V=25% T=20% S=15%\n"
                   "⚠️ Rate limits: analyse 1–2 tickers\n"
                   "at a time for best results.")

    if deploy_btn:
        st.session_state["show_deploy"] = True

    if st.session_state.get("show_deploy"):
        render_deploy_guide()
        if st.button("← Back to Analyzer"):
            st.session_state.pop("show_deploy")
            st.rerun()
        return

    if not ticker_inputs and "results" not in st.session_state:
        st.markdown("# 📈 Stock Analyzer")
        st.markdown(
            "Comprehensive equity analysis for stocks across **30+ global exchanges**. "
            "Enter up to **5 tickers** in the sidebar and click **Analyze**.\n\n"
            "**Features:**\n"
            "- 🌍 **International peers** — auto-detected from US, Canada, Europe, Japan, Korea, China\n"
            "- 🧮 **4 DCF Models** — WACC, CAPM, Fixed Rate, Two-Stage FCF (side-by-side)\n"
            "- 📈 **All technical indicators** — each with ✅/🔴 signal feedback\n"
            "- ℹ️ **Hover tooltips** — click ⓘ next to any metric for definition + formula\n"
            "- 📅 **Earnings calendar** — dates, EPS estimate vs actual, beat/miss history\n"
            "- 📊 **Forecast tracking** — EPS revision trends, estimate vs consensus\n"
            "- 📱 **Phone access** — deploy guide included\n\n"
            "**European tickers:** ASML.AS · SHEL.L · SAP.DE · TTE.PA · NOVN.SW · 0700.HK · 005930.KS"
        )
        st.info("👈 Enter tickers in the sidebar, then click **Analyze**.")
        return

    if run_btn and ticker_inputs:
        with st.spinner("Fetching data and running analysis…"):
            rfr     = fetch_risk_free_rate()
            results = {}
            progress = st.progress(0, text="")
            n = len(ticker_inputs)

            for idx, ticker in enumerate(ticker_inputs):
                progress.progress(idx / n, text=f"Analysing {ticker}…")
                data = fetch_ticker_data(ticker)
                if data.get("error"):
                    results[ticker] = {"error": data["error"]}
                    continue

                info = data["info"]
                peers_list = get_peers(ticker, info, peer_overrides.get(ticker, ""))
                peer_data  = fetch_peer_metrics(tuple(peers_list)) if peers_list else {}

                fund       = analyze_fundamentals(data)
                dcf        = run_dcf_models(data, rfr, fixed_rate)
                ind        = calculate_indicators(data.get("hist_2y", pd.DataFrame()))
                cp         = get_current_price(info)
                sigs       = generate_signals(ind, cp)
                valuation  = get_valuation_ratios(info)
                comparison = compare_with_peers(valuation, peer_data, ticker)
                sentiment  = analyze_sentiment(data)

                cfs        = fund.get("cashflows", {})
                fs, fmsgs  = score_fundamental(info, cfs)
                vs, vmsgs  = score_valuation(info, dcf)
                tech_score = sigs.get("overall_score", 5.0)
                ss, smsgs  = score_sentiment(sentiment)
                scoring    = calculate_composite(fs, vs, tech_score, ss)
                scoring["fund_msgs"] = fmsgs; scoring["val_msgs"] = vmsgs; scoring["sent_msgs"] = smsgs

                results[ticker] = {
                    "data": data, "peers": peers_list, "peer_data": peer_data,
                    "fund": fund, "dcf": dcf, "indicators": ind, "signals": sigs,
                    "valuation": valuation, "comparison": comparison,
                    "sentiment": sentiment, "scoring": scoring,
                }

            progress.progress(1.0, text="Done!")
            st.session_state["results"] = results
            st.session_state["rfr"]     = rfr

    results = st.session_state.get("results", {})
    rfr     = st.session_state.get("rfr", 0.045)
    if not results: return

    for ticker, r in results.items():
        if r.get("error"):
            st.error(f"**{ticker}**: {r['error']}")

    valid = {t: r for t, r in results.items() if not r.get("error")}
    if not valid: return

    tab_labels = ["📊 Summary"] + [f"📈 {t}" for t in valid]
    top_tabs   = st.tabs(tab_labels)

    with top_tabs[0]:
        render_summary(valid, rfr)

    for i, (ticker, r) in enumerate(valid.items()):
        with top_tabs[i + 1]:
            render_stock_tab(
                ticker=ticker, data=r["data"], dcf=r["dcf"], fund=r["fund"],
                indicators=r["indicators"], signals=r["signals"], valuation=r["valuation"],
                comparison=r["comparison"], sentiment=r["sentiment"],
                scoring=r["scoring"], peer_data=r["peer_data"],
            )


if __name__ == "__main__":
    main()