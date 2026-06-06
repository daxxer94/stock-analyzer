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
from screener_ui         import render_screener_page
from portfolio_ui        import render_portfolio_page
from cca                 import run_cca, format_cca_peer_table
from financial_analysis  import (run_deep_analysis, DRIVER_COLORS, SENSITIVITY_COLORS,
                               SECTOR_MACRO_SENSITIVITY)
try:
    from financial_analysis import get_government_factors, RISK_COLORS
except ImportError:
    def get_government_factors(info): return {}
    RISK_COLORS = {"very high":"#EF5350","high":"#FF9800","moderate":"#FFC107","low":"#66BB6A"}

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
    """Safe formatter — coerces v to float, returns '—' on any bad value."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(v) or np.isinf(v):
        return "—"
    if pct:
        return f"{v:+.1%}" if abs(v) < 10 else f"{v:.1%}"
    if prefix == "$":
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
    """Area chart: full price history with clear currency label, date x-axis, clean legend."""
    if hist is None or hist.empty:
        return go.Figure()

    close  = hist["Close"].astype(float)
    dates  = hist.index
    is_up  = float(close.iloc[-1]) >= float(close.iloc[0])
    color  = "#26a69a" if is_up else "#ef5350"
    r, g, b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)

    # Currency info
    native_ccy  = info.get("currency", "USD")
    disp_ccy    = st.session_state.get("display_ccy_code", native_ccy)
    rates       = st.session_state.get("fx_rates", {})
    ccy_sym     = CURRENCY_SYMBOLS.get(disp_ccy, disp_ccy)
    showing_ccy = disp_ccy if disp_ccy != native_ccy else native_ccy

    cp   = float(close.iloc[-1])
    high = float(close.max())
    low  = float(close.min())
    chg  = (cp - float(close.iloc[0])) / float(close.iloc[0])
    cp_str   = fmt_currency(cp,   native_ccy, disp_ccy, rates)
    high_str = fmt_currency(high, native_ccy, disp_ccy, rates)
    low_str  = fmt_currency(low,  native_ccy, disp_ccy, rates)

    years = (dates[-1] - dates[0]).days / 365.25

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=close,
        name=f"{ticker} ({showing_ccy})",
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.08)",
        hovertemplate=f"{ccy_sym}%{{y:,.2f}}<br>%{{x|%b %d, %Y}}<extra></extra>",
    ))

    # Current price line
    fig.add_hline(y=cp, line=dict(color="white", width=1, dash="dot"),
                  annotation_text=f" {cp_str}", annotation_position="right",
                  annotation_font=dict(size=11, color="white"))

    fig.update_layout(
        height=280,
        template="plotly_dark",
        margin=dict(l=60, r=80, t=70, b=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0, y=1.18,          # above chart, no overlap with title
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=(
                f"<b>{ticker}</b>  ·  {cp_str}  "
                f"<span style='color:{'#26a69a' if is_up else '#ef5350'}'>"
                f"{'▲' if is_up else '▼'} {abs(chg):.1%}</span>"
                f"  <span style='font-size:11px;color:#718096'>({years:.1f}Y history · {showing_ccy})</span>"
            ),
            font=dict(size=13),
            x=0.01, y=0.97,
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
            tickformat="%b '%y",     # "Jan '23" format — no decimals
            dtick="M6",              # tick every 6 months
            tickangle=-30,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11),
            tickprefix=ccy_sym,
        ),
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )

    # High/low annotations
    hi_idx = close.idxmax(); lo_idx = close.idxmin()
    fig.add_annotation(x=hi_idx, y=high, text=f"▲ {high_str}",
        showarrow=True, arrowhead=2, arrowcolor="#FFD700",
        font=dict(color="#FFD700", size=10), bgcolor="#0e1117",
        bordercolor="#FFD700", borderwidth=1, ay=-28)
    fig.add_annotation(x=lo_idx, y=low, text=f"▼ {low_str}",
        showarrow=True, arrowhead=2, arrowcolor="#ef5350",
        font=dict(color="#ef5350", size=10), bgcolor="#0e1117",
        bordercolor="#ef5350", borderwidth=1, ay=28)
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



def build_football_field(football_field: list, dcf_results: dict,
                          current_price: float, native_ccy: str) -> go.Figure:
    """
    Football field valuation chart combining CCA ranges + DCF values.
    Horizontal bars show low (p25) to high (p75) with median dot.
    """
    disp_ccy = st.session_state.get("display_ccy_code", native_ccy)
    rates    = st.session_state.get("fx_rates", {})

    rows = []
    # CCA rows
    for f in football_field:
        rows.append({
            "label": f["label"],
            "low":   f["low"],
            "mid":   f["mid"],
            "high":  f["high"],
            "color": "#42A5F5",
            "type":  "CCA",
        })
    # DCF rows
    dcf_labels = [("wacc","WACC DCF"),("capm","CAPM DCF"),
                  ("fixed","Fixed Rate"),("two_stage","Two-Stage")]
    for key, lbl in dcf_labels:
        m = dcf_results.get(key, {})
        iv = m.get("intrinsic_value")
        if iv and iv > 0 and not m.get("error"):
            rows.append({"label": lbl, "low": iv*0.85, "mid": iv, "high": iv*1.15,
                         "color": "#66BB6A", "type": "DCF"})

    if not rows:
        return go.Figure()

    labels = [r["label"] for r in rows]
    lows   = [r["low"]   for r in rows]
    mids   = [r["mid"]   for r in rows]
    highs  = [r["high"]  for r in rows]
    colors = [r["color"] for r in rows]

    fig = go.Figure()

    # Range bars (low → high)
    for i, row in enumerate(rows):
        lo  = fmt_currency(row["low"],  native_ccy, disp_ccy, rates)
        hi  = fmt_currency(row["high"], native_ccy, disp_ccy, rates)
        mid = fmt_currency(row["mid"],  native_ccy, disp_ccy, rates)
        fig.add_trace(go.Bar(
            x=[row["high"] - row["low"]], y=[row["label"]],
            base=[row["low"]], orientation="h",
            marker_color=row["color"], opacity=0.35,
            showlegend=False,
            hovertemplate=f"{row['label']}<br>Range: {lo} – {hi}<br>Mid: {mid}<extra></extra>",
        ))
        # Median dot
        fig.add_trace(go.Scatter(
            x=[row["mid"]], y=[row["label"]],
            mode="markers",
            marker=dict(color=row["color"], size=10, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"{row['label']}<br>Midpoint: {mid}<extra></extra>",
        ))

    # Current price line
    fig.add_vline(x=current_price,
                  line=dict(color="#FC8181", width=2, dash="dash"),
                  annotation_text=f"  Current {fmt_currency(current_price, native_ccy, disp_ccy, rates)}",
                  annotation_font=dict(color="#FC8181", size=11))

    # Legend annotations
    fig.add_annotation(x=0.02, y=1.04, xref="paper", yref="paper",
        text="<span style='color:#42A5F5'>■</span> CCA Range (P25–P75)  "
             "<span style='color:#66BB6A'>■</span> DCF ±15%  "
             "<span style='color:#FC8181'>|</span> Current Price",
        showarrow=False, font=dict(size=11), xanchor="left")

    all_vals = [r["high"] for r in rows] + [r["low"] for r in rows] + [current_price]
    x_min = max(0, min(all_vals) * 0.80)
    x_max = max(all_vals) * 1.15

    fig.update_layout(
        title=dict(text="⚽ Football Field Valuation — CCA + DCF Ranges",
                   font_size=13, x=0.01),
        height=max(300, len(rows) * 52 + 100),
        template="plotly_dark",
        barmode="overlay",
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        margin=dict(l=160, r=60, t=60, b=40),
        xaxis=dict(
            title="Implied Share Price",
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11),
            range=[x_min, x_max],
        ),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    return fig


def build_sensitivity_chart(sens: dict, current_price: float, native_ccy: str) -> go.Figure:
    """Heat-map style table: WACC × Terminal Growth → Implied Price."""
    if not sens or not sens.get("table"):
        return go.Figure()

    wacc_lbls = [f"{w:.1f}%" for w in sens["wacc_values"]]
    tg_lbls   = [f"{t:.1f}%" for t in sens["tg_values"]]
    table     = sens["table"]

    # Color each cell relative to current price
    z     = table
    texts = [[f"${table[r][c]:.2f}" for c in range(len(tg_lbls))]
             for r in range(len(wacc_lbls))]

    fig = go.Figure(go.Heatmap(
        z=z, x=tg_lbls, y=wacc_lbls,
        text=texts, texttemplate="%{text}",
        colorscale=[
            [0.0,  "#B71C1C"],
            [0.35, "#EF5350"],
            [0.50, "#FFC107"],
            [0.65, "#66BB6A"],
            [1.0,  "#00C853"],
        ],
        zmid=current_price,
        showscale=True,
        colorbar=dict(title="Implied Price", tickfont=dict(size=10)),
        hovertemplate="WACC: %{y}<br>Terminal g: %{x}<br>Implied: %{text}<extra></extra>",
    ))

    # Highlight the base case cell
    bi = sens.get("base_wacc_idx", 2)
    ti = sens.get("base_tg_idx",   2)
    if bi < len(wacc_lbls) and ti < len(tg_lbls):
        fig.add_shape(type="rect",
            x0=ti-0.5, x1=ti+0.5, y0=bi-0.5, y1=bi+0.5,
            line=dict(color="white", width=2))

    fig.update_layout(
        title=dict(text="Sensitivity: WACC × Terminal Growth Rate → Implied Share Price",
                   font_size=12, x=0.01),
        height=260,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        margin=dict(l=60, r=60, t=50, b=40),
        xaxis=dict(title="Terminal Growth Rate (g)", tickfont=dict(size=11)),
        yaxis=dict(title="WACC", tickfont=dict(size=11)),
        font=dict(size=11),
    )
    return fig



def build_stock_benchmark_chart(ticker: str, hist: pd.DataFrame,
                                  info: dict) -> None:
    """
    Render an interactive stock vs benchmark comparison chart.
    Called inside the Technical Analysis tab.
    """
    from portfolio import BENCHMARKS, TIMEFRAMES
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    native_ccy = info.get("currency","USD")
    name       = info.get("shortName") or ticker

    st.markdown(section_header("📊 Price vs Benchmark"), unsafe_allow_html=True)
    st.caption("Both lines rebased to 100 at the start of the selected period.")

    col1, col2, col3 = st.columns([2.5, 2, 0.8])
    with col1:
        bench_name = st.selectbox(
            "Benchmark", list(BENCHMARKS.keys()), index=0,
            key=f"stock_bench_{ticker}")
        bench_ticker = BENCHMARKS[bench_name]
    with col2:
        tf_options = list(TIMEFRAMES.keys())
        tf = st.selectbox(
            "Timeframe", tf_options,
            index=tf_options.index("1Y") if "1Y" in tf_options else 4,
            key=f"stock_tf_{ticker}")
        days = TIMEFRAMES[tf]
    with col3:
        show_vol = st.checkbox("Volume", value=False, key=f"vol_{ticker}")

    # Determine yfinance period string
    period_map = {7:"5d",30:"1mo",90:"3mo",180:"6mo",
                  365:"1y",730:"2y",1095:"3y",None:"ytd"}
    period = period_map.get(days, "1y")

    import yfinance as yf

    @st.cache_data(ttl=3600, show_spinner=False)
    def _fetch(sym, per):
        try:
            h = yf.Ticker(sym).history(period=per)
            return h["Close"].dropna() if h is not None and not h.empty else pd.Series()
        except Exception:
            return pd.Series()

    stock_hist = _fetch(ticker, period)
    bench_hist = _fetch(bench_ticker, period)

    # YTD filter
    if days is None:
        ytd = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
        def _tz_filter(s, t):
            if s.empty: return s
            t2 = t.tz_localize(s.index.tz) if s.index.tz else t.tz_localize(None)
            return s[s.index >= t2]
        stock_hist = _tz_filter(stock_hist, ytd)
        bench_hist = _tz_filter(bench_hist, ytd)
    elif days:
        now = pd.Timestamp.now()
        cutoff = now - pd.Timedelta(days=days)
        def _day_filter(s):
            if s.empty: return s
            c2 = cutoff.tz_localize(s.index.tz) if s.index.tz else cutoff.tz_localize(None)
            return s[s.index >= c2]
        stock_hist = _day_filter(stock_hist)
        bench_hist = _day_filter(bench_hist)

    if stock_hist.empty:
        st.warning("No price history available for this ticker.")
        return

    # Rebase to 100
    s_rebased = stock_hist / stock_hist.iloc[0] * 100
    b_rebased = bench_hist / bench_hist.iloc[0] * 100 if not bench_hist.empty else pd.Series()

    rows = 2 if show_vol else 1
    fig  = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                         vertical_spacing=0.06,
                         row_heights=[0.75,0.25] if show_vol else [1.0])

    s_ret  = (s_rebased.iloc[-1]/100 - 1)*100 if len(s_rebased)>1 else 0
    s_col  = "#00C853" if s_ret >= 0 else "#EF5350"
    fig.add_trace(go.Scatter(
        x=s_rebased.index, y=s_rebased.values,
        name=f"{ticker}  ({s_ret:+.1f}%)",
        line=dict(color=s_col, width=2.5), mode="lines",
        hovertemplate="%{y:.1f}<extra>" + ticker + "</extra>",
    ), row=1, col=1)

    if not b_rebased.empty:
        b_ret  = (b_rebased.iloc[-1]/100 - 1)*100 if len(b_rebased)>1 else 0
        b_col  = "#42A5F5"
        fig.add_trace(go.Scatter(
            x=b_rebased.index, y=b_rebased.values,
            name=f"{bench_name}  ({b_ret:+.1f}%)",
            line=dict(color=b_col, width=2, dash="dot"), mode="lines",
            hovertemplate="%{y:.1f}<extra>" + bench_name + "</extra>",
        ), row=1, col=1)

    fig.add_hline(y=100, line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
                  row=1, col=1)

    if show_vol and not hist.empty and "Volume" in hist.columns:
        vols  = hist["Volume"].tail(len(s_rebased))
        vcols = ["#00C853" if c >= o else "#EF5350"
                 for c, o in zip(hist["Close"].tail(len(s_rebased)),
                                 hist["Open"].tail(len(s_rebased)))]
        fig.add_trace(go.Bar(
            x=hist.index[-len(s_rebased):], y=vols.values,
            name="Volume", marker_color=vcols, showlegend=False,
            hovertemplate="%{y:,.0f}<extra>Volume</extra>",
        ), row=2, col=1)

    fig.update_layout(
        height=400 if not show_vol else 520,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        margin=dict(l=50, r=30, t=30, b=40),
        legend=dict(orientation="h", x=0, y=1.05,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        xaxis=dict(showgrid=False, tickfont=dict(size=10),
                   tickformat="%d %b '%y",
                   rangeslider=dict(visible=True, thickness=0.04)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   tickfont=dict(size=10), title="Rebased to 100"),
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    if show_vol:
        fig.update_yaxes(tickfont=dict(size=9), row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, key=f"pc_{ticker}_1")

def build_financials_chart(fund, ticker):
    rev = fund.get("revenue_series", {}); ni = fund.get("net_income_series", {})
    gm  = fund.get("gross_margin_series", {}); om = fund.get("op_margin_series", {}); nm = fund.get("net_margin_series", {})
    years = sorted(set(list(rev.keys()) + list(ni.keys())), reverse=True)[:6]; years.sort()

    # Get display currency
    native_ccy = "USD"
    disp_ccy   = st.session_state.get("display_ccy_code", "USD")
    ccy_sym    = CURRENCY_SYMBOLS.get(disp_ccy, disp_ccy)

    # Use two completely independent figures stacked — cleanest legend control
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.22,
        row_heights=[0.55, 0.45],
        subplot_titles=["", ""],   # we use annotations instead
    )

    # ── Panel 1: Revenue & Net Income ────────────────────────────────────
    fig.add_trace(go.Bar(
        x=years, y=[rev.get(y) for y in years],
        name="Revenue", marker_color="#42A5F5", opacity=0.85,
        legendgroup="g1", legendgrouptitle=dict(text=""),
        hovertemplate=f"{ccy_sym}%{{y:,.0f}}<extra>Revenue</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=years, y=[ni.get(y) for y in years],
        name="Net Income", marker_color="#66BB6A", opacity=0.85,
        legendgroup="g1",
        hovertemplate=f"{ccy_sym}%{{y:,.0f}}<extra>Net Income</extra>",
    ), row=1, col=1)

    # ── Panel 2: Margins ──────────────────────────────────────────────────
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
                legendgroup="g2",
                hovertemplate="%{y:.1f}%<extra>" + lbl + "</extra>",
            ), row=2, col=1)

    # Subplot title annotations — manually placed to avoid overlap
    fig.add_annotation(text=f"<b>Revenue & Net Income</b>  ({disp_ccy})",
        xref="paper", yref="paper", x=0, y=1.02,
        xanchor="left", showarrow=False,
        font=dict(size=12, color="#e2e8f0"))
    fig.add_annotation(text="<b>Profit Margins</b>",
        xref="paper", yref="paper", x=0, y=0.44,
        xanchor="left", showarrow=False,
        font=dict(size=12, color="#e2e8f0"))

    fig.update_layout(
        height=580,
        template="plotly_dark",
        barmode="group",
        margin=dict(l=70, r=30, t=50, b=50),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
        # Legend 1 (income): just above panel 1
        # Legend 2 (margins): just above panel 2
        # Plotly supports only one legend — use legend2 for second group
        legend=dict(
            orientation="h", x=0.55, y=1.02,
            xanchor="left", yanchor="bottom",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
            tracegroupgap=0,
        ),
        legend2=dict(
            orientation="h", x=0.55, y=0.44,
            xanchor="left", yanchor="bottom",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
            tracegroupgap=0,
        ),
    )

    # Assign traces to legend2 (margins = g2)
    for trace in fig.data:
        if trace.legendgroup == "g2":
            trace.legend = "legend2"

    # Format axes
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     tickfont=dict(size=11), tickprefix=ccy_sym, row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     tickfont=dict(size=11), ticksuffix="%", row=2, col=1)
    fig.update_xaxes(tickfont=dict(size=11), tickformat="%Y",
                     dtick="M12", showgrid=False)
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
            st.plotly_chart(fig_trend, use_container_width=True, key=f"pc_{ticker}_2")

    # ── Historical Earnings Surprise ─────────────────────────────────────────
    past = [r for r in earn_rows if not r["is_future"]]
    st.markdown(section_header("📊 EPS Surprise History"), unsafe_allow_html=True)
    if past:
        fig_sur = build_earnings_surprise_chart(earn_rows, ticker)
        if fig_sur.data:
            st.plotly_chart(fig_sur, use_container_width=True, key=f"pc_{ticker}_3")

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
    """Recent news + earnings summary + source links."""
    import urllib.parse, datetime
    company_name = info.get("shortName", ticker)
    sector       = info.get("sector","")
    tick_enc     = urllib.parse.quote(ticker)

    # ── Earnings summary from available data ──────────────────────────────────
    st.markdown(section_header("📊 Recent Earnings Summary"), unsafe_allow_html=True)

    eps_ttm   = info.get("trailingEps")
    eps_fwd   = info.get("forwardEps")
    rev       = info.get("totalRevenue")
    rev_g     = info.get("revenueGrowth")
    eps_g     = info.get("earningsGrowth")
    net_m     = info.get("profitMargins")
    gross_m   = info.get("grossMargins")
    op_m      = info.get("operatingMargins")
    ebitda    = info.get("ebitda")
    fcf       = info.get("freeCashflow")
    rec_mean  = info.get("recommendationMean")
    n_ana     = info.get("numberOfAnalystOpinions",0)
    tgt_mean  = info.get("targetMeanPrice")
    cp        = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    e1,e2,e3,e4 = st.columns(4)
    if eps_ttm is not None:
        e1.metric("EPS (TTM)",  f"${eps_ttm:.2f}" if eps_ttm else "N/A",
                  f"Fwd: ${eps_fwd:.2f}" if eps_fwd else "")
    if rev:
        rev_lbl = f"${rev/1e9:.1f}B" if rev >= 1e9 else f"${rev/1e6:.0f}M"
        e2.metric("Revenue (TTM)", rev_lbl,
                  f"{rev_g:+.1%} YoY" if rev_g else "")
    if net_m is not None:
        m_delta = f"GM: {gross_m:.1%}" if gross_m else ""
        e3.metric("Net Margin", f"{net_m:.1%}", m_delta)
    if fcf:
        fcf_lbl = f"${fcf/1e9:.1f}B" if abs(fcf)>=1e9 else f"${fcf/1e6:.0f}M"
        e4.metric("Free Cash Flow", fcf_lbl)

    # Analyst consensus
    if tgt_mean and cp:
        upside = (tgt_mean - cp) / cp
        c_up   = "#00C853" if upside > 0.05 else ("#EF5350" if upside < -0.05 else "#FFC107")
        rec_labels = {1:"Strong Buy",2:"Buy",3:"Hold",4:"Sell",5:"Strong Sell"}
        rec_lbl    = rec_labels.get(round(rec_mean), "Hold") if rec_mean else "N/A"
        st.markdown(
            f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;"
            f"padding:12px 16px;margin:8px 0;display:flex;gap:24px;flex-wrap:wrap'>"
            f"<div><div style='font-size:11px;color:#718096'>Analyst Consensus</div>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0'>{rec_lbl}</div></div>"
            f"<div><div style='font-size:11px;color:#718096'>Price Target (mean)</div>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0'>${tgt_mean:.2f}</div></div>"
            f"<div><div style='font-size:11px;color:#718096'>Implied upside</div>"
            f"<div style='font-size:16px;font-weight:700;color:{c_up}'>{upside:+.1%}</div></div>"
            f"<div><div style='font-size:11px;color:#718096'>Analysts covering</div>"
            f"<div style='font-size:16px;font-weight:700;color:#e2e8f0'>{n_ana}</div></div>"
            f"</div>", unsafe_allow_html=True)

    # Earnings quality assessment
    if eps_ttm is not None or net_m is not None:
        with st.expander("📋 Earnings Quality Assessment"):
            items = []
            if eps_ttm and eps_ttm > 0:
                items.append(("✅","Profitable","Company generating positive EPS"))
            elif eps_ttm and eps_ttm < 0:
                items.append(("⚠️","Loss-making",f"EPS: ${eps_ttm:.2f} — monitor path to profitability"))
            if fcf and fcf > 0:
                items.append(("✅","Positive FCF","Cash generation supports operations"))
            elif fcf and fcf < 0:
                items.append(("⚠️","Negative FCF","Company burning cash — check runway"))
            if gross_m and gross_m > 0.40:
                items.append(("✅","Strong gross margins",f"{gross_m:.1%} — pricing power evident"))
            elif gross_m and gross_m < 0.15:
                items.append(("⚠️","Thin margins",f"{gross_m:.1%} — vulnerable to cost inflation"))
            if rev_g and rev_g > 0.15:
                items.append(("🚀","High revenue growth",f"{rev_g:+.1%} YoY"))
            elif rev_g and rev_g < 0:
                items.append(("🔴","Revenue declining",f"{rev_g:+.1%} YoY"))
            for icon, lbl, desc in items:
                st.markdown(
                    f"<div style='padding:5px 0;font-size:12px'>"
                    f"<span>{icon} <b>{lbl}</b></span> — "
                    f"<span style='color:#94a3b8'>{desc}</span></div>",
                    unsafe_allow_html=True)
            if not items:
                st.info("Insufficient earnings data for quality assessment.")

    st.divider()

    # ── Free news source links ────────────────────────────────────────────────
    st.markdown(section_header("🔗 News Sources"), unsafe_allow_html=True)
    links = build_news_search_links(ticker, company_name)
    cols  = st.columns(4)
    for i, link in enumerate(links):
        with cols[i % 4]:
            st.markdown(
                f"""<a href="{link['url']}" target="_blank"
                   style="display:block;background:#1a1d2e;border:1px solid #3a3f5c;
                   border-radius:8px;padding:8px 10px;text-decoration:none;
                   color:#90cdf4;font-size:12px;margin-bottom:6px;text-align:center">
                   {link['icon']} {link['source']}</a>""",
                unsafe_allow_html=True)

    st.divider()

    # ── Recent news feed ──────────────────────────────────────────────────────
    st.markdown(section_header("📰 Latest News"), unsafe_allow_html=True)

    if news_items:
        # Sort by date (most recent first)
        def _ts(item):
            return item.get("providerPublishTime", 0) or 0
        sorted_news = sorted(news_items, key=_ts, reverse=True)

        now = datetime.datetime.now()
        for item in sorted_news[:15]:
            title     = item.get("title","")
            link      = item.get("link","")
            publisher = item.get("publisher","")
            ts        = item.get("providerPublishTime", 0)
            if not title or not link:
                continue

            date_str  = ""
            age_badge = ""
            if ts:
                try:
                    dt       = datetime.datetime.fromtimestamp(ts)
                    days_ago = (now - dt).days
                    date_str = dt.strftime("%b %d, %Y")
                    if days_ago == 0:
                        age_badge = "<span style='background:#00C853;color:#000;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:6px'>Today</span>"
                    elif days_ago <= 3:
                        age_badge = f"<span style='background:#FFC107;color:#000;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:6px'>{days_ago}d ago</span>"
                except Exception:
                    pass

            # Highlight earnings-related articles
            t_lower   = title.lower()
            is_earn   = any(w in t_lower for w in ["earnings","revenue","eps","profit","loss","quarterly","results","guidance"])
            border_c  = "#FFC107" if is_earn else "#2d3748"
            st.markdown(
                f"""<div style='background:#1a1d2e;border:1px solid {border_c};
                    border-radius:8px;padding:12px 16px;margin-bottom:6px'>
                  <a href="{link}" target="_blank"
                     style="color:#90cdf4;font-size:13px;font-weight:600;text-decoration:none">
                    {title}
                  </a>{age_badge}
                  <div style="font-size:11px;color:#718096;margin-top:4px">
                    {publisher}{"  ·  " + date_str if date_str else ""}
                    {"  ·  📊 <i>Earnings related</i>" if is_earn else ""}
                  </div>
                </div>""",
                unsafe_allow_html=True)
    else:
        st.info("No recent news found via Yahoo Finance. Use the source links above.")

    # ── Earnings calendar links ───────────────────────────────────────────────
    st.markdown(section_header("📅 Earnings Calendar"), unsafe_allow_html=True)
    st.markdown(
        f"[📅 Earnings Whispers](https://www.earningswhispers.com/stocks/{ticker.lower()})  ·  "
        f"[📊 Seeking Alpha Earnings](https://seekingalpha.com/symbol/{tick_enc}/earnings)  ·  "
        f"[🗓️ Yahoo Finance Financials](https://finance.yahoo.com/quote/{tick_enc}/financials/)  ·  "
        f"[📈 MarketBeat Earnings](https://www.marketbeat.com/stocks/{info.get('exchange','NASDAQ')}/{ticker}/earnings/)"
    )



def render_cca_tab(ticker, info, peer_data, dcf, current_price):
    """
    Comparable Company Analysis tab.
    BIWS methodology: peer multiples → implied value ranges → football field.
    """
    native_ccy = info.get("currency", "USD")
    disp_ccy   = st.session_state.get("display_ccy_code", native_ccy)
    rates      = st.session_state.get("fx_rates", {})

    if not peer_data:
        st.info("No peer data available — run analysis with peers to enable CCA.")
        return

    cca = run_cca(info, peer_data)
    if not cca:
        st.warning("Could not compute CCA — insufficient peer data.")
        return

    summ = cca.get("summary", {})

    # ── Header: CCA Implied Range ─────────────────────────────────────────────
    ov_low  = summ.get("overall_low")
    ov_mid  = summ.get("overall_mid")
    ov_high = summ.get("overall_high")
    upside  = summ.get("implied_upside")

    if ov_mid:
        udcolor = "#00C853" if (upside or 0) >= 0 else "#EF5350"
        st.markdown(f"""
        <div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:12px;
                     padding:18px 24px;margin-bottom:16px'>
          <div style='font-size:12px;color:#718096;margin-bottom:4px'>
            Comparable Company Analysis — {summ.get("n_peers",0)} peers · {summ.get("n_multiples",0)} multiples
          </div>
          <div style='display:flex;gap:28px;flex-wrap:wrap;align-items:center'>
            <div>
              <div style='font-size:11px;color:#718096'>Current Price</div>
              <div style='font-size:20px;font-weight:800;color:#e2e8f0'>
                {fmt_currency(current_price, native_ccy, disp_ccy, rates)}
              </div>
            </div>
            <div style='font-size:20px;color:#718096'>→</div>
            <div>
              <div style='font-size:11px;color:#718096'>CCA Median Implied</div>
              <div style='font-size:20px;font-weight:800;color:#e2e8f0'>
                {fmt_currency(ov_mid, native_ccy, disp_ccy, rates)}
              </div>
              {f"<div style='font-size:13px;color:{udcolor};font-weight:700'>{'▲' if (upside or 0)>=0 else '▼'} {abs(upside):.1%} implied {'upside' if (upside or 0)>=0 else 'downside'}</div>" if upside is not None else ""}
            </div>
            <div>
              <div style='font-size:11px;color:#718096'>CCA Range (P25–P75)</div>
              <div style='font-size:15px;color:#94a3b8;font-weight:600'>
                {fmt_currency(ov_low, native_ccy, disp_ccy, rates)} – {fmt_currency(ov_high, native_ccy, disp_ccy, rates)}
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Football Field ────────────────────────────────────────────────────────
    ff = cca.get("football_field", [])
    if ff:
        fig_ff = build_football_field(ff, dcf, current_price, native_ccy)
        if fig_ff.data:
            st.plotly_chart(fig_ff, use_container_width=True, key=f"pc_{ticker}_4")

    # ── Methodology explanation ───────────────────────────────────────────────
    with st.expander("📖 How CCA works (BIWS / JPMorgan methodology)", expanded=False):
        st.markdown("""
**Comparable Company Analysis** values a company based on what similar public companies trade at in the market right now.

**The 4-step process (per BIWS / Breaking Into Wall Street):**

1. **Select peers** — companies in the same industry, similar geography and size (5–10 is ideal)
2. **Choose multiples** — EV/Revenue, EV/EBITDA (EV-based), P/E, Forward P/E (equity-based)
3. **Calculate peer statistics** — 25th percentile, median, 75th percentile for each multiple
4. **Apply to your company** — multiply the peer percentile by your company's metric to get an implied price

**EV-based multiples** (EV/Revenue, EV/EBITDA):
```
Implied EV = Peer Multiple × Target's Metric (Revenue or EBITDA)
Implied Equity Value = Implied EV + Cash − Debt
Implied Share Price = Implied Equity Value ÷ Shares Outstanding
```

**Equity-based multiples** (P/E, P/B):
```
Implied Share Price = Peer Multiple × Target's EPS (or Book Value per Share)
```

**Reading the Football Field:**
- Each bar shows the P25–P75 range of implied prices
- The diamond shows the median
- The red line is the current market price
- Bars to the **right** of the current price suggest undervaluation
- Bars to the **left** suggest overvaluation

**CCA vs DCF:**
- CCA reflects what the market is paying *right now* for comparable businesses
- DCF reflects the intrinsic value based on your long-term cash flow projections
- When CCA and DCF agree → high conviction. When they diverge → investigate why.
        """)

    # ── Peer Multiples Table ──────────────────────────────────────────────────
    st.markdown(section_header("📊 Peer Multiples Table"), unsafe_allow_html=True)
    st.caption("Target's own multiples shown for comparison. "
               "Peer statistics: Min | P25 | **Median** | P75 | Max")

    peer_stats = cca.get("peer_stats", {})
    target_own = cca.get("target_multiples", {})

    multiple_defs = [
        ("ev_revenue",  "EV / Revenue"),
        ("ev_ebitda",   "EV / EBITDA"),
        ("pe_trailing", "P/E (TTM)"),
        ("pe_forward",  "Forward P/E"),
        ("price_book",  "P / Book"),
        ("price_sales", "P / Sales"),
    ]

    # Header row
    hcols = st.columns([1.5, 0.8, 0.8, 0.9, 0.8, 0.8, 0.8])
    for hcol, htxt in zip(hcols, ["Multiple", "Target", "Min", "P25", "**Median**", "P75", "Max"]):
        hcol.markdown(
            f"<div style='font-size:11px;font-weight:700;color:#718096;"
            f"padding:4px 0;border-bottom:2px solid #3a3f5c'>{htxt}</div>",
            unsafe_allow_html=True
        )

    for key, lbl in multiple_defs:
        stats  = peer_stats.get(key, {})
        target_v = target_own.get(key)
        if not stats:
            continue

        med = stats.get("median")
        # Is target cheap (below median for higher-is-worse multiples)?
        is_cheap = target_v and med and target_v < med

        c1,c2,c3,c4,c5,c6,c7 = st.columns([1.5, 0.8, 0.8, 0.9, 0.8, 0.8, 0.8])
        c1.markdown(f"<div style='font-size:12px;font-weight:600;color:#e2e8f0'>{lbl}</div>",
                    unsafe_allow_html=True)

        tv_color = "#68d391" if is_cheap else "#e2e8f0"
        c2.markdown(
            f"<div style='font-size:12px;font-weight:700;color:{tv_color}'>"
            f"{fmt(target_v, dec=1) if target_v else '—'}</div>",
            unsafe_allow_html=True
        )
        for col, pct in [(c3,"min"),(c4,"p25"),(c5,"median"),(c6,"p75"),(c7,"max")]:
            v = stats.get(pct)
            is_median = pct == "median"
            col.markdown(
                f"<div style='font-size:{'13' if is_median else '11'}px;"
                f"font-weight:{'700' if is_median else '400'};"
                f"color:{'#e2e8f0' if is_median else '#94a3b8'}'>"
                f"{fmt(v, dec=1) if v else '—'}</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # ── Implied Price Table ───────────────────────────────────────────────────
    st.markdown(section_header("💰 Implied Share Prices"), unsafe_allow_html=True)
    st.caption("Applying peer percentile multiples to the target company's own metrics")

    implied = cca.get("implied_prices", {})
    if implied:
        ip_cols = st.columns([1.5, 0.9, 0.9, 0.9, 0.9])
        for hcol, htxt in zip(ip_cols, ["Multiple", "P25 Implied", "Median Implied", "P75 Implied", "vs Current"]):
            hcol.markdown(f"<div style='font-size:11px;font-weight:700;color:#718096;"
                          f"padding:4px 0;border-bottom:2px solid #3a3f5c'>{htxt}</div>",
                          unsafe_allow_html=True)

        for key, lbl in multiple_defs:
            prices = implied.get(key, {})
            if not prices:
                continue
            mid    = prices.get("median", 0)
            vs_cur = (mid - current_price) / current_price if (mid and current_price > 0) else None
            color  = "#68d391" if (vs_cur or 0) >= 0.05 else ("#fc8181" if (vs_cur or 0) <= -0.05 else "#fbd38d")

            c1,c2,c3,c4,c5 = st.columns([1.5, 0.9, 0.9, 0.9, 0.9])
            c1.markdown(f"<div style='font-size:12px;color:#e2e8f0'>{lbl}</div>",
                        unsafe_allow_html=True)
            c2.markdown(f"<div style='font-size:12px;color:#94a3b8'>"
                        f"{fmt_currency(prices.get('p25',0), native_ccy, disp_ccy, rates)}</div>",
                        unsafe_allow_html=True)
            c3.markdown(f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>"
                        f"{fmt_currency(mid, native_ccy, disp_ccy, rates)}</div>",
                        unsafe_allow_html=True)
            c4.markdown(f"<div style='font-size:12px;color:#94a3b8'>"
                        f"{fmt_currency(prices.get('p75',0), native_ccy, disp_ccy, rates)}</div>",
                        unsafe_allow_html=True)
            c5.markdown(f"<div style='font-size:13px;font-weight:700;color:{color}'>"
                        f"{'▲' if (vs_cur or 0)>=0 else '▼'} {abs(vs_cur or 0):.1%}</div>",
                        unsafe_allow_html=True)



def render_deep_analysis_tab(ticker, info, deep, fund, data):
    """Deep Financial Analysis tab — statements, scores, macro."""
    if not deep:
        st.info("Run analysis to see deep financial metrics.")
        return

    native_ccy = info.get("currency","USD")
    disp_ccy   = st.session_state.get("display_ccy_code", native_ccy)
    rates      = st.session_state.get("fx_rates", {})
    sector     = info.get("sector","")

    # ── Summary scorecard row ────────────────────────────────────────────────
    piot  = deep.get("piotroski", {})
    alt   = deep.get("altman", {})
    dup   = deep.get("dupont", {})
    cap   = deep.get("cap_eff", {})
    macro = deep.get("macro", {})
    fwd   = deep.get("fwd_growth", {})
    opl   = deep.get("op_leverage", {})

    # Quick 4-metric scorecard
    st.markdown("### Financial Health Scores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    if piot.get("total") is not None:
        sc1.metric("Piotroski F-Score",
                   f"{piot['total']}/9 — {piot['signal'].split()[0]}",
                   help="0-9 score: ≥7 strong, 4-6 neutral, <4 weak")
    if alt.get("z_score"):
        sc2.metric("Altman Z-Score",
                   f"{alt['z_score']} — {alt['zone'].split()[0]}",
                   help="Z>2.99 safe, 1.81-2.99 grey, <1.81 distress")
    if dup.get("roe_direct"):
        sc3.metric("ROE (DuPont)",
                   f"{dup['roe_direct']:.1%}",
                   f"Driven by {dup.get('primary_driver','?')}")
    if cap.get("roic"):
        sc4.metric("ROIC",
                   f"{cap['roic']['value']:.1%}",
                   cap["roic"]["note"][:30])

    st.divider()

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # ── Piotroski F-Score breakdown ───────────────────────────────────
        if piot.get("scores"):
            st.markdown(section_header(f"📊 Piotroski F-Score: {piot['total']}/9"), unsafe_allow_html=True)
            st.caption("Financial health checklist developed by Stanford professor Joseph Piotroski (2000)")
            groups = [
                ("Profitability", ["F1_roa_positive","F2_ocf_positive","F3_roa_improving","F4_accruals"]),
                ("Leverage & Liquidity", ["F5_leverage_improving","F6_liquidity","F7_no_dilution"]),
                ("Operating Efficiency", ["F8_gross_margin","F9_asset_turnover"]),
            ]
            for grp_name, keys in groups:
                st.markdown(f"**{grp_name}**")
                for k in keys:
                    s = piot["scores"].get(k, {})
                    icon  = "✅" if s.get("pass") else "❌"
                    color = "#68d391" if s.get("pass") else "#fc8181"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:3px 0;font-size:12px;border-bottom:1px solid #1e2130'>"
                        f"<span style='color:#a0aec0'>{icon} {s.get('label','')}</span>"
                        f"<span style='color:{color};font-weight:600'>{s.get('value','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # ── DuPont Decomposition ──────────────────────────────────────────
        if dup and not dup.get("error"):
            st.markdown(section_header("⚗️ DuPont ROE Decomposition"), unsafe_allow_html=True)
            st.caption("ROE = Net Margin × Asset Turnover × Equity Multiplier")
            components = [
                ("Net Margin",       dup.get("net_margin"),        "Profitability — how much of each revenue dollar becomes profit"),
                ("Asset Turnover",   dup.get("asset_turnover"),    "Efficiency — revenue generated per dollar of assets"),
                ("Equity Multiplier",dup.get("equity_multiplier"), "Leverage — ratio of assets to equity (higher = more debt)"),
            ]
            for lbl, val, desc in components:
                is_primary = lbl == dup.get("primary_driver")
                fmt_val = f"{val:.2%}" if lbl == "Net Margin" else f"{val:.2f}×"
                bg = "#1a2a1a" if is_primary else "transparent"
                st.markdown(
                    f"<div style='background:{bg};padding:6px 10px;border-radius:6px;margin:3px 0'>"
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='font-size:13px;color:#e2e8f0;font-weight:{"700" if is_primary else "400"}'>"
                    f"{'★ ' if is_primary else ''}{lbl}</span>"
                    f"<span style='font-size:14px;font-weight:700;color:#68d391'>{fmt_val}</span>"
                    f"</div>"
                    f"<div style='font-size:11px;color:#718096;margin-top:2px'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:8px;"
                f"padding:8px 12px;margin-top:8px;font-size:12px;color:#a0aec0'>"
                f"💡 <b style='color:#e2e8f0'>Primary driver:</b> {dup.get('driver_note','')}"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_right:
        # ── Altman Z-Score ────────────────────────────────────────────────
        if alt and not alt.get("error"):
            st.markdown(section_header("⚡ Altman Z-Score"), unsafe_allow_html=True)
            st.caption("Bankruptcy risk model — developed by Edward Altman (1968)")
            zc = alt["color"]
            st.markdown(
                f"<div style='background:#1a1d2e;border:2px solid {zc};border-radius:10px;"
                f"padding:16px;text-align:center;margin-bottom:12px'>"
                f"<div style='font-size:32px;font-weight:900;color:{zc}'>{alt['z_score']}</div>"
                f"<div style='font-size:13px;color:{zc};font-weight:700'>{alt['zone']}</div>"
                f"<div style='font-size:11px;color:#718096;margin-top:4px'>{alt['interpretation'][:120]}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            thresholds = [
                ("> 2.99", "Safe Zone",     "#00C853"),
                ("1.81–2.99", "Grey Zone",  "#FFC107"),
                ("< 1.81", "Distress Zone", "#EF5350"),
            ]
            for thr, lbl, tc in thresholds:
                st.markdown(
                    f"<div style='font-size:11px;color:{tc};padding:2px 0'>"
                    f"Z {thr} → {lbl}</div>",
                    unsafe_allow_html=True
                )

        # ── Capital Efficiency ────────────────────────────────────────────
        if cap:
            st.markdown(section_header("💎 Capital Efficiency"), unsafe_allow_html=True)
            for key, metric in cap.items():
                if not isinstance(metric, dict) or "value" not in metric:
                    continue
                v   = metric["value"]
                lbl = metric["label"]
                note= metric["note"]
                is_pct = v < 2 and v != 0
                disp = f"{v:.1%}" if is_pct else f"{v:.2f}×"
                st.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid #1e2130'>"
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='font-size:12px;color:#a0aec0'>{lbl}</span>"
                    f"<span style='font-size:13px;font-weight:700;color:#e2e8f0'>{disp}</span>"
                    f"</div>"
                    f"<div style='font-size:11px;color:#718096'>{note}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Operating Leverage ────────────────────────────────────────────
        if opl and not opl.get("error") and opl.get("dol") is not None:
            st.markdown(section_header("⚙️ Operating Leverage"), unsafe_allow_html=True)
            dol = opl["dol"]
            dol_c = "#EF5350" if dol > 3 else ("#FFC107" if dol > 1.5 else "#68d391")
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;padding:8px 0'>"
                f"<div style='font-size:28px;font-weight:800;color:{dol_c}'>{dol:.2f}×</div>"
                f"<div style='font-size:12px;color:#a0aec0'>Degree of Operating Leverage (DOL)</div>"
                f"</div>"
                f"<div style='font-size:12px;color:#718096;padding:4px 0'>{opl['note']}</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # ── Forward Growth Analysis ───────────────────────────────────────────────
    st.markdown("### 📈 Forward Growth & Analyst Estimates")
    if fwd:
        fg1, fg2, fg3 = st.columns(3)
        if fwd.get("rev_growth"):
            fg1.metric("Revenue Growth (TTM)", f"{fwd['rev_growth']:.1%}")
        if fwd.get("eps_growth"):
            fg2.metric("EPS Growth (TTM)", f"{fwd['eps_growth']:.1%}")
        if fwd.get("fwd_pe"):
            fg3.metric("Forward P/E", f"{fwd['fwd_pe']:.1f}×")

        # Estimate revision trend
        rev_4w = fwd.get("estimate_revision_4w")
        rev_3m = fwd.get("estimate_revision_3m")
        if rev_4w is not None or rev_3m is not None:
            st.markdown(section_header("📐 EPS Estimate Revisions"), unsafe_allow_html=True)
            rc1, rc2 = st.columns(2)
            if rev_4w is not None:
                c4 = "#68d391" if rev_4w > 0 else "#fc8181"
                rc1.markdown(f"<div style='font-size:12px;color:#a0aec0'>4-week revision</div>"
                            f"<div style='font-size:20px;font-weight:700;color:{c4}'>"
                            f"{'▲' if rev_4w>0 else '▼'} {abs(rev_4w):.1%}</div>",
                            unsafe_allow_html=True)
            if rev_3m is not None:
                c3 = "#68d391" if rev_3m > 0 else "#fc8181"
                rc2.markdown(f"<div style='font-size:12px;color:#a0aec0'>3-month revision</div>"
                            f"<div style='font-size:20px;font-weight:700;color:{c3}'>"
                            f"{'▲' if rev_3m>0 else '▼'} {abs(rev_3m):.1%}</div>",
                            unsafe_allow_html=True)
            st.caption("Positive revisions = analysts raising EPS estimates (bullish signal). "
                       "Negative = estimates being cut (bearish signal).")

    st.divider()

    # ── Macro & Industry Sensitivity ──────────────────────────────────────────
    st.markdown("### 🌍 Market, Industry & Macro Factors")
    macro_profile = SECTOR_MACRO_SENSITIVITY.get(sector, {})

    if macro_profile:
        # Sensitivity ratings row
        st.markdown(f"**{sector} sector** — key macro sensitivities:")
        sens_cols = st.columns(4)
        for col, (lbl, key) in zip(sens_cols, [
            ("Recession", "recession_sensitivity"),
            ("Interest Rates", "rate_sensitivity"),
            ("FX / USD", "fx_sensitivity"),
            ("Cyclicality", "cyclicality"),
        ]):
            val = macro_profile.get(key, "unknown")
            c   = SENSITIVITY_COLORS.get(val, "#718096")
            col.markdown(
                f"<div style='background:{c}22;border:1px solid {c}44;border-radius:8px;"
                f"padding:8px;text-align:center'>"
                f"<div style='font-size:10px;color:#718096'>{lbl}</div>"
                f"<div style='font-size:13px;font-weight:700;color:{c}'>{val.title()}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Market sensitivity (beta)
        beta = macro.get("beta", 1.0)
        bc   = "#EF5350" if beta > 1.5 else ("#FFC107" if beta > 1.0 else "#68d391")
        st.markdown(
            f"<div style='margin:12px 0 8px;padding:10px 14px;background:#1a1d2e;"
            f"border:1px solid #3a3f5c;border-radius:8px'>"
            f"<div style='font-size:12px;color:#718096'>Market Sensitivity (β = {beta:.2f})</div>"
            f"<div style='font-size:13px;color:{bc};font-weight:600;margin-top:3px'>"
            f"{macro.get('market_sensitivity','')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Interest rate sensitivity from balance sheet
        nde = macro.get("net_debt_ebitda")
        ic  = macro.get("interest_coverage")
        if nde is not None:
            nde_c = "#EF5350" if nde > 3 else ("#FFC107" if nde > 1.5 else "#68d391")
            ic_str = f"  ·  Interest coverage: {ic:.1f}×" if ic else ""
            st.markdown(
                f"<div style='font-size:12px;color:#a0aec0;padding:4px 0'>"
                f"Net Debt/EBITDA: <span style='color:{nde_c};font-weight:700'>{nde:.1f}×</span>{ic_str}"
                f" — {'High leverage = significant rate risk' if nde > 3 else 'Moderate leverage' if nde > 1.5 else 'Low leverage = minimal rate risk'}"
                f"</div>",
                unsafe_allow_html=True
            )

        # Key macro drivers table
        st.markdown(f"**Key macro drivers for {sector}:**")
        drivers = macro_profile.get("drivers", [])
        for driver, direction, description in drivers:
            dc = DRIVER_COLORS.get(direction, "#718096")
            dir_lbl = {"positive":"▲ Positive","negative":"▼ Negative",
                       "cyclical":"↕ Cyclical","mixed":"↔ Mixed",
                       "moderate":"→ Moderate","low":"→ Low"}.get(direction, direction)
            st.markdown(
                f"<div style='padding:8px 12px;border-left:3px solid {dc};"
                f"margin:4px 0;background:#0e1117;border-radius:0 6px 6px 0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:13px;font-weight:600;color:#e2e8f0'>{driver}</span>"
                f"<span style='font-size:11px;color:{dc};font-weight:700'>{dir_lbl}</span>"
                f"</div>"
                f"<div style='font-size:12px;color:#718096;margin-top:3px'>{description}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info(f"No sector macro profile available for '{sector}'. "
                "The deep analysis above (Piotroski, Altman, DuPont) is still valid.")

    st.divider()

    # ── Government & External Factors ────────────────────────────────────────
    st.markdown("### 🏛️ Government, Regulation & External Factors")
    gov = get_government_factors(info)
    risk_level = gov.get("regulatory_risk","moderate")
    risk_color = RISK_COLORS.get(risk_level,"#FFC107")
    gov_factors = gov.get("factors",[])

    st.markdown(
        f"<div style='background:{risk_color}22;border:1px solid {risk_color}44;"
        f"border-radius:10px;padding:10px 16px;margin-bottom:12px;display:flex;gap:16px'>"
        f"<div><div style='font-size:11px;color:#718096'>Sector Regulatory Risk</div>"
        f"<div style='font-size:18px;font-weight:800;color:{risk_color}'>"
        f"{risk_level.replace('_',' ').title()}</div></div>"
        f"<div style='font-size:12px;color:#a0aec0;align-self:center'>"
        f"Regulatory exposure for the {gov.get('sector','')} sector based on "
        f"current legislative and enforcement environment.</div>"
        f"</div>", unsafe_allow_html=True)

    if gov_factors:
        for factor_name, factor_desc in gov_factors:
            st.markdown(
                f"<div style='padding:10px 14px;border-left:3px solid {risk_color};"
                f"margin:5px 0;background:#0e1117;border-radius:0 8px 8px 0'>"
                f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>{factor_name}</div>"
                f"<div style='font-size:12px;color:#94a3b8;margin-top:4px;line-height:1.5'>"
                f"{factor_desc}</div>"
                f"</div>", unsafe_allow_html=True)
    else:
        st.info(f"No specific regulatory profile for the '{gov.get('sector','')}' sector. "
                "General regulatory environment applies.")

    st.markdown("""
> **Data sources:** Financial ratios from yfinance (Yahoo Finance).
> Macro sensitivity profiles compiled from Bloomberg sector factor research,
> Damodaran industry analysis, and Fed/OECD sector studies.
> Regulatory factors based on current legislative environment (2024-2025).
> Piotroski F-Score: Piotroski (2000) — *Journal of Accounting Research*.
> Altman Z-Score: Altman (1968, 1983) — *Journal of Finance*.
    """)

# ─── Per-stock Deep Dive ──────────────────────────────────────────────────────

def render_stock_tab(ticker, data, dcf, fund, indicators, signals,
                     valuation, comparison, sentiment, scoring, peer_data, deep=None):
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

    tabs = st.tabs(["📋 Overview", "💰 Financials & DCF", "📊 Comparable Analysis",
                     "📊 Valuation & Peers", "📈 Technical Analysis", "🎯 Sentiment",
                     "📅 Earnings & Forecasts", "🏢 Intelligence", "📰 News",
                     "🔮 Prediction", "🧠 Deep Analysis"])

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
            # Stock type classification
            st_label, st_color, st_desc = classify_stock(info)
            st.markdown(
                f"<div style='margin:8px 0'>"
                f"<span style='background:{st_color}22;color:{st_color};"
                f"border:1px solid {st_color}44;border-radius:12px;"
                f"padding:3px 12px;font-size:12px;font-weight:600'>{st_label}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            with st.expander("What does this mean?", expanded=False):
                st.markdown(f"<div style='font-size:12px;color:#a0aec0'>{st_desc}</div>",
                            unsafe_allow_html=True)

            # Earnings track record
            if beat_pct is not None:
                color = "#00C853" if beat_pct > 0.6 else ("#F44336" if beat_pct < 0.4 else "#FFC107")
                st.markdown(f"<div style='margin-top:10px;font-size:12px;color:{color}'>"
                            f"{'✅' if beat_pct>0.6 else ('🔴' if beat_pct<0.4 else '🟡')} "
                            f"EPS beat rate: {beat_pct:.0%} ({len(past_surp)} quarters)</div>", unsafe_allow_html=True)

        with col2:
            # Price chart spanning full width above metrics
            # Use 5y history for price chart (more context)
            hist_price = data.get("hist_5y", pd.DataFrame())
            if hist_price.empty:
                hist_price = data.get("hist_2y", pd.DataFrame())
            if not hist_price.empty:
                fig_price = build_current_price_chart(hist_price, ticker, info)
                st.plotly_chart(fig_price, use_container_width=True, key=f"pc_{ticker}_5")
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
        st.plotly_chart(build_financials_chart(fund, ticker), use_container_width=True, key=f"pc_{ticker}_6")
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

        # ── Path-to-Profitability model (shown for loss-making companies) ────
        p2p = dcf.get("p2p", {})
        if p2p and not p2p.get("error") and p2p.get("intrinsic_value"):
            # Currency helpers (scoped here since P2P section needs them)
            _native = info.get("currency", "USD")
            _disp   = st.session_state.get("display_ccy_code", _native)
            _rates  = st.session_state.get("fx_rates", {})
            with st.expander("🌱 Path-to-Profitability DCF — Revenue-Based Model", expanded=True):
                iv_p2p  = p2p["intrinsic_value"]
                up_p2p  = (iv_p2p - cp) / cp if cp > 0 else 0
                uc      = "#00C853" if up_p2p >= 0 else "#EF5350"
                disp_iv = fmt_currency(iv_p2p, _native, _disp, _rates)

                col1, col2, col3 = st.columns(3)
                col1.metric("P2P Implied Price", disp_iv,
                            f"{'▲' if up_p2p>=0 else '▼'} {abs(up_p2p):.1%} vs current")
                col2.metric("Break-even Year",
                            f"Yr {p2p.get('first_positive_yr','?')}" if p2p.get("first_positive_yr") else "Beyond 10Y",
                            "From today")
                col3.metric("Target Margin",
                            f"{p2p.get('target_margin',0):.1%}",
                            p2p.get("margin_rationale",""))

                st.caption(p2p.get("rate_label",""))

                # Revenue + margin projection table
                details = p2p.get("cashflow_details", [])
                if details:
                    df_p2p = pd.DataFrame([{
                        "Year":        d["year"],
                        "Revenue":     fmt_currency(d["revenue"], _native, _disp, _rates),
                        "Rev Growth":  f"{d['rev_growth']:.1%}",
                        "Op Margin":   f"{d['op_margin']:.1%}",
                        "Est. FCF":    fmt_currency(d["fcf"], _native, _disp, _rates),
                        "PV of FCF":   fmt_currency(d["pv"],  _native, _disp, _rates),
                    } for d in details])
                    st.dataframe(df_p2p, use_container_width=True, hide_index=True)

                st.markdown("""
> **Methodology:** This model projects revenue forward (not FCF, which is currently negative),
> models margin improvement toward the industry benchmark, then estimates FCF from
> projected EBITDA minus normalised CapEx. Analyst growth estimates are blended with
> industry benchmarks. The discount rate uses CAPM with a size premium.
> Terminal value is only added once the company reaches sustainable positive FCF.
                """)

        # Sensitivity table
        sens = dcf.get("sensitivity", {})
        if sens:
            with st.expander("🌡️ Sensitivity Analysis — WACC × Terminal Growth Rate", expanded=False):
                st.caption("Each cell = implied share price. "
                           "White box = base case. Green = above current, red = below.")
                fig_sens = build_sensitivity_chart(sens, dcf.get("current_price",0) or cp, 
                                                    info.get("currency","USD"))
                if fig_sens.data:
                    st.plotly_chart(fig_sens, use_container_width=True, key=f"pc_{ticker}_7")

        # Growth rate sources
        gr = dcf.get("growth_rates", {})
        if gr:
            with st.expander("📐 Growth Rate Assumptions", expanded=False):
                st.markdown(f"""
| Component | Value | Source |
|---|---|---|
| Stage 1 Growth (years 1–5) | **{gr.get('near', 0):.1%}** | Blended: historical UFCF + analyst estimates |
| Stage 2 Growth (years 6–10) | **{gr.get('fade', 0):.1%}** | Fade = Stage 1 × 50% |
| Terminal Growth (perpetuity) | **{gr.get('terminal', 0):.1%}** | ≈ long-run nominal GDP |
| Historical UFCF Growth | {gr.get('hist_ufcf', 0):.1%} | From financial statements |
| Analyst EPS Growth | {f"{gr.get('analyst_eps'):.1%}" if gr.get('analyst_eps') else '—'} | Yahoo Finance consensus |
                """)

        # FCF type note
        fcf_type = dcf.get("fcf_type", "FCF")
        if fcf_type == "UFCF":
            st.info("💡 **Improvement:** These models now use **Unlevered Free Cash Flow (UFCF)** "
                    "= EBIT×(1−t) + D&A − CapEx − ΔNWC, consistent with JPMorgan M&A methodology. "
                    "UFCF is more accurate than reported operating cash flow as it removes "
                    "working capital noise and financing effects.", icon="📐")

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

    # ── Comparable Company Analysis ──────────────────────────────────────────
    with tabs[2]:
        render_cca_tab(ticker, info, peer_data, dcf, cp)

    # ── Valuation & Peers ──────────────────────────────────────────────────
    with tabs[3]:
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
                    # Values are already normalized decimals (0.25 = 25%) — multiply by 100 for display
                    def _pct_str(v):
                        if v is None: return "—"
                        try:
                            f = float(v)
                            # Extra guard: if value > 5 it's already a percentage (shouldn't happen after normalization)
                            if f > 5: f = f / 100
                            return f"{f*100:.1f}%"
                        except Exception:
                            return "—"
                    t_str = _pct_str(t) if is_pct else (fmt(t, dec=2) if t else "—")
                    m_str = _pct_str(m) if is_pct else (fmt(m, dec=2) if m else "—")
                    sig = r.get("signal","")
                    hb  = r.get("higher_better", True)
                    # "Above avg" is a beat only if higher is better (margins, ROE, growth).
                    # For valuation multiples (P/E, PEG, EV/EBITDA) LOWER is better,
                    # so "Below avg" = beat and "Above avg" = miss.
                    if "In line" in sig:   fs = "inline"
                    elif "Above" in sig:   fs = "beat" if hb else "miss"
                    elif "Below" in sig:   fs = "miss" if hb else "beat"
                    else:                  fs = ""
                    st.markdown(tooltip_html(r["key"], r["metric"], f"{t_str}  (peers: {m_str})", fs),
                                unsafe_allow_html=True)
            with c2:
                fig = build_peer_chart(comparison, ticker)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True, key=f"pc_{ticker}_8")

        if peer_data:
            st.markdown(section_header("Full Peer Table"), unsafe_allow_html=True)
            st.caption("Click **▶ Analyze** to add a peer as a new analysis tab.")

            # Column headers
            hcols = st.columns([1.8, 1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 1.2])
            for hcol, htxt in zip(hcols, ["Ticker / Name","Country · Mkt Cap","P/E","Fwd P/E","Net Mgn","ROE","Rev Grw","Action"]):
                hcol.markdown(f"<div style='font-size:11px;font-weight:700;color:#94a3b8;padding:4px 0;border-bottom:2px solid #3a3f5c'>{htxt}</div>",
                              unsafe_allow_html=True)

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
                        st.session_state["pending_add_ticker"] = pt
                        st.rerun()
                st.divider()

    # ── Technical Analysis ────────────────────────────────────────────────────
    with tabs[4]:
        hist = data.get("hist_2y", pd.DataFrame())
        build_stock_benchmark_chart(ticker, hist, info)
        st.divider()
        if hist.empty or not indicators:
            st.warning("Insufficient price history for technical analysis.")
        else:
            st.plotly_chart(build_price_chart(hist, indicators, ticker), use_container_width=True, key=f"pc_{ticker}_9")

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
    with tabs[5]:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_sent = build_sentiment_donut(sentiment, ticker)
            if fig_sent.data:
                st.plotly_chart(fig_sent, use_container_width=True, key=f"pc_{ticker}_10")
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
                st.plotly_chart(fig_earn, use_container_width=True, key=f"pc_{ticker}_11")

            # ── Recent analyst rating changes ─────────────────────────────
            recs_df = data.get("recommendations")
            if recs_df is not None and isinstance(recs_df, pd.DataFrame) and not recs_df.empty:
                st.markdown(section_header("Recent Analyst Rating Changes"), unsafe_allow_html=True)
                try:
                    recs_show = recs_df.reset_index()

                    # Normalise column names (yfinance 1.4.0 uses these after rename)
                    col_map = {}
                    for c in recs_show.columns:
                        cl = c.lower()
                        if "grade" in cl and "date" in cl: col_map[c] = "_raw_date"
                        elif c in ("Firm","firm"):          col_map[c] = "Analyst Firm"
                        elif c in ("ToGrade","toGrade"):    col_map[c] = "Rating"
                        elif c in ("FromGrade","fromGrade"):col_map[c] = "Previous"
                        elif c in ("Action","action"):      col_map[c] = "Action"
                        elif c == "GradeDate":              col_map[c] = "_raw_date"
                    recs_show = recs_show.rename(columns=col_map)

                    # Parse dates — try multiple strategies, discard if invalid (epoch 0)
                    has_valid_dates = False
                    MIN_VALID = pd.Timestamp("2000-01-01")   # anything before 2000 = bad data

                    if "_raw_date" in recs_show.columns:
                        raw = recs_show["_raw_date"]
                        parsed = None
                        for kwargs in [{"unit": "ms"}, {"unit": "s"}, {}]:
                            try:
                                candidate = pd.to_datetime(raw, errors="coerce", **kwargs)
                                if candidate.notna().any() and candidate.max() > MIN_VALID:
                                    parsed = candidate
                                    break
                            except Exception:
                                pass

                        if parsed is not None and parsed.max() > MIN_VALID:
                            recs_show["Date"] = parsed
                            has_valid_dates = True
                        recs_show = recs_show.drop(columns=["_raw_date"], errors="ignore")

                    # Filter to last 10, sort newest first
                    if has_valid_dates:
                        recs_show = recs_show.dropna(subset=["Date"])
                        recs_show = recs_show[recs_show["Date"] > MIN_VALID]
                        recs_show = recs_show.sort_values("Date", ascending=False).head(10)
                        cutoff_90d = pd.Timestamp.now() - pd.Timedelta(days=90)
                        recent_ct  = int((recs_show["Date"] >= cutoff_90d).sum())
                        recs_show["Date"] = recs_show["Date"].dt.strftime("%d %b %Y")
                    else:
                        # Dates not available — show ratings without date column
                        recent_ct = 0
                        recs_show = recs_show.head(10)

                    keep_cols = ([c for c in ["Date","Analyst Firm","Rating","Previous","Action"]
                                  if c in recs_show.columns])
                    if keep_cols:
                        st.dataframe(recs_show[keep_cols], use_container_width=True, hide_index=True)

                    # Freshness banner
                    if not has_valid_dates:
                        st.caption("ℹ️ Date information not available from Yahoo Finance for this ticker.")
                    elif recent_ct > 0:
                        st.success(f"✅ {recent_ct} rating change{'s' if recent_ct>1 else ''} "
                                   "in the last 90 days — analyst coverage is recent.")
                    else:
                        latest = recs_show["Date"].iloc[0] if has_valid_dates and len(recs_show) > 0 else None
                        if latest:
                            st.warning(f"⚠️ Most recent analyst change: {latest} — "
                                       "price targets may be stale.")
                        else:
                            st.info("No recent analyst rating changes available.")
                except Exception:
                    pass

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
    with tabs[6]:
        render_earnings_section(data, ticker, earn_rows, cal_info, estimates, eps_trend_data)

    # ── Intelligence (SWOT + Customers/Suppliers + Filings) ───────────────────
    with tabs[7]:
        render_intelligence_tab(ticker, info, scoring, signals, sentiment, fund, comparison)

    # ── News ──────────────────────────────────────────────────────────────────
    with tabs[8]:
        render_news_tab(ticker, info, data.get("news", []))

    # ── Prediction ────────────────────────────────────────────────────────────
    with tabs[9]:
        render_prediction_tab(ticker, info, dcf, sentiment, scoring, signals, fund)

    # ── Deep Financial Analysis ───────────────────────────────────────────────
    with tabs[10]:
        render_deep_analysis_tab(ticker, info, deep or {}, fund, data)



def classify_stock(info: dict) -> tuple[str, str, str]:
    """
    Classify stock as Growth / Value / Dividend / GARP / Speculative etc.
    Returns (label, color, description)
    """
    pe      = info.get("trailingPE") or 0
    fpe     = info.get("forwardPE") or 0
    peg     = info.get("pegRatio") or 0
    rg      = info.get("revenueGrowth") or 0
    eg      = info.get("earningsGrowth") or 0
    div_y   = info.get("dividendYield") or 0
    pb      = info.get("priceToBook") or 0
    beta    = info.get("beta") or 1.0
    mktcap  = info.get("marketCap") or 0
    nm      = info.get("profitMargins") or 0

    # Dividend stock
    if div_y > 0.035 and pe < 35:
        return ("💰 Dividend", "#FFD700",
                f"Dividend yield {div_y:.1%} — income-focused stock. "
                f"Typically mature, stable companies returning cash to shareholders.")

    # High-growth / momentum
    if rg > 0.20 and (pe > 40 or fpe > 30):
        return ("🚀 Growth", "#00C853",
                f"Revenue growing {rg:.0%} YoY with premium valuation (P/E {pe:.0f}x). "
                f"Market pricing in high future growth — expect volatility.")

    # GARP (Growth at Reasonable Price)
    if 0 < peg < 1.5 and rg > 0.08 and pe < 35:
        return ("⚖️ GARP", "#42A5F5",
                f"PEG {peg:.2f} — growth available at a reasonable price. "
                f"Balanced profile: {rg:.0%} revenue growth, P/E {pe:.0f}x.")

    # Deep value
    if pe and 0 < pe < 12 and pb and 0 < pb < 1.5:
        return ("💎 Deep Value", "#FF6B35",
                f"Low P/E ({pe:.0f}x) and P/B ({pb:.2f}x) suggest potential undervaluation. "
                f"May be cyclical, out-of-favour, or a turnaround play.")

    # Value
    if pe and 0 < pe < 18 and rg < 0.12:
        return ("🏛️ Value", "#81C784",
                f"Below-market P/E ({pe:.0f}x) with modest growth ({rg:.0%}). "
                f"Typically mature business with stable earnings and reasonable valuation.")

    # Speculative / unprofitable
    if nm and nm < 0:
        return ("⚡ Speculative", "#EF5350",
                f"Currently unprofitable (net margin {nm:.1%}). "
                f"Valuation depends entirely on future growth expectations — high risk/reward.")

    # Small/micro cap growth
    if mktcap < 2e9 and rg > 0.10:
        return ("🌱 Small-Cap Growth", "#AB47BC",
                f"Small-cap ({fmt(mktcap, prefix='$')}) with {rg:.0%} revenue growth. "
                f"Higher risk but potentially higher return than large-cap peers.")

    # Blend / balanced default
    return ("🔵 Blend", "#90CAF9",
            f"Balanced characteristics — neither a pure growth nor pure value stock. "
            f"P/E {pe:.0f}x, revenue growth {rg:.0%}.")


def build_price_prediction(ticker, info, dcf, sentiment, scoring, signals, fund) -> dict:
    """
    Build a 1-3 year price prediction by combining:
      - DCF intrinsic value range (4 models)
      - Analyst price targets
      - Technical trend direction
      - Earnings growth trajectory
    Returns a dict with prediction data.
    """
    cp = get_current_price(info)
    if not cp:
        return {}

    targets = []

    # DCF targets
    dcf_values = []
    for key in ["wacc","capm","two_stage","fixed"]:
        m = dcf.get(key, {})
        iv = m.get("intrinsic_value")
        if iv and iv > 0 and not m.get("error"):
            dcf_values.append(iv)
    if dcf_values:
        dcf_low  = min(dcf_values)
        dcf_high = max(dcf_values)
        dcf_mid  = np.mean(dcf_values)
        targets.append(("DCF Models (avg)",   dcf_mid,  "#42A5F5"))
        targets.append(("DCF Conservative",   dcf_low,  "#90CAF9"))
        targets.append(("DCF Optimistic",     dcf_high, "#1E88E5"))

    # Analyst consensus target
    analyst_target = sentiment.get("target_mean")
    analyst_high   = sentiment.get("target_high")
    analyst_low    = sentiment.get("target_low")
    if analyst_target:
        targets.append(("Analyst Consensus",  analyst_target, "#FFD700"))
    if analyst_high:
        targets.append(("Analyst High",       analyst_high,   "#66BB6A"))
    if analyst_low:
        targets.append(("Analyst Low",        analyst_low,    "#EF9A9A"))

    # EPS-based price target (forward P/E × forward EPS × peer multiple)
    fpe  = info.get("forwardPE")  or info.get("trailingPE") or 0
    feps = info.get("forwardEps") or info.get("trailingEps") or 0
    eg   = info.get("earningsGrowth") or 0
    if fpe and feps and feps > 0:
        # Project forward EPS 2 years at estimated growth rate
        growth = max(-0.20, min(0.50, eg or 0.05))
        eps_2y = feps * (1 + growth) ** 2
        price_2y = eps_2y * fpe
        targets.append(("EPS Growth Model (2Y)", price_2y, "#AB47BC"))

    if not targets:
        return {}

    # Consensus view weighted by source reliability
    all_prices = [t[1] for t in targets if t[1] > 0]
    consensus  = float(np.median(all_prices)) if all_prices else cp
    low_est    = float(np.percentile(all_prices, 25)) if len(all_prices) >= 2 else min(all_prices)
    high_est   = float(np.percentile(all_prices, 75)) if len(all_prices) >= 2 else max(all_prices)

    upside_mid  = (consensus - cp) / cp
    upside_high = (high_est  - cp) / cp
    upside_low  = (low_est   - cp) / cp

    # Overall signal
    score = scoring.get("composite", 5)
    if upside_mid > 0.25 and score >= 6:
        outlook = ("🟢 Bullish", "#00C853")
    elif upside_mid > 0.10:
        outlook = ("🟡 Moderately Bullish", "#FFC107")
    elif upside_mid > -0.10:
        outlook = ("🟡 Neutral", "#FFC107")
    elif upside_mid > -0.25:
        outlook = ("🔴 Moderately Bearish", "#FF9800")
    else:
        outlook = ("🔴 Bearish", "#EF5350")

    return {
        "current_price": cp,
        "targets":       targets,
        "consensus":     consensus,
        "low_est":       low_est,
        "high_est":      high_est,
        "upside_mid":    upside_mid,
        "upside_high":   upside_high,
        "upside_low":    upside_low,
        "outlook":       outlook,
        "dcf_values":    dcf_values,
        "analyst_target": analyst_target,
    }


def render_prediction_tab(ticker, info, dcf, sentiment, scoring, signals, fund):
    """1-3 Year Price Prediction tab."""
    pred = build_price_prediction(ticker, info, dcf, sentiment, scoring, signals, fund)
    if not pred:
        st.info("Insufficient data to generate a price prediction for this ticker.")
        return

    cp          = pred["current_price"]
    native_ccy  = info.get("currency", "USD")
    disp_ccy    = st.session_state.get("display_ccy_code", native_ccy)
    rates       = st.session_state.get("fx_rates", {})
    ccy_sym     = CURRENCY_SYMBOLS.get(disp_ccy, disp_ccy)

    # ── Header ────────────────────────────────────────────────────────────────
    outlook_label, outlook_color = pred["outlook"]
    consensus_str = fmt_currency(pred["consensus"], native_ccy, disp_ccy, rates)
    low_str       = fmt_currency(pred["low_est"],   native_ccy, disp_ccy, rates)
    high_str      = fmt_currency(pred["high_est"],  native_ccy, disp_ccy, rates)

    st.markdown(f"""
    <div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:12px;
                padding:20px 24px;margin-bottom:16px'>
      <div style='font-size:13px;color:#94a3b8;margin-bottom:4px'>
        1–3 Year Price Outlook  ·  {ticker}
      </div>
      <div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap'>
        <div>
          <div style='font-size:11px;color:#718096'>Current Price</div>
          <div style='font-size:22px;font-weight:800;color:#fff'>
            {fmt_currency(cp, native_ccy, disp_ccy, rates)}
          </div>
        </div>
        <div style='font-size:24px;color:#718096'>→</div>
        <div>
          <div style='font-size:11px;color:#718096'>Consensus Target</div>
          <div style='font-size:22px;font-weight:800;color:#e2e8f0'>{consensus_str}</div>
          <div style='font-size:12px;color:{"#00C853" if pred["upside_mid"]>=0 else "#EF5350"}'>
            {"▲" if pred["upside_mid"]>=0 else "▼"} {abs(pred["upside_mid"]):.1%} implied
            {"upside" if pred["upside_mid"]>=0 else "downside"}
          </div>
        </div>
        <div>
          <div style='font-size:11px;color:#718096'>Range (Low – High)</div>
          <div style='font-size:16px;font-weight:600;color:#94a3b8'>
            {low_str} – {high_str}
          </div>
        </div>
        <div>
          <div style='font-size:11px;color:#718096'>Overall Outlook</div>
          <div style='font-size:16px;font-weight:700;color:{outlook_color}'>{outlook_label}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Target breakdown chart ────────────────────────────────────────────────
    if pred["targets"]:
        labels = [t[0] for t in pred["targets"]]
        values = [t[1] for t in pred["targets"]]
        colors = [t[2] for t in pred["targets"]]
        upsides = [(v - cp) / cp * 100 for v in values]

        fig = go.Figure()
        # Horizontal bars showing target vs current price
        fig.add_trace(go.Bar(
            x=upsides, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{fmt_currency(v, native_ccy, disp_ccy, rates)}  ({u:+.1f}%)"
                  for v, u in zip(values, upsides)],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.add_vline(x=0, line=dict(color="white", width=2))
        fig.add_vline(x=-20, line=dict(color="#EF5350", width=1, dash="dash"),
                      annotation_text="-20%", annotation_font_size=10)
        fig.add_vline(x=20,  line=dict(color="#00C853", width=1, dash="dash"),
                      annotation_text="+20%", annotation_font_size=10)
        fig.update_layout(
            title=dict(text=f"Price Targets vs Current ({fmt_currency(cp, native_ccy, disp_ccy, rates)})",
                       font_size=13, x=0.01),
            height=max(300, len(labels)*55 + 80),
            template="plotly_dark",
            margin=dict(l=200, r=120, t=50, b=40),
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
            xaxis=dict(ticksuffix="%", showgrid=True,
                       gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pc_{ticker}_12")

    # ── Methodology breakdown ────────────────────────────────────────────────
    st.markdown(section_header("📐 How this prediction is built"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Inputs used:**")
        inputs = []
        if pred.get("dcf_values"):
            inputs.append(f"✅ DCF Models ({len(pred['dcf_values'])} models, range: "
                          f"{fmt_currency(min(pred['dcf_values']), native_ccy, disp_ccy, rates)} – "
                          f"{fmt_currency(max(pred['dcf_values']), native_ccy, disp_ccy, rates)})")
        if pred.get("analyst_target"):
            inputs.append(f"✅ Analyst consensus target: "
                          f"{fmt_currency(pred['analyst_target'], native_ccy, disp_ccy, rates)} "
                          f"({sentiment.get('num_analysts',0)} analysts)")
        feps = info.get("forwardEps")
        if feps:
            inputs.append(f"✅ Forward EPS: ${feps:.2f} × Forward P/E model")
        for item in inputs:
            st.markdown(f"<div style='font-size:12px;color:#c6f6d5;padding:2px 0'>{item}</div>",
                        unsafe_allow_html=True)

    with c2:
        st.markdown("**Supporting signals:**")
        sc = scoring.get("scores", {})
        signals_list = [
            (f"Composite Score: {scoring.get('composite',0):.1f}/10 — {scoring.get('signal','')}", scoring.get("composite",5) >= 5),
            (f"Fundamental Score: {sc.get('fundamental',0):.1f}/10", sc.get("fundamental",5) >= 5),
            (f"Valuation Score: {sc.get('valuation',0):.1f}/10", sc.get("valuation",5) >= 5),
            (f"Technical Score: {sc.get('technical',0):.1f}/10", sc.get("technical",5) >= 5),
            (f"Sentiment Score: {sc.get('sentiment',0):.1f}/10", sc.get("sentiment",5) >= 5),
        ]
        for lbl, positive in signals_list:
            icon = "✅" if positive else "🔴"
            st.markdown(f"<div style='font-size:12px;padding:2px 0'>{icon} {lbl}</div>",
                        unsafe_allow_html=True)

    st.markdown("""
    > ⚠️ **Disclaimer:** This prediction combines quantitative models and analyst estimates.
    > It is for informational purposes only and does not constitute financial advice.
    > Actual stock prices may differ materially from these estimates.
    > Always do your own due diligence.
    """)

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
    # ── Process pending ticker additions BEFORE any widgets render ────────────
    # Handles: peer table "Analyze" button, and any other programmatic adds
    # ── Process pending slot clear BEFORE widgets render ─────────────────────
    clear_slot = st.session_state.pop("pending_clear_slot", None)
    if clear_slot is not None:
        slots = st.session_state.get("ticker_slots", ["","","","",""])
        slots[clear_slot] = ""
        st.session_state["ticker_slots"] = slots
        # Pop the widget key so it re-renders empty
        st.session_state.pop(f"t{clear_slot}", None)

    pending = st.session_state.pop("pending_add_ticker", None)
    if pending:
        slots = st.session_state.get("ticker_slots", ["","","","",""])
        if pending not in slots:
            for si in range(5):
                if not slots[si]:
                    slots[si] = pending
                    st.session_state["ticker_slots"] = slots
                    break
        # Clear ALL widget keys so the seeding loop fills them fresh from ticker_slots
        # This fixes the bug where stale t0-t4 keys prevent the new ticker from showing
        for _ki in range(5):
            st.session_state.pop(f"t{_ki}", None)
        st.session_state["auto_analyze"] = True
        st.session_state["nav_page"]     = "analyzer"

    with st.sidebar:
        st.markdown("# 📈 Stock Analyzer")

        # ── Page navigation ───────────────────────────────────────────────────
        # Resolve current page — force_nav takes priority over radio
        # (used when screener sends a ticker to the analyzer)
        if "force_nav" in st.session_state:
            forced = st.session_state.pop("force_nav")
            st.session_state["nav_page"] = forced

        nav_page = st.session_state.get("nav_page", "analyzer")
        _idx_map  = {"analyzer": 0, "screener": 1, "portfolio": 2}
        nav_sel   = st.radio(
            "Navigation",
            options=["📈 Analyzer", "🔍 Screener", "💼 Portfolio"],
            index=_idx_map.get(nav_page, 0),
            horizontal=True,
            key="nav_radio",
            label_visibility="collapsed",
        )
        new_nav_from_radio = ("screener" if "Screener" in nav_sel
                              else "portfolio" if "Portfolio" in nav_sel
                              else "analyzer")
        if new_nav_from_radio != nav_page:
            st.session_state["nav_page"] = new_nav_from_radio
            nav_page = new_nav_from_radio

        st.markdown("---")

        # Only show analyzer controls on analyzer page
        _cur = st.session_state.get("nav_page","analyzer")
        if _cur == "screener":
            st.caption("Use the screener panel on the right to find stocks.")
        elif _cur == "portfolio":
            st.caption("Upload your Yahoo Finance portfolio CSV on the right.")
        else:
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

        # Seed widget keys ONLY if they don't exist yet (first load or after
        # programmatic clear). Never overwrite a key that already exists —
        # that would revert user edits made in the current render cycle.
        for i in range(5):
            if f"t{i}" not in st.session_state:
                st.session_state[f"t{i}"] = slots[i] if i < len(slots) else ""

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
                    # Also try live name from yfinance if not in our lookup dict
                    if not disp_name and val.strip():
                        disp_name = st.session_state.get(
                            f"live_name_{val.strip().upper()}", "")
                    if disp_name:
                        st.markdown(
                            f"<div style='font-size:11px;color:#68d391;"
                            f"margin-top:2px;line-height:1.3'>"
                            f"↳ {disp_name}</div>",
                            unsafe_allow_html=True
                        )
            with col_x:
                has_val = val.strip() != ""
                if has_val and st.button("✕", key=f"clear_{i}", help="Remove"):
                    # Store which slot to clear — processed at top of next run
                    # BEFORE widgets render (same pattern as peer/screener buttons)
                    st.session_state["pending_clear_slot"] = i
                    st.rerun()
                elif not has_val:
                    st.empty()
            v = val.strip().upper()
            new_slots.append(v)
            if v:
                ticker_inputs.append(v)
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

    # ── Page routing ─────────────────────────────────────────────────────────
    if st.session_state.get("nav_page") == "screener":
        render_screener_page()
        return
    if st.session_state.get("nav_page") == "portfolio":
        render_portfolio_page()
        return

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

    auto_analyze = st.session_state.pop("auto_analyze", False)
    if (run_btn or auto_analyze) and ticker_inputs:
        with st.spinner("Fetching data and running analysis…"):
            rfr     = fetch_risk_free_rate()
            results = {}
            progress = st.progress(0, text="")
            n = len(ticker_inputs)

            for idx, ticker in enumerate(ticker_inputs):
                progress.progress(idx / n, text=f"Analysing {ticker}…")
                data = fetch_ticker_data(ticker)
                # Guard: data could be None if TTL cache had an exception
                if not data or not isinstance(data, dict):
                    results[ticker] = {"error": f"Failed to fetch data for '{ticker}'. Please try again."}
                    continue
                if data.get("error"):
                    results[ticker] = {"error": data["error"]}
                    continue

                info = data.get("info") or {}
                # Guard: info must be a non-empty dict
                if not info or not isinstance(info, dict):
                    results[ticker] = {"error": f"No data returned for '{ticker}'. Check the ticker symbol."}
                    continue

                try:
                    peers_list = get_peers(ticker, info, peer_overrides.get(ticker, ""))
                    peer_data  = fetch_peer_metrics(tuple(peers_list)) if peers_list else {}

                    fund       = analyze_fundamentals(data)
                    dcf        = run_dcf_models(data, rfr, fixed_rate)
                    ind        = calculate_indicators(data.get("hist_2y", pd.DataFrame()))
                    cp         = get_current_price(info)
                    sigs       = generate_signals(ind, cp)
                    valuation  = get_valuation_ratios(info)
                    comparison = compare_with_peers(valuation, peer_data, ticker)
                    sentiment  = analyze_sentiment(data) or {}

                    cfs        = (fund or {}).get("cashflows", {})
                    fs, fmsgs  = score_fundamental(info, cfs)
                    vs, vmsgs  = score_valuation(info, dcf)
                    tech_score = (sigs or {}).get("overall_score", 5.0)
                    ss, smsgs  = score_sentiment(sentiment)
                    scoring    = calculate_composite(fs, vs, tech_score, ss)
                    scoring["fund_msgs"] = fmsgs
                    scoring["val_msgs"]  = vmsgs
                    scoring["sent_msgs"] = smsgs

                    deep = run_deep_analysis(data, info)
                    results[ticker] = {
                        "data": data, "peers": peers_list, "peer_data": peer_data,
                        "fund": fund, "dcf": dcf, "indicators": ind, "signals": sigs,
                        "valuation": valuation, "comparison": comparison,
                        "sentiment": sentiment, "scoring": scoring,
                        "deep": deep,
                    }
                except Exception as _err:
                    results[ticker] = {"error": f"{ticker}: {_err}"}

            progress.progress(1.0, text="Done!")
            st.session_state["results"] = results
            st.session_state["rfr"]     = rfr
            # Cache company names for sidebar display under each ticker field
            for _t, _r in results.items():
                if not _r.get("error"):
                    _info = (_r.get("data") or {}).get("info") or {}
                    _name = _info.get("shortName") or _info.get("longName") or ""
                    if _name:
                        st.session_state[f"live_name_{_t.upper()}"] = _name
            # Feedback for peer analysis
            new_peer = st.session_state.pop("pending_analyze", None)
            if new_peer:
                if new_peer in results and not results[new_peer].get("error"):
                    st.success(f"✅ {new_peer} analyzed — see the new tab below.")
                elif new_peer in results:
                    st.error(f"Could not analyze {new_peer}: {results[new_peer].get('error','Unknown error')}")

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
                deep=r.get("deep", {}),
            )


if __name__ == "__main__":
    main()