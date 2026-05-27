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

# ─── Page config ─────────────────────────────────────────────────────────────
favicon = Image.open("favicon.png")
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

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

def fmt(v, pct=False, dec=2, prefix=""):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if pct:
        return f"{v:+.1%}" if abs(v) < 10 else f"{v:.1%}"
    if prefix == "$":
        if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
        return f"${v:,.{dec}f}"
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
    fig.update_layout(height=700, template="plotly_dark", title=f"{ticker} — Technical Analysis",
        title_font_size=15, margin=dict(l=50, r=30, t=50, b=30),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.06, x=0),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(showgrid=False)
    return fig


def build_financials_chart(fund, ticker):
    rev = fund.get("revenue_series", {}); ni = fund.get("net_income_series", {})
    gm  = fund.get("gross_margin_series", {}); om = fund.get("op_margin_series", {}); nm = fund.get("net_margin_series", {})
    years = sorted(set(list(rev.keys()) + list(ni.keys())), reverse=True)[:5]; years.sort()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         row_heights=[0.6, 0.4], subplot_titles=("Revenue & Net Income", "Margins"))
    fig.add_trace(go.Bar(x=years, y=[rev.get(y) for y in years], name="Revenue", marker_color="#42A5F5", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=[ni.get(y)  for y in years], name="Net Income", marker_color="#66BB6A", opacity=0.85), row=1, col=1)
    for d, col, lbl in [(gm,"#FFD700","Gross"),(om,"#FF6B35","Op."),(nm,"#AB47BC","Net")]:
        vals = [d.get(y) for y in years]
        if any(v for v in vals):
            fig.add_trace(go.Scatter(x=years, y=vals, name=lbl+" Margin", line=dict(color=col, width=2)), row=2, col=1)
    fig.update_layout(height=420, template="plotly_dark", barmode="group",
        margin=dict(l=40, r=20, t=40, b=20), paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
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
                     "📈 Technical Analysis", "🎯 Sentiment", "📅 Earnings & Forecasts"])

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
            with st.expander("⚙️ WACC Components"):
                w1,w2,w3,w4 = st.columns(4)
                w1.metric("Beta",           fmt(wacc_c.get("beta"), dec=2))
                w2.metric("Cost of Equity", fmt(wacc_c.get("cost_of_equity"), pct=True))
                w3.metric("Cost of Debt",   fmt(wacc_c.get("cost_of_debt"), pct=True))
                w4.metric("WACC",           fmt(wacc_c.get("wacc"), pct=True))
                w1.metric("Risk-Free Rate", fmt(wacc_c.get("rfr"), pct=True))
                w2.metric("Equity Weight",  fmt(wacc_c.get("w_equity"), pct=True))
                w3.metric("Debt Weight",    fmt(wacc_c.get("w_debt"), pct=True))
                w4.metric("Tax Rate",       fmt(wacc_c.get("tax_rate"), pct=True))
                with st.container():
                    st.markdown(tooltip_html("wacc","WACC Definition","", position="up"), unsafe_allow_html=True)
                    st.markdown(tooltip_html("capm","CAPM Definition","", position="up"), unsafe_allow_html=True)

        df_dcf = build_dcf_comparison(dcf, dcf.get("current_price", 0) or cp)
        st.dataframe(df_dcf, use_container_width=True, hide_index=True)
        st.caption(f"Current price: **{fmt(dcf.get('current_price',0) or cp, prefix='$')}** · "
                   f"FCF base: {fmt((dcf.get('fcf_list',[None])[0]), prefix='$') if dcf.get('fcf_list') else '—'}")

        with st.expander("ℹ️ How DCF Models Work"):
            st.markdown(tooltip_html("terminal_value","Terminal Value","",""), unsafe_allow_html=True)
            st.markdown("""
            | Model | Rate Used | Best For |
            |---|---|---|
            | **WACC DCF** | Blended debt+equity cost | Companies with meaningful debt |
            | **CAPM DCF** | Cost of equity (Ke = Rfr + β×5.5%) | Pure equity / no debt analysis |
            | **Fixed Rate** | Your chosen rate (sidebar) | Sensitivity testing |
            | **Two-Stage FCF** | CAPM rate, 2 growth phases | High-growth companies |
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
            rows = []
            for pt, pd_info in peer_data.items():
                rows.append({
                    "Ticker":    pt,
                    "Name":      pd_info.get("name","")[:20],
                    "Country":   pd_info.get("country",""),
                    "Mkt Cap":   fmt(pd_info.get("market_cap"), prefix="$"),
                    "P/E":       fmt(pd_info.get("pe_trailing"), dec=1),
                    "Fwd P/E":   fmt(pd_info.get("pe_forward"), dec=1),
                    "EV/EBITDA": fmt(pd_info.get("ev_ebitda"), dec=1),
                    "Net Margin":f"{pd_info.get('net_margin',0)*100:.1f}%" if pd_info.get("net_margin") else "—",
                    "ROE":       f"{pd_info.get('roe',0)*100:.1f}%"        if pd_info.get("roe")        else "—",
                    "Rev Growth":f"{pd_info.get('revenue_growth',0)*100:.1f}%" if pd_info.get("revenue_growth") else "—",
                    "D/E":       fmt((pd_info.get("debt_equity") or 0)/100, dec=2),
                    "Beta":      fmt(pd_info.get("beta"), dec=2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
        ("Market Cap",      lambda i: i.get("marketCap"),                           True),
    ]
    rows = []
    for label, fn, is_pct in metrics:
        row = {"Metric": label}
        for t, r in valid:
            v = fn(r["data"]["info"])
            row[t] = f"{v*100:.1f}%" if is_pct and v is not None else fmt(v, dec=1)
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
        st.image("logo.png", width=180)
        st.markdown("# 📈 Stock Analyzer")
        st.markdown("---")
        st.markdown("**Enter up to 5 tickers**")
        st.caption("US: AAPL · EU: ASML.AS SHEL.L SAP.DE TTE.PA · JP: 7203.T · KR: 005930.KS · HK: 0700.HK")

        ticker_inputs = []
        for i in range(5):
            t = st.text_input(f"Ticker {i+1}", key=f"t{i}",
                              label_visibility="collapsed", placeholder=f"Ticker {i+1}")
            if t.strip():
                ticker_inputs.append(t.strip().upper())

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
        st.caption("Data: Yahoo Finance · Cache: 1h\n"
                   "Peers: 60+ industries · 30+ exchanges\n"
                   "Scores: F=40% V=25% T=20% S=15%")

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
