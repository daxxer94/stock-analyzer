"""
portfolio_ui.py — Portfolio Analysis page UI.

Separate navigation page alongside Analyzer and Screener.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from portfolio import parse_yahoo_csv, enrich_portfolio, calc_portfolio_metrics
from currency import fmt_currency, CURRENCY_SYMBOLS


# ─── Colour helpers ───────────────────────────────────────────────────────────

QUOTE_TYPE_COLORS = {
    "EQUITY":    "#42A5F5",
    "ETF":       "#66BB6A",
    "MUTUALFUND":"#FFC107",
    "CRYPTO":    "#AB47BC",
    "FUTURE":    "#FF6B35",
    "INDEX":     "#90CAF9",
    "CURRENCY":  "#EF5350",
}
QUOTE_TYPE_ICONS = {
    "EQUITY":    "📈",
    "ETF":       "🧺",
    "MUTUALFUND":"💼",
    "CRYPTO":    "🔷",
    "FUTURE":    "⚡",
    "INDEX":     "📊",
    "CURRENCY":  "💱",
}


def _pnl_color(v):
    return "#00C853" if v >= 0 else "#EF5350"


def _fmt_pct(v, dec=1):
    if v is None: return "—"
    try:
        return f"{'+'if v>=0 else ''}{float(v):.{dec}f}%"
    except Exception:
        return "—"


def _fmt_money(v, ccy="$"):
    if v is None: return "—"
    try:
        f = float(v)
        sym = CURRENCY_SYMBOLS.get(ccy, ccy)
        if abs(f) >= 1e9:  return f"{sym}{f/1e9:,.2f}B"
        if abs(f) >= 1e6:  return f"{sym}{f/1e6:,.2f}M"
        if abs(f) >= 1e3:  return f"{sym}{f/1e3:,.1f}K"
        return f"{sym}{f:,.2f}"
    except Exception:
        return "—"


# ─── Upload section ───────────────────────────────────────────────────────────

def render_upload_section():
    st.markdown("""
    <div style='background:#1a1d2e;border:1px dashed #3a3f5c;border-radius:12px;
                padding:24px;text-align:center;margin-bottom:20px'>
      <div style='font-size:32px'>📂</div>
      <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin:8px 0'>
        Import your Yahoo Finance portfolio
      </div>
      <div style='font-size:12px;color:#718096'>
        Export from Yahoo Finance → My Portfolio → Download → CSV<br>
        Required columns: Symbol · Quantity · Purchase Price · Trade Date
      </div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Yahoo Finance Portfolio CSV",
        type=["csv"],
        key="portfolio_csv_upload",
        label_visibility="collapsed",
    )

    # Manual input fallback
    with st.expander("Or enter positions manually", expanded=False):
        st.markdown("Enter one position per line: `TICKER,QUANTITY,PURCHASE_PRICE`")
        manual = st.text_area("Positions (CSV format)", height=150,
                              placeholder="AAPL,10,178.50\nASML.AS,5,820.00\nMSFT,8,420.00")
        if st.button("Load manual positions", key="load_manual"):
            if manual.strip():
                lines = ["Symbol,Quantity,Purchase Price"]
                lines += [l.strip() for l in manual.strip().splitlines() if l.strip()]
                st.session_state["portfolio_csv_text"] = "\n".join(lines)
                st.session_state["portfolio_analyzed"] = False
                st.rerun()

    return uploaded


# ─── Summary Dashboard ────────────────────────────────────────────────────────

def render_summary(metrics: dict, enriched: pd.DataFrame):
    m  = metrics
    pnl_c = _pnl_color(m.get("total_pnl", 0))

    # Top metrics row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Invested",  _fmt_money(m.get("total_cost")))
    c2.metric("Market Value",    _fmt_money(m.get("total_value")))
    delta_str = f"{'+'if m.get('total_pnl',0)>=0 else ''}{_fmt_money(m.get('total_pnl'))} ({_fmt_pct(m.get('total_pnl_pct'))})"
    c3.metric("Unrealised P&L",  delta_str,
              delta_color="normal" if m.get("total_pnl",0) >= 0 else "inverse")
    c4.metric("Positions",       str(m.get("n_positions",0)))
    c5.metric("Portfolio Beta",  f"{m.get('portfolio_beta',0):.2f}")

    # Second row
    c6,c7,c8,c9 = st.columns(4)
    c6.metric("Top 5 Concentration", f"{m.get('top5_pct',0):.1f}%",
              help="% of portfolio in top 5 holdings. >40% = concentrated")
    c7.metric("HHI Diversification",
              f"{'High' if m['hhi']>0.25 else 'Moderate' if m['hhi']>0.10 else 'Low'} ({m.get('hhi',0):.3f})",
              help="Herfindahl-Hirschman Index. Lower = more diversified")
    if m.get("var_95_daily"):
        c8.metric("VaR 95% (1 day)", _fmt_money(m["var_95_daily"]),
                  help="Estimated max 1-day loss at 95% confidence")
    if m.get("div_income_ann"):
        c9.metric("Est. Annual Dividends", _fmt_money(m["div_income_ann"]))


# ─── Allocation Charts ────────────────────────────────────────────────────────

def render_allocation_charts(metrics: dict, enriched: pd.DataFrame):
    st.markdown("### 🥧 Portfolio Allocation")

    tab1, tab2, tab3, tab4 = st.tabs(["By Asset Type","By Sector","By Geography","By Currency"])

    with tab1:
        _donut_chart(metrics.get("type_alloc",{}), "Asset Type Allocation",
                     list(QUOTE_TYPE_COLORS.values()))

    with tab2:
        sa = metrics.get("sector_alloc",{})
        if sa:
            _donut_chart(sa, "Sector Allocation (Equities)",
                         px.colors.qualitative.Set2)
        else:
            st.info("Sector data not available (may require equity positions).")

    with tab3:
        ga = metrics.get("geo_alloc",{})
        if ga:
            _donut_chart(ga, "Geographic Allocation", px.colors.qualitative.Pastel)
        else:
            st.info("Country data unavailable.")

    with tab4:
        ca = metrics.get("ccy_alloc",{})
        if ca:
            _donut_chart(ca, "Currency Exposure", px.colors.qualitative.Bold)
        else:
            st.info("Currency data unavailable.")


def _donut_chart(data: dict, title: str, colors):
    if not data:
        st.info("No data.")
        return
    labels = [k if k else "Unknown" for k in data.keys()]
    values = list(data.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        texttemplate="%{label}<br><b>%{value:.1f}%</b>",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        marker=dict(colors=colors[:len(labels)],
                    line=dict(color="#0e1117", width=2)),
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13, x=0.01),
        height=300, showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=20,r=20,t=50,b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Position Table ───────────────────────────────────────────────────────────

def render_positions_table(enriched: pd.DataFrame):
    st.markdown("### 📋 Positions")
    st.caption("Click **▶ Analyze** to open a full analysis for any holding.")

    if enriched.empty:
        st.warning("No positions to display.")
        return

    # Sort controls
    sort_col = st.selectbox("Sort by", ["Market Value","P&L %","P&L €","Weight"],
                            label_visibility="collapsed", key="pos_sort")
    sort_map = {"Market Value":"market_value","P&L %":"pnl_pct",
                "P&L €":"unrealized_pnl","Weight":"weight_pct"}
    df = enriched.sort_values(sort_map[sort_col], ascending=False)

    # Column headers
    hcols = st.columns([0.6,1.6,0.8,0.7,0.8,0.9,1.0,1.0,0.8,1.1])
    for hcol, htxt in zip(hcols, ["Type","Symbol / Name","Qty","Avg Cost",
                                    "Price","Market Val","Cost Basis","P&L","Weight","Action"]):
        hcol.markdown(
            f"<div style='font-size:11px;font-weight:700;color:#718096;"
            f"padding:4px 0;border-bottom:2px solid #3a3f5c'>{htxt}</div>",
            unsafe_allow_html=True)

    for _, pos in df.iterrows():
        qt    = str(pos.get("quote_type","EQUITY"))
        icon  = QUOTE_TYPE_ICONS.get(qt, "📈")
        color = QUOTE_TYPE_COLORS.get(qt, "#90CAF9")
        pnl_c = _pnl_color(pos.get("unrealized_pnl",0))
        ccy   = pos.get("currency","USD")
        sym   = pos.get("symbol","")

        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = st.columns([0.6,1.6,0.8,0.7,0.8,0.9,1.0,1.0,0.8,1.1])

        c1.markdown(
            f"<span style='background:{color}22;color:{color};border:1px solid {color}44;"
            f"border-radius:6px;padding:2px 6px;font-size:11px'>{icon} {qt[:3]}</span>",
            unsafe_allow_html=True)

        c2.markdown(
            f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>{sym}</div>"
            f"<div style='font-size:10px;color:#718096'>{str(pos.get('name',''))[:22]}</div>",
            unsafe_allow_html=True)

        c3.markdown(f"<div style='font-size:12px'>{pos.get('quantity',0):.4g}</div>", unsafe_allow_html=True)
        c4.markdown(f"<div style='font-size:12px'>{_fmt_money(pos.get('avg_cost'), ccy)}</div>", unsafe_allow_html=True)
        c5.markdown(f"<div style='font-size:12px;font-weight:600'>{_fmt_money(pos.get('price'), ccy)}</div>", unsafe_allow_html=True)
        c6.markdown(f"<div style='font-size:12px;font-weight:700'>{_fmt_money(pos.get('market_value'), ccy)}</div>", unsafe_allow_html=True)
        c7.markdown(f"<div style='font-size:12px'>{_fmt_money(pos.get('cost_basis'), ccy)}</div>", unsafe_allow_html=True)

        pnl   = pos.get("unrealized_pnl",0)
        pnlp  = pos.get("pnl_pct",0)
        c8.markdown(
            f"<div style='font-size:12px;font-weight:700;color:{pnl_c}'>"
            f"{_fmt_money(pnl,ccy)}</div>"
            f"<div style='font-size:10px;color:{pnl_c}'>{_fmt_pct(pnlp)}</div>",
            unsafe_allow_html=True)

        c9.markdown(f"<div style='font-size:12px'>{pos.get('weight_pct',0):.1f}%</div>",
                    unsafe_allow_html=True)

        if c10.button("▶ Analyze", key=f"port_analyze_{sym}", use_container_width=True):
            st.session_state["pending_add_ticker"] = sym
            st.session_state["auto_analyze"]       = True
            st.session_state["nav_page"]           = "analyzer"
            st.rerun()

        st.divider()


# ─── ETF Details ──────────────────────────────────────────────────────────────

def render_etf_details(enriched: pd.DataFrame):
    etfs = enriched[enriched["quote_type"] == "ETF"]
    if etfs.empty:
        return

    st.markdown("### 🧺 ETF Details")
    st.caption("Fund-specific information for ETF positions")

    for _, etf in etfs.iterrows():
        with st.expander(f"{etf.get('symbol','')} — {etf.get('name','')} ({etf.get('weight_pct',0):.1f}% of portfolio)"):
            c1, c2, c3 = st.columns(3)
            if etf.get("etf_expense_ratio"):
                c1.metric("Expense Ratio", f"{etf['etf_expense_ratio']:.2%}")
            if etf.get("etf_total_assets"):
                c2.metric("AUM", _fmt_money(etf["etf_total_assets"], etf.get("currency","USD")))
            if etf.get("etf_yield"):
                c3.metric("Distribution Yield", f"{etf['etf_yield']:.2%}")

            c4, c5, c6 = st.columns(3)
            if etf.get("etf_ytd_return"):
                c4.metric("YTD Return", _fmt_pct(etf["etf_ytd_return"]*100))
            if etf.get("etf_3yr_return"):
                c5.metric("3Y Avg Return", _fmt_pct(etf["etf_3yr_return"]*100))
            if etf.get("etf_5yr_return"):
                c6.metric("5Y Avg Return", _fmt_pct(etf["etf_5yr_return"]*100))

            info_cols = [
                ("Fund Family",   etf.get("etf_fund_family","")),
                ("Category",      etf.get("etf_category","")),
                ("Legal Type",    etf.get("etf_legal_type","")),
            ]
            for lbl, val in info_cols:
                if val:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:12px;padding:3px 0;border-bottom:1px solid #1e2130'>"
                        f"<span style='color:#718096'>{lbl}</span>"
                        f"<span style='color:#e2e8f0;font-weight:600'>{val}</span>"
                        f"</div>",
                        unsafe_allow_html=True)


# ─── Performance Chart ────────────────────────────────────────────────────────

def render_performance_chart(enriched: pd.DataFrame):
    if enriched.empty:
        return

    # Waterfall chart: P&L contribution per position
    df_sorted = enriched.sort_values("unrealized_pnl", ascending=False)
    symbols = df_sorted["symbol"].tolist()
    pnls    = df_sorted["unrealized_pnl"].tolist()
    colors  = ["#00C853" if p >= 0 else "#EF5350" for p in pnls]

    fig = go.Figure(go.Bar(
        x=symbols, y=pnls,
        marker_color=colors,
        text=[f"{_fmt_pct(p)}" for p in df_sorted["pnl_pct"].tolist()],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="white", width=1))
    fig.update_layout(
        title=dict(text="Unrealised P&L by Position", font_size=13, x=0.01),
        height=300, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1d2e",
        margin=dict(l=60,r=30,t=50,b=60),
        xaxis=dict(tickfont=dict(size=10), tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Risk Summary ─────────────────────────────────────────────────────────────

def render_risk_section(metrics: dict, enriched: pd.DataFrame):
    st.markdown("### ⚡ Risk Overview")
    m = metrics

    c1, c2, c3 = st.columns(3)

    # Beta interpretation
    beta = m.get("portfolio_beta", 1.0)
    if beta > 1.3:   beta_lbl, beta_c = "Aggressive (>1.3)", "#EF5350"
    elif beta > 0.8: beta_lbl, beta_c = "Market-like", "#FFC107"
    else:            beta_lbl, beta_c = "Defensive (<0.8)", "#66BB6A"

    c1.markdown(
        f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;padding:16px'>"
        f"<div style='font-size:12px;color:#718096'>Portfolio Beta</div>"
        f"<div style='font-size:28px;font-weight:800;color:{beta_c}'>{beta:.2f}</div>"
        f"<div style='font-size:11px;color:{beta_c}'>{beta_lbl}</div>"
        f"<div style='font-size:11px;color:#718096;margin-top:4px'>"
        f"A 10% market move → ~{beta*10:.1f}% portfolio move</div>"
        f"</div>",
        unsafe_allow_html=True)

    # Concentration
    hhi = m.get("hhi", 0)
    if hhi > 0.25:   hhi_lbl, hhi_c = "Highly concentrated", "#EF5350"
    elif hhi > 0.10: hhi_lbl, hhi_c = "Moderately diversified", "#FFC107"
    else:            hhi_lbl, hhi_c = "Well diversified", "#66BB6A"

    c2.markdown(
        f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;padding:16px'>"
        f"<div style='font-size:12px;color:#718096'>Diversification (HHI)</div>"
        f"<div style='font-size:28px;font-weight:800;color:{hhi_c}'>{hhi:.3f}</div>"
        f"<div style='font-size:11px;color:{hhi_c}'>{hhi_lbl}</div>"
        f"<div style='font-size:11px;color:#718096;margin-top:4px'>"
        f"Top 5 holdings: {m.get('top5_pct',0):.1f}% of portfolio</div>"
        f"</div>",
        unsafe_allow_html=True)

    # VaR
    var = m.get("var_95_daily")
    c3.markdown(
        f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;padding:16px'>"
        f"<div style='font-size:12px;color:#718096'>Value at Risk (95%, 1-day)</div>"
        f"<div style='font-size:22px;font-weight:800;color:#FF9800'>{_fmt_money(var) if var else 'N/A'}</div>"
        f"<div style='font-size:11px;color:#718096;margin-top:4px'>"
        f"Estimated maximum daily loss at 95% confidence.<br>"
        f"Based on weighted volatility proxy — indicative only.</div>"
        f"</div>",
        unsafe_allow_html=True)

    # Best / worst performers
    st.markdown("---")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**🟢 Best performers**")
        for p in m.get("best",[]):
            c = _pnl_color(p["pnl_pct"])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:3px 0'>"
                f"<span style='color:#e2e8f0;font-weight:600'>{p['symbol']}</span>"
                f"<span style='color:{c};font-weight:700'>{_fmt_pct(p['pnl_pct'])}</span>"
                f"</div>", unsafe_allow_html=True)
    with pc2:
        st.markdown("**🔴 Worst performers**")
        for p in m.get("worst",[]):
            c = _pnl_color(p["pnl_pct"])
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:13px;padding:3px 0'>"
                f"<span style='color:#e2e8f0;font-weight:600'>{p['symbol']}</span>"
                f"<span style='color:{c};font-weight:700'>{_fmt_pct(p['pnl_pct'])}</span>"
                f"</div>", unsafe_allow_html=True)


# ─── Main Portfolio Page ──────────────────────────────────────────────────────

def render_portfolio_page():
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1d2e,#0d1117);
                border:1px solid #3a3f5c;border-radius:12px;padding:20px 24px;margin-bottom:20px'>
      <div style='font-size:22px;font-weight:800;color:#e2e8f0'>💼 Portfolio Analysis</div>
      <div style='font-size:13px;color:#718096;margin-top:4px'>
        Import your Yahoo Finance portfolio CSV for full position analysis,
        risk metrics, asset allocation and P&L breakdown.
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Upload / load ─────────────────────────────────────────────────────────
    uploaded = render_upload_section()

    if uploaded is not None:
        text = uploaded.read().decode("utf-8")
        # Only reset analysis if the file content actually changed
        import hashlib
        new_hash = hashlib.md5(text.encode()).hexdigest()
        if new_hash != st.session_state.get("portfolio_csv_hash"):
            st.session_state["portfolio_csv_text"] = text
            st.session_state["portfolio_csv_hash"] = new_hash
            st.session_state["portfolio_analyzed"]  = False
            st.session_state.pop("portfolio_enriched", None)
            st.session_state.pop("portfolio_metrics",  None)

    csv_text = st.session_state.get("portfolio_csv_text")
    if not csv_text:
        return

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        positions_df = parse_yahoo_csv(csv_text)
    except Exception as e:
        st.error(f"CSV parse error: {e}")
        return

    if positions_df.empty:
        st.warning("No valid positions found in the CSV. "
                   "Check that Symbol, Quantity and Purchase Price columns are present.")
        return

    # ── Run enrichment (check BEFORE the preview block so the return doesn't block it)
    if st.session_state.get("portfolio_running"):
        progress = st.progress(0, "Fetching live data…")
        status   = st.empty()

        def _prog(i, total, sym):
            pct = int((i / max(total, 1)) * 100)
            progress.progress(pct / 100, f"Fetching {sym} ({i+1}/{total})…")
            status.caption(f"Fetching {sym}…")

        _analysis_error = None
        try:
            enriched = enrich_portfolio(positions_df, progress_cb=_prog)
            metrics  = calc_portfolio_metrics(enriched)
            st.session_state["portfolio_enriched"] = enriched
            st.session_state["portfolio_metrics"]  = metrics
            st.session_state["portfolio_analyzed"] = True
        except Exception as e:
            _analysis_error = e
        finally:
            st.session_state["portfolio_running"] = False

        # st.rerun() raises RerunException — must be OUTSIDE try/except
        if _analysis_error:
            st.error(f"Analysis error: {_analysis_error}")
        else:
            st.rerun()
        return

    # Preview before first analysis
    if not st.session_state.get("portfolio_analyzed"):
        st.markdown(
            f"<div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;"
            f"padding:14px 18px;margin-bottom:12px'>"
            f"<div style='font-size:14px;font-weight:700;color:#e2e8f0'>"
            f"✅ {len(positions_df)} positions loaded</div>"
            f"<div style='font-size:12px;color:#94a3b8;margin-top:4px'>"
            f"{', '.join(positions_df['symbol'].tolist())}"
            f"</div></div>",
            unsafe_allow_html=True
        )
        if st.button("🔍 Run Portfolio Analysis", type="primary",
                     use_container_width=True, key="run_portfolio"):
            st.session_state["portfolio_running"]  = True
            st.session_state["portfolio_analyzed"] = False
            st.rerun()
        return

    # ── Display results ───────────────────────────────────────────────────────
    enriched = st.session_state.get("portfolio_enriched", pd.DataFrame())
    metrics  = st.session_state.get("portfolio_metrics", {})

    if enriched.empty or not metrics:
        st.info("No portfolio data. Upload a CSV and click Run.")
        return

    # Refresh button
    col_r, col_e = st.columns([1, 5])
    with col_r:
        if st.button("🔄 Refresh prices", key="refresh_portfolio"):
            st.session_state["portfolio_analyzed"] = False
            st.session_state["portfolio_running"]  = True
            st.rerun()
    with col_e:
        if st.button("🗑️ Clear portfolio", key="clear_portfolio"):
            for k in ["portfolio_csv_text","portfolio_analyzed",
                      "portfolio_enriched","portfolio_metrics"]:
                st.session_state.pop(k, None)
            st.rerun()

    # Summary
    render_summary(metrics, enriched)
    st.divider()

    # Allocation + performance
    col_a, col_p = st.columns([1.2, 1])
    with col_a:
        render_allocation_charts(metrics, enriched)
    with col_p:
        render_performance_chart(enriched)

    st.divider()

    # Risk
    render_risk_section(metrics, enriched)
    st.divider()

    # Positions table
    render_positions_table(enriched)
    st.divider()

    # ETF details
    render_etf_details(enriched)

    st.markdown("""
    > ⚠️ **Disclaimer:** Portfolio data is for informational purposes only.
    > P&L calculations are based on import data and live Yahoo Finance prices.
    > VaR is a statistical estimate — actual losses can exceed this figure.
    > This does not constitute financial advice.
    """)