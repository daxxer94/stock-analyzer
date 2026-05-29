"""
screener_ui.py — Screener page UI.

Renders as a completely separate page from the analyzer.
Called from app.py when user selects "Screener" in navigation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from screener import (
    run_screener, ScreenerCriteria, get_preset_criteria,
    SECTORS, REGIONS, CONTINENTS, MARKET_CAP_BUCKETS,
    INDUSTRIES_BY_SECTOR, _classify_stock_type,
)
from currency import CURRENCY_SYMBOLS, fmt_currency


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(v, pct=False, dec=1, prefix=""):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if pct:
        return f"{v:+.1%}" if abs(v) < 10 else f"{v:.0%}"
    if prefix == "$":
        if abs(v) >= 1e12: return f"${v/1e12:.1f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
        return f"${v:,.{dec}f}"
    return f"{v:.{dec}f}"


STOCK_TYPE_COLORS = {
    "Growth":          "#00C853",
    "GARP":            "#42A5F5",
    "Deep Value":      "#FF6B35",
    "Value":           "#81C784",
    "Dividend":        "#FFD700",
    "Speculative":     "#EF5350",
    "Small-Cap Growth":"#AB47BC",
    "Blend":           "#90CAF9",
}

STOCK_TYPE_ICONS = {
    "Growth":          "🚀",
    "GARP":            "⚖️",
    "Deep Value":      "💎",
    "Value":           "🏛️",
    "Dividend":        "💰",
    "Speculative":     "⚡",
    "Small-Cap Growth":"🌱",
    "Blend":           "🔵",
}

SIGNAL_COLORS = {
    1: "#00C853", 2: "#4CAF50", 3: "#8BC34A",
    4: "#FFC107", 5: "#FF9800", 6: "#F44336", 7: "#B71C1C",
}


def score_color(s):
    if s >= 7:  return "#00C853"
    if s >= 5.5: return "#FFC107"
    return "#F44336"


def score_badge(s, size=13):
    c = score_color(s)
    return (f"<span style='background:{c}22;color:{c};border:1px solid {c}44;"
            f"border-radius:8px;padding:2px 8px;font-size:{size}px;font-weight:700'>{s:.1f}</span>")


def type_badge(stock_type):
    c = STOCK_TYPE_COLORS.get(stock_type, "#90CAF9")
    ic = STOCK_TYPE_ICONS.get(stock_type, "🔵")
    return (f"<span style='background:{c}22;color:{c};border:1px solid {c}44;"
            f"border-radius:8px;padding:2px 8px;font-size:11px;font-weight:600'>{ic} {stock_type}</span>")


def upside_badge(upside):
    if upside > 0.15:  c = "#00C853"; label = f"▲ {upside:.0%}"
    elif upside > 0:   c = "#8BC34A"; label = f"▲ {upside:.0%}"
    elif upside > -0.10: c = "#FFC107"; label = f"▼ {abs(upside):.0%}"
    else:              c = "#EF5350"; label = f"▼ {abs(upside):.0%}"
    return (f"<span style='color:{c};font-weight:700;font-size:12px'>{label}</span>")


# ─── Criteria Panel ───────────────────────────────────────────────────────────

def render_criteria_panel() -> ScreenerCriteria:
    """Render the left-side criteria panel and return a ScreenerCriteria object."""
    criteria = ScreenerCriteria()

    # ── Preset buttons ────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Quick Presets")
    presets = [
        "🚀 High Growth + High Upside",
        "📈 Revenue Accelerators",
        "🎯 Analyst Favourites",
        "High Growth Tech",
        "Dividend Income",
        "European Value",
        "Asian Growth",
        "Quality Compounders",
        "Undervalued Gems",
        "Healthcare Innovation",
        "High Conviction (Global)",
    ]
    # 3 columns for presets
    preset_cols = st.columns(3)
    for i, p in enumerate(presets):
        if preset_cols[i % 3].button(p, key=f"preset_{p}", use_container_width=True):
            st.session_state["screener_preset"] = p
            st.rerun()

    # Apply preset if selected
    active_preset = st.session_state.get("screener_preset")
    if active_preset:
        preset_c = get_preset_criteria(active_preset)
        st.success(f"Preset loaded: **{active_preset}** — adjust below or click Run")
        if st.button("✕ Clear preset", key="clear_preset"):
            st.session_state.pop("screener_preset", None)
            st.rerun()
        criteria = preset_c
        # Show the preset params but allow modification via expander
        with st.expander("Adjust preset filters", expanded=False):
            criteria = _render_all_filters(criteria)
    else:
        criteria = _render_all_filters(criteria)

    return criteria


def _render_all_filters(c: ScreenerCriteria) -> ScreenerCriteria:
    """Render all filter controls, return updated criteria."""

    # ── Geography ─────────────────────────────────────────────────────────────
    st.markdown("#### 🌍 Geography")

    geo_mode = st.radio("Select by", ["Continent", "Individual Countries"],
                         horizontal=True, key="geo_mode")

    if geo_mode == "Continent":
        continent = st.selectbox("Continent", list(CONTINENTS.keys()),
                                  index=3, key="continent_sel")
        region_codes = CONTINENTS[continent]
        if region_codes:
            c.regions = region_codes
        # else: empty = no filter (All Global)
    else:
        region_labels = {code: label for code, label in REGIONS.items()}
        selected = st.multiselect(
            "Countries",
            options=list(REGIONS.keys()),
            format_func=lambda x: REGIONS[x],
            key="countries_sel",
        )
        c.regions = selected

    # ── Sector & Industry ─────────────────────────────────────────────────────
    st.markdown("#### 🏭 Sector & Industry")
    selected_sectors = st.multiselect(
        "Sector (leave empty = all)",
        options=SECTORS,
        default=getattr(c, "sectors", []),
        key="sectors_sel",
    )
    c.sectors = selected_sectors

    # Industry (only show if single sector selected)
    if len(selected_sectors) == 1:
        available_industries = INDUSTRIES_BY_SECTOR.get(selected_sectors[0], [])
        if available_industries:
            selected_industries = st.multiselect(
                "Industry (optional)",
                options=sorted(available_industries),
                key="industries_sel",
            )
            c.industries = selected_industries

    # ── Market Cap ────────────────────────────────────────────────────────────
    st.markdown("#### 💰 Market Cap")
    cap_options = list(MARKET_CAP_BUCKETS.keys())
    cap_sel = st.multiselect(
        "Market cap range (multiple OK)",
        options=cap_options,
        default=["Large ($10B–$200B)", "Mid ($2B–$10B)"],
        key="cap_sel",
    )
    if cap_sel and "Any" not in cap_sel:
        all_lo = [MARKET_CAP_BUCKETS[k][0] for k in cap_sel]
        all_hi = [MARKET_CAP_BUCKETS[k][1] for k in cap_sel]
        c.min_mktcap = min(all_lo)
        c.max_mktcap = max(all_hi)

    # ── Stock Type ────────────────────────────────────────────────────────────
    st.markdown("#### 🏷️ Stock Type")
    type_options = ["Growth","GARP","Value","Deep Value","Dividend","Speculative","Small-Cap Growth","Blend"]
    selected_types = st.multiselect(
        "Stock type (leave empty = all)",
        options=type_options,
        default=getattr(c, "stock_types", []),
        key="types_sel",
    )
    c.stock_types = selected_types

    # ── Valuation ─────────────────────────────────────────────────────────────
    st.markdown("#### 📊 Valuation")
    col1, col2 = st.columns(2)
    with col1:
        max_pe = st.number_input("Max Trailing P/E",
            min_value=0.0, max_value=500.0,
            value=float(min(c.max_pe, 100)) if c.max_pe < 999 else 100.0,
            step=5.0, key="max_pe")
        c.max_pe = max_pe if max_pe < 100 else 999

        max_fwd_pe = st.number_input("Max Forward P/E",
            min_value=0.0, max_value=200.0,
            value=float(min(c.max_fwd_pe, 50)) if c.max_fwd_pe < 999 else 50.0,
            step=5.0, key="max_fwd_pe")
        c.max_fwd_pe = max_fwd_pe if max_fwd_pe < 50 else 999

        max_peg = st.number_input("Max PEG Ratio",
            min_value=0.0, max_value=10.0,
            value=float(min(c.max_peg, 3.0)) if c.max_peg < 999 else 3.0,
            step=0.25, key="max_peg")
        c.max_peg = max_peg if max_peg < 3.0 else 999

    with col2:
        max_pb = st.number_input("Max Price/Book",
            min_value=0.0, max_value=50.0,
            value=float(min(c.max_pb, 15.0)) if c.max_pb < 999 else 15.0,
            step=1.0, key="max_pb")
        c.max_pb = max_pb if max_pb < 15.0 else 999

        max_ev_ebitda = st.number_input("Max EV/EBITDA",
            min_value=0.0, max_value=100.0,
            value=float(min(c.max_ev_ebitda, 30.0)) if c.max_ev_ebitda < 999 else 30.0,
            step=2.0, key="max_ev_ebitda")
        c.max_ev_ebitda = max_ev_ebitda if max_ev_ebitda < 30.0 else 999

    # ── Growth ────────────────────────────────────────────────────────────────
    st.markdown("#### 📈 Growth")
    col1, col2 = st.columns(2)
    with col1:
        min_rev_g = st.slider("Min Revenue Growth (%)",
            -50, 100,
            value=int(max(c.min_rev_growth * 100, -50)),
            step=5, key="min_rev_g")
        c.min_rev_growth = min_rev_g / 100
    with col2:
        min_eps_g = st.slider("Min EPS Growth (%)",
            -50, 100,
            value=int(max(c.min_eps_growth * 100, -50)),
            step=5, key="min_eps_g")
        c.min_eps_growth = min_eps_g / 100

    # ── Profitability ─────────────────────────────────────────────────────────
    st.markdown("#### 💵 Profitability")
    col1, col2 = st.columns(2)
    with col1:
        min_gm = st.slider("Min Gross Margin (%)",
            -20, 90, value=int(max(c.min_gross_margin * 100, -20)), step=5, key="min_gm")
        c.min_gross_margin = min_gm / 100

        min_nm = st.slider("Min Net Margin (%)",
            -30, 50, value=int(max(c.min_net_margin * 100, -30)), step=5, key="min_nm")
        c.min_net_margin = min_nm / 100
    with col2:
        min_roe = st.slider("Min ROE (%)",
            -20, 60, value=int(max(c.min_roe * 100, -20)), step=5, key="min_roe")
        c.min_roe = min_roe / 100

        min_div = st.slider("Min Dividend Yield (%)",
            0, 15, value=int(c.min_dividend_yield * 100), step=1, key="min_div")
        c.min_dividend_yield = min_div / 100

    # ── Risk ──────────────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Risk (Beta)")
    beta_range = st.slider("Beta range",
        0.0, 3.0, value=(float(c.min_beta), min(float(c.max_beta), 3.0)),
        step=0.1, key="beta_range")
    c.min_beta = beta_range[0]
    c.max_beta = beta_range[1]

    # ── Quality threshold ─────────────────────────────────────────────────────
    st.markdown("#### 🏆 Minimum Quality Score")
    c.min_score = st.slider("Min composite score (0–10)",
        0.0, 9.0, value=float(c.min_score), step=0.5, key="min_score")

    # ── Result count ──────────────────────────────────────────────────────────
    st.markdown("#### 📋 Results")
    c.max_results = st.selectbox("Max results to show",
        [10, 20, 30, 50, 75, 100], index=2, key="max_results")

    return c


# ─── Results Panel ────────────────────────────────────────────────────────────

def render_results(results: list):
    """Render the screener results table."""
    if not results:
        st.warning("No stocks matched your criteria. Try relaxing the filters.")
        return

    disp_ccy = st.session_state.get("display_ccy_code", "USD")
    rates    = st.session_state.get("fx_rates", {})

    st.markdown(f"### Found **{len(results)}** matching stocks")
    st.caption("Click **▶ Analyze** to open a full analysis in the Analyzer tab.")

    # Summary cards (top 3)
    st.markdown("#### 🏆 Top Picks")
    top_cols = st.columns(min(3, len(results)))
    for i, result_row in enumerate(results[:3]):
        ticker, m, score, stype, upside = result_row[0], result_row[1], result_row[2], result_row[3], result_row[4]
        cp = m.get("current_price", 0)
        cp_str = fmt_currency(cp, m.get("currency","USD"), disp_ccy, rates)
        with top_cols[i]:
            sc = score_color(score)
            tc = STOCK_TYPE_COLORS.get(stype, "#90CAF9")
            st.markdown(f"""
            <div style='background:#1a1d2e;border:1px solid #3a3f5c;border-radius:10px;
                         padding:14px 16px;text-align:center'>
              <div style='font-size:18px;font-weight:800;color:#e2e8f0'>{ticker}</div>
              <div style='font-size:11px;color:#718096;margin:2px 0'>{m.get('name','')[:22]}</div>
              <div style='font-size:24px;font-weight:800;color:{sc};margin:6px 0'>{score:.1f}</div>
              <div style='margin:4px 0'>
                <span style='background:{tc}22;color:{tc};border:1px solid {tc}44;
                  border-radius:8px;padding:2px 8px;font-size:10px;font-weight:600'>
                  {STOCK_TYPE_ICONS.get(stype,"🔵")} {stype}
                </span>
              </div>
              <div style='font-size:13px;color:#e2e8f0;margin-top:6px'>{cp_str}</div>
              {f"<div style='font-size:11px;color:{'#00C853' if upside>0 else '#EF5350'}'>{'▲' if upside>0 else '▼'} {abs(upside):.0%} analyst upside</div>" if upside else ""}
            </div>""", unsafe_allow_html=True)
            if st.button(f"▶ Analyze {ticker}", key=f"top_analyze_{ticker}_{i}",
                         use_container_width=True):
                _send_to_analyzer(ticker)

    st.markdown("---")

    # ── Full results table ────────────────────────────────────────────────────
    st.markdown("#### 📋 All Results")

    # Column headers
    hcols = st.columns([1.4, 1.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 1.1])
    headers = ["Ticker","Name / Type","Score","G+U","Mkt Cap","P/E","Fwd P/E","Rev Grw","Net Mgn","Analyst↑","Action"]
    for hcol, htxt in zip(hcols, headers):
        hcol.markdown(
            f"<div style='font-size:11px;font-weight:700;color:#718096;"
            f"padding:4px 0;border-bottom:2px solid #3a3f5c'>{htxt}</div>",
            unsafe_allow_html=True
        )

    for rank, result_row in enumerate(results):
        ticker, m, score, stype, upside = result_row[0], result_row[1], result_row[2], result_row[3], result_row[4]
        gu_score = result_row[5] if len(result_row) > 5 else None
        cp     = m.get("current_price", 0)
        ccy    = m.get("currency", "USD")
        cp_str = fmt_currency(cp, ccy, disp_ccy, rates)

        c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11 = st.columns([1.4,1.6,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,1.1])

        c1.markdown(
            f"<div style='font-size:13px;font-weight:700;color:#e2e8f0'>{ticker}</div>"
            f"<div style='font-size:10px;color:#718096'>{m.get('country','')}"
            f"  ·  {m.get('sector','')[:14]}</div>",
            unsafe_allow_html=True
        )
        tc = STOCK_TYPE_COLORS.get(stype, "#90CAF9")
        ic = STOCK_TYPE_ICONS.get(stype, "🔵")
        c2.markdown(
            f"<div style='font-size:11px;color:#e2e8f0'>{m.get('name','')[:20]}</div>"
            f"<span style='background:{tc}22;color:{tc};border:1px solid {tc}33;"
            f"border-radius:6px;padding:1px 6px;font-size:10px'>{ic} {stype}</span>",
            unsafe_allow_html=True
        )
        sc = score_color(score)
        c3.markdown(f"<div style='font-size:14px;font-weight:700;color:{sc}'>{score:.1f}</div>",
                    unsafe_allow_html=True)
        # G+U = Growth + Upside sub-score
        if gu_score is not None:
            gu_c = score_color(gu_score)
            c4.markdown(f"<div style='font-size:13px;font-weight:600;color:{gu_c}' title='Growth+Upside score'>{gu_score:.1f}</div>",
                        unsafe_allow_html=True)
        else:
            c4.markdown("—")
        c5.markdown(f"<div style='font-size:12px'>{_fmt(m.get('market_cap'), prefix='$')}</div>",
                    unsafe_allow_html=True)
        c6.markdown(f"<div style='font-size:12px'>{_fmt(m.get('pe_trailing'), dec=1)}</div>",
                    unsafe_allow_html=True)
        c7.markdown(f"<div style='font-size:12px'>{_fmt(m.get('pe_forward'), dec=1)}</div>",
                    unsafe_allow_html=True)
        rg = m.get("revenue_growth")
        c8.markdown(
            f"<div style='font-size:12px;color:{'#66BB6A' if rg and rg>0 else '#EF5350' if rg else '#718096'}'>"
            f"{_fmt(rg, pct=True)}</div>",
            unsafe_allow_html=True
        )
        nm = m.get("net_margin")
        c9.markdown(
            f"<div style='font-size:12px;color:{'#66BB6A' if nm and nm>0 else '#EF5350' if nm else '#718096'}'>"
            f"{_fmt(nm, pct=True)}</div>",
            unsafe_allow_html=True
        )
        c10.markdown(upside_badge(upside) if upside else "—", unsafe_allow_html=True)

        if c11.button("▶ Analyze", key=f"res_analyze_{ticker}_{rank}",
                      use_container_width=True):
            _send_to_analyzer(ticker)

        st.divider()

    # ── Export to CSV ─────────────────────────────────────────────────────────
    with st.expander("📥 Export results as CSV"):
        rows = []
        for result_row in results:
            ticker, m, score, stype, upside = result_row[0], result_row[1], result_row[2], result_row[3], result_row[4]
            rows.append({
                "Ticker":        ticker,
                "Name":          m.get("name",""),
                "Type":          stype,
                "Score":         score,
                "Country":       m.get("country",""),
                "Sector":        m.get("sector",""),
                "Industry":      m.get("industry",""),
                "Market Cap":    m.get("market_cap"),
                "Price":         m.get("current_price"),
                "Currency":      m.get("currency","USD"),
                "P/E":           m.get("pe_trailing"),
                "Fwd P/E":       m.get("pe_forward"),
                "PEG":           m.get("peg"),
                "EV/EBITDA":     m.get("ev_ebitda"),
                "Revenue Growth":m.get("revenue_growth"),
                "Net Margin":    m.get("net_margin"),
                "Gross Margin":  m.get("gross_margin"),
                "ROE":           m.get("roe"),
                "Div Yield":     m.get("dividend_yield"),
                "Beta":          m.get("beta"),
                "Analyst Target":m.get("target_mean"),
                "Analyst Upside":f"{upside:.1%}" if upside else "",
                "# Analysts":    m.get("num_analysts",0),
            })
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name="screener_results.csv",
            mime="text/csv",
        )


def _send_to_analyzer(ticker: str):
    """
    Send a ticker to the analyzer page and auto-run analysis.
    Preserves any existing tickers already in the analyzer slots.
    """
    slots = st.session_state.get("ticker_slots", ["","","","",""])
    # Add only if not already present
    if ticker not in slots:
        added = False
        for si in range(5):
            if not slots[si]:
                slots[si] = ticker
                added = True
                break
        if not added:
            # All 5 slots full: insert at top, shift others
            slots = [ticker] + list(slots[:4])
    st.session_state["ticker_slots"] = slots
    # Clear widget keys so the seeding loop re-applies fresh values
    for ki in range(5):
        st.session_state.pop(f"t{ki}", None)
    st.session_state["nav_page"]     = "analyzer"
    st.session_state["auto_analyze"] = True
    st.rerun()


# ─── Chart: Score distribution ────────────────────────────────────────────────

def render_results_chart(results: list):
    """Show a scatter chart of score vs analyst upside for all results."""
    if len(results) < 3:
        return

    tickers  = [r[0] for r in results]
    scores   = [r[2] for r in results]
    upsides  = [r[4] * 100 if r[4] else 0 for r in results]
    types    = [r[3] for r in results]
    names    = [r[1].get("name","") for r in results]
    mktcaps  = [r[1].get("market_cap") or 1e9 for r in results]
    colors   = [STOCK_TYPE_COLORS.get(t, "#90CAF9") for t in types]

    # Normalize bubble size
    max_cap = max(mktcaps)
    sizes   = [max(8, min(40, (mc / max_cap) ** 0.3 * 40)) for mc in mktcaps]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores, y=upsides,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        marker=dict(
            color=colors,
            size=sizes,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        customdata=list(zip(names, types,
                            [r[1].get("market_cap") for r in results],
                            [r[1].get("revenue_growth") for r in results])),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "%{customdata[0]}<br>"
            "Type: %{customdata[1]}<br>"
            "Score: %{x:.1f}/10<br>"
            "Analyst upside: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Quadrant lines
    fig.add_hline(y=0,  line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))
    fig.add_vline(x=6.5, line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"))

    # Quadrant labels
    for x, y, txt in [(9, max(upsides)*0.85, "🟢 High Score\nHigh Upside"),
                       (4, max(upsides)*0.85, "🟡 Low Score\nHigh Upside"),
                       (9, min(upsides)*0.85, "🟡 High Score\nLow Upside"),
                       (4, min(upsides)*0.85, "🔴 Low Score\nLow Upside")]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color="#4a5568"), xanchor="center")

    fig.update_layout(
        title=dict(text="Quality Score vs Analyst Upside  (bubble size = market cap)",
                   font_size=12, x=0.01),
        height=420,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d2e",
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(
            title="Composite Score (0–10)",
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11), range=[0, 10],
        ),
        yaxis=dict(
            title="Analyst Upside (%)",
            showgrid=True, gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(size=11), ticksuffix="%",
        ),
        hoverlabel=dict(bgcolor="#1a1d2e", font_size=12),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Main screener page ───────────────────────────────────────────────────────

def render_screener_page():
    """Entry point: renders the full screener page."""

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1a1d2e,#0d1117);
                border:1px solid #3a3f5c;border-radius:12px;
                padding:20px 24px;margin-bottom:20px'>
      <div style='font-size:22px;font-weight:800;color:#e2e8f0'>🔍 Stock Screener</div>
      <div style='font-size:13px;color:#718096;margin-top:4px'>
        Define your criteria → let the system find matching stocks globally →
        click any result to run a full analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Layout: criteria left, results right ──────────────────────────────────
    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown("### Criteria")
        criteria = render_criteria_panel()
        st.markdown("---")
        run_screen = st.button("🔍 Run Screener", type="primary",
                               use_container_width=True, key="run_screener_btn")
        if run_screen:
            st.session_state.pop("screener_results", None)  # clear old results
            st.session_state["screener_criteria"] = criteria
            st.session_state["screener_running"]  = True
            st.rerun()

        # Info box
        st.markdown("""
        <div style='background:#1a1d2e;border:1px solid #2d3748;border-radius:8px;
                     padding:12px;margin-top:12px;font-size:11px;color:#718096'>
          <b style='color:#a0aec0'>How it works:</b><br>
          1. Yahoo Finance screener filters by sector, region, market cap &amp; P/E<br>
          2. Detailed metrics fetched for each candidate<br>
          3. Growth, margin &amp; style filters applied<br>
          4. Stocks scored 0–10 using our quality engine<br>
          5. Results ranked — best opportunities first<br><br>
          ⚠️ Screener may take 1–3 minutes depending on filters.
          Rate limits are handled automatically.
        </div>
        """, unsafe_allow_html=True)

    with right:
        # Running state
        if st.session_state.get("screener_running"):
            criteria = st.session_state.get("screener_criteria", ScreenerCriteria())
            progress_bar = st.progress(0, text="Starting screener…")
            status_text  = st.empty()

            def progress_cb(pct, total, msg):
                progress_bar.progress(min(pct, 100) / 100, text=msg)
                status_text.caption(msg)

            try:
                results = run_screener(criteria, progress_cb=progress_cb)
                st.session_state["screener_results"]  = results
                st.session_state["screener_running"]  = False
                st.rerun()
            except Exception as e:
                st.error(f"Screener error: {e}")
                st.session_state["screener_running"] = False

        elif "screener_results" in st.session_state:
            results = st.session_state["screener_results"]
            if results:
                render_results_chart(results)
                render_results(results)
            else:
                st.info("No results matched your criteria. Try relaxing the filters.")
        else:
            # Empty state
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#4a5568'>
              <div style='font-size:48px'>🔍</div>
              <div style='font-size:16px;font-weight:600;color:#718096;margin-top:12px'>
                Set your criteria and click Run Screener
              </div>
              <div style='font-size:13px;margin-top:8px'>
                Or pick a preset on the left to get started instantly
              </div>
            </div>
            """, unsafe_allow_html=True)
