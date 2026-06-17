"""
landing.py — Home screen for the Stock Analyzer.

Shows:
  1. Animated ticker tape (horizontal marquee of AI supply chain + major stocks)
  2. 20 major global indices with live prices and change
  3. News grouped by outlet (5 articles per source)
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
try:
    from search import TICKER_NAMES as _NAMES
except ImportError:
    _NAMES = {}
import pandas as pd
import numpy as np
import time
import datetime

# ─── Major indices — 20 across US / Europe / Asia ─────────────────────────

INDICES = {
    "United States": [
        ("S&P 500",       "^GSPC"),
        ("Nasdaq 100",    "^NDX"),
        ("Dow Jones",     "^DJI"),
        ("Russell 2000",  "^RUT"),
        ("S&P 600",       "^SML"),
        ("NYSE Comp.",    "^NYA"),
        ("VIX",           "^VIX"),
    ],
    "Europe": [
        ("FTSE 100",      "^FTSE"),
        ("DAX 40",        "^GDAXI"),
        ("CAC 40",        "^FCHI"),
        ("AEX 25",        "^AEX"),
        ("Euro Stoxx 50", "^STOXX50E"),
        ("SMI",           "^SSMI"),
        ("IBEX 35",       "^IBEX"),
    ],
    "Asia-Pacific": [
        ("Nikkei 225",    "^N225"),
        ("Hang Seng",     "^HSI"),
        ("Kospi",         "^KS11"),
        ("ASX 200",       "^AXJO"),
        ("Sensex",        "^BSESN"),
        ("Shanghai Comp.","000001.SS"),
    ],
}

# Ticker tape stocks (AI supply chain + blue chips)
TAPE_TICKERS = [
    "NVDA","MSFT","AAPL","GOOG","META","AMD","INTC","TSM","ASML","CAMT",
    "AEHR","FORM","AAOI","VRT","RMBS","CRDO","SMCI","ENTG","SKYT","ICHR",
    "AVGO","QCOM","MU","AMAT","LRCX","KLAC","MRVL","NXPI","ON","TXN",
]


@st.cache_data(ttl=180, show_spinner=False)
def fetch_index_data() -> dict:
    """Fetch all 20 index quotes. Cached 3 minutes."""
    result = {}
    all_tickers = [sym for region in INDICES.values() for _, sym in region]
    try:
        data = yf.download(
            tickers=all_tickers,
            period="2d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        closes = data.get("Close", pd.DataFrame())
        for sym in all_tickers:
            try:
                col = closes[sym] if sym in closes.columns else closes.get(sym)
                if col is not None and not col.dropna().empty:
                    vals  = col.dropna()
                    price = float(vals.iloc[-1])
                    prev  = float(vals.iloc[-2]) if len(vals) >= 2 else price
                    chg   = ((price - prev) / prev) * 100 if prev else 0
                    result[sym] = {"price": price, "change_pct": chg}
            except Exception:
                pass
    except Exception:
        pass
    return result


@st.cache_data(ttl=120, show_spinner=False)
def fetch_tape_quotes() -> list:
    """Fetch price + change for ticker tape. Cached 2 minutes."""
    out = []
    try:
        data = yf.download(TAPE_TICKERS, period="2d", auto_adjust=True,
                           progress=False, threads=True)
        closes = data.get("Close", pd.DataFrame())
        for t in TAPE_TICKERS:
            try:
                col  = closes[t] if t in closes.columns else closes.get(t)
                vals = col.dropna()
                if len(vals) >= 1:
                    p   = float(vals.iloc[-1])
                    prev= float(vals.iloc[-2]) if len(vals) >= 2 else p
                    chg = (p - prev) / prev * 100 if prev else 0
                    out.append({"ticker": t, "price": p, "change": chg})
            except Exception:
                pass
    except Exception:
        pass
    return out


def render_ticker_tape(quotes: list):
    """Render an infinite-scroll horizontal ticker tape using HTML/CSS animation."""
    if not quotes:
        return

    def item_html(q):
        c    = "#10b981" if q["change"] >= 0 else "#ef4444"
        arrow = "▲" if q["change"] >= 0 else "▼"
        t    = q["ticker"]
        name = _NAMES.get(t, "")
        title_attr = f'title="{name}"' if name else ""
        return (
            f"<span class='tape-item' {title_attr}>"
            f"<span class='tape-sym'>{t}</span>"
            f"<span class='tape-price'>${q['price']:,.2f}</span>"
            f"<span class='tape-chg' style='color:{c}'>{arrow}{abs(q['change']):.2f}%</span>"
            f"</span>"
        )

    items_html = "".join(item_html(q) for q in quotes)
    # Duplicate for seamless loop
    tape_html = items_html * 3

    components.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600&display=swap');
  .tape-wrap {{
    width: 100%;
    overflow: hidden;
    background: #111318;
    border-top: 1px solid #1e2433;
    border-bottom: 1px solid #1e2433;
    padding: 8px 0;
    margin-bottom: 4px;
  }}
  .tape-track {{
    display: inline-flex;
    gap: 0;
    animation: scroll-left 60s linear infinite;
    white-space: nowrap;
  }}
  .tape-wrap:hover .tape-track {{ animation-play-state: paused; }}
  @keyframes scroll-left {{
    0%   {{ transform: translateX(0); }}
    100% {{ transform: translateX(-33.333%); }}
  }}
  .tape-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0 20px;
    border-right: 1px solid #1e2433;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
  }}
  .tape-sym   {{ font-weight: 700; color: #e2e8f0; letter-spacing: 0.03em; }}
  .tape-price {{ font-weight: 500; color: #94a3b8; }}
  .tape-chg   {{ font-weight: 600; font-size: 11px; }}
  @media (max-width: 768px) {{
    .tape-item {{ padding: 0 12px; font-size: 11px; gap: 4px; }}
    .tape-chg  {{ font-size: 10px; }}
    .tape-track {{ animation-duration: 40s; }}
  }}
</style>
<div class="tape-wrap">
  <div class="tape-track">{tape_html}</div>
</div>
""", height=42, scrolling=False)


def render_indices(index_data: dict):
    """
    Render the 20 major indices as a single responsive CSS grid.
    Uses components.html so the grid reflows correctly on mobile
    (Streamlit columns don't stack gracefully for dense data).
    """
    cards = []
    for region, indices in INDICES.items():
        cards.append(
            f"<div class='region-label'>{region}</div>")
        for name, sym in indices:
            q = index_data.get(sym, {})
            price = q.get("price")
            chg   = q.get("change_pct")
            if price is None:
                price_str, chg_str, chg_color = "—", "", "#475569"
            else:
                chg_color = "#10b981" if (chg or 0) >= 0 else "#ef4444"
                arrow = "&#9650;" if (chg or 0) >= 0 else "&#9660;"
                price_str = f"{price:,.0f}" if price >= 100 else f"{price:,.2f}"
                chg_str = (f"<span style='color:{chg_color}'>{arrow} {abs(chg):.2f}%</span>"
                           if chg is not None else "")
            cards.append(
                f"<div class='idx-card'>"
                f"<div class='idx-name'>{name}</div>"
                f"<div class='idx-price'>{price_str}</div>"
                f"<div class='idx-chg'>{chg_str}</div>"
                f"</div>")

    grid_html = "".join(cards)
    components.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');
  * {{ box-sizing: border-box; }}
  .idx-wrap {{
    font-family: 'Inter', sans-serif;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }}
  .region-label {{
    grid-column: 1 / -1;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #3b82f6;
    margin: 10px 0 2px;
  }}
  .idx-card {{
    background: #14161c;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 10px 12px;
    min-width: 0;
  }}
  .idx-name  {{ font-size: 11px; color: #8b95a8; font-weight: 500;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .idx-price {{ font-size: 16px; color: #f1f5f9; font-weight: 700;
                letter-spacing: -0.02em; margin: 2px 0; }}
  .idx-chg   {{ font-size: 11px; font-weight: 600; }}

  @media (max-width: 768px) {{
    .idx-wrap {{ grid-template-columns: repeat(2, 1fr); gap: 6px; }}
    .idx-price {{ font-size: 15px; }}
  }}
  @media (max-width: 380px) {{
    .idx-wrap {{ grid-template-columns: repeat(2, 1fr); }}
    .idx-card {{ padding: 8px 10px; }}
  }}
</style>
<div class="idx-wrap">{grid_html}</div>
""", height=560, scrolling=False)


def render_grouped_news(news_items: list, ticker: str, info: dict):
    """
    Group news articles by publisher and show up to 5 per outlet.
    Renders after the standard news links.
    """
    if not news_items:
        return

    import datetime

    now = datetime.datetime.now()

    # Group by publisher
    by_publisher: dict = {}
    for item in news_items:
        pub = item.get("publisher", "Unknown")
        by_publisher.setdefault(pub, []).append(item)

    # Sort publishers by most recent article
    def latest_ts(articles):
        ts = [a.get("providerPublishTime", 0) for a in articles if a.get("providerPublishTime")]
        return max(ts) if ts else 0

    sorted_pubs = sorted(by_publisher.items(), key=lambda x: latest_ts(x[1]), reverse=True)

    st.markdown(
        "<p style='font-size:11px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.08em;color:#475569;margin:16px 0 10px'>News by Outlet</p>",
        unsafe_allow_html=True)

    for pub, articles in sorted_pubs:
        # Sort articles within each outlet newest first, take up to 5
        articles_sorted = sorted(articles,
                                  key=lambda a: a.get("providerPublishTime", 0),
                                  reverse=True)[:5]

        with st.expander(f"{pub}  ({len(articles_sorted)} articles)", expanded=False):
            for article in articles_sorted:
                title = article.get("title", "")
                link  = article.get("link", "")
                ts    = article.get("providerPublishTime", 0)
                if not title or not link:
                    continue

                date_str = ""
                age_badge = ""
                if ts:
                    try:
                        dt = datetime.datetime.fromtimestamp(ts)
                        days_ago = (now - dt).days
                        hrs_ago  = int((now - dt).total_seconds() / 3600)
                        date_str = dt.strftime("%d %b %Y")
                        if days_ago == 0:
                            age_str = f"{hrs_ago}h ago"
                            age_color = "#3b82f6"
                        elif days_ago <= 3:
                            age_str = f"{days_ago}d ago"
                            age_color = "#f59e0b"
                        else:
                            age_str = date_str
                            age_color = "#475569"
                        age_badge = (
                            f"<span style='font-size:10px;font-weight:600;"
                            f"color:{age_color}'>{age_str}</span>"
                        )
                    except Exception:
                        pass

                t_lower = title.lower()
                is_earnings = any(w in t_lower for w in
                                  ["earnings","revenue","eps","profit","loss","quarterly","guidance"])
                left_border = "3px solid #f59e0b" if is_earnings else "3px solid #1e2433"

                st.markdown(
                    f"<div style='border-left:{left_border};padding:8px 12px;"
                    f"margin:4px 0;background:#111318;border-radius:0 8px 8px 0'>"
                    f"<a href='{link}' target='_blank' style='color:#93c5fd;"
                    f"font-size:13px;font-weight:500;text-decoration:none;"
                    f"line-height:1.4;display:block'>{title}</a>"
                    f"<div style='margin-top:4px'>{age_badge}"
                    f"{'  ·  <span style=\"font-size:10px;color:#f59e0b\">Earnings</span>' if is_earnings else ''}"
                    f"</div></div>",
                    unsafe_allow_html=True)


# ─── Market Movers — winners, losers, most active ─────────────────────────

# A representative universe per region to compute daily movers from.
MOVERS_UNIVERSE = {
    "US": ["AAPL","MSFT","NVDA","AMZN","GOOG","META","TSLA","AMD","INTC","NFLX",
           "JPM","BAC","XOM","CVX","PFE","KO","DIS","BA","WMT","CRM",
           "AVGO","ORCL","ADBE","PYPL","UBER","COIN","PLTR","SOFI","F","GM"],
    "Europe": ["ASML.AS","SAP.DE","SHEL.L","NESN.SW","NOVN.SW","MC.PA","OR.PA",
               "SIE.DE","AIR.PA","BNP.PA","AZN.L","HSBA.L","ADYEN.AS","BAYN.DE",
               "ABI.BR","INGA.AS","BP.L","VOW3.DE","ALV.DE","DTE.DE"],
    "Asia": ["7203.T","6758.T","9984.T","005930.KS","0700.HK","9988.HK",
             "1299.HK","6861.T","000660.KS","8306.T","9433.T","BABA","TCEHY"],
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_movers(region: str = "US") -> dict:
    """
    Compute the day's biggest winners, losers, and most active stocks
    from a representative universe. Cached 5 minutes.
    """
    universe = MOVERS_UNIVERSE.get(region, MOVERS_UNIVERSE["US"])
    rows = []
    try:
        data = yf.download(universe, period="2d", auto_adjust=True,
                           progress=False, threads=True)
        closes = data.get("Close", pd.DataFrame())
        volumes = data.get("Volume", pd.DataFrame())
        for t in universe:
            try:
                col = closes[t] if t in closes.columns else None
                if col is None: continue
                vals = col.dropna()
                if len(vals) < 2: continue
                price = float(vals.iloc[-1]); prev = float(vals.iloc[-2])
                chg = (price - prev) / prev * 100 if prev else 0
                vol = 0
                try:
                    vcol = volumes[t] if t in volumes.columns else None
                    if vcol is not None:
                        vol = float(vcol.dropna().iloc[-1])
                except Exception:
                    pass
                rows.append({"ticker": t, "price": price, "change": chg,
                             "volume": vol, "dollar_vol": price * vol})
            except Exception:
                pass
    except Exception:
        pass

    if not rows:
        return {}
    winners = sorted(rows, key=lambda x: -x["change"])[:5]
    losers  = sorted(rows, key=lambda x: x["change"])[:5]
    active  = sorted(rows, key=lambda x: -x["dollar_vol"])[:5]
    return {"winners": winners, "losers": losers, "active": active}


def render_market_movers():
    """Render winners / losers / most active with a region selector."""
    st.markdown(
        "<p style='font-size:11px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.08em;color:#475569;margin:20px 0 8px'>Market Movers</p>",
        unsafe_allow_html=True)

    region = st.radio("Region", ["US", "Europe", "Asia"],
                      horizontal=True, key="movers_region",
                      label_visibility="collapsed")

    with st.spinner("Loading movers…"):
        movers = fetch_market_movers(region)

    if not movers:
        st.caption("Market movers data temporarily unavailable.")
        return

    c1, c2, c3 = st.columns(3, gap="medium")

    def _movers_card(col, title, items, value_key, accent):
        with col:
            st.markdown(
                f"<p style='font-size:11px;font-weight:700;color:{accent};"
                f"margin:0 0 6px'>{title}</p>", unsafe_allow_html=True)
            for it in items:
                if value_key == "change":
                    chg = it["change"]
                    vc  = "#10b981" if chg >= 0 else "#ef4444"
                    arrow = "▲" if chg >= 0 else "▼"
                    right = f"<span style='color:{vc};font-weight:600'>{arrow} {abs(chg):.1f}%</span>"
                else:
                    dv = it["dollar_vol"]
                    dv_str = (f"${dv/1e9:.1f}B" if dv >= 1e9 else
                              f"${dv/1e6:.0f}M" if dv >= 1e6 else f"${dv:,.0f}")
                    right = f"<span style='color:#8b95a8;font-weight:600'>{dv_str}</span>"
                name_m = _NAMES.get(it['ticker'], '')
                title_m = f'title="{name_m}"' if name_m else ''
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid #1a1f2e;font-size:12px'>"
                    f"<span style='color:#e2e8f0;font-weight:600' {title_m}>{it['ticker']}</span>"
                    f"{right}</div>", unsafe_allow_html=True)

    _movers_card(c1, "Top Gainers",  movers["winners"], "change", "#10b981")
    _movers_card(c2, "Top Losers",   movers["losers"],  "change", "#ef4444")
    _movers_card(c3, "Most Active",  movers["active"],  "dollar_vol", "#3b82f6")
