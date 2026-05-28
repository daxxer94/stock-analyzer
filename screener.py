"""
screener.py — Stock Screener using yfinance EquityQuery + scoring engine.

Strategy:
  1. Build EquityQuery from user criteria (sector, region, market cap, P/E, beta)
  2. Run Yahoo's screener to get up to 100 candidate tickers quickly
  3. Fetch basic metrics for all candidates (batched, rate-limited)
  4. Apply additional client-side filters (stock type, growth, margins, dividend)
  5. Return ranked results ready for display

All network calls use the same _ttl / retry pattern as data.py.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random

# ─── TTL cache (same pattern as other modules) ────────────────────────────────
_SC_CACHE: dict = {}

def _ttl(key, ttl_secs, fn):
    now = time.time()
    if key in _SC_CACHE:
        val, ts = _SC_CACHE[key]
        if now - ts < ttl_secs:
            return val
    result = fn()
    _SC_CACHE[key] = (result, now)
    return result


def _retry(fn, retries=3, base_delay=2.0):
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if any(x in msg for x in ["429","too many","rate","502","503","timeout"]):
                wait = base_delay * (2 ** attempt) + random.uniform(0.3, 1.0)
                time.sleep(wait)
            else:
                raise
    raise last_err


# ─── Valid Yahoo Finance screener values ──────────────────────────────────────

SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Communication Services",
    "Basic Materials", "Real Estate", "Utilities",
]

# Region codes → display labels
REGIONS = {
    "us": "🇺🇸 United States",
    "gb": "🇬🇧 United Kingdom",
    "de": "🇩🇪 Germany",
    "nl": "🇳🇱 Netherlands",
    "fr": "🇫🇷 France",
    "ch": "🇨🇭 Switzerland",
    "se": "🇸🇪 Sweden",
    "no": "🇳🇴 Norway",
    "dk": "🇩🇰 Denmark",
    "it": "🇮🇹 Italy",
    "es": "🇪🇸 Spain",
    "jp": "🇯🇵 Japan",
    "kr": "🇰🇷 South Korea",
    "hk": "🇭🇰 Hong Kong",
    "cn": "🇨🇳 China",
    "ca": "🇨🇦 Canada",
    "au": "🇦🇺 Australia",
    "in": "🇮🇳 India",
    "sg": "🇸🇬 Singapore",
    "br": "🇧🇷 Brazil",
}

# Continent groupings
CONTINENTS = {
    "🌎 North America": ["us", "ca"],
    "🌍 Europe":        ["gb","de","nl","fr","ch","se","no","dk","it","es","be","at","fi","pt"],
    "🌏 Asia/Pacific":  ["jp","kr","hk","cn","au","sg","in","tw","nz"],
    "🌍 All Global":    [],  # empty = no region filter
}

# Market cap buckets (USD)
MARKET_CAP_BUCKETS = {
    "Nano (< $50M)":         (0,       50e6),
    "Micro ($50M–$300M)":    (50e6,    300e6),
    "Small ($300M–$2B)":     (300e6,   2e9),
    "Mid ($2B–$10B)":        (2e9,     10e9),
    "Large ($10B–$200B)":    (10e9,    200e9),
    "Mega (> $200B)":        (200e9,   1e15),
    "Any":                   (0,       1e15),
}

# Industries grouped by sector (subset — most relevant)
INDUSTRIES_BY_SECTOR = {
    "Technology": [
        "Semiconductors","Software—Application","Software—Infrastructure",
        "Information Technology Services","Computer Hardware","Consumer Electronics",
        "Semiconductor Equipment & Materials","Electronic Components","Solar",
    ],
    "Healthcare": [
        "Drug Manufacturers—General","Biotechnology","Medical Devices",
        "Healthcare Plans","Diagnostics & Research","Medical Instruments & Supplies",
        "Drug Manufacturers—Specialty & Generic",
    ],
    "Financial Services": [
        "Banks—Diversified","Banks—Regional","Asset Management",
        "Credit Services","Insurance—Property & Casualty","Insurance—Life",
        "Capital Markets","Financial Data & Stock Exchanges",
    ],
    "Consumer Cyclical": [
        "Auto Manufacturers","Specialty Retail","Apparel Retail","Restaurants",
        "Internet Retail","Leisure","Residential Construction","Lodging",
    ],
    "Consumer Defensive": [
        "Packaged Foods","Beverages—Non-Alcoholic","Household & Personal Products",
        "Grocery Stores","Tobacco","Discount Stores",
    ],
    "Energy": [
        "Oil & Gas Integrated","Oil & Gas E&P","Oil & Gas Midstream",
        "Oil & Gas Refining & Marketing","Oil & Gas Equipment & Services",
    ],
    "Industrials": [
        "Aerospace & Defense","Specialty Industrial Machinery","Airlines",
        "Railroads","Trucking","Engineering & Construction","Waste Management",
    ],
    "Communication Services": [
        "Internet Content & Information","Entertainment","Telecom Services",
        "Electronic Gaming & Multimedia","Advertising Agencies",
    ],
    "Basic Materials": [
        "Chemicals","Specialty Chemicals","Gold","Steel","Copper","Aluminum",
    ],
    "Real Estate": [
        "REIT—Industrial","REIT—Retail","REIT—Specialty","Real Estate Services",
    ],
    "Utilities": [
        "Utilities—Regulated Electric","Utilities—Renewable","Utilities—Diversified",
    ],
}

# ─── Screener criteria dataclass ──────────────────────────────────────────────

class ScreenerCriteria:
    """Holds all user-defined screening criteria."""
    def __init__(self):
        # Geography
        self.regions: list     = []        # list of region codes e.g. ["us","gb"]
        # Sector / Industry
        self.sectors: list     = []        # e.g. ["Technology"]
        self.industries: list  = []        # e.g. ["Semiconductors"]
        # Market cap
        self.min_mktcap: float = 0
        self.max_mktcap: float = 1e15
        # Valuation
        self.max_pe: float     = 999
        self.max_fwd_pe: float = 999
        self.min_peg: float    = 0
        self.max_peg: float    = 999
        self.max_pb: float     = 999
        self.max_ev_ebitda: float = 999
        # Growth
        self.min_rev_growth: float = -1.0
        self.min_eps_growth: float = -1.0
        # Profitability
        self.min_gross_margin: float   = -1.0
        self.min_net_margin: float     = -1.0
        self.min_roe: float            = -1.0
        # Income
        self.min_dividend_yield: float = 0.0
        # Risk
        self.min_beta: float  = 0.0
        self.max_beta: float  = 10.0
        # Stock type filter
        self.stock_types: list = []   # e.g. ["Growth","Value","Dividend"]
        # Score threshold
        self.min_score: float = 0.0
        # Result count
        self.max_results: int = 50


# ─── Core screener logic ──────────────────────────────────────────────────────

def _build_equity_query(criteria: ScreenerCriteria):
    """Build a yfinance EquityQuery from criteria. Returns None if no filters."""
    operands = []

    # Region
    if len(criteria.regions) == 1:
        operands.append(yf.EquityQuery('eq', ['region', criteria.regions[0]]))
    elif len(criteria.regions) > 1:
        region_queries = [yf.EquityQuery('eq', ['region', r]) for r in criteria.regions]
        operands.append(yf.EquityQuery('or', region_queries))

    # Sector
    if len(criteria.sectors) == 1:
        operands.append(yf.EquityQuery('eq', ['sector', criteria.sectors[0]]))
    elif len(criteria.sectors) > 1:
        sector_queries = [yf.EquityQuery('eq', ['sector', s]) for s in criteria.sectors]
        operands.append(yf.EquityQuery('or', sector_queries))

    # Industry
    if len(criteria.industries) == 1:
        operands.append(yf.EquityQuery('eq', ['industry', criteria.industries[0]]))
    elif len(criteria.industries) > 1:
        ind_queries = [yf.EquityQuery('eq', ['industry', i]) for i in criteria.industries]
        operands.append(yf.EquityQuery('or', ind_queries))

    # Market cap
    if criteria.min_mktcap > 0 or criteria.max_mktcap < 1e14:
        lo = max(criteria.min_mktcap, 1e6)
        hi = min(criteria.max_mktcap, 1e14)
        operands.append(yf.EquityQuery('btwn', ['intradaymarketcap', lo, hi]))

    # P/E ratio
    if criteria.max_pe < 999:
        operands.append(yf.EquityQuery('lt', ['peratio.lasttwelvemonths', criteria.max_pe]))

    # Beta
    if criteria.min_beta > 0 or criteria.max_beta < 9:
        operands.append(yf.EquityQuery('btwn', ['beta', criteria.min_beta, criteria.max_beta]))

    if not operands:
        # At minimum filter to equities with market cap > $10M
        operands.append(yf.EquityQuery('gt', ['intradaymarketcap', 10e6]))

    if len(operands) == 1:
        return operands[0]
    return yf.EquityQuery('and', operands)


def _run_yf_screener(query, size: int = 100) -> list:
    """Run Yahoo screener and return list of ticker symbols."""
    try:
        result = _retry(lambda: yf.screen(query, sortField='intradaymarketcap',
                                           sortAsc=False, size=min(size, 100)))
        if result and 'quotes' in result:
            return [q['symbol'] for q in result['quotes'] if q.get('symbol')]
        return []
    except Exception:
        return []


def _fetch_candidate_metrics(tickers: list, progress_cb=None) -> dict:
    """
    Fetch basic metrics for a list of candidate tickers.
    Returns dict: {ticker: {metric: value, ...}}
    """
    results = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        if progress_cb:
            progress_cb(i, total, ticker)
        # Stagger to avoid rate limiting
        if i > 0:
            time.sleep(0.5 if i % 5 != 0 else 1.2)
        try:
            info = _retry(lambda t=ticker: yf.Ticker(t).info)
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            if not price or float(price) <= 0:
                continue
            results[ticker] = {
                "name":             info.get("shortName", ticker),
                "sector":           info.get("sector", ""),
                "industry":         info.get("industry", ""),
                "country":          info.get("country", ""),
                "currency":         info.get("currency", "USD"),
                "current_price":    float(price),
                "market_cap":       info.get("marketCap"),
                "pe_trailing":      info.get("trailingPE"),
                "pe_forward":       info.get("forwardPE"),
                "peg":              info.get("pegRatio"),
                "ev_ebitda":        info.get("enterpriseToEbitda"),
                "price_book":       info.get("priceToBook"),
                "price_sales":      info.get("priceToSalesTrailingTwelveMonths"),
                "gross_margin":     info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin":       info.get("profitMargins"),
                "roe":              info.get("returnOnEquity"),
                "roa":              info.get("returnOnAssets"),
                "revenue_growth":   info.get("revenueGrowth"),
                "earnings_growth":  info.get("earningsGrowth"),
                "debt_equity":      info.get("debtToEquity"),
                "current_ratio":    info.get("currentRatio"),
                "dividend_yield":   info.get("dividendYield"),
                "beta":             info.get("beta"),
                "52w_high":         info.get("fiftyTwoWeekHigh"),
                "52w_low":          info.get("fiftyTwoWeekLow"),
                "target_mean":      info.get("targetMeanPrice"),
                "rec_mean":         info.get("recommendationMean"),
                "num_analysts":     info.get("numberOfAnalystOpinions", 0),
                "free_cashflow":    info.get("freeCashflow"),
                "forward_eps":      info.get("forwardEps"),
                "trailing_eps":     info.get("trailingEps"),
                "short_pct":        info.get("shortPercentOfFloat"),
            }
        except Exception:
            pass
    return results


def _apply_client_filters(candidates: dict, criteria: ScreenerCriteria) -> dict:
    """Apply filters that EquityQuery can't handle (margins, growth, stock type)."""
    filtered = {}
    for ticker, m in candidates.items():
        # Forward P/E
        fpe = m.get("pe_forward")
        if fpe and fpe > 0 and criteria.max_fwd_pe < 999:
            if fpe > criteria.max_fwd_pe:
                continue

        # PEG
        peg = m.get("peg")
        if peg and peg > 0 and criteria.max_peg < 999:
            if peg > criteria.max_peg:
                continue

        # P/B
        pb = m.get("price_book")
        if pb and criteria.max_pb < 999:
            if pb > criteria.max_pb:
                continue

        # EV/EBITDA
        ev_eb = m.get("ev_ebitda")
        if ev_eb and criteria.max_ev_ebitda < 999:
            if ev_eb > criteria.max_ev_ebitda:
                continue

        # Revenue growth
        rg = m.get("revenue_growth") or 0
        if rg < criteria.min_rev_growth:
            continue

        # EPS growth
        eg = m.get("earnings_growth") or 0
        if eg < criteria.min_eps_growth:
            continue

        # Gross margin
        gm = m.get("gross_margin") or 0
        if gm < criteria.min_gross_margin:
            continue

        # Net margin
        nm = m.get("net_margin") or 0
        if nm < criteria.min_net_margin:
            continue

        # ROE
        roe = m.get("roe") or 0
        if roe < criteria.min_roe:
            continue

        # Dividend yield
        dy = m.get("dividend_yield") or 0
        if dy < criteria.min_dividend_yield:
            continue

        # Stock type filter
        if criteria.stock_types:
            st = _classify_stock_type(m)
            if not any(t.lower() in st.lower() for t in criteria.stock_types):
                continue

        filtered[ticker] = m
    return filtered


def _classify_stock_type(m: dict) -> str:
    """Quick stock type classification from metrics dict."""
    pe     = m.get("pe_trailing") or 0
    fpe    = m.get("pe_forward") or 0
    peg    = m.get("peg") or 0
    rg     = m.get("revenue_growth") or 0
    eg     = m.get("earnings_growth") or 0
    dy     = m.get("dividend_yield") or 0
    pb     = m.get("price_book") or 0
    nm     = m.get("net_margin") or 0
    mktcap = m.get("market_cap") or 0

    if dy > 0.035 and pe < 35:
        return "Dividend"
    if rg > 0.20 and (pe > 40 or fpe > 30):
        return "Growth"
    if 0 < peg < 1.5 and rg > 0.08 and pe < 35:
        return "GARP"
    if pe and 0 < pe < 12 and pb and 0 < pb < 1.5:
        return "Deep Value"
    if pe and 0 < pe < 18 and rg < 0.12:
        return "Value"
    if nm and nm < 0:
        return "Speculative"
    if mktcap < 2e9 and rg > 0.10:
        return "Small-Cap Growth"
    return "Blend"


def _score_candidates(candidates: dict) -> list:
    """
    Apply a simplified scoring to each candidate (no full DCF — too slow for 50+ stocks).
    Returns sorted list of (ticker, metrics, score, stock_type, upside_pct) tuples.
    """
    scored = []
    for ticker, m in candidates.items():
        score = 5.0
        pts = 0; max_pts = 0

        def add(p, mx):
            nonlocal pts, max_pts
            pts += p; max_pts += mx

        # Growth
        rg = m.get("revenue_growth") or 0
        eg = m.get("earnings_growth") or 0
        if rg > 0.20: add(2, 2)
        elif rg > 0.08: add(1.5, 2)
        elif rg > 0: add(1, 2)
        else: add(0, 2)

        if eg > 0.20: add(2, 2)
        elif eg > 0.08: add(1.5, 2)
        elif eg > 0: add(1, 2)
        else: add(0, 2)

        # Margins
        nm = m.get("net_margin") or 0
        gm = m.get("gross_margin") or 0
        if nm > 0.20: add(2, 2)
        elif nm > 0.08: add(1.5, 2)
        elif nm > 0: add(1, 2)
        else: add(0, 2)

        if gm > 0.50: add(2, 2)
        elif gm > 0.30: add(1.5, 2)
        elif gm > 0.15: add(1, 2)
        else: add(0.5, 2)

        # Valuation
        pe  = m.get("pe_trailing") or 0
        fpe = m.get("pe_forward") or 0
        peg = m.get("peg") or 0
        if pe and 0 < pe < 15: add(2, 2)
        elif pe and 0 < pe < 25: add(1.5, 2)
        elif pe and pe < 40: add(1, 2)
        else: add(0, 2)

        if peg and 0 < peg < 1: add(2, 2)
        elif peg and 0 < peg < 2: add(1.5, 2)
        else: add(1, 2)

        # Analyst upside
        cp = m.get("current_price") or 0
        tp = m.get("target_mean") or 0
        upside = (tp - cp) / cp if (tp and cp > 0) else 0
        if upside > 0.30: add(2, 2)
        elif upside > 0.15: add(1.5, 2)
        elif upside > 0: add(1, 2)
        else: add(0.5, 2)

        # ROE
        roe = m.get("roe") or 0
        if roe > 0.25: add(2, 2)
        elif roe > 0.12: add(1.5, 2)
        elif roe > 0: add(1, 2)
        else: add(0, 2)

        # D/E
        de = (m.get("debt_equity") or 0) / 100
        if de < 0.3: add(2, 2)
        elif de < 0.8: add(1.5, 2)
        elif de < 1.5: add(1, 2)
        else: add(0, 2)

        # Analyst consensus (1=Strong Buy, 5=Strong Sell)
        rec = m.get("rec_mean") or 3
        if rec <= 1.8: add(2, 2)
        elif rec <= 2.5: add(1.5, 2)
        elif rec <= 3.0: add(1, 2)
        else: add(0, 2)

        # FCF
        fcf = m.get("free_cashflow") or 0
        mktcap = m.get("market_cap") or 1
        if fcf > 0 and mktcap > 0:
            fcf_yield = fcf / mktcap
            if fcf_yield > 0.05: add(2, 2)
            elif fcf_yield > 0.02: add(1.5, 2)
            elif fcf_yield > 0: add(1, 2)
            else: add(0, 2)

        if max_pts > 0:
            score = (pts / max_pts) * 10

        stock_type = _classify_stock_type(m)
        scored.append((ticker, m, round(score, 2), stock_type, upside))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def run_screener(criteria: ScreenerCriteria, progress_cb=None) -> list:
    """
    Main screener entry point.
    Returns list of (ticker, metrics, score, stock_type, upside) tuples.
    """
    # Step 1: Build query
    query = _build_equity_query(criteria)

    # Step 2: Get candidates from Yahoo screener
    if progress_cb:
        progress_cb(0, 100, "Querying Yahoo Finance screener…")

    candidates_raw = []
    try:
        candidates_raw = _run_yf_screener(query, size=min(criteria.max_results * 3, 100))
    except Exception:
        pass

    if not candidates_raw:
        return []

    # Limit to reasonable number for fetching
    fetch_limit = min(len(candidates_raw), criteria.max_results * 2, 80)
    candidates_raw = candidates_raw[:fetch_limit]

    if progress_cb:
        progress_cb(10, 100, f"Found {len(candidates_raw)} candidates, fetching metrics…")

    # Step 3: Fetch metrics
    def _prog(i, total, t):
        if progress_cb:
            pct = 10 + int((i / max(total, 1)) * 70)
            progress_cb(pct, 100, f"Fetching {t} ({i+1}/{total})…")

    metrics = _fetch_candidate_metrics(candidates_raw, progress_cb=_prog)

    if progress_cb:
        progress_cb(80, 100, "Applying filters and scoring…")

    # Step 4: Apply client-side filters
    filtered = _apply_client_filters(metrics, criteria)

    # Step 5: Score and sort
    scored = _score_candidates(filtered)

    # Step 6: Apply minimum score filter
    if criteria.min_score > 0:
        scored = [s for s in scored if s[2] >= criteria.min_score]

    # Step 7: Limit results
    result = scored[:criteria.max_results]

    if progress_cb:
        progress_cb(100, 100, f"Done — {len(result)} stocks found.")

    return result


# ─── Preset screens ───────────────────────────────────────────────────────────

def get_preset_criteria(preset_name: str) -> ScreenerCriteria:
    """Return pre-configured criteria for common screen types."""
    c = ScreenerCriteria()

    if preset_name == "High Growth Tech":
        c.sectors       = ["Technology"]
        c.min_rev_growth = 0.15
        c.max_fwd_pe    = 60
        c.stock_types   = ["Growth", "GARP"]
        c.min_score     = 6.0

    elif preset_name == "Dividend Income":
        c.min_dividend_yield = 0.025
        c.max_pe        = 25
        c.max_beta      = 1.2
        c.stock_types   = ["Dividend", "Value"]
        c.min_score     = 5.0

    elif preset_name == "European Value":
        c.regions       = ["gb","de","nl","fr","ch","se","no"]
        c.max_pe        = 18
        c.max_pb        = 3.0
        c.stock_types   = ["Value", "Deep Value", "GARP"]
        c.min_score     = 5.5

    elif preset_name == "Asian Growth":
        c.regions       = ["jp","kr","hk","sg","tw"]
        c.min_rev_growth = 0.10
        c.stock_types   = ["Growth", "GARP", "Small-Cap Growth"]
        c.min_score     = 5.5

    elif preset_name == "Quality Compounders":
        c.min_roe       = 0.15
        c.min_net_margin = 0.10
        c.min_rev_growth = 0.05
        c.max_pb        = 8.0
        c.min_score     = 6.5

    elif preset_name == "Undervalued Gems":
        c.max_peg       = 1.2
        c.max_pe        = 20
        c.min_roe       = 0.10
        c.min_score     = 6.0

    elif preset_name == "Healthcare Innovation":
        c.sectors       = ["Healthcare"]
        c.min_rev_growth = 0.08
        c.min_score     = 5.5

    elif preset_name == "High Conviction (Global)":
        c.min_rev_growth = 0.10
        c.min_net_margin = 0.08
        c.min_roe       = 0.12
        c.max_pe        = 40
        c.min_score     = 7.0
        c.max_results   = 20

    return c
