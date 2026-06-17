"""
sec_data.py — Company intelligence module.

Sources:
  - SEC EDGAR (US stocks): filing links, 10-K metadata via public API
  - International equivalents: links to relevant regulatory bodies
  - Yahoo Finance news via yfinance
  - Algorithmic SWOT from scoring data
  - Customers/suppliers extracted from business descriptions + filings
"""
import urllib.request
import urllib.parse
import json
import re
import time
import streamlit as st
import yfinance as yf


# ─── Simple TTL cache ─────────────────────────────────────────────────────────
import time as _time
_MOD_CACHE: dict = {}

def _ttl(key, ttl_secs, fn):
    now = _time.time()
    if key in _MOD_CACHE:
        val, ts = _MOD_CACHE[key]
        if now - ts < ttl_secs:
            return val
    result = fn()
    _MOD_CACHE[key] = (result, now)
    return result


# ─── Regulatory body URLs by country ─────────────────────────────────────────
REGULATORY_LINKS = {
    "United States": {
        "name": "SEC EDGAR",
        "search_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={name}&CIK=&type=10-K&dateb=&owner=include&count=5",
        "base": "https://www.sec.gov",
        "icon": "🇺🇸",
    },
    "United Kingdom": {
        "name": "Companies House / FCA",
        "search_url": "https://find-and-update.company-information.service.gov.uk/search?q={name}",
        "base": "https://find-and-update.company-information.service.gov.uk",
        "icon": "🇬🇧",
    },
    "Germany": {
        "name": "Bundesanzeiger",
        "search_url": "https://www.bundesanzeiger.de/pub/de/suchergebnis?0-2.-top~content~contentheader~searchform&fulltext={name}",
        "base": "https://www.bundesanzeiger.de",
        "icon": "🇩🇪",
    },
    "Netherlands": {
        "name": "AFM / KvK",
        "search_url": "https://www.afm.nl/en/professionals/registers/search?q={name}",
        "base": "https://www.afm.nl",
        "icon": "🇳🇱",
    },
    "France": {
        "name": "AMF",
        "search_url": "https://www.amf-france.org/en/recherche?q={name}",
        "base": "https://www.amf-france.org",
        "icon": "🇫🇷",
    },
    "Japan": {
        "name": "EDINET (FSA)",
        "search_url": "https://disclosure2.edinet-fsa.go.jp/WZEK0040.aspx?S1={name}",
        "base": "https://disclosure2.edinet-fsa.go.jp",
        "icon": "🇯🇵",
    },
    "South Korea": {
        "name": "DART (FSS)",
        "search_url": "https://dart.fss.or.kr/dsab001/search.ax?textCrpNm={name}",
        "base": "https://dart.fss.or.kr",
        "icon": "🇰🇷",
    },
    "Hong Kong": {
        "name": "HKEXnews",
        "search_url": "https://www.hkexnews.hk/listedco/listconews/advancedsearch/search_active_main.aspx",
        "base": "https://www.hkexnews.hk",
        "icon": "🇭🇰",
    },
    "China": {
        "name": "CNINFO",
        "search_url": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=data/search&keywords={name}",
        "base": "https://www.cninfo.com.cn",
        "icon": "🇨🇳",
    },
    "Canada": {
        "name": "SEDAR+",
        "search_url": "https://www.sedarplus.ca/csa-party/party/search.html?search={name}",
        "base": "https://www.sedarplus.ca",
        "icon": "🇨🇦",
    },
    "Switzerland": {
        "name": "SIX / FINMA",
        "search_url": "https://www.six-group.com/exchanges/shares/search.html?searchString={name}",
        "base": "https://www.six-group.com",
        "icon": "🇨🇭",
    },
    "Sweden": {
        "name": "Finansinspektionen",
        "search_url": "https://fi.se/en/search/?query={name}",
        "base": "https://fi.se",
        "icon": "🇸🇪",
    },
    "Australia": {
        "name": "ASX / ASIC",
        "search_url": "https://www.asx.com.au/asx/research/companyInfo.do?by=asxCode&allinfo=false&asxCode={name}",
        "base": "https://www.asx.com.au",
        "icon": "🇦🇺",
    },
}


# ─── SEC EDGAR ────────────────────────────────────────────────────────────────

def _fetch_sec_cik_impl(ticker: str) -> str | None:
    """Look up SEC CIK number for a US ticker via EDGAR company search."""
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
        req = urllib.request.Request(url, headers={
            "User-Agent": "StockAnalyzer research@example.com",
            "Accept": "application/json",
        })
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("entity_id")
    except Exception:
        pass

    # Try the company tickers JSON (EDGAR provides this publicly)
    try:
        req2 = urllib.request.Request(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "StockAnalyzer research@example.com"}
        )
        companies = json.loads(urllib.request.urlopen(req2, timeout=8).read())
        t_upper = ticker.upper()
        for _, info in companies.items():
            if info.get("ticker", "").upper() == t_upper:
                return str(info["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


def _fetch_sec_filings_impl(cik: str, form_type: str = "10-K", count: int = 5) -> list:
    """Fetch recent SEC filings for a CIK."""
    if not cik:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        req = urllib.request.Request(url, headers={
            "User-Agent": "StockAnalyzer research@example.com"
        })
        data = json.loads(urllib.request.urlopen(req, timeout=8).read())
        filings = data.get("filings", {}).get("recent", {})
        forms   = filings.get("form",         [])
        dates   = filings.get("filingDate",   [])
        acc_nos = filings.get("accessionNumber", [])
        names   = filings.get("primaryDocument", [])
        results = []
        for i, form in enumerate(forms):
            if form.upper() in [form_type.upper(), form_type.upper() + "/A"]:
                acc = acc_nos[i].replace("-", "")
                doc = names[i] if i < len(names) else ""
                url_filing = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
                results.append({
                    "form":   form,
                    "date":   dates[i] if i < len(dates) else "",
                    "url":    url_filing,
                    "index":  f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count={count}",
                })
                if len(results) >= count:
                    break
        return results
    except Exception:
        return []


# ─── News ─────────────────────────────────────────────────────────────────────

def _parse_news_date(item: dict, content: dict) -> int:
    """Extract a unix timestamp from the various date formats yfinance returns."""
    import datetime as _dt
    # New format: content.pubDate is ISO 8601 string
    for src in (content, item):
        pub_date = src.get("pubDate") or src.get("displayTime") or src.get("providerPublishTime")
        if pub_date:
            if isinstance(pub_date, (int, float)) and pub_date > 0:
                return int(pub_date)
            if isinstance(pub_date, str):
                try:
                    # ISO 8601 e.g. "2026-06-10T14:30:00Z"
                    dt = _dt.datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    pass
    return 0


def _fetch_news_impl(ticker: str) -> list:
    """
    Fetch recent news from Yahoo Finance.
    yfinance 1.4.0 changed the structure: each item now nests data under
    item['content'] with canonicalUrl / clickThroughUrl for the link.
    This handles both old (flat) and new (nested) formats.
    """
    results = []
    try:
        news = yf.Ticker(ticker).get_news(count=20) or []
    except Exception:
        news = []

    for item in news:
        if not isinstance(item, dict):
            continue

        # New nested format
        content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}

        # Title
        title = (item.get("title") or content.get("title") or "").strip()

        # Link — try every known location
        link = (
            item.get("link")
            or content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else ""
        )
        if not link:
            ctu = content.get("clickThroughUrl")
            if isinstance(ctu, dict):
                link = ctu.get("url", "")
        if not link:
            cu = content.get("canonicalUrl")
            if isinstance(cu, dict):
                link = cu.get("url", "")
        if not link:
            link = item.get("link", "")

        # Publisher
        pub = item.get("publisher", "")
        if not pub:
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                pub = provider.get("displayName", "")

        ts = _parse_news_date(item, content)

        if title:
            # If still no link, build a Yahoo Finance search fallback so it's never dead
            if not link:
                import urllib.parse as _up
                link = f"https://finance.yahoo.com/quote/{_up.quote(ticker)}/news/"
            results.append({
                "title":               title,
                "link":                link,
                "publisher":           pub or "Yahoo Finance",
                "timestamp":           ts,
                "providerPublishTime": ts,   # both keys for compatibility
            })

    return results[:15]


def build_news_search_links(ticker: str, company_name: str) -> list:
    """Build links to free news search pages for a stock."""
    name_enc = urllib.parse.quote(company_name or ticker)
    tick_enc = urllib.parse.quote(ticker)
    return [
        {
            "source": "Reuters",
            "url": f"https://www.reuters.com/search/news?blob={name_enc}",
            "icon": "📰",
        },
        {
            "source": "Yahoo Finance News",
            "url": f"https://finance.yahoo.com/quote/{tick_enc}/news/",
            "icon": "💹",
        },
        {
            "source": "Seeking Alpha",
            "url": f"https://seekingalpha.com/symbol/{tick_enc}/news",
            "icon": "📈",
        },
        {
            "source": "MarketWatch",
            "url": f"https://www.marketwatch.com/investing/stock/{tick_enc}",
            "icon": "📊",
        },
        {
            "source": "Investopedia",
            "url": f"https://www.investopedia.com/search#q={name_enc}",
            "icon": "📚",
        },
        {
            "source": "Google Finance",
            "url": f"https://www.google.com/finance/quote/{tick_enc}",
            "icon": "🔍",
        },
    ]


# ─── Customers & Suppliers ───────────────────────────────────────────────────

CUSTOMER_KEYWORDS = [
    r"(?:our\s+)?(?:largest?|major|key|top|primary|principal|significant)\s+customers?",
    r"customers?\s+(?:include|such as|like)",
    r"(?:sold|provided|delivered)\s+to\s+([A-Z][A-Za-z\s&,\.]+)",
    r"customer\s+concentration",
    r"no\s+single\s+customer",
]
SUPPLIER_KEYWORDS = [
    r"(?:our\s+)?(?:largest?|major|key|top|primary|principal)\s+suppliers?",
    r"suppliers?\s+(?:include|such as)",
    r"(?:sourced?|purchased?|procured?)\s+from\s+([A-Z][A-Za-z\s&,\.]+)",
    r"supply\s+chain",
    r"sole\s+source\s+supplier",
]


def extract_customers_suppliers(business_summary: str) -> dict:
    """
    Extract customer and supplier mentions from a business description.
    Returns {customers: [...], suppliers: [...], customer_notes: str, supplier_notes: str}
    """
    if not business_summary:
        return {"customers": [], "suppliers": [], "customer_notes": "", "supplier_notes": ""}

    text = business_summary

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cust_sentences = []
    supp_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(re.search(kw, sent_lower) for kw in ["customer", "client", "buyer", "user", "consumer"]):
            cust_sentences.append(sent.strip())
        if any(re.search(kw, sent_lower) for kw in ["supplier", "vendor", "manufacturer", "provider", "source"]):
            supp_sentences.append(sent.strip())

    # Try to extract named entities (simple capitalized word groups)
    def extract_names(sentences_list):
        names = []
        for s in sentences_list:
            # Find capitalized proper noun sequences
            matches = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}(?:\s+(?:Inc|Corp|Ltd|LLC|Group|Co|PLC|AG|NV|SA|SE)\.?)?\b', s)
            for m in matches:
                if len(m) > 3 and m not in ["We", "Our", "The", "This", "These", "Such"]:
                    names.append(m)
        return list(dict.fromkeys(names))[:8]

    return {
        "customers":      extract_names(cust_sentences),
        "suppliers":      extract_names(supp_sentences),
        "customer_notes": " ".join(cust_sentences[:3]),
        "supplier_notes": " ".join(supp_sentences[:3]),
    }


# ─── SWOT Analysis ───────────────────────────────────────────────────────────

def generate_swot(info: dict, scoring: dict, signals: dict,
                  sentiment: dict, fund: dict, comparison: list) -> dict:
    """
    Generate a dynamic SWOT analysis based on all available data.
    Returns {strengths, weaknesses, opportunities, threats} as lists of strings.
    """
    scores   = scoring.get("scores", {})
    f_score  = scores.get("fundamental", 5)
    v_score  = scores.get("valuation", 5)
    t_score  = scores.get("technical", 5)
    s_score  = scores.get("sentiment", 5)
    sig      = scoring.get("signal", "HOLD")

    strengths     = []
    weaknesses    = []
    opportunities = []
    threats       = []

    # ── STRENGTHS ─────────────────────────────────────────────────────────────
    if f_score >= 7:
        strengths.append("💪 Strong fundamentals — solid revenue growth, margins, and balance sheet")
    gm = info.get("grossMargins", 0) or 0
    if gm > 0.50:
        strengths.append(f"💰 High gross margin ({gm:.0%}) indicating strong pricing power")
    roe = info.get("returnOnEquity", 0) or 0
    if roe > 0.20:
        strengths.append(f"📈 Exceptional ROE of {roe:.0%} — efficient use of shareholder capital")
    cr = info.get("currentRatio", 0) or 0
    if cr > 2.0:
        strengths.append(f"🏦 Strong liquidity (current ratio {cr:.1f}x) — well-positioned for downturns")
    rg = info.get("revenueGrowth", 0) or 0
    if rg > 0.15:
        strengths.append(f"🚀 Strong revenue growth of {rg:.0%} YoY")
    fcf = fund.get("cashflows", {}).get("fcf", [None])[0] if fund.get("cashflows", {}).get("fcf") else None
    if fcf and fcf > 0:
        strengths.append("💵 Positive free cash flow — self-funding business model")
    beats = sentiment.get("beat_count", 0)
    total_q = beats + sentiment.get("miss_count", 0)
    if total_q >= 4 and beats / total_q >= 0.75:
        strengths.append(f"✅ Consistent earnings beats ({beats}/{total_q} quarters) — strong execution")
    de = (info.get("debtToEquity") or 0) / 100
    if de < 0.3:
        strengths.append(f"🔒 Low leverage (D/E {de:.2f}x) — financial resilience")
    inst = sentiment.get("inst_pct", 0) or 0
    if inst > 0.70:
        strengths.append(f"🏛️ High institutional ownership ({inst:.0%}) — professional investor confidence")

    # Peer outperformance
    above_peers = [r for r in comparison if "Above" in r.get("signal", "") and
                   r["key"] in ["gross_margin", "net_margin", "roe", "revenue_growth"]]
    if len(above_peers) >= 2:
        metrics = ", ".join(r["metric"] for r in above_peers[:3])
        strengths.append(f"🏆 Above-peer performance in: {metrics}")

    # ── WEAKNESSES ────────────────────────────────────────────────────────────
    if f_score < 4:
        weaknesses.append("⚠️ Weak fundamentals — declining revenue, thin margins, or leverage concerns")
    nm = info.get("profitMargins", 0) or 0
    if nm and nm < 0.05:
        weaknesses.append(f"📉 Thin net margin ({nm:.1%}) — limited profitability buffer")
    if nm and nm < 0:
        weaknesses.append(f"🔴 Negative net margin ({nm:.1%}) — currently unprofitable")
    if de > 1.5:
        weaknesses.append(f"⚠️ High leverage (D/E {de:.2f}x) — elevated financial risk")
    if cr and cr < 1.0:
        weaknesses.append(f"🔴 Current ratio below 1.0 ({cr:.2f}x) — potential short-term liquidity risk")
    pe = info.get("trailingPE") or 0
    fpe = info.get("forwardPE") or 0
    if pe and pe > 50 and fpe and fpe > 40:
        weaknesses.append(f"💸 High P/E ({pe:.0f}x trailing, {fpe:.0f}x forward) — expensive valuation")
    rg_dec = info.get("revenueGrowth", 0) or 0
    if rg_dec < -0.05:
        weaknesses.append(f"📉 Revenue declining ({rg_dec:.1%} YoY) — top-line pressure")
    short_pct = sentiment.get("short_pct", 0) or 0
    if short_pct > 0.10:
        weaknesses.append(f"🩳 High short interest ({short_pct:.0%} of float) — bearish market sentiment")
    below_peers = [r for r in comparison if "Below" in r.get("signal", "") and
                   r["key"] in ["gross_margin", "operating_margin", "roe", "revenue_growth"]]
    if len(below_peers) >= 2:
        metrics = ", ".join(r["metric"] for r in below_peers[:3])
        weaknesses.append(f"📊 Below-peer performance in: {metrics}")

    # ── OPPORTUNITIES ─────────────────────────────────────────────────────────
    if v_score >= 6.5:
        opportunities.append("💎 Attractive valuation — stock may be undervalued vs intrinsic value")
    upside = sentiment.get("target_upside", 0) or 0
    if upside > 0.15:
        opportunities.append(f"🎯 Analyst consensus target implies {upside:.0%} upside from current price")
    eg = info.get("earningsGrowth", 0) or 0
    if eg > 0.15:
        opportunities.append(f"📈 Strong earnings growth outlook ({eg:.0%}) — expanding profitability")
    # DCF upside from first valid model
    for model_key in ["wacc", "capm", "two_stage"]:
        m = scoring.get(f"dcf_{model_key}") or {}
        iv = m.get("intrinsic_value")
        cp = info.get("currentPrice", 0) or 0
        if iv and cp > 0 and (iv - cp) / cp > 0.20:
            opportunities.append(f"🔢 DCF models suggest {(iv-cp)/cp:.0%} potential upside to intrinsic value")
            break
    peg = info.get("pegRatio", 0) or 0
    if peg and 0 < peg < 1.0:
        opportunities.append(f"🌱 PEG ratio {peg:.2f} < 1 — growth available at a reasonable price")
    # Technical breakout
    sr_sigs = signals.get("support_resistance", {}).get("signals", [])
    if any("Breakout" in s[1] for s in sr_sigs):
        opportunities.append("📊 Technical breakout above resistance — potential momentum continuation")
    sma_score = signals.get("sma_ema", {}).get("score", 0)
    if sma_score >= 5:
        opportunities.append("📈 Strong technical trend — price above key moving averages")
    div_yield = info.get("dividendYield", 0) or 0
    if div_yield and div_yield > 0.03:
        opportunities.append(f"💰 Dividend yield of {div_yield:.1%} provides income while waiting for appreciation")

    # ── THREATS ───────────────────────────────────────────────────────────────
    beta = info.get("beta", 1) or 1
    if beta > 1.5:
        threats.append(f"⚡ High beta ({beta:.2f}) — significantly more volatile than the market")
    if t_score < 4:
        threats.append("📉 Weak technical signals — price trending below key moving averages")
    miss_count = sentiment.get("miss_count", 0)
    if miss_count >= 3:
        threats.append(f"❌ History of earnings misses ({miss_count} in recent quarters) — execution risk")
    avg_surp = sentiment.get("avg_surprise", 0) or 0
    if avg_surp and avg_surp < -2:
        threats.append(f"📉 Average earnings surprise is negative ({avg_surp:.1f}%) — guidance issues")
    if de > 2.0:
        threats.append(f"💣 Very high leverage (D/E {de:.2f}x) — vulnerable to rising interest rates")
    rec = sentiment.get("rec_mean", 3) or 3
    if rec >= 3.5:
        threats.append(f"🐻 Cautious analyst consensus (mean rating {rec:.1f}/5) — limited bullish conviction")
    obv_score = signals.get("obv", {}).get("score", 0)
    if obv_score <= -2:
        threats.append("📦 OBV declining — distribution / institutional selling detected")
    # Macro / valuation risk
    if pe and pe > 40:
        threats.append(f"📊 High valuation (P/E {pe:.0f}x) leaves little room for earnings disappointments")
    pct_from_high = sentiment.get("pct_from_52h", 0) or 0
    if pct_from_high and pct_from_high < -0.30:
        threats.append(f"📉 Stock is {abs(pct_from_high):.0%} below its 52-week high — sustained downtrend")

    # ── News & strategy-driven layer (dynamic, recent) ───────────────────────
    news_items = (sentiment.get("_news_items") or []) if isinstance(sentiment, dict) else []
    if news_items:
        import datetime as _dt
        now = _dt.datetime.now()
        recent_titles = []
        for it in news_items[:12]:
            ts = it.get("timestamp") or it.get("providerPublishTime") or 0
            try:
                age_days = (now - _dt.datetime.fromtimestamp(ts)).days if ts else 999
            except Exception:
                age_days = 999
            if age_days <= 30:
                recent_titles.append(it.get("title", "").lower())

        joined = " ".join(recent_titles)
        # Opportunity signals from recent headlines
        opp_kw = {
            "upgrade":        "Recent analyst upgrade reported in the news flow",
            "beat":           "Recent earnings beat highlighted in headlines",
            "raises guidance":"Company recently raised guidance",
            "new product":    "New product / launch generating coverage",
            "partnership":    "New partnership or strategic deal announced",
            "contract":       "New contract win covered in recent news",
            "expansion":      "Expansion into new markets reported",
            "buyback":        "Share buyback / capital return announced",
            "record":         "Record results or metrics reported recently",
        }
        for kw, msg in opp_kw.items():
            if kw in joined:
                opportunities.append(f"📰 {msg}")
                break
        # Threat signals from recent headlines
        thr_kw = {
            "downgrade":      "Recent analyst downgrade in the news flow",
            "miss":           "Recent earnings miss highlighted in headlines",
            "cuts guidance":  "Company recently cut guidance",
            "lawsuit":        "Litigation / legal action reported recently",
            "investigation":  "Regulatory investigation covered in news",
            "recall":         "Product recall reported recently",
            "layoff":         "Layoffs / restructuring reported — execution pressure",
            "probe":          "Regulatory probe mentioned in recent coverage",
            "fine":           "Regulatory fine or penalty reported",
        }
        for kw, msg in thr_kw.items():
            if kw in joined:
                threats.append(f"📰 {msg}")
                break

    # ── Sector / industry context layer ──────────────────────────────────────
    sector = info.get("sector", "")
    sector_dynamics = {
        "Technology":         ("AI/cloud capex cycle and digital transformation tailwinds",
                               "Rapid technology shifts and intensifying competition / regulation"),
        "Healthcare":         ("Aging demographics and structural healthcare demand growth",
                               "Drug pricing pressure and binary regulatory/clinical outcomes"),
        "Financial Services": ("Higher-rate environment supports net interest margins",
                               "Credit cycle and capital regulation (Basel) constrain returns"),
        "Energy":             ("Commodity price strength and energy security investment",
                               "Energy transition and stranded-asset risk over the long run"),
        "Consumer Cyclical":  ("Consumer spending recovery and brand pricing power",
                               "Highly sensitive to recession and consumer-confidence swings"),
        "Industrials":        ("Reshoring, infrastructure spend, and defense budget tailwinds",
                               "Cyclical demand tied to PMI and capital-spending cycles"),
        "Communication Services": ("Streaming/ad recovery and AI-driven ARPU expansion",
                               "Antitrust scrutiny and content-cost arms race"),
        "Basic Materials":    ("Green-transition demand for copper, lithium, and key metals",
                               "China-demand dependence and commodity-price cyclicality"),
        "Utilities":          ("Electrification and rate-based capex growth",
                               "Rate sensitivity — bond-proxy status hurts when rates rise"),
        "Real Estate":        ("Inflation-hedge characteristics and rent escalation",
                               "Direct rate sensitivity compresses REIT valuations"),
    }
    if sector in sector_dynamics:
        opp_msg, thr_msg = sector_dynamics[sector]
        opportunities.append(f"🏭 Sector tailwind: {opp_msg}")
        threats.append(f"🏭 Sector risk: {thr_msg}")

    # Ensure we have content in each category
    if not strengths:    strengths.append("No standout strengths identified from available data")
    if not weaknesses:   weaknesses.append("No major weaknesses identified from available data")
    if not opportunities:opportunities.append("No clear near-term opportunities identified from available data")
    if not threats:      threats.append("No significant threats identified from available data")

    return {
        "strengths":     strengths[:7],
        "weaknesses":    weaknesses[:7],
        "opportunities": opportunities[:7],
        "threats":       threats[:7],
    }


def fetch_sec_cik(ticker: str):
    """Cached 24h."""
    return _ttl(f"sec_cik:{ticker}", 86400, lambda: _fetch_sec_cik_impl(ticker))


def fetch_sec_filings(cik: str, form_type: str = "10-K", count: int = 5) -> list:
    """Cached 24h."""
    return _ttl(f"sec_filings:{cik}:{form_type}:{count}", 86400,
                lambda: _fetch_sec_filings_impl(cik, form_type, count))


def _fetch_google_news_rss(ticker: str, company_name: str = "") -> list:
    """
    Fallback news source: Google News RSS feed.
    Free, reliable, returns real recent articles with working links.
    Used when yfinance news fails (which is frequent).
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    import urllib.parse as _up

    # Build a query: company name + ticker for relevance
    query = company_name or ticker
    q_enc = _up.quote(f"{query} stock")
    url   = f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en"

    results = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        for item in root.iter("item"):
            title_el = item.find("title")
            link_el  = item.find("link")
            date_el  = item.find("pubDate")
            src_el   = item.find("source")

            title = title_el.text if title_el is not None else ""
            link  = link_el.text  if link_el  is not None else ""
            pub   = src_el.text   if src_el   is not None else "Google News"

            # Parse RFC 822 date
            ts = 0
            if date_el is not None and date_el.text:
                try:
                    from email.utils import parsedate_to_datetime
                    ts = int(parsedate_to_datetime(date_el.text).timestamp())
                except Exception:
                    pass

            if title and link:
                results.append({
                    "title":               title,
                    "link":                link,
                    "publisher":           pub,
                    "timestamp":           ts,
                    "providerPublishTime": ts,
                })
        return results[:15]
    except Exception:
        return []


def _fetch_news_combined(ticker: str, company_name: str = "") -> list:
    """Try yfinance first, fall back to Google News RSS if it returns nothing."""
    items = _fetch_news_impl(ticker)
    if items and len(items) >= 3:
        return items
    # yfinance failed or returned too few — use Google News RSS
    rss = _fetch_google_news_rss(ticker, company_name)
    # Merge, dedupe by title
    seen = {it["title"] for it in items}
    for it in rss:
        if it["title"] not in seen:
            items.append(it)
            seen.add(it["title"])
    return items[:15]


def fetch_news(ticker: str, company_name: str = "") -> list:
    """Cached 30 min. Combines yfinance + Google News RSS fallback."""
    return _ttl(f"news:{ticker}", 1800,
                lambda: _fetch_news_combined(ticker, company_name))


def get_regulatory_info(country: str, company_name: str) -> dict:
    """Get regulatory filing links for a given country."""
    info = REGULATORY_LINKS.get(country, {})
    if not info:
        # Try partial match
        for k, v in REGULATORY_LINKS.items():
            if k.lower() in country.lower() or country.lower() in k.lower():
                info = v
                break
    if not info:
        return {}
    name_enc = urllib.parse.quote((company_name or "")[:50])
    return {
        "name":   info["name"],
        "url":    info["search_url"].format(name=name_enc),
        "icon":   info.get("icon", "🌐"),
        "country": country,
    }
