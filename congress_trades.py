"""
congress_trades.py — U.S. Congress stock trade tracking.

Data source: the public House & Senate Stock Watcher datasets, which are
built directly from official STOCK Act disclosure filings (the same filings
the r/tradewithcongress subreddit summarises). These are free JSON feeds with
no API key required.

We deliberately use the official-disclosure-derived datasets rather than
scraping Reddit, because:
  - Reddit blocks automated access and its HTML changes frequently
  - The underlying data (ticker, member, transaction type, date) is identical
  - These feeds are the canonical machine-readable form of the same filings

Fields extracted per trade:
  ticker, company, member name, party/chamber, transaction type (buy/sell),
  amount range, transaction date, disclosure date.
"""

import urllib.request
import json
import time
import datetime
from collections import defaultdict


# Public dataset endpoints (mirrors of STOCK Act filings).
# Multiple candidates — we try each in order for resilience.
HOUSE_SOURCES = [
    "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json",
    "https://housestockwatcher.com/api/transactions",
]
SENATE_SOURCES = [
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json",
    "https://senatestockwatcher.com/api/transactions",
]


def _http_get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockAnalyzer)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _fetch_source_list(sources: list) -> list:
    """Try each candidate URL until one returns data."""
    for url in sources:
        try:
            data = _http_get_json(url)
            if isinstance(data, list) and data:
                return data
            if isinstance(data, dict) and data.get("data"):
                return data["data"]
        except Exception:
            continue
    return []


def _parse_date(s: str):
    """Parse the various date formats the feeds use → datetime or None."""
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.datetime.strptime(s.strip()[:19], fmt)
        except Exception:
            continue
    return None


def _normalise_txn(raw: dict, chamber: str) -> dict:
    """Map a raw feed record to our standard schema. Handles field-name variants."""
    def g(*keys, default=""):
        for k in keys:
            if k in raw and raw[k] not in (None, ""):
                return raw[k]
        return default

    ticker = str(g("ticker", "Ticker", "symbol", default="")).upper().strip()
    # Filter out non-stock placeholders the feeds use
    if ticker in ("", "--", "N/A", "NA", "<C>", "FALSE", "—"):
        ticker = ""

    txn_type_raw = str(g("type", "Transaction", "transaction_type", default="")).lower()
    if "purchase" in txn_type_raw or "buy" in txn_type_raw:
        txn_type = "Buy"
    elif "sale" in txn_type_raw or "sell" in txn_type_raw:
        txn_type = "Sell"
    elif "exchange" in txn_type_raw:
        txn_type = "Exchange"
    else:
        txn_type = txn_type_raw.title() or "—"

    txn_date  = _parse_date(g("transaction_date", "Transaction Date", "transactionDate"))
    disc_date = _parse_date(g("disclosure_date", "Disclosure Date", "disclosureDate"))

    return {
        "ticker":       ticker,
        "company":      g("asset_description", "Asset", "company", default=""),
        "member":       g("representative", "senator", "member", "Name", default="Unknown"),
        "chamber":      chamber,
        "party":        g("party", "Party", default=""),
        "type":         txn_type,
        "amount":       g("amount", "Amount", "amount_range", default=""),
        "txn_date":     txn_date,
        "disc_date":    disc_date,
        "txn_date_str": txn_date.strftime("%Y-%m-%d") if txn_date else "",
    }


# Module-level cache (the UI also wraps this in st.cache_data)
_CACHE = {}


def fetch_congress_trades(days_back: int = 90, force: bool = False) -> list:
    """
    Fetch and normalise recent congressional trades from House + Senate feeds.
    Returns a list of standardised trade dicts, most recent first.

    days_back: only include trades with a transaction date within this window.
    """
    cache_key = f"congress:{days_back}"
    if not force and cache_key in _CACHE:
        val, ts = _CACHE[cache_key]
        if time.time() - ts < 3600:
            return val

    house  = _fetch_source_list(HOUSE_SOURCES)
    senate = _fetch_source_list(SENATE_SOURCES)

    trades = []
    for raw in house:
        if isinstance(raw, dict):
            trades.append(_normalise_txn(raw, "House"))
    for raw in senate:
        if isinstance(raw, dict):
            trades.append(_normalise_txn(raw, "Senate"))

    # Filter: valid ticker + within date window
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days_back)
    filtered = [
        t for t in trades
        if t["ticker"] and t["txn_date"] and t["txn_date"] >= cutoff
    ]
    # Sort most recent transaction first
    filtered.sort(key=lambda t: t["txn_date"], reverse=True)

    _CACHE[cache_key] = (filtered, time.time())
    return filtered


def aggregate_by_ticker(trades: list) -> list:
    """
    Aggregate trades by ticker to find the most-traded names.
    Returns a list of dicts sorted by total trade count, descending.
    """
    by_ticker = defaultdict(lambda: {
        "ticker": "", "company": "", "buys": 0, "sells": 0,
        "members": set(), "latest_date": None, "trades": []
    })

    for t in trades:
        tk = t["ticker"]
        rec = by_ticker[tk]
        rec["ticker"]  = tk
        rec["company"] = rec["company"] or t["company"]
        if t["type"] == "Buy":
            rec["buys"] += 1
        elif t["type"] == "Sell":
            rec["sells"] += 1
        rec["members"].add(t["member"])
        if t["txn_date"] and (rec["latest_date"] is None or t["txn_date"] > rec["latest_date"]):
            rec["latest_date"] = t["txn_date"]
        rec["trades"].append(t)

    out = []
    for tk, rec in by_ticker.items():
        out.append({
            "ticker":       tk,
            "company":      rec["company"],
            "total_trades": rec["buys"] + rec["sells"],
            "buys":         rec["buys"],
            "sells":        rec["sells"],
            "n_members":    len(rec["members"]),
            "members":      sorted(rec["members"])[:8],
            "latest_date":  rec["latest_date"],
            "latest_str":   rec["latest_date"].strftime("%Y-%m-%d") if rec["latest_date"] else "",
            "net_bias":     "Buying" if rec["buys"] > rec["sells"] else
                            "Selling" if rec["sells"] > rec["buys"] else "Mixed",
        })

    # Sort: most-traded first, then most recent
    out.sort(key=lambda r: (r["total_trades"], r["latest_date"] or datetime.datetime.min),
             reverse=True)
    return out
