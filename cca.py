"""
cca.py — Comparable Company Analysis (CCA).

Based on Breaking Into Wall Street / JPMorgan M&A methodology:

Process:
  1. Use the already-fetched peer metrics (5–6 companies)
  2. Calculate LTM and Forward multiples for all peers:
       EV/Revenue, EV/EBITDA, P/E, Forward P/E, EV/EBIT, P/FCF
  3. Compute peer statistics: min, 25th percentile, median, 75th, max
  4. Apply peer multiples to target company's metrics
  5. EV → Equity Value → implied share price for each multiple
  6. Football field: range of implied prices across all methods

Reference:
  - BIWS CCA Tutorial: breakingintowallstreet.com/kb/valuation/comparable-company-analysis-cca
  - JPMorgan M&A: DCF Valuation and Merger Analysis
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─── Multiple Definitions ────────────────────────────────────────────────────

# (key, label, is_ev_based, metric_key_on_peer, metric_key_on_target, higher_is_worse)
MULTIPLES = [
    ("ev_revenue",  "EV / Revenue",   True,  "ev_revenue",  "ev_revenue",   True),
    ("ev_ebitda",   "EV / EBITDA",    True,  "ev_ebitda",   "ev_ebitda",    True),
    ("pe_trailing", "P/E (TTM)",      False, "pe_trailing", "pe_trailing",  True),
    ("pe_forward",  "Forward P/E",    False, "pe_forward",  "pe_forward",   True),
    ("price_book",  "P / Book",       False, "price_book",  "price_book",   True),
    ("price_sales", "P / Sales",      False, "price_sales", "price_sales",  True),
]


def _safe(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f) or f <= 0 or f > 500:
            return None
        return f
    except Exception:
        return None


def _pct_stats(values: list) -> dict:
    """Compute min / 25th / median / 75th / max for a list of floats."""
    clean = sorted([v for v in values if v is not None])
    if not clean:
        return {}
    n = len(clean)
    return {
        "min":    clean[0],
        "p25":    float(np.percentile(clean, 25)),
        "median": float(np.median(clean)),
        "p75":    float(np.percentile(clean, 75)),
        "max":    clean[-1],
        "mean":   float(np.mean(clean)),
        "n":      n,
    }


# ─── Main CCA Engine ─────────────────────────────────────────────────────────

def run_cca(target_info: dict, peer_data: dict) -> dict:
    """
    Run Comparable Company Analysis.

    Args:
        target_info:  yfinance info dict for the target company
        peer_data:    {ticker: {metric: value, ...}} from fetch_peer_metrics

    Returns:
        {
          "peer_multiples":   {multiple_key: [peer_values], ...},
          "peer_stats":       {multiple_key: {min, p25, median, p75, max}, ...},
          "implied_prices":   {multiple_key: {p25, median, p75}, ...},
          "summary":          {overall_low, overall_mid, overall_high, current_price},
          "football_field":   [{label, low, mid, high}, ...]  for chart
          "target_multiples": {multiple_key: value}           target's own multiples
          "peers_used":       [ticker, ...]
        }
    """
    if not peer_data:
        return {}

    cp        = float(target_info.get("currentPrice") or target_info.get("regularMarketPrice") or 0)
    mktcap    = float(target_info.get("marketCap") or 0)
    ev        = float(target_info.get("enterpriseValue") or 0)
    shares    = float(target_info.get("sharesOutstanding") or target_info.get("floatShares") or 0)

    # Cash and debt for EV → Equity Value bridge
    cash      = float(target_info.get("totalCash")  or 0)
    debt      = float(target_info.get("totalDebt")  or 0)
    net_cash  = cash - debt

    if cp <= 0 or mktcap <= 0 or shares <= 0:
        return {}

    # Target's own metrics (denominators for applying multiples)
    target_metrics = {
        "revenue":   float(target_info.get("totalRevenue") or 0) or None,
        "ebitda":    float(target_info.get("ebitda") or 0) or None,
        "net_income":float((target_info.get("trailingEps") or 0) * shares) or None,
        "eps":       float(target_info.get("trailingEps") or 0) or None,
        "fwd_eps":   float(target_info.get("forwardEps") or 0) or None,
        "book_val":  float(target_info.get("bookValue") or 0) * shares if target_info.get("bookValue") else None,
        "ev":        ev,
        "mktcap":    mktcap,
        "shares":    shares,
        "net_cash":  net_cash,
        "cp":        cp,
    }

    # Target's own multiples (for comparison)
    target_own = {
        "ev_revenue":  _safe(target_info.get("enterpriseToRevenue")),
        "ev_ebitda":   _safe(target_info.get("enterpriseToEbitda")),
        "pe_trailing": _safe(target_info.get("trailingPE")),
        "pe_forward":  _safe(target_info.get("forwardPE")),
        "price_book":  _safe(target_info.get("priceToBook")),
        "price_sales": _safe(target_info.get("priceToSalesTrailingTwelveMonths")),
    }

    # Collect peer multiples
    peer_multiples: dict = {k: [] for k in ["ev_revenue","ev_ebitda","pe_trailing",
                                              "pe_forward","price_book","price_sales"]}
    peers_used = []

    for ticker, pm in peer_data.items():
        has_data = False
        for key in peer_multiples:
            v = _safe(pm.get(key))
            if v is not None:
                peer_multiples[key].append(v)
                has_data = True
        if has_data:
            peers_used.append(ticker)

    # Statistics per multiple
    peer_stats = {key: _pct_stats(vals) for key, vals in peer_multiples.items() if vals}

    # Implied prices: apply peer percentile multiples to target's metrics
    implied_prices: dict = {}
    football_field: list = []

    # EV-based multiples → EV → subtract debt, add cash → equity → /shares
    ev_multiple_map = {
        "ev_revenue": target_metrics.get("revenue"),
        "ev_ebitda":  target_metrics.get("ebitda"),
    }
    for mult_key, target_metric in ev_multiple_map.items():
        stats = peer_stats.get(mult_key, {})
        if not stats or target_metric is None or target_metric <= 0:
            continue
        prices = {}
        for pct in ["p25", "median", "p75"]:
            m = stats.get(pct)
            if m:
                implied_ev    = m * target_metric
                implied_eq    = implied_ev + net_cash
                implied_price = implied_eq / shares
                prices[pct]   = round(max(0, implied_price), 2)
        if prices:
            implied_prices[mult_key] = prices

    # Equity-based multiples → multiply by EPS / BV per share
    eq_multiple_map = {
        "pe_trailing": target_metrics.get("eps"),
        "pe_forward":  target_metrics.get("fwd_eps"),
        "price_book":  (target_info.get("bookValue") or None),  # P/B uses BV per share
        "price_sales": (target_metrics.get("revenue") / shares
                        if target_metrics.get("revenue") and shares else None),
    }
    for mult_key, target_metric in eq_multiple_map.items():
        stats = peer_stats.get(mult_key, {})
        if not stats or target_metric is None or target_metric <= 0:
            continue
        prices = {}
        for pct in ["p25", "median", "p75"]:
            m = stats.get(pct)
            if m:
                implied_price = m * target_metric
                prices[pct]   = round(max(0, implied_price), 2)
        if prices:
            implied_prices[mult_key] = prices

    # Football field rows
    multiple_labels = {
        "ev_revenue":  "EV / Revenue",
        "ev_ebitda":   "EV / EBITDA",
        "pe_trailing": "P/E (TTM)",
        "pe_forward":  "Forward P/E",
        "price_book":  "P / Book",
        "price_sales": "P / Sales",
    }
    for key, prices in implied_prices.items():
        lo  = prices.get("p25",    0)
        mid = prices.get("median", 0)
        hi  = prices.get("p75",    0)
        if lo > 0 and hi > 0:
            football_field.append({
                "label":   multiple_labels.get(key, key),
                "key":     key,
                "low":     lo,
                "mid":     mid,
                "high":    hi,
                "vs_current": (mid - cp) / cp if cp > 0 else 0,
            })

    # Overall CCA implied range (across all methods)
    all_mids  = [f["mid"]  for f in football_field if f["mid"]  > 0]
    all_lows  = [f["low"]  for f in football_field if f["low"]  > 0]
    all_highs = [f["high"] for f in football_field if f["high"] > 0]

    summary = {
        "current_price":  cp,
        "overall_low":    float(np.percentile(all_lows,  25)) if all_lows  else None,
        "overall_mid":    float(np.median(all_mids))          if all_mids  else None,
        "overall_high":   float(np.percentile(all_highs, 75)) if all_highs else None,
        "implied_upside": (float(np.median(all_mids)) - cp) / cp if (all_mids and cp > 0) else None,
        "peers_used":     peers_used,
        "n_peers":        len(peers_used),
        "n_multiples":    len(football_field),
    }

    return {
        "peer_multiples":   peer_multiples,
        "peer_stats":       peer_stats,
        "target_multiples": target_own,
        "implied_prices":   implied_prices,
        "football_field":   football_field,
        "summary":          summary,
        "peers_used":       peers_used,
        "target_metrics":   target_metrics,
    }


def format_cca_peer_table(peer_data: dict, target_ticker: str) -> pd.DataFrame:
    """Build a comparison DataFrame for the peer multiples table."""
    rows = []
    for ticker, pm in peer_data.items():
        rows.append({
            "Ticker":      ticker,
            "Name":        pm.get("name", ticker)[:22],
            "Country":     pm.get("country", ""),
            "Mkt Cap":     pm.get("market_cap"),
            "EV/Rev":      _safe(pm.get("ev_revenue")),
            "EV/EBITDA":   _safe(pm.get("ev_ebitda")),
            "P/E (TTM)":   _safe(pm.get("pe_trailing")),
            "Fwd P/E":     _safe(pm.get("pe_forward")),
            "P/B":         _safe(pm.get("price_book")),
            "P/S":         _safe(pm.get("price_sales")),
            "Net Margin":  pm.get("net_margin"),
            "Rev Growth":  pm.get("revenue_growth"),
        })
    return pd.DataFrame(rows)
