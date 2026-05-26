"""
valuation.py  - Relative valuation ratios + peer comparison.
sentiment.py  - Analyst recommendations, price targets, earnings history.
(Combined in one file for convenience.)
"""
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  VALUATION
# ═══════════════════════════════════════════════════════════════════════════════

RATIO_META = [
    # (key, label, higher_is_better)
    ("pe_trailing",      "P/E (TTM)",           False),
    ("pe_forward",       "Forward P/E",          False),
    ("peg",              "PEG Ratio",            False),
    ("ev_ebitda",        "EV/EBITDA",            False),
    ("ev_revenue",       "EV/Revenue",           False),
    ("price_book",       "P/B Ratio",            False),
    ("price_sales",      "P/S Ratio",            False),
    ("gross_margin",     "Gross Margin",         True),
    ("operating_margin", "Operating Margin",     True),
    ("net_margin",       "Net Margin",           True),
    ("roe",              "Return on Equity",     True),
    ("roa",              "Return on Assets",     True),
    ("revenue_growth",   "Revenue Growth",       True),
    ("earnings_growth",  "Earnings Growth",      True),
    ("debt_equity",      "Debt/Equity",          False),
    ("current_ratio",    "Current Ratio",        True),
    ("beta",             "Beta",                 None),
    ("dividend_yield",   "Dividend Yield",       True),
]


def get_valuation_ratios(info: dict) -> dict:
    """Extract all valuation ratios from yfinance info dict."""
    return {
        "pe_trailing":      _clean(info.get("trailingPE")),
        "pe_forward":       _clean(info.get("forwardPE")),
        "peg":              _clean(info.get("pegRatio")),
        "ev_ebitda":        _clean(info.get("enterpriseToEbitda")),
        "ev_revenue":       _clean(info.get("enterpriseToRevenue")),
        "price_book":       _clean(info.get("priceToBook")),
        "price_sales":      _clean(info.get("priceToSalesTrailingTwelveMonths")),
        "gross_margin":     _clean(info.get("grossMargins")),
        "operating_margin": _clean(info.get("operatingMargins")),
        "net_margin":       _clean(info.get("profitMargins")),
        "roe":              _clean(info.get("returnOnEquity")),
        "roa":              _clean(info.get("returnOnAssets")),
        "revenue_growth":   _clean(info.get("revenueGrowth")),
        "earnings_growth":  _clean(info.get("earningsGrowth")),
        "debt_equity":      _clean(info.get("debtToEquity")),
        "current_ratio":    _clean(info.get("currentRatio")),
        "beta":             _clean(info.get("beta")),
        "dividend_yield":   _clean(info.get("dividendYield")),
        "payout_ratio":     _clean(info.get("payoutRatio")),
        "eps_ttm":          _clean(info.get("trailingEps")),
        "eps_forward":      _clean(info.get("forwardEps")),
        "book_value":       _clean(info.get("bookValue")),
        "market_cap":       info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
    }


def compare_with_peers(target: dict, peer_data: dict, target_ticker: str) -> list:
    """
    Compare target stock to peers on all RATIO_META metrics.
    Returns list of comparison rows (dicts).
    """
    rows = []
    for key, label, higher_better in RATIO_META:
        t_val = target.get(key)
        peer_vals = {t: peer_data[t].get(key) for t in peer_data
                     if peer_data[t].get(key) is not None and _is_valid(peer_data[t].get(key))}
        if not peer_vals and t_val is None:
            continue

        vals = list(peer_vals.values())
        p_median = float(np.median(vals)) if vals else None
        p_mean   = float(np.mean(vals))   if vals else None

        # Percentile rank (of target among all vals including itself)
        all_vals = vals + ([t_val] if t_val is not None else [])
        percentile = None
        if t_val is not None and len(all_vals) > 1:
            percentile = sum(1 for v in all_vals if v < t_val) / len(all_vals)

        # vs median signal
        signal = "—"
        if t_val is not None and p_median is not None and p_median != 0:
            diff = (t_val - p_median) / abs(p_median)
            if higher_better is True:
                signal = "✅ Above avg" if diff > 0.10 else ("🔴 Below avg" if diff < -0.10 else "🟡 In line")
            elif higher_better is False:
                signal = "✅ Below avg" if diff < -0.10 else ("🔴 Above avg" if diff > 0.10 else "🟡 In line")
            else:
                signal = "🟡 N/A"

        rows.append({
            "metric":      label,
            "key":         key,
            "target":      t_val,
            "peer_median": p_median,
            "peer_mean":   p_mean,
            "peers":       peer_vals,
            "signal":      signal,
            "percentile":  percentile,
            "higher_better": higher_better,
        })
    return rows


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def _is_valid(v):
    if v is None:
        return False
    try:
        f = float(v)
        return not (np.isnan(f) or np.isinf(f) or abs(f) > 1e9)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(data: dict) -> dict:
    """Extract and structure all sentiment / consensus data."""
    info = data.get("info", {})
    recs = data.get("recommendations")
    targets = data.get("analyst_targets")
    earn_hist = data.get("earnings_history")

    result = {
        "rec_mean":    _clean(info.get("recommendationMean")),
        "rec_key":     info.get("recommendationKey", "N/A"),
        "num_analysts": info.get("numberOfAnalystOpinions", 0),
        "target_high":  _clean(info.get("targetHighPrice")),
        "target_low":   _clean(info.get("targetLowPrice")),
        "target_mean":  _clean(info.get("targetMeanPrice")),
        "target_median":_clean(info.get("targetMedianPrice")),
        "current_price":_clean(info.get("currentPrice") or info.get("regularMarketPrice")),
        "52w_high":     _clean(info.get("fiftyTwoWeekHigh")),
        "52w_low":      _clean(info.get("fiftyTwoWeekLow")),
        "short_ratio":  _clean(info.get("shortRatio")),
        "short_pct":    _clean(info.get("shortPercentOfFloat")),
        "inst_pct":     _clean(info.get("institutionPercentHeld")),
        "insider_pct":  _clean(info.get("heldPercentInsiders")),
        # Counts filled below
        "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0,
        "earnings_surprises": [],
        "avg_surprise": None,
        "beat_count": 0,
        "miss_count": 0,
    }

    # 52-week position
    cp = result["current_price"]
    if cp and result["52w_high"]:
        result["pct_from_52h"] = (cp - result["52w_high"]) / result["52w_high"]
    if cp and result["52w_low"]:
        result["pct_from_52l"] = (cp - result["52w_low"]) / result["52w_low"]

    # Analyst rating breakdown
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        try:
            # New yfinance format has period + counts
            if "period" in recs.columns:
                latest = recs[recs["period"] == "0m"] if "0m" in recs["period"].values else recs.tail(1)
                for col, key in [("strongBuy","strong_buy"),("buy","buy"),("hold","hold"),
                                  ("sell","sell"),("strongSell","strong_sell")]:
                    if col in latest.columns:
                        result[key] = int(latest[col].sum())
            else:
                # Older format: 'Firm', 'To Grade', 'Date'
                grade_map = {
                    "Strong Buy": "strong_buy", "Buy": "buy", "Outperform": "buy",
                    "Overweight": "buy", "Accumulate": "buy",
                    "Neutral": "hold", "Hold": "hold", "Market Perform": "hold",
                    "Equal-Weight": "hold", "Sector Perform": "hold",
                    "Underperform": "sell", "Sell": "sell", "Underweight": "sell",
                    "Strong Sell": "strong_sell", "Reduce": "sell",
                }
                col = "To Grade" if "To Grade" in recs.columns else (
                      "Action" if "Action" in recs.columns else None)
                if col:
                    for grade in recs[col].tail(30):
                        k = grade_map.get(grade, "hold")
                        result[k] = result.get(k, 0) + 1
        except Exception:
            pass

    # Earnings surprise history
    if isinstance(earn_hist, pd.DataFrame) and not earn_hist.empty:
        try:
            surprises = []
            for col in ["surprisePercent", "epsSurprisePct", "surprise"]:
                if col in earn_hist.columns:
                    surprises = earn_hist[col].dropna().tolist()
                    break
            if surprises:
                result["earnings_surprises"] = [float(s) for s in surprises[-8:]]
                result["avg_surprise"] = float(np.mean(result["earnings_surprises"]))
                result["beat_count"]   = sum(1 for s in result["earnings_surprises"] if s > 0)
                result["miss_count"]   = sum(1 for s in result["earnings_surprises"] if s < 0)
        except Exception:
            pass

    # Target upside
    if result["current_price"] and result["target_mean"]:
        result["target_upside"] = (result["target_mean"] - result["current_price"]) / result["current_price"]

    return result
