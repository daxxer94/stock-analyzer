"""
etf_data.py — ETF-specific data fetching and analysis.

ETFs are NOT operating companies — they have no income statement, balance sheet,
or DCF. Instead they hold a basket of securities. This module extracts the
information that actually matters for an ETF:

  - Fund overview: category, family, expense ratio, AUM, inception, yield
  - Top holdings (the actual companies/bonds the ETF owns)
  - Sector weightings
  - Asset class allocation (stocks / bonds / cash)
  - Bond holdings & credit ratings (for fixed-income ETFs)
  - Performance returns (YTD / 3yr / 5yr)
  - Major institutional holders of the ETF itself

Primary source: yfinance funds_data (Yahoo Finance).
Secondary: yfinance info dict + fast_info.
All wrapped in graceful fallbacks — a missing field never breaks the page.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import random


def _retry(fn, retries=3, base=1.5):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if any(x in str(e).lower() for x in ["429", "rate", "timeout", "503"]):
                time.sleep(base * (2 ** attempt) + random.uniform(0.2, 0.8))
            else:
                raise
    if last:
        raise last
    return None


def _safe_float(v, default=None):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def detect_quote_type(ticker: str, info: dict = None) -> str:
    """
    Detect the asset type for a ticker.
    Returns one of: EQUITY, ETF, MUTUALFUND, INDEX, FUTURE, CRYPTOCURRENCY,
                    CURRENCY, OPTION, or UNKNOWN.
    """
    if info and info.get("quoteType"):
        return str(info["quoteType"]).upper()
    try:
        t  = yf.Ticker(ticker)
        qt = None
        try:
            fi = t.fast_info
            qt = getattr(fi, "quote_type", None) or (fi.get("quoteType") if hasattr(fi, "get") else None)
        except Exception:
            pass
        if not qt:
            raw = t.info or {}
            qt  = raw.get("quoteType")
        return str(qt).upper() if qt else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def fetch_etf_data(ticker: str) -> dict:
    """
    Fetch comprehensive ETF data. Returns a dict with everything available;
    missing fields are simply absent or None.
    """
    result = {"ticker": ticker.upper(), "quote_type": "ETF"}

    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        return {**result, "error": f"Could not load ticker: {e}"}

    # ── Price / basic info ────────────────────────────────────────────────────
    info = {}
    try:
        raw = _retry(lambda: t.info)
        if isinstance(raw, dict):
            info = raw
    except Exception:
        pass

    price = None
    try:
        fi = t.fast_info
        for attr in ("last_price", "previous_close"):
            v = getattr(fi, attr, None)
            if v and float(v) > 0:
                price = float(v); break
        if not price:
            for k in ("lastPrice", "previousClose"):
                v = fi.get(k) if hasattr(fi, "get") else None
                if v and float(v) > 0:
                    price = float(v); break
    except Exception:
        pass
    if not price:
        try:
            h = t.history(period="5d")
            if not h.empty:
                price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass

    result.update({
        "name":         info.get("longName") or info.get("shortName") or ticker.upper(),
        "price":        price or 0,
        "currency":     info.get("currency", "USD"),
        "exchange":     info.get("exchange", ""),
        "category":     info.get("category", ""),
        "fund_family":  info.get("fundFamily", ""),
        "total_assets": _safe_float(info.get("totalAssets")),
        "nav":          _safe_float(info.get("navPrice")),
        "expense_ratio":_safe_float(info.get("annualReportExpenseRatio") or
                                    info.get("netExpenseRatio")),
        "yield":        _safe_float(info.get("yield")),
        "ytd_return":   _safe_float(info.get("ytdReturn")),
        "three_yr_return": _safe_float(info.get("threeYearAverageReturn")),
        "five_yr_return":  _safe_float(info.get("fiveYearAverageReturn")),
        "beta_3yr":     _safe_float(info.get("beta3Year")),
        "inception":    info.get("fundInceptionDate", ""),
        "legal_type":   info.get("legalType", ""),
        "52w_high":     _safe_float(info.get("fiftyTwoWeekHigh")),
        "52w_low":      _safe_float(info.get("fiftyTwoWeekLow")),
        "50d_avg":      _safe_float(info.get("fiftyDayAverage")),
        "200d_avg":     _safe_float(info.get("twoHundredDayAverage")),
        "volume":       _safe_float(info.get("volume") or info.get("averageVolume")),
    })

    # ── Rich funds_data (holdings, sectors, asset classes) ────────────────────
    try:
        fd = t.funds_data
        if fd is not None:
            # Description
            try:
                desc = fd.description
                if desc and isinstance(desc, str):
                    result["description"] = desc.strip()
            except Exception:
                pass

            # Fund overview dict
            try:
                ov = fd.fund_overview
                if isinstance(ov, dict):
                    result["fund_overview"] = {k: v for k, v in ov.items() if v}
            except Exception:
                pass

            # Fund operations (expense ratios vs category)
            try:
                ops = fd.fund_operations
                if isinstance(ops, pd.DataFrame) and not ops.empty:
                    result["fund_operations"] = ops
            except Exception:
                pass

            # Top holdings (the actual companies the ETF owns)
            try:
                th = fd.top_holdings
                if isinstance(th, pd.DataFrame) and not th.empty:
                    result["top_holdings"] = th
            except Exception:
                pass

            # Sector weightings
            try:
                sw = fd.sector_weightings
                if isinstance(sw, dict) and sw:
                    # Clean & sort descending
                    cleaned = {k: _safe_float(v) for k, v in sw.items()
                               if _safe_float(v) is not None}
                    result["sector_weightings"] = dict(
                        sorted(cleaned.items(), key=lambda x: -x[1]))
            except Exception:
                pass

            # Asset class allocation
            try:
                ac = fd.asset_classes
                if isinstance(ac, dict) and ac:
                    cleaned = {k: _safe_float(v) for k, v in ac.items()
                               if _safe_float(v) is not None}
                    result["asset_classes"] = cleaned
            except Exception:
                pass

            # Equity holdings characteristics (P/E, P/B of underlying)
            try:
                eh = fd.equity_holdings
                if isinstance(eh, pd.DataFrame) and not eh.empty:
                    result["equity_holdings"] = eh
                elif isinstance(eh, dict) and eh:
                    result["equity_holdings_dict"] = eh
            except Exception:
                pass

            # Bond holdings (for fixed-income ETFs)
            try:
                bh = fd.bond_holdings
                if isinstance(bh, pd.DataFrame) and not bh.empty:
                    result["bond_holdings"] = bh
                elif isinstance(bh, dict) and bh:
                    result["bond_holdings_dict"] = bh
            except Exception:
                pass

            # Bond credit ratings
            try:
                br = fd.bond_ratings
                if isinstance(br, dict) and br:
                    cleaned = {k: _safe_float(v) for k, v in br.items()
                               if _safe_float(v) is not None}
                    result["bond_ratings"] = cleaned
            except Exception:
                pass
    except Exception:
        pass

    # ── Institutional holders of the ETF itself ───────────────────────────────
    try:
        ih = _retry(lambda: t.institutional_holders)
        if isinstance(ih, pd.DataFrame) and not ih.empty:
            result["institutional_holders"] = ih.head(10)
    except Exception:
        pass

    return result


def get_etf_summary_metrics(etf: dict) -> list:
    """Return a list of (label, value, help) tuples for the summary cards."""
    metrics = []

    er = etf.get("expense_ratio")
    if er is not None:
        metrics.append(("Expense Ratio", f"{er:.2%}",
                        "Annual fee as % of assets. Lower is better; <0.10% is excellent for index ETFs."))

    aum = etf.get("total_assets")
    if aum:
        if aum >= 1e9:   aum_str = f"${aum/1e9:.1f}B"
        elif aum >= 1e6: aum_str = f"${aum/1e6:.0f}M"
        else:            aum_str = f"${aum:,.0f}"
        metrics.append(("Assets Under Mgmt", aum_str,
                        "Total fund size. Larger funds usually have better liquidity and tighter spreads."))

    y = etf.get("yield")
    if y is not None:
        metrics.append(("Distribution Yield", f"{y:.2%}",
                        "Trailing 12-month income distributions as % of price."))

    ytd = etf.get("ytd_return")
    if ytd is not None:
        metrics.append(("YTD Return", f"{ytd:+.1%}", "Year-to-date total return."))

    return metrics
