"""
data.py - Data fetching from Yahoo Finance via yfinance.
All functions are cached with st.cache_data to avoid redundant API calls.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker: str) -> dict:
    """Fetch all relevant data for a single ticker."""
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if not price or float(price) == 0:
            return {"ticker": ticker, "error": f"Could not find price data for '{ticker}'. "
                    "Check the symbol — EU examples: ASML.AS, SHEL.L, SAP.DE, TTE.PA"}

        hist_2y = stock.history(period="2y")
        hist_5y = stock.history(period="5y")

        income_stmt   = _safe_fetch(stock, "income_stmt")
        balance_sheet = _safe_fetch(stock, "balance_sheet")
        cash_flow     = _safe_fetch(stock, "cashflow")
        q_income      = _safe_fetch(stock, "quarterly_income_stmt")
        q_balance     = _safe_fetch(stock, "quarterly_balance_sheet")
        q_cashflow    = _safe_fetch(stock, "quarterly_cashflow")

        recommendations  = _safe_fetch(stock, "recommendations")
        analyst_targets  = _safe_fetch(stock, "analyst_price_targets")
        earnings_history = _safe_fetch(stock, "earnings_history")
        institutional    = _safe_fetch(stock, "institutional_holders")

        # ── Earnings dates (EPS estimate vs actual) ───────────────────────────
        earnings_dates = pd.DataFrame()
        try:
            earnings_dates = stock.get_earnings_dates(limit=12)
        except Exception:
            try:
                earnings_dates = stock.earnings_dates
            except Exception:
                pass

        # ── Upcoming earnings calendar ────────────────────────────────────────
        calendar = {}
        try:
            calendar = stock.get_calendar() or {}
        except Exception:
            pass

        # ── Analyst estimates ─────────────────────────────────────────────────
        earnings_estimate = pd.DataFrame()
        revenue_estimate  = pd.DataFrame()
        eps_trend         = pd.DataFrame()
        growth_estimates  = pd.DataFrame()
        try:
            earnings_estimate = stock.get_earnings_estimate() or pd.DataFrame()
            revenue_estimate  = stock.get_revenue_estimate()  or pd.DataFrame()
            eps_trend         = stock.get_eps_trend()          or pd.DataFrame()
            growth_estimates  = stock.get_growth_estimates()   or pd.DataFrame()
        except Exception:
            pass

        return {
            "ticker": ticker,
            "info":   info,
            "hist_2y":  hist_2y,
            "hist_5y":  hist_5y,
            "income_stmt":   income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow":     cash_flow,
            "q_income":   q_income,
            "q_balance":  q_balance,
            "q_cashflow": q_cashflow,
            "recommendations":   recommendations,
            "analyst_targets":   analyst_targets,
            "earnings_history":  earnings_history,
            "institutional":     institutional,
            "earnings_dates":    earnings_dates,
            "calendar":          calendar,
            "earnings_estimate": earnings_estimate,
            "revenue_estimate":  revenue_estimate,
            "eps_trend":         eps_trend,
            "growth_estimates":  growth_estimates,
            "error": None,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def _safe_fetch(stock, attr: str):
    try:
        val = getattr(stock, attr)
        if val is None:
            return pd.DataFrame()
        return val if isinstance(val, pd.DataFrame) else val
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_risk_free_rate() -> float:
    """Fetch current 10-year US Treasury yield (^TNX). Falls back to 4.5%."""
    try:
        hist = yf.Ticker("^TNX").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sp500() -> pd.DataFrame:
    try:
        return yf.Ticker("^GSPC").history(period="2y")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer_metrics(tickers_tuple: tuple) -> dict:
    """Fetch key valuation & profitability metrics for peer list."""
    result = {}
    for t in tickers_tuple:
        try:
            info  = yf.Ticker(t).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            if not price:
                continue
            result[t] = {
                "name":             info.get("shortName", t),
                "sector":           info.get("sector", ""),
                "industry":         info.get("industry", ""),
                "country":          info.get("country", ""),
                "currency":         info.get("currency", "USD"),
                "market_cap":       info.get("marketCap"),
                "pe_trailing":      info.get("trailingPE"),
                "pe_forward":       info.get("forwardPE"),
                "peg":              info.get("pegRatio"),
                "ev_ebitda":        info.get("enterpriseToEbitda"),
                "ev_revenue":       info.get("enterpriseToRevenue"),
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
                "beta":             info.get("beta"),
                "dividend_yield":   info.get("dividendYield"),
                "current_price":    float(price),
            }
            time.sleep(0.05)
        except Exception:
            pass
    return result


# ─── Earnings / Forecast Helpers ─────────────────────────────────────────────

def parse_earnings_dates(df: pd.DataFrame) -> list:
    """
    Parse earnings_dates DataFrame into a list of dicts.
    Returns: [{date, eps_estimate, eps_actual, surprise_pct, is_future}, ...]
    Sorted newest first.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    rows = []
    try:
        # Column names vary across yfinance versions
        col_est = next((c for c in df.columns if "estimate" in c.lower()), None)
        col_act = next((c for c in df.columns if "reported" in c.lower() or "actual" in c.lower() or "eps" in c.lower() and "estimate" not in c.lower()), None)
        col_sur = next((c for c in df.columns if "surprise" in c.lower() and "%" in c.lower()), None)

        import datetime
        today = pd.Timestamp.now(tz="UTC")

        for idx, row in df.iterrows():
            ts = idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx, errors="coerce")
            if pd.isna(ts):
                continue
            # Ensure timezone-aware comparison
            ts_aware = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            is_future = ts_aware > today

            eps_est  = _safe_float(row[col_est] if col_est else None)
            eps_act  = _safe_float(row[col_act] if col_act else None)
            surp     = _safe_float(row[col_sur] if col_sur else None)

            if surp is None and eps_est is not None and eps_act is not None and eps_est != 0:
                surp = (eps_act - eps_est) / abs(eps_est) * 100

            rows.append({
                "date":        ts.strftime("%Y-%m-%d"),
                "eps_estimate": eps_est,
                "eps_actual":   eps_act,
                "surprise_pct": surp,
                "is_future":    is_future,
            })
    except Exception:
        pass
    return sorted(rows, key=lambda x: x["date"], reverse=True)


def parse_calendar(cal: dict) -> dict:
    """Extract next earnings date and estimated EPS range from calendar dict."""
    if not cal:
        return {}
    result = {}
    try:
        # Yahoo Finance calendar format varies
        if "Earnings Date" in cal:
            ed = cal["Earnings Date"]
            if isinstance(ed, (list, tuple)):
                result["next_earnings_date"] = ed[0].strftime("%Y-%m-%d") if hasattr(ed[0], "strftime") else str(ed[0])
            else:
                result["next_earnings_date"] = str(ed)
        for k in ["Earnings High", "Earnings Low", "Earnings Average", "Revenue High", "Revenue Low", "Revenue Average"]:
            if k in cal:
                result[k.lower().replace(" ", "_")] = _safe_float(cal[k])
    except Exception:
        pass
    return result


def parse_estimates(earnings_est: pd.DataFrame, revenue_est: pd.DataFrame) -> dict:
    """Parse analyst estimate tables into a structured dict."""
    result = {}
    try:
        if earnings_est is not None and not earnings_est.empty:
            # Rows = periods (0q=current quarter, 1q=next quarter, 0y, 1y)
            for period in earnings_est.index[:4]:
                row = earnings_est.loc[period]
                result[f"eps_est_{period}"] = {
                    "avg":   _safe_float(row.get("avg")),
                    "low":   _safe_float(row.get("low")),
                    "high":  _safe_float(row.get("high")),
                    "count": _safe_float(row.get("numberOfAnalysts")),
                    "growth": _safe_float(row.get("growth")),
                }
    except Exception:
        pass
    try:
        if revenue_est is not None and not revenue_est.empty:
            for period in revenue_est.index[:4]:
                row = revenue_est.loc[period]
                result[f"rev_est_{period}"] = {
                    "avg":  _safe_float(row.get("avg")),
                    "low":  _safe_float(row.get("low")),
                    "high": _safe_float(row.get("high")),
                    "growth": _safe_float(row.get("growth")),
                }
    except Exception:
        pass
    return result


def parse_eps_trend(eps_trend: pd.DataFrame) -> dict:
    """Parse EPS revision trend — how estimates have changed over time."""
    result = {}
    try:
        if eps_trend is not None and not eps_trend.empty:
            for period in eps_trend.index[:2]:  # current & next quarter
                row = eps_trend.loc[period]
                result[f"eps_trend_{period}"] = {
                    "current":    _safe_float(row.get("current")),
                    "7daysAgo":   _safe_float(row.get("7daysAgo")),
                    "30daysAgo":  _safe_float(row.get("30daysAgo")),
                    "60daysAgo":  _safe_float(row.get("60daysAgo")),
                    "90daysAgo":  _safe_float(row.get("90daysAgo")),
                }
    except Exception:
        pass
    return result


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def get_current_price(info: dict) -> float:
    for k in ["currentPrice", "regularMarketPrice", "previousClose", "open"]:
        v = info.get(k)
        if v and float(v) > 0:
            return float(v)
    return 0.0


def safe_val(df: pd.DataFrame, row_names, col: int = 0):
    if df is None or df.empty:
        return None
    names = [row_names] if isinstance(row_names, str) else row_names
    for name in names:
        try:
            if name in df.index:
                v = df.loc[name].iloc[col]
                if pd.notna(v):
                    return float(v)
        except Exception:
            pass
    return None
