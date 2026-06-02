"""
data.py - Data fetching from Yahoo Finance via yfinance.
Includes retry logic with exponential backoff to handle rate limiting.
Uses a simple TTL dict cache instead of st.cache_data to avoid
CacheReplayClosureError in Streamlit 1.57+.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import functools
import streamlit as st   # only used for @st.cache_data on peer metrics

# ─── Simple TTL cache (replaces st.cache_data for data fetching functions) ────
_CACHE: dict = {}

def _ttl_cache(key: str, ttl: int, fn):
    """Fetch from cache if fresh, else call fn() and store result.
    Only caches non-None results so a failure doesn't poison the cache."""
    now = time.time()
    if key in _CACHE:
        val, ts = _CACHE[key]
        if now - ts < ttl:
            return val
    result = fn()
    if result is not None:
        _CACHE[key] = (result, now)
    return result


# ─── Retry helper ─────────────────────────────────────────────────────────────

def _retry(fn, retries=4, base_delay=2.0, label=""):
    """
    Call fn() with exponential backoff on rate-limit / server errors.
    Delays: 2s, 4s, 8s, 16s + random jitter.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # Only retry on rate-limit or server errors
            if any(x in msg for x in ["429", "too many", "rate", "502", "503", "504", "timeout"]):
                wait = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)  # no st.toast here — can't use UI calls inside cached functions
            else:
                raise  # non-rate-limit error — don't retry
    raise last_err


def _safe_get(stock, attr):
    """Fetch a yfinance attribute, returning empty DataFrame on any error."""
    try:
        val = getattr(stock, attr)
        return val if val is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _safe_get_retry(stock, attr, ticker=""):
    """Fetch a yfinance attribute with retry on rate limiting."""
    try:
        return _retry(lambda: _safe_get(stock, attr), label=f"{ticker}/{attr}")
    except Exception:
        return pd.DataFrame()


# ─── Main ticker fetch ────────────────────────────────────────────────────────

def _fetch_ticker_data_impl(ticker: str) -> dict:
    """Fetch all relevant data for a single ticker, with rate-limit retries."""
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)

        # info is the most likely to be rate-limited
        info = _retry(lambda: stock.info, label=ticker)

        # Guard: yfinance can return None for info on some tickers
        if not info or not isinstance(info, dict):
            return {"ticker": ticker, "error":
                    f"No data returned for '{ticker}'. "
                    "The ticker may be delisted, or try adding the exchange suffix "
                    "(e.g. ASML.AS, SHEL.L, SAP.DE)."}

        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if not price or float(price) == 0:
            return {"ticker": ticker, "error":
                    f"Could not find price data for '{ticker}'. "
                    "Check the symbol — EU examples: ASML.AS, SHEL.L, SAP.DE, TTE.PA"}

        # Stagger requests to avoid hammering Yahoo Finance
        hist_2y = _retry(lambda: stock.history(period="2y"), label=f"{ticker}/hist_2y")
        hist_5y = _retry(lambda: stock.history(period="5y"), label=f"{ticker}/hist_5y")

        income_stmt   = _safe_get_retry(stock, "income_stmt",   ticker)
        balance_sheet = _safe_get_retry(stock, "balance_sheet", ticker)
        cash_flow     = _safe_get_retry(stock, "cashflow",      ticker)
        q_income      = _safe_get_retry(stock, "quarterly_income_stmt",   ticker)
        q_balance     = _safe_get_retry(stock, "quarterly_balance_sheet", ticker)
        q_cashflow    = _safe_get_retry(stock, "quarterly_cashflow",      ticker)

        recommendations  = _safe_get_retry(stock, "recommendations",   ticker)
        analyst_targets  = _safe_get_retry(stock, "analyst_price_targets", ticker)
        earnings_history = _safe_get_retry(stock, "earnings_history",  ticker)
        institutional    = _safe_get_retry(stock, "institutional_holders", ticker)

        # Earnings dates
        earnings_dates = pd.DataFrame()
        try:
            earnings_dates = _retry(lambda: stock.get_earnings_dates(limit=12), label=f"{ticker}/earn_dates")
        except Exception:
            try:
                earnings_dates = stock.earnings_dates or pd.DataFrame()
            except Exception:
                pass
        time.sleep(0.2)

        # Calendar
        calendar = {}
        try:
            calendar = _retry(lambda: stock.get_calendar() or {}, label=f"{ticker}/calendar")
        except Exception:
            pass
        time.sleep(0.2)

        # Analyst estimates
        earnings_estimate = pd.DataFrame()
        revenue_estimate  = pd.DataFrame()
        eps_trend         = pd.DataFrame()
        growth_estimates  = pd.DataFrame()
        for attr, var_name in [("get_earnings_estimate", "earnings_estimate"),
                                ("get_revenue_estimate",  "revenue_estimate"),
                                ("get_eps_trend",         "eps_trend"),
                                ("get_growth_estimates",  "growth_estimates")]:
            try:
                val = _retry(lambda a=attr: getattr(stock, a)() or pd.DataFrame(), label=f"{ticker}/{attr}")
                if attr == "get_earnings_estimate": earnings_estimate = val
                elif attr == "get_revenue_estimate":  revenue_estimate  = val
                elif attr == "get_eps_trend":         eps_trend         = val
                elif attr == "get_growth_estimates":  growth_estimates  = val
            except Exception:
                pass
            time.sleep(0.15)

        # News
        news_items = []
        try:
            news_items = _retry(lambda: yf.Ticker(ticker).get_news(count=15) or [], label=f"{ticker}/news")
        except Exception:
            pass

        return {
            "ticker": ticker, "info": info,
            "hist_2y": hist_2y, "hist_5y": hist_5y,
            "news": news_items,
            "income_stmt": income_stmt, "balance_sheet": balance_sheet, "cash_flow": cash_flow,
            "q_income": q_income, "q_balance": q_balance, "q_cashflow": q_cashflow,
            "recommendations": recommendations, "analyst_targets": analyst_targets,
            "earnings_history": earnings_history, "institutional": institutional,
            "earnings_dates": earnings_dates, "calendar": calendar,
            "earnings_estimate": earnings_estimate, "revenue_estimate": revenue_estimate,
            "eps_trend": eps_trend, "growth_estimates": growth_estimates,
            "error": None,
        }

    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ["429", "too many", "rate"]):
            return {"ticker": ticker, "error":
                    f"Yahoo Finance rate limit hit for '{ticker}'. "
                    "Please wait 30–60 seconds and try again. "
                    "Tip: analyse 1–2 tickers at a time to avoid this."}
        return {"ticker": ticker, "error": str(e)}


# ─── Risk-free rate ───────────────────────────────────────────────────────────

def fetch_ticker_data(ticker: str) -> dict:
    """Public cached entry point for fetch_ticker_data."""
    key = f"ticker:{ticker.upper().strip()}"
    return _ttl_cache(key, 3600, lambda: _fetch_ticker_data_impl(ticker))


def _fetch_risk_free_rate_impl() -> float:
    """Fetch 10-year US Treasury yield. Falls back to 4.5%."""
    try:
        hist = _retry(lambda: yf.Ticker("^TNX").history(period="5d"), label="^TNX")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.045


def fetch_risk_free_rate() -> float:
    """Cached: 10Y Treasury rate."""
    return _ttl_cache("rfr", 86400, _fetch_risk_free_rate_impl)


def _fetch_sp500_impl() -> pd.DataFrame:
    try:
        return _retry(lambda: yf.Ticker("^GSPC").history(period="2y"), label="^GSPC")
    except Exception:
        return pd.DataFrame()


# ─── Peer metrics ─────────────────────────────────────────────────────────────

def fetch_sp500() -> pd.DataFrame:
    """Cached: S&P 500 history."""
    return _ttl_cache("sp500", 3600, _fetch_sp500_impl)


@st.cache_data(ttl=3600)
def fetch_peer_metrics(tickers_tuple: tuple) -> dict:
    """
    Fetch valuation metrics for peer list.
    Capped at 6 peers, 0.3s stagger to keep it fast.
    """
    result = {}
    # Cap at 6 peers to bound the time (6 × ~1s = ~6s max)
    for i, t in enumerate(tickers_tuple[:6]):
        if i > 0:
            time.sleep(0.3)
        try:
            info = _retry(lambda ticker=t: yf.Ticker(ticker).info, label=t)
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
        except Exception:
            pass  # Skip peers that fail — don't let one bad ticker kill the rest
    return result


# ─── Earnings / Forecast Helpers ─────────────────────────────────────────────

def parse_earnings_dates(df: pd.DataFrame) -> list:
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    rows = []
    try:
        col_est = next((c for c in df.columns if "estimate" in c.lower()), None)
        col_act = next((c for c in df.columns
                        if ("reported" in c.lower() or "actual" in c.lower()
                            or ("eps" in c.lower() and "estimate" not in c.lower()))), None)
        col_sur = next((c for c in df.columns if "surprise" in c.lower() and "%" in c.lower()), None)

        today = pd.Timestamp.now(tz="UTC")
        for idx, row in df.iterrows():
            ts = idx if isinstance(idx, pd.Timestamp) else pd.to_datetime(idx, errors="coerce")
            if pd.isna(ts):
                continue
            ts_aware  = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            is_future = ts_aware > today
            eps_est   = _safe_float(row[col_est] if col_est else None)
            eps_act   = _safe_float(row[col_act] if col_act else None)
            surp      = _safe_float(row[col_sur] if col_sur else None)
            if surp is None and eps_est and eps_act and eps_est != 0:
                surp = (eps_act - eps_est) / abs(eps_est) * 100
            rows.append({"date": ts.strftime("%Y-%m-%d"), "eps_estimate": eps_est,
                         "eps_actual": eps_act, "surprise_pct": surp, "is_future": is_future})
    except Exception:
        pass
    return sorted(rows, key=lambda x: x["date"], reverse=True)


def parse_calendar(cal: dict) -> dict:
    if not cal:
        return {}
    result = {}
    try:
        if "Earnings Date" in cal:
            ed = cal["Earnings Date"]
            if isinstance(ed, (list, tuple)):
                result["next_earnings_date"] = ed[0].strftime("%Y-%m-%d") if hasattr(ed[0], "strftime") else str(ed[0])
            else:
                result["next_earnings_date"] = str(ed)
        for k in ["Earnings High","Earnings Low","Earnings Average","Revenue High","Revenue Low","Revenue Average"]:
            if k in cal:
                result[k.lower().replace(" ", "_")] = _safe_float(cal[k])
    except Exception:
        pass
    return result


def parse_estimates(earnings_est: pd.DataFrame, revenue_est: pd.DataFrame) -> dict:
    result = {}
    try:
        if earnings_est is not None and not earnings_est.empty:
            for period in earnings_est.index[:4]:
                row = earnings_est.loc[period]
                result[f"eps_est_{period}"] = {
                    "avg": _safe_float(row.get("avg")), "low": _safe_float(row.get("low")),
                    "high": _safe_float(row.get("high")), "count": _safe_float(row.get("numberOfAnalysts")),
                    "growth": _safe_float(row.get("growth")),
                }
    except Exception:
        pass
    try:
        if revenue_est is not None and not revenue_est.empty:
            for period in revenue_est.index[:4]:
                row = revenue_est.loc[period]
                result[f"rev_est_{period}"] = {
                    "avg": _safe_float(row.get("avg")), "low": _safe_float(row.get("low")),
                    "high": _safe_float(row.get("high")), "growth": _safe_float(row.get("growth")),
                }
    except Exception:
        pass
    return result


def parse_eps_trend(eps_trend: pd.DataFrame) -> dict:
    result = {}
    try:
        if eps_trend is not None and not eps_trend.empty:
            for period in eps_trend.index[:2]:
                row = eps_trend.loc[period]
                result[f"eps_trend_{period}"] = {
                    "current":   _safe_float(row.get("current")),
                    "7daysAgo":  _safe_float(row.get("7daysAgo")),
                    "30daysAgo": _safe_float(row.get("30daysAgo")),
                    "60daysAgo": _safe_float(row.get("60daysAgo")),
                    "90daysAgo": _safe_float(row.get("90daysAgo")),
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