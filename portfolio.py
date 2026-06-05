"""
portfolio.py — Portfolio analysis from Yahoo Finance CSV export.

CSV columns expected:
    Symbol, Current Price, Date, Time, Change, Open, High, Low, Volume,
    Trade Date, Purchase Price, Quantity, Commission, High Limit, Low Limit,
    Comment, Transaction Type

Process:
  1. Parse CSV → group transactions by symbol (buys/sells → net position + avg cost)
  2. Fetch live price + metadata via yfinance fast_info + info
  3. Compute per-position metrics
  4. Compute portfolio-level metrics (beta, volatility, concentration, VaR)
  5. Build allocation breakdowns (type, sector, geography)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
import random


# ─── CSV Parser ───────────────────────────────────────────────────────────────

def parse_yahoo_csv(file_content: str) -> pd.DataFrame:
    """
    Parse Yahoo Finance portfolio CSV — handles duplicate column names.
    Yahoo Finance CSVs often have two columns both named 'Date'.
    We deduplicate headers before handing to pandas.
    """
    from io import StringIO
    import csv as _csv

    lines = [l for l in file_content.splitlines() if l.strip()]
    if not lines:
        raise ValueError("CSV file is empty")

    # Parse the header row and deduplicate
    try:
        raw_headers = list(next(_csv.reader([lines[0]])))
    except Exception:
        raw_headers = lines[0].split(",")
    raw_headers = [h.strip() for h in raw_headers]

    seen = {}
    clean_headers = []
    for h in raw_headers:
        if h not in seen:
            seen[h] = 1
            clean_headers.append(h)
        else:
            seen[h] += 1
            clean_headers.append(h + "_" + str(seen[h]))

    newline = chr(10)
    clean_csv = ",".join(clean_headers) + newline + newline.join(lines[1:])

    try:
        df = pd.read_csv(StringIO(clean_csv), on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(StringIO(clean_csv))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    # Normalise column names (strip spaces, title-case)
    df.columns = [c.strip() for c in df.columns]

    # Map known column name variants
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "")
        if cl == "symbol":                           col_map[c] = "Symbol"
        elif cl in ("purchaseprice","buyprice"):     col_map[c] = "Purchase Price"
        elif cl == "quantity":                       col_map[c] = "Quantity"
        elif cl == "tradedate":                      col_map[c] = "Trade Date"
        elif cl == "date" and "Trade Date" not in col_map.values(): col_map[c] = "Trade Date"
        elif cl == "commission":                     col_map[c] = "Commission"
        elif cl in ("transactiontype","type","txntype"): col_map[c] = "Transaction Type"
        elif cl == "currentprice":                   col_map[c] = "Current Price"
        elif cl == "comment":                        col_map[c] = "Comment"
    df = df.rename(columns=col_map)

    # Drop rows with missing symbol
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].str.strip().str.upper()

    # Parse numeric columns safely
    for col in ["Purchase Price","Quantity","Commission","Current Price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse Trade Date
    if "Trade Date" in df.columns:
        col = df["Trade Date"]
        # If rename created a DataFrame (2 cols), take the first one
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        df["Trade Date"] = pd.to_datetime(col, errors="coerce")

    # Transaction Type default = Buy
    if "Transaction Type" not in df.columns:
        df["Transaction Type"] = "Buy"
    df["Transaction Type"] = df["Transaction Type"].fillna("Buy").str.strip().str.title()

    # Commission default = 0
    if "Commission" not in df.columns:
        df["Commission"] = 0.0
    df["Commission"] = df["Commission"].fillna(0.0)

    # ── Aggregate to net positions per symbol ─────────────────────────────────
    positions = []
    for symbol, grp in df.groupby("Symbol"):
        buys  = grp[grp["Transaction Type"].str.lower().isin(["buy","b"])]
        sells = grp[grp["Transaction Type"].str.lower().isin(["sell","s"])]

        buy_qty   = buys["Quantity"].sum()   if not buys.empty  else 0
        sell_qty  = sells["Quantity"].sum()  if not sells.empty else 0
        net_qty   = buy_qty - sell_qty

        if net_qty <= 0:
            continue   # fully sold out — skip

        # Weighted average cost basis
        if "Purchase Price" in grp.columns and buys["Purchase Price"].notna().any():
            prices    = buys["Purchase Price"].fillna(0)
            quantities = buys["Quantity"].fillna(0)
            total_cost = (prices * quantities).sum()
            avg_cost   = total_cost / buy_qty if buy_qty > 0 else 0
        else:
            avg_cost = 0

        total_commission = grp["Commission"].sum()

        # First/last trade date
        trade_dates = grp["Trade Date"].dropna()
        first_trade = trade_dates.min() if not trade_dates.empty else None
        last_trade  = trade_dates.max() if not trade_dates.empty else None

        # Current price from CSV (will be refreshed from live feed)
        csv_price = grp["Current Price"].dropna().iloc[-1] if (
            "Current Price" in grp.columns and grp["Current Price"].notna().any()) else None

        positions.append({
            "symbol":            symbol,
            "quantity":          net_qty,
            "avg_cost":          avg_cost,
            "total_commission":  total_commission,
            "cost_basis":        avg_cost * net_qty + total_commission,
            "first_trade_date":  first_trade,
            "last_trade_date":   last_trade,
            "csv_price":         csv_price,
            "comment":           grp["Comment"].dropna().iloc[-1] if "Comment" in grp.columns and grp["Comment"].notna().any() else "",
        })

    return pd.DataFrame(positions)


# ─── Live Data Enrichment ─────────────────────────────────────────────────────

def _safe_float(v, default=None):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _retry(fn, retries=3, base=1.5):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            if any(x in str(e).lower() for x in ["429","rate","timeout","503"]):
                time.sleep(base * (2 ** attempt) + random.uniform(0.2, 0.8))
            else:
                raise
    raise last


def fetch_position_data(symbol: str) -> dict:
    """
    Fetch live price, metadata, and asset-class-specific fields for one symbol.
    Handles stocks, ETFs, and indices.
    """
    try:
        t    = yf.Ticker(symbol)
        fi   = t.fast_info
        info = {}
        try:
            raw = _retry(lambda: t.info)
            if isinstance(raw, dict):
                info = raw
        except Exception:
            pass

        # Price from fast_info (most reliable in yfinance 1.4.0)
        price = None
        for py_attr in ("last_price","previous_close"):
            try:
                v = getattr(fi, py_attr)
                if v and float(v) > 0:
                    price = float(v); break
            except Exception:
                pass
        if not price:
            for dict_key in ("lastPrice","previousClose","regularMarketPreviousClose"):
                try:
                    v = fi.get(dict_key)
                    if v and float(v) > 0:
                        price = float(v); break
                except Exception:
                    pass
        if not price:
            price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

        # Market cap
        mktcap = None
        try: mktcap = _safe_float(getattr(fi, "market_cap"))
        except Exception: pass
        if not mktcap:
            try: mktcap = _safe_float(fi.get("marketCap") or info.get("marketCap"))
            except Exception: pass

        # Currency
        currency = "USD"
        try: currency = str(getattr(fi, "currency") or fi.get("currency") or info.get("currency") or "USD")
        except Exception: pass

        # Quote type
        quote_type = str(info.get("quoteType","EQUITY")).upper()

        result = {
            "symbol":      symbol,
            "name":        info.get("shortName") or info.get("longName") or symbol,
            "price":       price or 0,
            "currency":    currency,
            "quote_type":  quote_type,
            "sector":      info.get("sector",""),
            "industry":    info.get("industry",""),
            "country":     info.get("country",""),
            "exchange":    info.get("exchange",""),
            "beta":        _safe_float(info.get("beta")),
            "market_cap":  mktcap,
            "52w_high":    _safe_float(info.get("fiftyTwoWeekHigh") or (fi.get("yearHigh") if hasattr(fi,"get") else None)),
            "52w_low":     _safe_float(info.get("fiftyTwoWeekLow")  or (fi.get("yearLow")  if hasattr(fi,"get") else None)),
            "pe_trailing": _safe_float(info.get("trailingPE")),
            "pe_forward":  _safe_float(info.get("forwardPE")),
            "div_yield":   _safe_float(info.get("dividendYield")),
            "year_change": _safe_float(info.get("52WeekChange") or (fi.get("yearChange") if hasattr(fi,"get") else None)),
        }

        # ── ETF-specific fields ──────────────────────────────────────────────
        if quote_type == "ETF":
            result.update({
                "etf_expense_ratio":  _safe_float(info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")),
                "etf_total_assets":   _safe_float(info.get("totalAssets")),
                "etf_fund_family":    info.get("fundFamily",""),
                "etf_inception_date": info.get("fundInceptionDate",""),
                "etf_category":       info.get("category",""),
                "etf_legal_type":     info.get("legalType",""),
                "etf_yield":          _safe_float(info.get("yield")),
                "etf_ytd_return":     _safe_float(info.get("ytdReturn")),
                "etf_3yr_return":     _safe_float(info.get("threeYearAverageReturn")),
                "etf_5yr_return":     _safe_float(info.get("fiveYearAverageReturn")),
                "etf_nav":            _safe_float(info.get("navPrice")),
            })
        return result

    except Exception as e:
        return {"symbol": symbol, "name": symbol, "price": 0, "error": str(e),
                "quote_type": "EQUITY", "currency": "USD", "sector": "", "country": ""}


def enrich_portfolio(positions_df: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
    """
    Fetch live data for all positions and compute per-position metrics.
    """
    rows = []
    total = len(positions_df)

    for i, row in positions_df.iterrows():
        symbol = row["symbol"]
        if progress_cb:
            progress_cb(i, total, symbol)
        if i > 0:
            time.sleep(0.4)

        live = fetch_position_data(symbol)

        price    = live.get("price") or row.get("csv_price") or 0
        qty      = row["quantity"]
        avg_cost = row["avg_cost"]
        cost     = row["cost_basis"]

        market_val  = price * qty
        unrealized  = market_val - cost
        pnl_pct     = (unrealized / cost * 100) if cost > 0 else 0

        merged = {**row.to_dict(), **live,
                  "market_value":    market_val,
                  "unrealized_pnl":  unrealized,
                  "pnl_pct":         pnl_pct,
                  "cost_basis":      cost,
                  }
        rows.append(merged)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Portfolio weight
    total_val = df["market_value"].sum()
    df["weight_pct"] = (df["market_value"] / total_val * 100).round(2) if total_val > 0 else 0

    return df


# ─── Portfolio-Level Metrics ─────────────────────────────────────────────────

def calc_portfolio_metrics(enriched: pd.DataFrame) -> dict:
    """
    Compute aggregate portfolio metrics.
    """
    if enriched.empty:
        return {}

    total_cost  = enriched["cost_basis"].sum()
    total_val   = enriched["market_value"].sum()
    total_pnl   = total_val - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
    n_positions = len(enriched)

    # Weighted portfolio beta
    weights = enriched["market_value"] / total_val
    betas   = enriched["beta"].fillna(1.0)
    port_beta = float((weights * betas).sum())

    # Concentration (Herfindahl-Hirschman Index)
    w_sq = (enriched["weight_pct"] / 100) ** 2
    hhi  = float(w_sq.sum())   # 0 = perfectly diversified, 1 = one position
    top5_pct = float(enriched.nlargest(min(5, n_positions), "market_value")["weight_pct"].sum())

    # Best / worst performers
    best  = enriched.nlargest(3,  "pnl_pct")[["symbol","pnl_pct","unrealized_pnl"]]
    worst = enriched.nsmallest(3, "pnl_pct")[["symbol","pnl_pct","unrealized_pnl"]]

    # Asset type breakdown
    type_alloc = (enriched.groupby("quote_type")["market_value"]
                  .sum().sort_values(ascending=False))
    type_alloc_pct = (type_alloc / total_val * 100).round(1)

    # Sector breakdown (stocks only)
    stocks = enriched[enriched["quote_type"] == "EQUITY"]
    sector_alloc = {}
    if not stocks.empty and "sector" in stocks.columns:
        sa = (stocks.groupby("sector")["market_value"]
              .sum().sort_values(ascending=False))
        sector_alloc = (sa / total_val * 100).round(1).to_dict()

    # Geography
    geo_alloc = {}
    if "country" in enriched.columns:
        ga = (enriched.groupby("country")["market_value"]
              .sum().sort_values(ascending=False))
        geo_alloc = (ga / total_val * 100).round(1).to_dict()

    # Currency breakdown
    ccy_alloc = {}
    if "currency" in enriched.columns:
        ca = (enriched.groupby("currency")["market_value"]
              .sum().sort_values(ascending=False))
        ccy_alloc = (ca / total_val * 100).round(1).to_dict()

    # Simple VaR estimate: assume normal distribution, 1-day 95%
    # Use weighted avg of individual 52w returns as proxy volatility
    year_changes = enriched["year_change"].dropna()
    if len(year_changes) > 0:
        port_vol_ann = float((weights * enriched["year_change"].fillna(0)).std() * np.sqrt(252) + 0.18)
        daily_vol    = port_vol_ann / np.sqrt(252)
        var_95_daily = total_val * daily_vol * 1.645
    else:
        port_vol_ann = None
        var_95_daily = None

    # Dividend income (annualised estimate)
    div_income = 0
    for _, pos in enriched.iterrows():
        dy = pos.get("div_yield") or 0
        if dy and dy > 0:
            div_income += pos["market_value"] * dy

    return {
        "total_cost":     total_cost,
        "total_value":    total_val,
        "total_pnl":      total_pnl,
        "total_pnl_pct":  total_pnl_pct,
        "n_positions":    n_positions,
        "portfolio_beta": port_beta,
        "hhi":            hhi,
        "top5_pct":       top5_pct,
        "best":           best.to_dict("records"),
        "worst":          worst.to_dict("records"),
        "type_alloc":     type_alloc_pct.to_dict(),
        "sector_alloc":   sector_alloc,
        "geo_alloc":      geo_alloc,
        "ccy_alloc":      ccy_alloc,
        "port_vol_ann":   port_vol_ann,
        "var_95_daily":   var_95_daily,
        "div_income_ann": div_income,
    }

# ─── Benchmark tickers ────────────────────────────────────────────────────────

BENCHMARKS = {
    "S&P 500":       "^GSPC",
    "Dow Jones":     "^DJI",
    "Nasdaq 100":    "^NDX",
    "FTSE 100":      "^FTSE",
    "DAX 40":        "^GDAXI",
    "AEX 25":        "^AEX",
    "CAC 40":        "^FCHI",
    "Euro Stoxx 50": "^STOXX50E",
    "Nikkei 225":    "^N225",
    "Hang Seng":     "^HSI",
    "ASX 200":       "^AXJO",
    "SMI (Swiss)":   "^SSMI",
    "IBEX 35":       "^IBEX",
    "Kospi":         "^KS11",
}

TIMEFRAMES = {
    "1W":  7,
    "1M":  30,
    "3M":  90,
    "6M":  180,
    "YTD": None,   # special case
    "1Y":  365,
    "2Y":  730,
    "3Y":  1095,
}


# ─── Historical price fetching ────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "1y") -> pd.Series:
    """Fetch closing price history for a ticker."""
    try:
        hist = _retry(lambda: yf.Ticker(ticker).history(period=period))
        if hist is not None and not hist.empty:
            return hist["Close"].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def fetch_portfolio_history(enriched: pd.DataFrame,
                             period: str = "1y") -> pd.Series:
    """
    Build portfolio value history from weighted position price histories.
    Each position contributes: (current_shares × historical_price).
    Rebased so the earliest available date = sum of cost bases.
    """
    if enriched.empty:
        return pd.Series(dtype=float)

    all_series = {}
    for _, pos in enriched.iterrows():
        sym = pos.get("symbol","")
        qty = pos.get("quantity", 0)
        if qty <= 0:
            continue
        hist = fetch_price_history(sym, period=period)
        if not hist.empty:
            all_series[sym] = hist * qty
        time.sleep(0.2)

    if not all_series:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_series)
    df = df.sort_index().ffill().dropna(how="all")
    portfolio_series = df.sum(axis=1)
    return portfolio_series


# ─── Advanced portfolio metrics ───────────────────────────────────────────────

def calc_advanced_metrics(enriched: pd.DataFrame, rfr: float = 0.045) -> dict:
    """
    Calculate Sharpe Ratio, Alpha, Beta (from returns), Max Drawdown.
    Uses 1Y of price history for each position + S&P 500 as benchmark.
    rfr: annual risk-free rate (default 4.5% = current approx US 10Y)
    """
    if enriched.empty:
        return {}

    results = {}

    # ── Portfolio return history ───────────────────────────────────────────
    port_hist = fetch_portfolio_history(enriched, period="1y")
    if port_hist.empty or len(port_hist) < 20:
        return {"error": "Insufficient price history for advanced metrics"}

    port_returns = port_hist.pct_change().dropna()

    # ── Benchmark (S&P 500) returns ────────────────────────────────────────
    bench_hist = fetch_price_history("^GSPC", period="1y")
    bench_returns = pd.Series(dtype=float)
    if not bench_hist.empty:
        bench_returns = bench_hist.pct_change().dropna()
        # Align to same dates
        common = port_returns.index.intersection(bench_returns.index)
        if len(common) > 20:
            port_r  = port_returns.loc[common]
            bench_r = bench_returns.loc[common]
        else:
            port_r  = port_returns
            bench_r = bench_returns
    else:
        port_r = port_returns

    n_days = len(port_r)

    # ── Annualised portfolio return ────────────────────────────────────────
    total_ret   = (port_hist.iloc[-1] / port_hist.iloc[0]) - 1
    ann_ret     = (1 + total_ret) ** (252 / max(n_days, 1)) - 1
    daily_vol   = float(port_r.std())
    ann_vol     = daily_vol * np.sqrt(252)

    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    daily_rfr  = rfr / 252
    excess_ret = port_r - daily_rfr
    sharpe     = float(excess_ret.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else None

    # ── Alpha & Beta (vs S&P 500) ─────────────────────────────────────────
    alpha = beta_calc = None
    if len(bench_returns) > 20 and len(port_r) > 20:
        try:
            cov_matrix  = np.cov(port_r.values, bench_r.values)
            bench_var   = float(bench_r.var())
            beta_calc   = float(cov_matrix[0][1] / bench_var) if bench_var > 0 else None
            bench_ann   = float((1 + bench_r.mean()) ** 252 - 1)
            if beta_calc is not None:
                alpha = float(ann_ret - (rfr + beta_calc * (bench_ann - rfr)))
        except Exception:
            pass

    # ── Max Drawdown ──────────────────────────────────────────────────────
    rolling_max = port_hist.cummax()
    drawdown    = (port_hist - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())
    dd_end      = drawdown.idxmin()
    dd_start    = port_hist[:dd_end].idxmax() if not port_hist[:dd_end].empty else None

    # ── Sortino Ratio (downside only) ─────────────────────────────────────
    neg_returns    = port_r[port_r < daily_rfr]
    downside_vol   = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 2 else ann_vol
    sortino        = float((ann_ret - rfr) / downside_vol) if downside_vol > 0 else None

    # ── Calmar Ratio ──────────────────────────────────────────────────────
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else None

    results = {
        "ann_return":      ann_ret,
        "ann_volatility":  ann_vol,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "alpha":           alpha,
        "beta_calc":       beta_calc,
        "max_drawdown":    max_dd,
        "max_dd_start":    dd_start,
        "max_dd_end":      dd_end,
        "calmar":          calmar,
        "rfr":             rfr,
        "n_days":          n_days,
        "port_hist":       port_hist,
        "port_returns":    port_r,
    }
    return results


def fetch_benchmark_comparison(enriched: pd.DataFrame,
                                benchmark_ticker: str,
                                days: int | None = None) -> dict:
    """
    Fetch portfolio and benchmark history for comparison chart.
    Both series rebased to 100 at the start date.
    days=None means YTD.
    """
    period = "1y"
    if days and days <= 7:    period = "5d"
    elif days and days <= 30: period = "1mo"
    elif days and days <= 90: period = "3mo"
    elif days and days <= 180:period = "6mo"
    elif days and days <= 365:period = "1y"
    elif days and days <= 730:period = "2y"
    else:                     period = "3y"

    port_hist  = fetch_portfolio_history(enriched, period=period)
    bench_hist = fetch_price_history(benchmark_ticker, period=period)

    if port_hist.empty:
        return {"error": "No portfolio history available"}

    # Apply date filter
    if days is not None:
        cutoff = pd.Timestamp.now(tz=port_hist.index.tz) - pd.Timedelta(days=days)
        port_hist  = port_hist[port_hist.index >= cutoff]
        if not bench_hist.empty:
            bench_hist = bench_hist[bench_hist.index >= cutoff]
    else:
        # YTD
        ytd_start = pd.Timestamp(pd.Timestamp.now().year, 1, 1)
        if port_hist.index.tz:
            ytd_start = ytd_start.tz_localize(port_hist.index.tz)
        port_hist  = port_hist[port_hist.index >= ytd_start]
        if not bench_hist.empty:
            if bench_hist.index.tz:
                ytd_start2 = ytd_start.tz_localize(bench_hist.index.tz)
            else:
                ytd_start2 = ytd_start.tz_localize(None)
            bench_hist = bench_hist[bench_hist.index >= ytd_start2]

    if port_hist.empty:
        return {"error": "No data for selected timeframe"}

    # Rebase to 100
    port_rebased  = port_hist  / port_hist.iloc[0]  * 100
    bench_rebased = bench_hist / bench_hist.iloc[0] * 100 if not bench_hist.empty else pd.Series()

    return {
        "portfolio":  port_rebased,
        "benchmark":  bench_rebased,
        "port_raw":   port_hist,
        "bench_raw":  bench_hist,
    }