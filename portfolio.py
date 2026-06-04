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
    Parse Yahoo Finance portfolio CSV.
    Returns a clean DataFrame with one row per symbol (net position).
    Handles multiple buy/sell transactions per symbol.
    """
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(file_content))
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
        elif cl in ("tradedate","date"):             col_map[c] = "Trade Date"
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
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

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
