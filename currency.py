"""
currency.py — Multi-currency support.

Fetches live FX rates via:
  1. yfinance FX tickers (EURUSD=X etc.)
  2. open.er-api.com (free, no key)
  3. cdn.jsdelivr.net/@fawazahmed0 (free, no key)
  4. Hardcoded fallback rates (updated 2025)

All monetary values in the app are stored in their native currency.
The user selects a display currency in the sidebar.
"""
import yfinance as yf
import streamlit as st
import urllib.request
import json
import time

# ─── Supported currencies ──────────────────────────────────────────────────
CURRENCIES = {
    "USD": "🇺🇸 US Dollar",
    "EUR": "🇪🇺 Euro",
    "GBP": "🇬🇧 British Pound",
    "JPY": "🇯🇵 Japanese Yen",
    "KRW": "🇰🇷 South Korean Won",
    "HKD": "🇭🇰 Hong Kong Dollar",
    "CAD": "🇨🇦 Canadian Dollar",
    "CHF": "🇨🇭 Swiss Franc",
    "SEK": "🇸🇪 Swedish Krona",
    "NOK": "🇳🇴 Norwegian Krone",
    "DKK": "🇩🇰 Danish Krone",
    "AUD": "🇦🇺 Australian Dollar",
    "CNY": "🇨🇳 Chinese Yuan",
    "SGD": "🇸🇬 Singapore Dollar",
    "INR": "🇮🇳 Indian Rupee",
    "BRL": "🇧🇷 Brazilian Real",
}

# Fallback rates vs USD (approximate, updated 2025)
FALLBACK_RATES_TO_USD = {
    "USD": 1.000, "EUR": 1.090, "GBP": 1.270,
    "JPY": 0.0067, "KRW": 0.00073, "HKD": 0.128,
    "CAD": 0.740, "CHF": 1.120, "SEK": 0.096,
    "NOK": 0.094, "DKK": 0.146, "AUD": 0.650,
    "CNY": 0.138, "SGD": 0.745, "INR": 0.0120,
    "BRL": 0.200,
}

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "KRW": "₩",
    "HKD": "HK$", "CAD": "C$", "CHF": "Fr", "SEK": "kr", "NOK": "kr",
    "DKK": "kr", "AUD": "A$", "CNY": "¥", "SGD": "S$", "INR": "₹",
    "BRL": "R$",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fx_rates() -> dict:
    """
    Fetch all FX rates relative to USD.
    Returns dict: {currency_code: rate_vs_usd}
    e.g. {"EUR": 1.09, "GBP": 1.27, ...}
    """
    rates = dict(FALLBACK_RATES_TO_USD)  # start with fallbacks

    # Method 1: yfinance FX tickers
    yf_pairs = {
        "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "JPY": "JPYUSD=X",
        "KRW": "KRWUSD=X", "HKD": "HKDUSD=X", "CAD": "CADUSD=X",
        "CHF": "CHFUSD=X", "SEK": "SEKUSD=X", "NOK": "NOKUSD=X",
        "DKK": "DKKUSD=X", "AUD": "AUDUSD=X", "CNY": "CNYUSD=X",
        "SGD": "SGDUSD=X", "INR": "INRUSD=X", "BRL": "BRLUSD=X",
    }
    fetched = 0
    for ccy, ticker in yf_pairs.items():
        try:
            h = yf.Ticker(ticker).history(period="2d")
            if not h.empty:
                rates[ccy] = float(h["Close"].iloc[-1])
                fetched += 1
            time.sleep(0.1)
        except Exception:
            pass
    if fetched >= 5:
        return rates  # yfinance worked well enough

    # Method 2: open.er-api.com (free, no key)
    try:
        req = urllib.request.Request(
            "https://open.er-api.com/v6/latest/USD",
            headers={"User-Agent": "StockAnalyzer/1.0"}
        )
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        r = data.get("rates", {})
        for ccy in FALLBACK_RATES_TO_USD:
            if ccy != "USD" and ccy in r:
                rates[ccy] = 1.0 / float(r[ccy])  # API gives USD per foreign, we want foreign per USD inverse
        return rates
    except Exception:
        pass

    # Method 3: jsdelivr fawazahmed0 currency API (free)
    try:
        req = urllib.request.Request(
            "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
            headers={"User-Agent": "StockAnalyzer/1.0"}
        )
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        usd_rates = data.get("usd", {})
        for ccy in FALLBACK_RATES_TO_USD:
            code = ccy.lower()
            if code in usd_rates and usd_rates[code] > 0:
                rates[ccy] = 1.0 / float(usd_rates[code])
        return rates
    except Exception:
        pass

    return rates  # fallback rates


def convert(value: float, from_ccy: str, to_ccy: str, rates: dict) -> float:
    """Convert value from one currency to another using rates vs USD."""
    if value is None or from_ccy == to_ccy:
        return value
    try:
        # Convert to USD first, then to target
        rate_from = rates.get(from_ccy.upper(), 1.0)  # foreign → USD
        rate_to   = rates.get(to_ccy.upper(), 1.0)    # foreign → USD
        usd_val   = value * rate_from
        return usd_val / rate_to
    except Exception:
        return value


def fmt_currency(value, ccy: str, display_ccy: str, rates: dict,
                 dec: int = 2, abbreviate: bool = True) -> str:
    """Format a monetary value in the display currency."""
    import numpy as np
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    converted = convert(float(value), ccy, display_ccy, rates)
    if converted is None:
        return "—"
    sym = CURRENCY_SYMBOLS.get(display_ccy, display_ccy + " ")
    if abbreviate:
        if abs(converted) >= 1e12: return f"{sym}{converted/1e12:.2f}T"
        if abs(converted) >= 1e9:  return f"{sym}{converted/1e9:.2f}B"
        if abs(converted) >= 1e6:  return f"{sym}{converted/1e6:.2f}M"
        if abs(converted) >= 1e3:  return f"{sym}{converted/1e3:.2f}K"
    return f"{sym}{converted:,.{dec}f}"


def get_ticker_currency(info: dict) -> str:
    """Get the native currency of a ticker from yfinance info."""
    return (info.get("currency") or "USD").upper()


def sidebar_currency_selector() -> tuple[str, dict, float]:
    """
    Render currency selector in sidebar.
    Returns (selected_currency, fx_rates, manual_rate_override)
    manual_rate_override: if user typed a custom rate, use it; else 0.0
    """
    st.markdown("**Display Currency**")
    ccy_options = list(CURRENCIES.keys())
    ccy_labels  = [f"{k} — {v}" for k, v in CURRENCIES.items()]
    sel_idx = st.selectbox("Currency", options=range(len(ccy_options)),
                            format_func=lambda i: ccy_labels[i],
                            key="display_ccy", label_visibility="collapsed")
    selected = ccy_options[sel_idx]

    rates = fetch_fx_rates()

    # Show current rate vs USD
    if selected != "USD":
        rate = rates.get(selected, FALLBACK_RATES_TO_USD.get(selected, 1.0))
        usd_in_sel = 1.0 / rate if rate != 0 else 1.0
        st.caption(f"1 USD ≈ {usd_in_sel:.4f} {selected}  ·  live rate")

        # Manual override option
        with st.expander("✏️ Override exchange rate", expanded=False):
            st.caption(f"Current: 1 USD = {usd_in_sel:.4f} {selected}")
            manual = st.number_input(
                f"Custom rate (1 USD = ? {selected})",
                min_value=0.0001, max_value=100000.0,
                value=float(f"{usd_in_sel:.4f}"),
                step=0.0001, format="%.4f",
                key=f"manual_rate_{selected}"
            )
            use_manual = st.checkbox("Use this rate", key=f"use_manual_{selected}")
            if use_manual and manual > 0:
                rates[selected] = 1.0 / manual  # store as foreign→USD
                st.success(f"Using 1 USD = {manual:.4f} {selected}")

    return selected, rates


def apply_currency_to_info(info: dict, display_ccy: str, rates: dict) -> dict:
    """
    Return a copy of info with all monetary fields converted to display_ccy.
    Non-monetary fields (ratios, %s, counts) are untouched.
    """
    native = get_ticker_currency(info)
    if native == display_ccy:
        return info

    money_keys = [
        "currentPrice", "regularMarketPrice", "previousClose", "open",
        "dayHigh", "dayLow", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "marketCap", "enterpriseValue", "totalRevenue", "grossProfits",
        "ebitda", "freeCashflow", "totalCash", "totalDebt",
        "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice",
        "trailingEps", "forwardEps", "bookValue",
        "revenuePerShare", "earningsPerShare",
    ]
    result = dict(info)
    for k in money_keys:
        v = result.get(k)
        if v is not None:
            try:
                result[k] = convert(float(v), native, display_ccy, rates)
            except Exception:
                pass
    result["_display_currency"] = display_ccy
    result["_native_currency"]  = native
    return result
