"""
yf_health.py — yfinance API health check.

Run at app startup to detect yfinance breaking changes immediately.
Validates that all critical fields are still available in the current
yfinance version. If something changed, shows a clear warning with
the exact fields that are missing.

Usage in app.py:
    from yf_health import run_health_check
    run_health_check()   # call once at top of main()
"""
import streamlit as st
import yfinance as yf


# ─── Field contract ───────────────────────────────────────────────────────────
# These are the fields our app depends on. If any go missing in a new
# yfinance version, the health check flags it immediately.

# Fields expected in stock.info (fundamentals — not price)
INFO_FIELDS_REQUIRED = [
    "trailingPE", "forwardPE", "pegRatio",
    "enterpriseToEbitda", "enterpriseToRevenue",
    "priceToBook", "priceToSalesTrailingTwelveMonths",
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnEquity", "returnOnAssets",
    "revenueGrowth", "earningsGrowth",
    "debtToEquity", "currentRatio",
    "trailingEps", "forwardEps",
    "ebitda", "totalRevenue",
    "recommendationMean", "targetMeanPrice",
    "sharesOutstanding",
]

# Fields expected in stock.fast_info (price data — moved here in yfinance 1.4.0)
FAST_INFO_FIELDS_REQUIRED = [
    "last_price",
    "market_cap",
    "year_high",
    "year_low",
    "currency",
    "exchange",
]

# Attributes expected on the Ticker object itself
TICKER_ATTRS_REQUIRED = [
    "info", "fast_info", "history",
    "income_stmt", "balance_sheet", "cashflow",
    "recommendations", "earnings_dates",
    "get_earnings_estimate", "get_revenue_estimate",
    "get_news",
]

# Test ticker (large liquid stock — unlikely to disappear)
TEST_TICKER = "MSFT"


@st.cache_data(ttl=86400, show_spinner=False)  # check once per day
def _run_checks() -> dict:
    """Perform all health checks. Cached for 24h to avoid slowing every load."""
    results = {
        "yfinance_version": yf.__version__,
        "ticker_attrs":   {"missing": [], "ok": []},
        "fast_info":      {"missing": [], "ok": []},
        "info_fields":    {"missing": [], "ok": [], "note": ""},
        "passed":         True,
        "warnings":       [],
    }

    try:
        t = yf.Ticker(TEST_TICKER)

        # Check 1: Ticker attributes
        for attr in TICKER_ATTRS_REQUIRED:
            if hasattr(t, attr):
                results["ticker_attrs"]["ok"].append(attr)
            else:
                results["ticker_attrs"]["missing"].append(attr)
                results["passed"] = False

        # Check 2: fast_info fields (critical for price)
        try:
            fi = t.fast_info
            for field in FAST_INFO_FIELDS_REQUIRED:
                try:
                    val = getattr(fi, field, None)
                    if val is not None:
                        results["fast_info"]["ok"].append(field)
                    else:
                        results["fast_info"]["missing"].append(field)
                        results["passed"] = False
                except Exception:
                    results["fast_info"]["missing"].append(field)
                    results["passed"] = False
        except Exception as e:
            results["warnings"].append(f"fast_info unavailable: {e}")
            results["passed"] = False

        # Check 3: info dict fields (fundamentals)
        # Note: yfinance 1.4.0 removed price fields from info — that's expected.
        # We only check fundamental/ratio fields here.
        try:
            info = t.info or {}
            available = set(info.keys())
            for field in INFO_FIELDS_REQUIRED:
                if field in available:
                    results["info_fields"]["ok"].append(field)
                else:
                    results["info_fields"]["missing"].append(field)
            if results["info_fields"]["missing"]:
                results["info_fields"]["note"] = (
                    f"Note: {len(results['info_fields']['missing'])} fundamental fields "
                    "missing from info — may indicate a yfinance API change."
                )
        except Exception as e:
            results["warnings"].append(f"info dict check failed: {e}")

    except Exception as e:
        results["passed"] = False
        results["warnings"].append(f"Health check failed entirely: {e}")

    return results


def run_health_check(show_if_ok: bool = False):
    """
    Run the health check and display a warning banner if anything changed.
    Call once at the top of main().

    Args:
        show_if_ok: If True, also show a green banner when everything passes.
    """
    try:
        r = _run_checks()
    except Exception:
        return  # Don't let health check crash the app

    version = r.get("yfinance_version", "?")
    passed  = r.get("passed", True)

    if not passed or r.get("warnings"):
        missing_fast  = r["fast_info"]["missing"]
        missing_info  = r["info_fields"]["missing"]
        missing_attrs = r["ticker_attrs"]["missing"]

        msg_parts = [f"⚠️ **yfinance {version} API health check detected issues:**"]

        if missing_attrs:
            msg_parts.append(f"• Ticker attributes missing: `{'`, `'.join(missing_attrs)}`")
        if missing_fast:
            msg_parts.append(
                f"• `fast_info` fields missing: `{'`, `'.join(missing_fast)}`  "
                "— price data may be unavailable"
            )
        if missing_info and len(missing_info) > 3:
            # Only warn if many fields missing (a few missing is normal)
            msg_parts.append(
                f"• {len(missing_info)} fundamental fields missing from `info` dict  "
                f"(e.g. `{missing_info[0]}`, `{missing_info[1]}`)"
            )
        for w in r.get("warnings", []):
            msg_parts.append(f"• {w}")

        msg_parts.append(
            f"\n**Action:** Check [yfinance releases](https://github.com/ranaroussi/yfinance/releases) "
            "for breaking changes, then update `data.py` field mappings."
        )

        st.warning("\n".join(msg_parts))

    elif show_if_ok:
        st.success(f"✅ yfinance {version} — all API fields verified")
