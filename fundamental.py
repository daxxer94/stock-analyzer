"""
fundamental.py - Financial statement analysis + DCF models.

DCF Models implemented:
  1. WACC-based DCF
  2. CAPM-based DCF
  3. Fixed-rate DCF (user supplied)
  4. Two-Stage FCF DCF
"""
import pandas as pd
import numpy as np
from data import safe_val


# ─── Financial Statement Helpers ────────────────────────────────────────────

def get_income_series(income: pd.DataFrame) -> dict:
    """Extract key income statement items across all available years."""
    if income is None or income.empty:
        return {}

    def get_series(row_names):
        vals = {}
        for i, col in enumerate(income.columns):
            v = safe_val(income, row_names, i)
            if v is not None:
                vals[col.year if hasattr(col, "year") else i] = v
        return vals

    return {
        "revenue":          get_series(["Total Revenue","Revenue"]),
        "gross_profit":     get_series(["Gross Profit"]),
        "operating_income": get_series(["Operating Income","Ebit","EBIT"]),
        "net_income":       get_series(["Net Income","Net Income Common Stockholders"]),
        "ebitda":           get_series(["EBITDA","Ebitda","Normalized EBITDA"]),
        "interest_expense": get_series(["Interest Expense","Interest Expense Non Operating"]),
        "tax_provision":    get_series(["Tax Provision","Income Tax Expense"]),
        "pretax_income":    get_series(["Pretax Income","Income Before Tax"]),
        "rd_expense":       get_series(["Research And Development","Research Development"]),
        "eps_basic":        get_series(["Basic EPS","Basic Eps"]),
        "eps_diluted":      get_series(["Diluted EPS","Diluted Eps"]),
    }


def get_balance_series(balance: pd.DataFrame) -> dict:
    if balance is None or balance.empty:
        return {}

    def get_latest(row_names):
        return safe_val(balance, row_names, 0)

    return {
        "total_assets":     get_latest(["Total Assets"]),
        "total_debt":       get_latest(["Total Debt","Long Term Debt And Capital Lease Obligation"]),
        "long_term_debt":   get_latest(["Long Term Debt"]),
        "total_equity":     get_latest(["Stockholders Equity","Common Stock Equity","Total Equity Gross Minority Interest"]),
        "cash":             get_latest(["Cash And Cash Equivalents","Cash Cash Equivalents And Short Term Investments"]),
        "current_assets":   get_latest(["Current Assets","Total Current Assets"]),
        "current_liab":     get_latest(["Current Liabilities","Total Current Liabilities Net Minority Interest"]),
        "total_liab":       get_latest(["Total Liabilities Net Minority Interest","Total Liabilities"]),
        "goodwill":         get_latest(["Goodwill","Goodwill And Other Intangible Assets"]),
        "inventory":        get_latest(["Inventory"]),
        "receivables":      get_latest(["Accounts Receivable","Receivables"]),
    }


def get_cashflow_list(cashflow: pd.DataFrame) -> dict:
    """Return lists of cash flow values (most recent first)."""
    if cashflow is None or cashflow.empty:
        return {}

    def series(row_names):
        vals = []
        for i in range(min(4, len(cashflow.columns))):
            v = safe_val(cashflow, row_names, i)
            vals.append(v)
        return [v for v in vals if v is not None]

    ocf   = series(["Operating Cash Flow","Cash From Operations","Total Cash From Operating Activities"])
    capex = series(["Capital Expenditure","Purchases Of Property Plant And Equipment","Capital Expenditures"])
    fcf   = [ocf[i] - abs(capex[i]) for i in range(min(len(ocf), len(capex)))]

    return {
        "operating_cf": ocf,
        "capex":        capex,
        "fcf":          fcf,
        "dividends_paid": series(["Payment Of Dividends","Common Stock Dividend Paid","Dividends Paid"]),
        "buybacks":     series(["Repurchase Of Capital Stock","Common Stock Repurchase"]),
    }


# ─── Growth & Margin Calculations ───────────────────────────────────────────

def growth_rates(series: dict) -> list:
    """Calculate YoY growth rates from a {year: value} dict (sorted descending)."""
    vals = [v for _, v in sorted(series.items(), reverse=True) if v]
    if len(vals) < 2:
        return []
    rates = []
    for i in range(len(vals) - 1):
        if vals[i + 1] and vals[i + 1] != 0:
            rates.append((vals[i] / vals[i + 1]) - 1)
    return rates


def margin_series(numerator: dict, denominator: dict) -> dict:
    """Calculate margin = num / denom for matching years."""
    result = {}
    for yr in numerator:
        if yr in denominator and denominator[yr] and denominator[yr] != 0:
            result[yr] = numerator[yr] / denominator[yr]
    return result


# ─── WACC Calculation ────────────────────────────────────────────────────────

def calculate_wacc(info: dict, income: pd.DataFrame, balance: pd.DataFrame, rfr: float) -> dict:
    beta = float(info.get("beta") or 1.0)
    mrp  = 0.055  # Market Risk Premium (historical avg)
    ke   = rfr + beta * mrp  # Cost of Equity (CAPM)

    # Cost of Debt
    interest = abs(safe_val(income, ["Interest Expense","Interest Expense Non Operating"]) or 0)
    total_debt = safe_val(balance, ["Total Debt","Long Term Debt And Capital Lease Obligation"]) or 0
    kd_pretax = (interest / total_debt) if total_debt > 0 and interest > 0 else 0.05

    # Effective Tax Rate
    tax  = safe_val(income, ["Tax Provision","Income Tax Expense"]) or 0
    ebt  = safe_val(income, ["Pretax Income","Income Before Tax"]) or 1
    tax_rate = max(0.0, min(0.40, tax / ebt)) if ebt > 0 else 0.21
    kd = kd_pretax * (1 - tax_rate)

    # Capital Structure Weights
    mkt_cap = float(info.get("marketCap") or 0)
    total_v = mkt_cap + total_debt
    w_e = mkt_cap / total_v if total_v > 0 else 1.0
    w_d = total_debt / total_v if total_v > 0 else 0.0

    wacc = w_e * ke + w_d * kd

    return {
        "wacc":           wacc,
        "cost_of_equity": ke,
        "cost_of_debt":   kd,
        "kd_pretax":      kd_pretax,
        "w_equity":       w_e,
        "w_debt":         w_d,
        "tax_rate":       tax_rate,
        "beta":           beta,
        "rfr":            rfr,
        "mrp":            mrp,
    }


# ─── Core DCF Engine ─────────────────────────────────────────────────────────

def _dcf_engine(base_fcf: float, discount_rate: float, stage1_growth: float,
                stage2_growth: float, terminal_growth: float,
                stage1_years: int = 5, stage2_years: int = 5) -> dict:
    """Internal DCF calculation used by all 4 models."""

    if discount_rate <= terminal_growth:
        # Use a safe floor to avoid division by zero or negative EV
        discount_rate = terminal_growth + 0.02

    fcf = base_fcf
    pv_stage1, pv_stage2 = 0.0, 0.0
    yr = 0

    # Stage 1
    stage1_cfs = []
    for i in range(1, stage1_years + 1):
        fcf = fcf * (1 + stage1_growth)
        pv = fcf / (1 + discount_rate) ** i
        pv_stage1 += pv
        stage1_cfs.append({"year": i, "fcf": fcf, "pv": pv})
        yr = i

    # Stage 2
    stage2_cfs = []
    for i in range(stage1_years + 1, stage1_years + stage2_years + 1):
        fcf = fcf * (1 + stage2_growth)
        pv = fcf / (1 + discount_rate) ** i
        pv_stage2 += pv
        stage2_cfs.append({"year": i, "fcf": fcf, "pv": pv})
        yr = i

    # Terminal Value
    terminal_fcf = fcf * (1 + terminal_growth)
    tv = terminal_fcf / (discount_rate - terminal_growth)
    pv_tv = tv / (1 + discount_rate) ** yr

    total_ev = pv_stage1 + pv_stage2 + pv_tv

    return {
        "discount_rate":   discount_rate,
        "stage1_growth":   stage1_growth,
        "stage2_growth":   stage2_growth,
        "terminal_growth": terminal_growth,
        "pv_stage1":       pv_stage1,
        "pv_stage2":       pv_stage2,
        "terminal_value":  tv,
        "pv_terminal":     pv_tv,
        "total_ev":        total_ev,
        "tv_pct":          pv_tv / total_ev * 100 if total_ev > 0 else 0,
        "stage1_cashflows": stage1_cfs,
        "stage2_cashflows": stage2_cfs,
    }


def _resolve_base_fcf(fcf_list: list) -> float | None:
    """Get a positive base FCF to use in projections."""
    if not fcf_list:
        return None
    if fcf_list[0] > 0:
        return fcf_list[0]
    # Try average of positive values
    pos = [f for f in fcf_list if f > 0]
    return float(np.mean(pos)) if pos else None


def _hist_fcf_growth(fcf_list: list) -> float:
    """Estimate historical FCF CAGR, capped for realism."""
    if len(fcf_list) < 2:
        return 0.08
    rates = []
    for i in range(len(fcf_list) - 1):
        if fcf_list[i + 1] > 0 and fcf_list[i] > 0:
            rates.append(fcf_list[i] / fcf_list[i + 1] - 1)
    if not rates:
        return 0.08
    g = float(np.median(rates))
    return max(-0.05, min(0.35, g))


# ─── 4 DCF Models ────────────────────────────────────────────────────────────

def dcf_wacc(fcf_list: list, wacc_components: dict, shares: float,
             net_cash: float = 0) -> dict:
    """Model 1 – WACC-based DCF."""
    base = _resolve_base_fcf(fcf_list)
    if base is None:
        return {"error": "Negative/zero FCF — WACC DCF not applicable"}

    r = wacc_components["wacc"]
    g1 = _hist_fcf_growth(fcf_list)
    g2 = max(terminal_g := 0.025, g1 * 0.5)
    result = _dcf_engine(base, r, g1, g2, terminal_g)
    result["model"] = "WACC"
    result["rate_label"] = f"WACC = {r:.2%}"
    _add_equity_value(result, shares, net_cash, wacc_components)
    return result


def dcf_capm(fcf_list: list, info: dict, rfr: float, shares: float,
             net_cash: float = 0) -> dict:
    """Model 2 – CAPM discount rate (cost of equity only)."""
    base = _resolve_base_fcf(fcf_list)
    if base is None:
        return {"error": "Negative/zero FCF — CAPM DCF not applicable"}

    beta = float(info.get("beta") or 1.0)
    r = rfr + beta * 0.055
    g1 = _hist_fcf_growth(fcf_list)
    g2 = max(terminal_g := 0.025, g1 * 0.5)
    result = _dcf_engine(base, r, g1, g2, terminal_g)
    result["model"] = "CAPM"
    result["rate_label"] = f"Ke = {r:.2%} (β={beta:.2f})"
    _add_equity_value(result, shares, net_cash)
    return result


def dcf_fixed(fcf_list: list, fixed_rate: float, shares: float,
              net_cash: float = 0) -> dict:
    """Model 3 – User-supplied fixed discount rate."""
    base = _resolve_base_fcf(fcf_list)
    if base is None:
        return {"error": "Negative/zero FCF — Fixed DCF not applicable"}

    g1 = _hist_fcf_growth(fcf_list)
    g2 = max(terminal_g := 0.025, g1 * 0.5)
    result = _dcf_engine(base, fixed_rate, g1, g2, terminal_g)
    result["model"] = "Fixed Rate"
    result["rate_label"] = f"r = {fixed_rate:.2%} (user)"
    _add_equity_value(result, shares, net_cash)
    return result


def dcf_two_stage(fcf_list: list, info: dict, rfr: float, shares: float,
                  net_cash: float = 0) -> dict:
    """Model 4 – Two-Stage FCF: high-growth phase then fading to terminal."""
    base = _resolve_base_fcf(fcf_list)
    if base is None:
        return {"error": "Negative/zero FCF — Two-Stage DCF not applicable"}

    beta = float(info.get("beta") or 1.0)
    r = rfr + beta * 0.055
    g1 = _hist_fcf_growth(fcf_list)
    g2 = max(terminal_g := 0.025, g1 * 0.4)
    # stage1 = 5y high growth, stage2 = 5y transition, then terminal
    result = _dcf_engine(base, r, g1, g2, terminal_g, stage1_years=5, stage2_years=5)
    result["model"] = "Two-Stage FCF"
    result["rate_label"] = f"r = {r:.2%} | g1={g1:.1%} → g2={g2:.1%}"
    _add_equity_value(result, shares, net_cash)
    return result


def _add_equity_value(result: dict, shares: float, net_cash: float,
                      wacc_components: dict = None):
    """Append equity value per share to a DCF result dict."""
    ev = result.get("total_ev", 0)
    equity_val = ev + net_cash  # Enterprise Value → Equity Value (simplified)
    result["equity_value"] = equity_val
    if shares and shares > 0:
        result["intrinsic_value"] = equity_val / shares
    else:
        result["intrinsic_value"] = None
    if wacc_components:
        result["wacc_components"] = wacc_components


# ─── Public Entry Point ──────────────────────────────────────────────────────

def run_dcf_models(data: dict, rfr: float, fixed_rate: float) -> dict:
    """Run all 4 DCF models and return results + metadata."""
    info    = data.get("info", {})
    income  = data.get("income_stmt", pd.DataFrame())
    balance = data.get("balance_sheet", pd.DataFrame())
    cf_data = get_cashflow_list(data.get("cash_flow", pd.DataFrame()))

    fcf_list = cf_data.get("fcf", [])
    shares   = float(info.get("sharesOutstanding") or info.get("floatShares") or 0)
    total_cash = float(safe_val(balance, ["Cash And Cash Equivalents",
                                           "Cash Cash Equivalents And Short Term Investments"]) or 0)
    total_debt = float(safe_val(balance, ["Total Debt"]) or 0)
    net_cash = total_cash - total_debt  # can be negative

    wacc_c = calculate_wacc(info, income, balance, rfr)

    return {
        "wacc":       dcf_wacc(fcf_list, wacc_c, shares, net_cash),
        "capm":       dcf_capm(fcf_list, info, rfr, shares, net_cash),
        "fixed":      dcf_fixed(fcf_list, fixed_rate, shares, net_cash),
        "two_stage":  dcf_two_stage(fcf_list, info, rfr, shares, net_cash),
        "wacc_components": wacc_c,
        "fcf_list":   fcf_list,
        "rfr":        rfr,
        "current_price": float(info.get("currentPrice") or info.get("regularMarketPrice") or 0),
        "shares":     shares,
        "net_cash":   net_cash,
    }


def analyze_fundamentals(data: dict) -> dict:
    """Aggregate all fundamental metrics into a single dict."""
    info    = data.get("info", {})
    income  = data.get("income_stmt", pd.DataFrame())
    balance = data.get("balance_sheet", pd.DataFrame())
    cashflow = data.get("cash_flow", pd.DataFrame())

    inc  = get_income_series(income)
    bal  = get_balance_series(balance)
    cfs  = get_cashflow_list(cashflow)

    rev  = inc.get("revenue", {})
    gp   = inc.get("gross_profit", {})
    oi   = inc.get("operating_income", {})
    ni   = inc.get("net_income", {})

    return {
        # From yfinance info (fast)
        "market_cap":       info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "current_price":    info.get("currentPrice") or info.get("regularMarketPrice"),
        "pe":               info.get("trailingPE"),
        "forward_pe":       info.get("forwardPE"),
        "peg":              info.get("pegRatio"),
        "ev_ebitda":        info.get("enterpriseToEbitda"),
        "price_book":       info.get("priceToBook"),
        "price_sales":      info.get("priceToSalesTrailingTwelveMonths"),
        "gross_margin":     info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin":       info.get("profitMargins"),
        "roe":              info.get("returnOnEquity"),
        "roa":              info.get("returnOnAssets"),
        "revenue_growth":   info.get("revenueGrowth"),
        "earnings_growth":  info.get("earningsGrowth"),
        "current_ratio":    info.get("currentRatio"),
        "quick_ratio":      info.get("quickRatio"),
        "debt_equity":      info.get("debtToEquity"),
        "eps_ttm":          info.get("trailingEps"),
        "eps_forward":      info.get("forwardEps"),
        "free_cashflow":    info.get("freeCashflow"),
        "ebitda":           info.get("ebitda"),
        "dividend_yield":   info.get("dividendYield"),
        "payout_ratio":     info.get("payoutRatio"),
        "beta":             info.get("beta"),
        # Computed from statements
        "revenue_series":       rev,
        "gross_profit_series":  gp,
        "op_income_series":     oi,
        "net_income_series":    ni,
        "gross_margin_series":  margin_series(gp, rev),
        "op_margin_series":     margin_series(oi, rev),
        "net_margin_series":    margin_series(ni, rev),
        "rev_growth_rates":     growth_rates(rev),
        "ni_growth_rates":      growth_rates(ni),
        "balance":  bal,
        "cashflows": cfs,
    }
