"""
fundamental.py — Financial statement analysis + DCF models.

DCF improvements based on JPMorgan M&A / CFA methodology:
  - Uses Unlevered Free Cash Flow (UFCF = EBIT×(1-t) + D&A - CapEx - ΔNWC)
    instead of reported operating cash flow, which includes working capital noise
  - Blends historical UFCF growth with analyst EPS forward estimates
  - Revenue-based projection option when FCF is unreliable
  - Sensitivity analysis table (WACC × terminal growth rate)
  - Proper EV → Equity Value bridge (+ cash, - debt, - minority interest)

4 DCF Models:
  1. WACC DCF (enterprise, UFCF)
  2. CAPM DCF (cost of equity, levered FCF)
  3. Fixed Rate DCF (user-supplied hurdle rate)
  4. Two-Stage FCF (high-growth fade to terminal)
"""

import pandas as pd
import numpy as np
from data import safe_val


# ─── Financial Statement Parsers ──────────────────────────────────────────────

def get_income_series(income: pd.DataFrame) -> dict:
    if income is None or income.empty:
        return {}

    def series(row_names):
        vals = {}
        for i, col in enumerate(income.columns):
            v = safe_val(income, row_names, i)
            if v is not None:
                yr = col.year if hasattr(col, "year") else i
                vals[yr] = v
        return vals

    return {
        "revenue":          series(["Total Revenue", "Revenue"]),
        "gross_profit":     series(["Gross Profit"]),
        "ebit":             series(["Operating Income", "EBIT", "Ebit"]),
        "ebitda":           series(["EBITDA", "Ebitda", "Normalized EBITDA"]),
        "net_income":       series(["Net Income", "Net Income Common Stockholders"]),
        "da":               series(["Reconciled Depreciation", "Depreciation Amortization Depletion"]),
        "interest_expense": series(["Interest Expense Non Operating", "Interest Expense"]),
        "tax_provision":    series(["Tax Provision", "Income Tax Expense"]),
        "pretax_income":    series(["Pretax Income", "Income Before Tax"]),
        "rd_expense":       series(["Research And Development"]),
        "eps_basic":        series(["Basic EPS"]),
        "eps_diluted":      series(["Diluted EPS"]),
    }


def get_balance_series(balance: pd.DataFrame) -> dict:
    if balance is None or balance.empty:
        return {}

    def latest(row_names):
        return safe_val(balance, row_names, 0)

    def series(row_names, n=4):
        vals = []
        for i in range(min(n, len(balance.columns))):
            v = safe_val(balance, row_names, i)
            vals.append(v)
        return vals

    # Working capital items (for ΔNWC calculation)
    ca_series   = series(["Current Assets", "Total Current Assets"])
    cl_series   = series(["Current Liabilities", "Total Current Liabilities Net Minority Interest"])
    # Exclude cash from current assets, exclude ST debt from current liabilities
    # for operating NWC
    cash_series = series(["Cash And Cash Equivalents"])
    std_series  = series(["Current Debt", "Current Portion Of Long Term Debt"])

    return {
        "total_assets":   latest(["Total Assets"]),
        "total_debt":     latest(["Total Debt", "Long Term Debt And Capital Lease Obligation"]),
        "long_term_debt": latest(["Long Term Debt"]),
        "total_equity":   latest(["Stockholders Equity", "Common Stock Equity",
                                   "Total Equity Gross Minority Interest"]),
        "cash":           latest(["Cash And Cash Equivalents",
                                   "Cash Cash Equivalents And Short Term Investments"]),
        "current_assets": latest(["Current Assets", "Total Current Assets"]),
        "current_liab":   latest(["Current Liabilities",
                                   "Total Current Liabilities Net Minority Interest"]),
        "total_liab":     latest(["Total Liabilities Net Minority Interest"]),
        "goodwill":       latest(["Goodwill And Other Intangible Assets", "Goodwill"]),
        "minority_int":   latest(["Minority Interest", "Non Controlling Interest"]),
        # Series for NWC delta calculation
        "ca_series":  ca_series,
        "cl_series":  cl_series,
        "cash_series": cash_series,
        "std_series":  std_series,
    }


def get_cashflow_data(cashflow: pd.DataFrame) -> dict:
    if cashflow is None or cashflow.empty:
        return {}

    def series(row_names, n=4):
        vals = []
        for i in range(min(n, len(cashflow.columns))):
            v = safe_val(cashflow, row_names, i)
            vals.append(v)
        return [v for v in vals if v is not None]

    ocf   = series(["Operating Cash Flow", "Cash From Operations",
                    "Total Cash From Operating Activities"])
    capex = series(["Capital Expenditure", "Purchases Of Property Plant And Equipment",
                    "Capital Expenditures"])
    da    = series(["Depreciation Amortization Depletion",
                    "Depreciation And Amortization"])
    nwc_c = series(["Change In Working Capital", "Changes In Working Capital"])

    fcf_reported = []
    for i in range(min(len(ocf), len(capex))):
        if ocf[i] is not None and capex[i] is not None:
            fcf_reported.append(ocf[i] - abs(capex[i]))

    return {
        "operating_cf": ocf,
        "capex":        capex,
        "da":           da,
        "nwc_change":   nwc_c,
        "fcf_reported": fcf_reported,
        "dividends":    series(["Payment Of Dividends", "Common Stock Dividend Paid"]),
        "buybacks":     series(["Repurchase Of Capital Stock", "Common Stock Repurchase"]),
    }


# ─── UFCF Calculator ──────────────────────────────────────────────────────────

def calculate_ufcf_series(income: pd.DataFrame, balance: pd.DataFrame,
                           cashflow: pd.DataFrame) -> list:
    """
    Unlevered Free Cash Flow = EBIT × (1 − Tax Rate) + D&A − CapEx − ΔNWC

    This is the correct base for an enterprise DCF (WACC model) because it
    represents cash flows available to ALL capital providers (debt + equity),
    before financing effects.

    Falls back to reported FCF if UFCF cannot be calculated.
    """
    inc = get_income_series(income)
    cf  = get_cashflow_data(cashflow)
    bal = get_balance_series(balance)

    n_periods = min(4, len(income.columns) if income is not None and not income.empty else 0)
    if n_periods == 0:
        return []

    # Effective tax rate from statements (capped 15–35%)
    tax_rates = []
    for yr, ebit in sorted(inc.get("ebit", {}).items(), reverse=True)[:4]:
        tp = inc.get("tax_provision", {}).get(yr)
        pi = inc.get("pretax_income", {}).get(yr)
        if tp is not None and pi and pi > 0:
            tr = max(0.15, min(0.35, tp / pi))
            tax_rates.append(tr)
    tax_rate = float(np.median(tax_rates)) if tax_rates else 0.21

    ufcf_list = []
    ebit_vals  = sorted(inc.get("ebit", {}).items(), reverse=True)
    da_vals    = sorted(inc.get("da", {}).items(), reverse=True)

    # D&A fallback: use cash flow statement D&A if income statement doesn't have it
    da_cf = cf.get("da", [])

    capex_list = cf.get("capex", [])
    nwc_list   = cf.get("nwc_change", [])

    for i in range(min(n_periods, len(ebit_vals))):
        yr, ebit = ebit_vals[i]

        # NOPAT = EBIT × (1 - t)
        nopat = ebit * (1 - tax_rate)

        # D&A: prefer income statement, fall back to cash flow
        da = None
        if i < len(da_vals):
            da = abs(da_vals[i][1])
        if da is None and i < len(da_cf):
            da = abs(da_cf[i])
        if da is None:
            # Estimate D&A as ~3% of revenue (industry average if not available)
            rev = list(inc.get("revenue", {}).values())
            da = abs(rev[i]) * 0.03 if i < len(rev) and rev[i] else 0

        # CapEx
        capex = abs(capex_list[i]) if i < len(capex_list) and capex_list[i] is not None else 0

        # ΔNWC: increase in NWC uses cash (negative for UFCF)
        # NWC = Operating Current Assets - Operating Current Liabilities
        # Operating CA = CA - Cash  /  Operating CL = CL - Short-term debt
        ca_s   = bal.get("ca_series", [])
        cl_s   = bal.get("cl_series", [])
        cash_s = bal.get("cash_series", [])
        std_s  = bal.get("std_series", [])

        delta_nwc = 0.0
        if (i + 1 < len(ca_s) and ca_s[i] is not None and ca_s[i+1] is not None
                and cl_s[i] is not None and cl_s[i+1] is not None):
            ca_now  = (ca_s[i]   or 0) - (cash_s[i]   if i < len(cash_s) and cash_s[i]   else 0)
            ca_prev = (ca_s[i+1] or 0) - (cash_s[i+1] if i+1 < len(cash_s) and cash_s[i+1] else 0)
            cl_now  = (cl_s[i]   or 0) - (std_s[i]   if i < len(std_s) and std_s[i]   else 0)
            cl_prev = (cl_s[i+1] or 0) - (std_s[i+1] if i+1 < len(std_s) and std_s[i+1] else 0)
            nwc_now  = ca_now  - cl_now
            nwc_prev = ca_prev - cl_prev
            delta_nwc = nwc_now - nwc_prev  # increase = uses cash
        elif i < len(nwc_list) and nwc_list[i] is not None:
            # Use cash flow statement NWC change directly
            delta_nwc = -nwc_list[i]  # CF statement shows it as inflow/outflow

        ufcf = nopat + da - capex - delta_nwc
        ufcf_list.append(ufcf)

    return ufcf_list, tax_rate


# ─── Growth Rate Estimation ────────────────────────────────────────────────────

def estimate_growth_rates(ufcf_list: list, info: dict) -> dict:
    """
    Blend three growth estimates:
      1. Historical UFCF CAGR (last 3-4 years)
      2. Analyst EPS forward growth estimate (from yfinance info)
      3. Analyst revenue growth estimate

    Returns {near: float, fade: float} where near = Stage 1, fade = Stage 2
    """
    # 1. Historical UFCF CAGR
    hist_g = _ufcf_cagr(ufcf_list)

    # 2. Analyst forward EPS growth
    eps_g = None
    for key in ["earningsGrowth", "revenueGrowth"]:
        v = info.get(key)
        if v is not None:
            try:
                eps_g = float(v)
                break
            except Exception:
                pass

    # 3. Long-term analyst growth (5y)
    ltg = None
    v = info.get("longTermPotentialGrowthRate") or info.get("targetGrowthRate5Year")
    if v:
        try:
            ltg = float(v)
        except Exception:
            pass

    # Blend: if analyst data exists, weight it more heavily (60/40)
    if eps_g is not None:
        blended_near = hist_g * 0.4 + eps_g * 0.6
    else:
        blended_near = hist_g

    # Cap unrealistic values
    blended_near = max(-0.10, min(0.40, blended_near))

    # Stage 2: fade rate (average of stage 1 and terminal)
    terminal_g   = 0.025
    blended_fade = max(terminal_g, blended_near * 0.5)

    # Stage 2 also blended with LTG if available
    if ltg is not None:
        blended_fade = max(terminal_g, (blended_fade + ltg) / 2)

    return {
        "near":        blended_near,
        "fade":        blended_fade,
        "terminal":    terminal_g,
        "hist_ufcf":   hist_g,
        "analyst_eps": eps_g,
        "ltg":         ltg,
    }


def _ufcf_cagr(ufcf_list: list) -> float:
    """Historical UFCF CAGR. Falls back to 5% if not calculable."""
    pos_vals = [v for v in ufcf_list if v and v > 0]
    if len(pos_vals) < 2:
        return 0.05
    rates = []
    for i in range(len(ufcf_list) - 1):
        try:
            a, b = ufcf_list[i], ufcf_list[i + 1]
            if a and b and abs(b) > 1 and a > 0:  # avoid div-by-zero and near-zero
                rates.append(a / b - 1)
        except (ZeroDivisionError, TypeError):
            pass
    if not rates:
        return 0.05
    return float(np.clip(np.median(rates), -0.50, 1.00))


# ─── WACC ─────────────────────────────────────────────────────────────────────

def calculate_wacc(info: dict, income: pd.DataFrame,
                   balance: pd.DataFrame, rfr: float) -> dict:
    """
    WACC with improvements:
    - Size premium for small-caps (market cap < $2B)
    - Iterative beta adjustment for financial leverage
    - Damodaran-style pre-tax cost of debt from synthetic spread
    """
    beta = float(info.get("beta") or 1.0)
    mrp  = 0.055  # Equity risk premium (Damodaran US estimate)

    # Size premium (Duff & Phelps approximate)
    mktcap = float(info.get("marketCap") or 1e10)
    if mktcap < 300e6:      size_prem = 0.040   # Micro-cap
    elif mktcap < 2e9:      size_prem = 0.025   # Small-cap
    elif mktcap < 10e9:     size_prem = 0.012   # Mid-cap
    else:                   size_prem = 0.000   # Large-cap

    ke = rfr + beta * mrp + size_prem  # Cost of equity

    # Cost of debt — prefer interest/debt, fall back to synthetic spread from coverage ratio
    interest = abs(safe_val(income, ["Interest Expense Non Operating",
                                      "Interest Expense"]) or 0)
    total_debt = abs(safe_val(balance, ["Total Debt",
                                        "Long Term Debt And Capital Lease Obligation"]) or 0)
    ebit_v = safe_val(income, ["Operating Income", "EBIT", "Ebit"])

    if total_debt > 0 and interest > 0:
        kd_pretax = min(interest / total_debt, 0.15)
    elif ebit_v is not None and total_debt > 0:
        # Synthetic rating from interest coverage ratio.
        # For loss-making companies ebit_v < 0 — use absolute EBIT as base,
        # ensure denominator is always > 0 with a floor of 1.
        safe_denom = max(abs(float(interest or 0)), abs(float(ebit_v)) * 0.01, 1.0)
        ic = float(ebit_v) / safe_denom  # negative IC signals distress → widest spread
        if ic > 12.5:   spread = 0.0050
        elif ic > 9.5:  spread = 0.0065
        elif ic > 7.5:  spread = 0.0085
        elif ic > 6.0:  spread = 0.0100
        elif ic > 4.5:  spread = 0.0130
        elif ic > 3.5:  spread = 0.0170
        elif ic > 2.5:  spread = 0.0250
        elif ic > 2.0:  spread = 0.0350
        elif ic > 1.5:  spread = 0.0490
        else:           spread = 0.0650
        kd_pretax = rfr + spread
    else:
        kd_pretax = rfr + 0.015  # fallback: Rfr + 150bps

    # Effective tax rate
    tax_rate = 0.21
    tp = safe_val(income, ["Tax Provision", "Income Tax Expense"])
    pi = safe_val(income, ["Pretax Income", "Income Before Tax"])
    if tp is not None and pi and pi > 0:
        tax_rate = max(0.10, min(0.35, tp / pi))

    kd = kd_pretax * (1 - tax_rate)  # After-tax cost of debt

    # Capital weights (market-value weights)
    market_cap = float(info.get("marketCap") or 0)
    total_v    = market_cap + total_debt
    w_e = market_cap / total_v if total_v > 1 else 1.0
    w_d = total_debt / total_v if total_v > 1 else 0.0

    wacc = w_e * ke + w_d * kd

    return {
        "wacc":           max(0.04, min(0.30, wacc)),  # bound to reasonable range
        "cost_of_equity": ke,
        "cost_of_debt":   kd,
        "kd_pretax":      kd_pretax,
        "w_equity":       w_e,
        "w_debt":         w_d,
        "tax_rate":       tax_rate,
        "beta":           beta,
        "rfr":            rfr,
        "mrp":            mrp,
        "size_premium":   size_prem,
    }


# ─── DCF Engine ───────────────────────────────────────────────────────────────

def _dcf_engine(base_fcf: float, discount_rate: float, stage1_growth: float,
                stage2_growth: float, terminal_growth: float,
                stage1_years: int = 5, stage2_years: int = 5) -> dict:
    """Core DCF: explicit forecast + terminal value (Gordon Growth)."""
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.02
    if not base_fcf or abs(base_fcf) < 1:
        return {"error": "Base FCF is zero — cannot compute DCF"}
    if discount_rate == terminal_growth:
        terminal_growth = discount_rate - 0.005

    fcf       = base_fcf
    pv_s1     = 0.0
    pv_s2     = 0.0
    s1_detail = []
    s2_detail = []
    yr        = 0

    for i in range(1, stage1_years + 1):
        fcf   = fcf * (1 + stage1_growth)
        pv    = fcf / (1 + discount_rate) ** i
        pv_s1 += pv
        s1_detail.append({"year": i, "fcf": fcf, "pv": pv, "pv_cumulative": pv_s1})
        yr = i

    for i in range(stage1_years + 1, stage1_years + stage2_years + 1):
        fcf   = fcf * (1 + stage2_growth)
        pv    = fcf / (1 + discount_rate) ** i
        pv_s2 += pv
        s2_detail.append({"year": i, "fcf": fcf, "pv": pv})
        yr = i

    # Terminal Value (Gordon Growth)
    terminal_fcf = fcf * (1 + terminal_growth)
    tv           = terminal_fcf / (discount_rate - terminal_growth)
    pv_tv        = tv / (1 + discount_rate) ** yr
    total_ev     = pv_s1 + pv_s2 + pv_tv

    return {
        "discount_rate":    discount_rate,
        "stage1_growth":    stage1_growth,
        "stage2_growth":    stage2_growth,
        "terminal_growth":  terminal_growth,
        "pv_stage1":        pv_s1,
        "pv_stage2":        pv_s2,
        "terminal_value":   tv,
        "pv_terminal":      pv_tv,
        "total_ev":         total_ev,
        "tv_pct":           pv_tv / total_ev * 100 if total_ev > 0 else 0,
        "stage1_cashflows": s1_detail,
        "stage2_cashflows": s2_detail,
    }


def _ev_to_equity(result: dict, shares: float, net_cash: float,
                   minority_int: float = 0) -> dict:
    """
    EV → Equity Value → Price per Share bridge.
    Equity Value = EV + Cash - Debt - Minority Interest
    (net_cash = Cash - Debt, can be negative)
    """
    ev          = result.get("total_ev", 0)
    equity_val  = ev + net_cash - (minority_int or 0)
    result["equity_value"]    = equity_val
    result["intrinsic_value"] = equity_val / shares if (shares and shares > 0) else None
    return result


# ─── 4 DCF Models ─────────────────────────────────────────────────────────────

def dcf_wacc(ufcf_list: list, wacc_components: dict, shares: float,
             net_cash: float, minority_int: float, growth_rates: dict) -> dict:
    """
    Model 1 — WACC DCF (enterprise, uses UFCF).
    Correct for companies with meaningful debt.
    """
    base = _positive_base(ufcf_list)
    if base is None:
        return {"error": "Negative/zero UFCF — WACC DCF not applicable"}

    r  = wacc_components["wacc"]
    g1 = growth_rates["near"]
    g2 = growth_rates["fade"]
    tg = growth_rates["terminal"]

    result = _dcf_engine(base, r, g1, g2, tg)
    result["model"]      = "WACC DCF"
    result["rate_label"] = f"WACC = {r:.2%}"
    result["base_fcf"]   = base
    result["fcf_type"]   = "UFCF"
    result["wacc_components"] = wacc_components
    return _ev_to_equity(result, shares, net_cash, minority_int)


def dcf_capm(ufcf_list: list, info: dict, rfr: float, shares: float,
             net_cash: float, minority_int: float, growth_rates: dict) -> dict:
    """
    Model 2 — CAPM / Equity DCF (cost of equity as discount rate).
    Correct for equity-funded or near-zero-debt companies.
    """
    base = _positive_base(ufcf_list)
    if base is None:
        return {"error": "Negative/zero UFCF — CAPM DCF not applicable"}

    beta      = float(info.get("beta") or 1.0)
    mktcap    = float(info.get("marketCap") or 1e10)
    mrp       = 0.055
    size_prem = 0.025 if mktcap < 2e9 else (0.012 if mktcap < 10e9 else 0.0)
    r = rfr + beta * mrp + size_prem

    g1 = growth_rates["near"]
    g2 = growth_rates["fade"]
    tg = growth_rates["terminal"]

    result = _dcf_engine(base, r, g1, g2, tg)
    result["model"]      = "CAPM DCF"
    result["rate_label"] = f"Ke = {r:.2%}  (β={beta:.2f}, size={size_prem:.1%})"
    result["base_fcf"]   = base
    result["fcf_type"]   = "UFCF"
    return _ev_to_equity(result, shares, net_cash, minority_int)


def dcf_fixed(ufcf_list: list, fixed_rate: float, shares: float,
              net_cash: float, minority_int: float, growth_rates: dict) -> dict:
    """Model 3 — Fixed Rate DCF (user hurdle rate)."""
    base = _positive_base(ufcf_list)
    if base is None:
        return {"error": "Negative/zero UFCF — Fixed DCF not applicable"}

    g1 = growth_rates["near"]
    g2 = growth_rates["fade"]
    tg = growth_rates["terminal"]

    result = _dcf_engine(base, fixed_rate, g1, g2, tg)
    result["model"]      = "Fixed Rate DCF"
    result["rate_label"] = f"r = {fixed_rate:.2%}  (user-set hurdle)"
    result["base_fcf"]   = base
    result["fcf_type"]   = "UFCF"
    return _ev_to_equity(result, shares, net_cash, minority_int)


def dcf_two_stage(ufcf_list: list, info: dict, rfr: float, shares: float,
                  net_cash: float, minority_int: float, growth_rates: dict) -> dict:
    """
    Model 4 — Two-Stage FCF (explicit high-growth + fading transition).
    More conservative: higher-growth Stage 1 fades more aggressively.
    """
    base = _positive_base(ufcf_list)
    if base is None:
        return {"error": "Negative/zero UFCF — Two-Stage DCF not applicable"}

    beta  = float(info.get("beta") or 1.0)
    r     = rfr + beta * 0.055
    g1    = growth_rates["near"]
    g2    = growth_rates["fade"]
    tg    = growth_rates["terminal"]

    result = _dcf_engine(base, r, g1, g2, tg, stage1_years=5, stage2_years=5)
    result["model"]      = "Two-Stage FCF"
    result["rate_label"] = f"r={r:.2%} | g1={g1:.1%} → g2={g2:.1%} → g∞={tg:.1%}"
    result["base_fcf"]   = base
    result["fcf_type"]   = "UFCF"
    return _ev_to_equity(result, shares, net_cash, minority_int)


def _positive_base(vals: list) -> float | None:
    """Most recent positive UFCF; fallback to 3Y average if latest negative."""
    if not vals:
        return None
    if vals[0] and vals[0] > 0:
        return float(vals[0])
    pos = [v for v in vals if v and v > 0]
    return float(np.mean(pos)) if pos else None


# ─── Sensitivity Table ────────────────────────────────────────────────────────

def build_sensitivity_table(base_fcf: float, wacc_base: float,
                              terminal_g_base: float, growth_rates: dict,
                              shares: float, net_cash: float) -> dict:
    """
    WACC sensitivity analysis: 5×5 table of intrinsic values.
    Rows = WACC ± steps, Cols = terminal growth ± steps.
    Returns {wacc_values: [], tg_values: [], table: [[price, ...], ...]}
    """
    if base_fcf is None or base_fcf <= 0 or shares <= 0:
        return {}

    wacc_range = [wacc_base + d for d in [-0.02, -0.01, 0, +0.01, +0.02]]
    tg_range   = [terminal_g_base + d for d in [-0.01, -0.005, 0, +0.005, +0.01]]

    g1 = growth_rates["near"]
    g2 = growth_rates["fade"]

    table = []
    for w in wacc_range:
        row = []
        for tg in tg_range:
            try:
                res = _dcf_engine(base_fcf, w, g1, g2, tg)
                equity_val = res["total_ev"] + net_cash
                price = equity_val / shares if shares > 0 else 0
                row.append(round(max(0, price), 2))
            except Exception:
                row.append(0.0)
        table.append(row)

    return {
        "wacc_values": [round(w * 100, 2) for w in wacc_range],
        "tg_values":   [round(tg * 100, 2) for tg in tg_range],
        "table":       table,
        "base_wacc_idx":   2,
        "base_tg_idx":     2,
    }



# ─── Industry Margin Benchmarks ───────────────────────────────────────────────

# Target mature net margins by sector (for Path-to-Profitability model)
SECTOR_TARGET_MARGINS = {
    "Technology":              0.22,
    "Software":                0.25,
    "Semiconductors":          0.20,
    "Healthcare":              0.15,
    "Biotechnology":           0.18,
    "Financial Services":      0.22,
    "Consumer Cyclical":       0.07,
    "Consumer Defensive":      0.08,
    "Energy":                  0.10,
    "Industrials":             0.10,
    "Communication Services":  0.14,
    "Basic Materials":         0.09,
    "Real Estate":             0.18,
    "Utilities":               0.12,
    "Automotive":              0.06,
    "Electric Vehicles":       0.07,
    "Aerospace & Defense":     0.08,
    "Retail":                  0.04,
    "Restaurant":              0.10,
    "E-commerce":              0.06,
}

# Revenue growth rate by maturity stage
INDUSTRY_GROWTH_BENCHMARKS = {
    "Technology":              0.14,
    "Software":                0.18,
    "Semiconductors":          0.12,
    "Healthcare":              0.10,
    "Biotechnology":           0.12,
    "Financial Services":      0.08,
    "Consumer Cyclical":       0.07,
    "Consumer Defensive":      0.05,
    "Energy":                  0.05,
    "Industrials":             0.07,
    "Communication Services":  0.08,
    "Electric Vehicles":       0.20,
    "Automotive":              0.05,
    "Retail":                  0.05,
    "default":                 0.08,
}


def _get_target_margin(info: dict) -> tuple[float, str]:
    """Return (target_net_margin, rationale) based on sector/industry."""
    sector   = info.get("sector", "")
    industry = info.get("industry", "")

    # Industry-specific overrides
    industry_l = industry.lower()
    if "electric" in industry_l or "ev" in industry_l:
        return 0.07, "EV/auto industry mature margin ~7%"
    if "software" in industry_l or "saas" in industry_l:
        return 0.25, "Software/SaaS mature margin ~25%"
    if "semiconductor" in industry_l:
        return 0.20, "Semiconductor mature margin ~20%"
    if "biotech" in industry_l:
        return 0.18, "Biotech/pharma mature margin ~18%"
    if "retail" in industry_l:
        return 0.04, "Retail mature margin ~4%"
    if "restaurant" in industry_l:
        return 0.10, "Restaurant mature margin ~10%"

    # Sector fallback
    target = SECTOR_TARGET_MARGINS.get(sector, 0.10)
    return target, f"{sector} sector mature margin ~{target:.0%}"


def dcf_path_to_profitability(data: dict, info: dict, rfr: float,
                               shares: float, net_cash: float) -> dict:
    """
    Path-to-Profitability DCF — for currently unprofitable or near-zero-FCF companies.

    Methodology (consistent with JPMorgan equity research practice):
      1. Start from current revenue (not FCF, which is negative/unreliable)
      2. Project revenue growth: blend analyst estimates (60%) + industry benchmarks (40%)
      3. Model operating margin improvement:
         - Current margin → target (sector benchmark) over a realistic timeline
         - More deeply unprofitable companies get longer timelines (7-10Y)
         - Near-breakeven companies get shorter timelines (3-5Y)
      4. Derive UFCF: Revenue × Op.Margin × (1-t) − CapEx − ΔNWC
      5. Discount at CAPM + size premium (loss-making companies carry higher risk)
      6. Terminal value only applied once FCF is sustainably positive
      7. Scenario analysis: base / bull / bear cases on margin trajectory

    Key loss-making adjustments:
      - Tax rate = 0% until EBIT positive (no tax benefit assumed on losses)
      - Negative FCF years contribute negative PV (cash burn)
      - Higher discount rate reflects uncertainty premium for pre-profit companies
      - NOL carryforward approximated (reduces tax for first profitable years)
    """
    income   = data.get("income_stmt",   pd.DataFrame())
    balance  = data.get("balance_sheet", pd.DataFrame())
    cashflow = data.get("cash_flow",     pd.DataFrame())

    inc = get_income_series(income)
    bal = get_balance_series(balance)
    cfs = get_cashflow_data(cashflow)

    # ── Current financials ───────────────────────────────────────────────────
    rev_series = sorted(inc.get("revenue", {}).items(), reverse=True)
    if not rev_series:
        return {"error": "No revenue data — Path-to-Profitability not available"}

    current_rev  = abs(rev_series[0][1]) if rev_series[0][1] else 0
    if current_rev <= 0:
        return {"error": "Zero/negative revenue — P2P not applicable"}

    current_nm   = float(info.get("profitMargins") or 0)
    current_om   = float(info.get("operatingMargins") or 0)
    capex_ratio  = 0.08  # default CapEx as % of revenue
    if cfs.get("capex") and current_rev > 1:
        try:
            vals = [abs(v) for v in cfs["capex"][:3] if v]
            if vals:
                avg_capex = np.mean(vals)
                capex_ratio = min(0.25, avg_capex / current_rev)
        except (ZeroDivisionError, Exception):
            pass

    # ── Growth assumptions ───────────────────────────────────────────────────
    sector   = info.get("sector", "")
    industry = info.get("industry", "")

    # Analyst revenue growth estimate (blended)
    analyst_rev_g = float(info.get("revenueGrowth") or 0)
    industry_g    = INDUSTRY_GROWTH_BENCHMARKS.get(sector,
                    INDUSTRY_GROWTH_BENCHMARKS["default"])

    # Stage 1 growth: weight analyst 60%, industry 40%
    if analyst_rev_g > 0:
        stage1_rev_g = analyst_rev_g * 0.60 + industry_g * 0.40
    else:
        stage1_rev_g = industry_g

    stage1_rev_g = max(0.0, min(0.60, stage1_rev_g))  # cap 0–60%

    # ── Margin trajectory ────────────────────────────────────────────────────
    target_margin, margin_rationale = _get_target_margin(info)

    # How many years to reach target margin?
    # Assumption: linear improvement from current to target over 7-10 years
    # Companies already close → faster; deeply unprofitable → slower
    margin_gap     = target_margin - current_om
    years_to_target = 8 if current_om < -0.15 else (6 if current_om < 0 else 5)

    # ── Discount rate (CAPM) ─────────────────────────────────────────────────
    beta      = float(info.get("beta") or 1.5)  # unprofitable firms tend higher beta
    mktcap    = float(info.get("marketCap") or 1e9)
    size_prem = 0.040 if mktcap < 300e6 else (0.025 if mktcap < 2e9 else 0.012)
    r         = rfr + beta * 0.055 + size_prem
    r         = max(0.08, min(0.25, r))  # bound to reasonable range

    terminal_g = 0.025

    # ── 10-year explicit forecast ─────────────────────────────────────────────
    rev         = current_rev
    stage2_g    = max(0.03, stage1_rev_g * 0.50)   # growth fades in years 6-10

    pv_total          = 0.0
    cf_details        = []
    first_positive_yr = None

    # Approximate NOL carryforward: cumulative losses reduce first profitable years' tax
    cumulative_loss   = 0.0
    nol_remaining     = abs(current_rev * max(0, -current_om) * 2)  # 2-year loss proxy

    for yr in range(1, 11):
        g   = stage1_rev_g if yr <= 5 else stage2_g
        rev = rev * (1 + g)

        # Margin ramp: S-curve approximation (slow start, accelerates mid-period)
        raw_progress = yr / years_to_target
        # S-curve: slow initial improvement, faster in middle, plateaus near target
        progress     = min(1.0, raw_progress ** 0.75)
        op_margin    = current_om + progress * margin_gap

        # Tax treatment for loss-making companies
        if op_margin <= 0:
            tax_rate     = 0.0        # no tax on losses
            cumulative_loss += rev * abs(op_margin)
        else:
            nopat_gross  = rev * op_margin
            # Use NOL carryforward to shelter early profits
            if nol_remaining > 0:
                nol_used     = min(nol_remaining, nopat_gross * 0.80)
                nol_remaining -= nol_used
                taxable       = nopat_gross - nol_used
                tax_rate      = 0.21 * (taxable / nopat_gross) if nopat_gross > 0 else 0
            else:
                tax_rate      = 0.21

        # UFCF = NOPAT + D&A_approx − CapEx − ΔNWC_approx
        nopat     = rev * op_margin * (1 - tax_rate)
        da_approx = rev * 0.04          # ~4% of revenue (typical D&A for asset-light models)
        nwc_delta = rev * g * 0.05      # working capital builds as revenue grows
        est_fcf   = nopat + da_approx - rev * capex_ratio - nwc_delta

        pv        = est_fcf / (1 + r) ** yr
        pv_total += pv

        if est_fcf > 0 and first_positive_yr is None:
            first_positive_yr = yr

        cf_details.append({
            "year":       yr,
            "revenue":    rev,
            "rev_growth": g,
            "op_margin":  op_margin,
            "tax_rate":   tax_rate,
            "fcf":        est_fcf,
            "pv":         pv,
        })

    # Terminal value (only if FCF is positive by year 10)
    final_fcf = cf_details[-1]["fcf"]
    if final_fcf > 0:
        tv    = final_fcf * (1 + terminal_g) / (r - terminal_g)
        pv_tv = tv / (1 + r) ** 10
    else:
        tv    = 0
        pv_tv = 0

    total_ev    = pv_total + pv_tv
    equity_val  = total_ev + net_cash
    iv          = equity_val / shares if shares > 0 else None

    return {
        "model":               "Path-to-Profitability DCF",
        "rate_label":          f"Ke={r:.2%} | Rev growth: {stage1_rev_g:.1%}→{stage2_g:.1%} | Margin: {current_om:.1%}→{target_margin:.1%}",
        "intrinsic_value":     iv,
        "equity_value":        equity_val,
        "total_ev":            total_ev,
        "pv_terminal":         pv_tv,
        "terminal_value":      tv,
        "tv_pct":              (pv_tv / total_ev * 100) if total_ev > 0 else 0,
        "current_revenue":     current_rev,
        "current_margin":      current_om,
        "target_margin":       target_margin,
        "margin_rationale":    margin_rationale,
        "first_positive_yr":   first_positive_yr,
        "stage1_growth":       stage1_rev_g,
        "stage2_growth":       stage2_g,
        "discount_rate":       r,
        "fcf_type":            "Revenue-based (P2P)",
        "cashflow_details":    cf_details,
        "capex_ratio":         capex_ratio,
        "industry":            industry,
        "sector":              sector,
    }

# ─── Public Entry Point ───────────────────────────────────────────────────────

def run_dcf_models(data: dict, rfr: float, fixed_rate: float) -> dict:
    """Run all 4 DCF models and return results + metadata."""
    info      = data.get("info", {})
    income    = data.get("income_stmt",   pd.DataFrame())
    balance   = data.get("balance_sheet", pd.DataFrame())
    cashflow  = data.get("cash_flow",     pd.DataFrame())

    # Shares and EV bridge
    shares       = float(info.get("sharesOutstanding") or info.get("floatShares") or 0)
    bal          = get_balance_series(balance)
    total_cash   = float(bal.get("cash") or info.get("totalCash") or 0)
    total_debt   = float(bal.get("total_debt") or info.get("totalDebt") or 0)
    minority_int = float(bal.get("minority_int") or 0)
    net_cash     = total_cash - total_debt  # can be negative

    # UFCF series
    ufcf_result = calculate_ufcf_series(income, balance, cashflow)
    if isinstance(ufcf_result, tuple):
        ufcf_list, tax_rate = ufcf_result
    else:
        ufcf_list = ufcf_result
        tax_rate  = 0.21

    # Fallback to reported FCF if UFCF fails
    cf_data      = get_cashflow_data(cashflow)
    fcf_reported = cf_data.get("fcf_reported", [])
    if not ufcf_list or all(v is None for v in ufcf_list):
        ufcf_list = fcf_reported

    # Growth rates (blended)
    g_rates = estimate_growth_rates(ufcf_list, info)

    # WACC
    wacc_c = calculate_wacc(info, income, balance, rfr)

    # Sensitivity table
    base_fcf = _positive_base(ufcf_list)
    sens     = build_sensitivity_table(
        base_fcf, wacc_c["wacc"], g_rates["terminal"],
        g_rates, shares, net_cash
    ) if base_fcf else {}

    # Detect loss-making companies → add Path-to-Profitability model
    current_nm = float(info.get("profitMargins") or 0)
    p2p_result = {}
    if current_nm < 0.02 or not _positive_base(ufcf_list):
        # Run P2P for unprofitable or near-zero FCF companies
        p2p_result = dcf_path_to_profitability(data, info, rfr, shares, net_cash)

    def _safe_model(fn, *args, name=""):
        try:
            return fn(*args)
        except Exception as e:
            return {"error": f"{name}: {e}"}

    return {
        "wacc":       _safe_model(dcf_wacc,       ufcf_list, wacc_c, shares, net_cash, minority_int, g_rates, name="WACC"),
        "capm":       _safe_model(dcf_capm,       ufcf_list, info, rfr, shares, net_cash, minority_int, g_rates, name="CAPM"),
        "fixed":      _safe_model(dcf_fixed,      ufcf_list, fixed_rate, shares, net_cash, minority_int, g_rates, name="Fixed"),
        "two_stage":  _safe_model(dcf_two_stage,  ufcf_list, info, rfr, shares, net_cash, minority_int, g_rates, name="Two-Stage"),
        "p2p":        p2p_result,
        "wacc_components":  wacc_c,
        "growth_rates":     g_rates,
        "ufcf_list":        ufcf_list,
        "fcf_list":         ufcf_list,       # backward compat alias
        "rfr":              rfr,
        "current_price":    float(info.get("currentPrice") or info.get("regularMarketPrice") or 0),
        "shares":           shares,
        "net_cash":         net_cash,
        "minority_int":     minority_int,
        "tax_rate":         tax_rate,
        "sensitivity":      sens,
        "fcf_type":         "UFCF",
    }


# ─── Fundamentals Aggregator ──────────────────────────────────────────────────

def growth_rates_from_series(series: dict) -> list:
    vals = [v for _, v in sorted(series.items(), reverse=True) if v]
    if len(vals) < 2:
        return []
    rates = []
    for i in range(len(vals) - 1):
        try:
            d = vals[i+1]
            if d and abs(d) > 1:   # avoid near-zero denominators
                rates.append((vals[i] / d) - 1)
        except (ZeroDivisionError, TypeError):
            pass
    return rates


def margin_series(numerator: dict, denominator: dict) -> dict:
    out = {}
    for yr in numerator:
        try:
            d = denominator.get(yr)
            if d and abs(d) > 1:
                out[yr] = numerator[yr] / d
        except (ZeroDivisionError, TypeError):
            pass
    return out


def analyze_fundamentals(data: dict) -> dict:
    info     = data.get("info", {})
    income   = data.get("income_stmt",   pd.DataFrame())
    balance  = data.get("balance_sheet", pd.DataFrame())
    cashflow = data.get("cash_flow",     pd.DataFrame())

    inc = get_income_series(income)
    bal = get_balance_series(balance)
    cfs = get_cashflow_data(cashflow)

    rev = inc.get("revenue", {})
    gp  = inc.get("gross_profit", {})
    oi  = inc.get("ebit", {})
    ni  = inc.get("net_income", {})

    return {
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
        "revenue_series":       rev,
        "gross_profit_series":  gp,
        "op_income_series":     oi,
        "net_income_series":    ni,
        "gross_margin_series":  margin_series(gp, rev),
        "op_margin_series":     margin_series(oi, rev),
        "net_margin_series":    margin_series(ni, rev),
        "rev_growth_rates":     growth_rates_from_series(rev),
        "ni_growth_rates":      growth_rates_from_series(ni),
        "balance":  bal,
        "cashflows": cfs,
    }
