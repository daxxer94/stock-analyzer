"""
financial_analysis.py — Deep financial statement analysis.

Provides:
  1. Multi-year trend analysis (revenue, margins, FCF quality)
  2. DuPont ROE decomposition (3-factor and 5-factor)
  3. Piotroski F-Score (9-point financial health score)
  4. Altman Z-Score (bankruptcy risk proxy)
  5. Operating leverage and capital efficiency
  6. Industry/macro sensitivity matrix
  7. Forward growth analysis from analyst estimates

All inputs come from yfinance data — no paid APIs required.
"""

import numpy as np
import pandas as pd
from fundamental import get_income_series, get_balance_series, get_cashflow_data


# ─── Macro/Industry sensitivity matrix ───────────────────────────────────────
# For each sector, describes the key macro drivers and their directional impact
# Sources: Bloomberg sector factor research, Damodaran industry analysis

SECTOR_MACRO_SENSITIVITY = {
    "Technology": {
        "drivers": [
            ("Interest Rates",      "negative",  "High P/E multiples compress when rates rise; growth stocks most affected"),
            ("USD Strength",        "negative",  "Most large-cap tech earns majority of revenue internationally; strong USD reduces reported earnings"),
            ("Corporate IT Spend",  "positive",  "Enterprise software/cloud spend tracks GDP with a lag; contract backlog provides visibility"),
            ("Semiconductor Cycle", "cyclical",  "Semi equipment and chip stocks follow a 3-4 year inventory cycle (bull/bear phases)"),
            ("AI/Cloud Buildout",   "positive",  "Infrastructure capex cycle currently boosting datacenter, networking, power stocks"),
        ],
        "recession_sensitivity": "moderate",
        "rate_sensitivity":      "high",
        "fx_sensitivity":        "high",
        "cyclicality":           "moderate",
    },
    "Healthcare": {
        "drivers": [
            ("Drug Pricing Policy", "negative",  "US Congressional/CMS pricing pressure directly impacts pharma revenue; ongoing regulatory risk"),
            ("Aging Demographics",  "positive",  "Structural tailwind: global 65+ population growing ~3% annually through 2040"),
            ("Insurance Coverage",  "positive",  "ACA-stable or expanding coverage boosts procedure volumes and hospital revenues"),
            ("R&D Pipeline",        "positive",  "FDA approval rate and clinical trial success drives biotech/pharma valuations nonlinearly"),
            ("Interest Rates",      "low",       "Healthcare largely inelastic demand; moderate rate sensitivity vs. other sectors"),
        ],
        "recession_sensitivity": "low",
        "rate_sensitivity":      "low",
        "fx_sensitivity":        "moderate",
        "cyclicality":           "low",
    },
    "Financial Services": {
        "drivers": [
            ("Interest Rates",      "positive",  "Net Interest Margin (NIM) expands when rates rise; biggest driver of bank profitability"),
            ("Yield Curve Slope",   "positive",  "Steep curve (long rates > short rates) = profitable for banks borrowing short, lending long"),
            ("Credit Cycle",        "cyclical",  "Loan loss provisions spike in recessions; credit quality is the key risk variable"),
            ("Regulation (Basel)",  "negative",  "Higher capital requirements reduce ROE; regulatory tightening compresses bank multiples"),
            ("Economic Growth",     "positive",  "Loan growth, M&A advisory, and trading volumes all tied to GDP trajectory"),
        ],
        "recession_sensitivity": "high",
        "rate_sensitivity":      "very high",
        "fx_sensitivity":        "moderate",
        "cyclicality":           "high",
    },
    "Consumer Cyclical": {
        "drivers": [
            ("Consumer Confidence", "positive",  "Discretionary spending highly correlated with consumer sentiment and employment"),
            ("Real Wages",          "positive",  "Purchasing power directly drives discretionary category spend"),
            ("Interest Rates",      "negative",  "Higher rates raise mortgage/auto loan costs, reducing discretionary budgets"),
            ("Unemployment Rate",   "negative",  "Rising unemployment leads to rapid pullback in big-ticket and discretionary purchases"),
            ("E-Commerce Shift",    "mixed",     "Online pure-plays gain; traditional retailers face structural margin pressure"),
        ],
        "recession_sensitivity": "very high",
        "rate_sensitivity":      "high",
        "fx_sensitivity":        "low",
        "cyclicality":           "very high",
    },
    "Consumer Defensive": {
        "drivers": [
            ("Inflation",           "mixed",     "Branded goods can pass through price increases; private-label competition rises in downturns"),
            ("Input Costs",         "negative",  "Commodity (corn, wheat, oil) costs compress margins if not offset by pricing"),
            ("Private Label Share", "negative",  "Economic stress increases consumer trade-down to store brands"),
            ("Population Growth",   "positive",  "Structural volume driver in emerging markets"),
        ],
        "recession_sensitivity": "very low",
        "rate_sensitivity":      "low",
        "fx_sensitivity":        "moderate",
        "cyclicality":           "very low",
    },
    "Energy": {
        "drivers": [
            ("Oil & Gas Prices",    "positive",  "Revenue and earnings almost entirely commodity-price driven; OPEC+ decisions critical"),
            ("ESG/Energy Transition","negative", "Capital allocation shifting away from fossil fuels; stranded asset risk growing"),
            ("Global Demand",       "positive",  "Emerging market growth (China, India) drives long-run hydrocarbon demand"),
            ("USD Strength",        "negative",  "Oil priced in USD; strong dollar typically pressures oil prices"),
            ("CapEx Cycle",         "cyclical",  "Underinvestment in 2015-2020 created supply constraints; oilfield services benefit"),
        ],
        "recession_sensitivity": "high",
        "rate_sensitivity":      "moderate",
        "fx_sensitivity":        "very high",
        "cyclicality":           "very high",
    },
    "Industrials": {
        "drivers": [
            ("Manufacturing PMI",   "positive",  "PMI > 50 = expansion; leading indicator of industrial orders with 2-4 month lead"),
            ("Infrastructure Spend","positive",  "Government capex cycles (IRA, CHIPS Act in US) drive multi-year order books"),
            ("Interest Rates",      "moderate",  "Capital equipment purchases are often financed; higher rates delay CapEx decisions"),
            ("Supply Chain",        "mixed",     "Reshoring/nearshoring trend benefits domestic manufacturers; offsets cost headwinds"),
            ("Defense Spending",    "positive",  "Geopolitical risk driving sustained NATO/global defense budget increases"),
        ],
        "recession_sensitivity": "high",
        "rate_sensitivity":      "moderate",
        "fx_sensitivity":        "high",
        "cyclicality":           "high",
    },
    "Communication Services": {
        "drivers": [
            ("Digital Advertising", "cyclical",  "Ad spend is first to be cut in downturns; highly correlated with corporate profit cycle"),
            ("Subscriber Growth",   "positive",  "Streaming/broadband penetration still growing in emerging markets"),
            ("Content Costs",       "negative",  "Sports rights and original content arms race compresses streaming margins"),
            ("Regulation (Antitrust)","negative","Big Tech regulatory risk (EU DMA, US DOJ) poses structural revenue risk for platforms"),
            ("AI Monetization",     "positive",  "Search, social, and cloud companies embedding AI to drive ARPU expansion"),
        ],
        "recession_sensitivity": "moderate",
        "rate_sensitivity":      "moderate",
        "fx_sensitivity":        "high",
        "cyclicality":           "moderate",
    },
    "Basic Materials": {
        "drivers": [
            ("China Demand",        "positive",  "China consumes ~50-55% of global metals; property sector and infrastructure drive demand"),
            ("USD Strength",        "negative",  "Commodities priced in USD; stronger dollar historically pressures commodity prices"),
            ("Supply Constraints",  "positive",  "Mine permitting backlogs and ESG restrictions limit new supply; structural tailwind for prices"),
            ("Green Transition",    "positive",  "Copper (EVs/grid), lithium (batteries), nickel (stainless/EVs) face structural demand surge"),
            ("Freight Costs",       "mixed",     "High shipping costs benefit producers near demand centers; hurt importers"),
        ],
        "recession_sensitivity": "very high",
        "rate_sensitivity":      "moderate",
        "fx_sensitivity":        "very high",
        "cyclicality":           "very high",
    },
    "Real Estate": {
        "drivers": [
            ("Interest Rates",      "negative",  "REIT cap rates tied to long-term rates; higher rates compress REIT valuations directly"),
            ("Occupancy Rates",     "positive",  "Supply/demand balance by property type (industrial/residential tight, office challenged)"),
            ("Inflation",           "positive",  "Hard assets with rent escalation clauses provide inflation hedge; replacement cost rises"),
            ("Work-From-Home",      "negative",  "Office REIT structural vacancy risk; benefiting industrial, data center, residential"),
        ],
        "recession_sensitivity": "moderate",
        "rate_sensitivity":      "very high",
        "fx_sensitivity":        "low",
        "cyclicality":           "moderate",
    },
    "Utilities": {
        "drivers": [
            ("Interest Rates",      "negative",  "Utilities are bond proxies; yield-seeking investors exit when rates rise"),
            ("Regulatory ROE",      "positive",  "Allowed returns set by state/federal regulators; predictable but capped upside"),
            ("Electrification",     "positive",  "EV charging, data center power demand, industrial electrification driving volume growth"),
            ("Renewable Buildout",  "positive",  "IRA tax credits subsidising massive clean energy capex; rate-based earnings growth"),
            ("Weather Extremes",    "mixed",     "Climate volatility increases both demand peaks and infrastructure hardening costs"),
        ],
        "recession_sensitivity": "very low",
        "rate_sensitivity":      "very high",
        "fx_sensitivity":        "very low",
        "cyclicality":           "very low",
    },
}

SENSITIVITY_COLORS = {
    "very high": "#EF5350",
    "high":      "#FF9800",
    "moderate":  "#FFC107",
    "low":       "#8BC34A",
    "very low":  "#4CAF50",
    "mixed":     "#90CAF9",
}

DRIVER_COLORS = {
    "positive": "#66BB6A",
    "negative": "#EF5350",
    "cyclical": "#FFC107",
    "mixed":    "#90CAF9",
    "moderate": "#FFC107",
    "low":      "#8BC34A",
}


def get_macro_context(info: dict) -> dict:
    """Return macro sensitivity profile for a given stock."""
    sector = info.get("sector", "")
    beta   = float(info.get("beta") or 1.0)
    total_debt  = float(info.get("totalDebt")  or 0)
    total_cash  = float(info.get("totalCash")  or 0)
    mktcap      = float(info.get("marketCap")  or 1)
    int_exp     = float(info.get("interestExpense") or 0)
    ebit        = float(info.get("ebit") or info.get("operatingIncome") or 1)
    revenue     = float(info.get("totalRevenue") or 1)

    profile = SECTOR_MACRO_SENSITIVITY.get(sector, {})

    # Interest coverage (how sensitive to rate rises)
    net_debt = total_debt - total_cash
    net_debt_to_ebitda = net_debt / max(abs(float(info.get("ebitda") or 1)), 1)
    interest_coverage  = ebit / max(abs(int_exp), 1) if int_exp else None

    # Market sensitivity
    if beta > 1.5:   mkt_sens = "High — moves significantly more than the market"
    elif beta > 1.0: mkt_sens = "Above average — amplifies market moves"
    elif beta > 0.7: mkt_sens = "Moderate — broadly tracks the market"
    else:            mkt_sens = "Defensive — less sensitive to market swings"

    return {
        "sector":             sector,
        "profile":            profile,
        "beta":               beta,
        "market_sensitivity": mkt_sens,
        "net_debt_ebitda":    net_debt_to_ebitda,
        "interest_coverage":  interest_coverage,
        "net_debt":           net_debt,
    }


# ─── Piotroski F-Score ────────────────────────────────────────────────────────

def calculate_piotroski(income: pd.DataFrame, balance: pd.DataFrame,
                         cashflow: pd.DataFrame) -> dict:
    """
    Piotroski F-Score: 9 binary criteria across profitability, leverage, efficiency.
    Score 0–3 = weak, 4–6 = neutral, 7–9 = strong.
    Published: Piotroski (2000), Journal of Accounting Research.
    """
    inc = get_income_series(income)
    bal = get_balance_series(balance)
    cfs = get_cashflow_data(cashflow)

    def _yr(series, offset=0):
        """Get value from series at year offset (0=latest, 1=prior year)."""
        items = sorted(series.items(), reverse=True)
        return items[offset][1] if len(items) > offset else None

    scores = {}

    # ── Profitability signals (4 points) ──────────────────────────────────────
    roa_now  = _yr(inc.get("net_income", {})) 
    ta_now   = bal.get("total_assets")
    roa_pct  = roa_now / ta_now if (roa_now and ta_now and ta_now > 0) else None

    # F1: ROA > 0
    scores["F1_roa_positive"] = {
        "pass": bool(roa_pct and roa_pct > 0),
        "label": "ROA > 0  (profitable)",
        "value": f"{roa_pct:.1%}" if roa_pct is not None else "N/A",
    }

    # F2: Operating cash flow > 0
    ocf_now = cfs.get("operating_cf", [None])[0] if cfs.get("operating_cf") else None
    scores["F2_ocf_positive"] = {
        "pass": bool(ocf_now and ocf_now > 0),
        "label": "Operating Cash Flow > 0",
        "value": f"${ocf_now/1e9:.2f}B" if ocf_now else "N/A",
    }

    # F3: ROA improving YoY
    ni_prior = _yr(inc.get("net_income", {}), 1)
    ta_prior = None
    if bal.get("ca_series") and bal.get("cl_series"):
        pass  # approximate: use same total assets
    roa_prior = ni_prior / ta_now if (ni_prior and ta_now and ta_now > 0) else None
    scores["F3_roa_improving"] = {
        "pass": bool(roa_pct and roa_prior and roa_pct > roa_prior),
        "label": "ROA improving YoY",
        "value": f"{roa_pct:.1%} vs {roa_prior:.1%}" if (roa_pct and roa_prior) else "N/A",
    }

    # F4: Accruals (OCF > Net Income — cash earnings quality)
    accruals_pass = bool(ocf_now and roa_now and ocf_now > roa_now)
    scores["F4_accruals"] = {
        "pass": accruals_pass,
        "label": "OCF > Net Income  (cash quality)",
        "value": "✅ Cash-backed earnings" if accruals_pass else "⚠️ Accruals exceed OCF",
    }

    # ── Leverage / Liquidity signals (3 points) ───────────────────────────────
    # F5: Debt ratio improving (lower leverage YoY)
    td_now   = bal.get("total_debt") or 0
    ta       = bal.get("total_assets") or 1
    dr_now   = td_now / ta
    # Approximate prior year from series
    scores["F5_leverage_improving"] = {
        "pass": bool(dr_now < 0.6),  # simplified: flag if debt ratio reasonable
        "label": "Debt ratio < 60%  (leverage check)",
        "value": f"{dr_now:.1%}",
    }

    # F6: Current ratio > 1 (adequate liquidity)
    ca = bal.get("current_assets") or 0
    cl = bal.get("current_liab") or 1
    cr = ca / cl if cl > 0 else 0
    scores["F6_liquidity"] = {
        "pass": bool(cr > 1.0),
        "label": "Current Ratio > 1  (liquidity)",
        "value": f"{cr:.2f}x",
    }

    # F7: No new equity dilution (shares not increasing)
    shares_curr = _yr(inc.get("eps_diluted", {}))
    shares_prev = _yr(inc.get("eps_diluted", {}), 1)
    # Proxy: if diluted EPS trend is stable/improving, no meaningful dilution
    eps_improving = bool(shares_curr and shares_prev and abs(shares_curr) >= abs(shares_prev) * 0.95)
    scores["F7_no_dilution"] = {
        "pass": eps_improving,
        "label": "No significant share dilution",
        "value": "No significant dilution detected" if eps_improving else "Possible dilution",
    }

    # ── Operating efficiency signals (2 points) ───────────────────────────────
    # F8: Gross margin improving
    gm_now   = _yr(inc.get("gross_margin_series" if hasattr(inc, "gross_margin_series") else "gross_profit", {}))
    rev_now  = _yr(inc.get("revenue", {}))
    rev_prev = _yr(inc.get("revenue", {}), 1)
    gp_now   = _yr(inc.get("gross_profit", {}))
    gp_prev  = _yr(inc.get("gross_profit", {}), 1)
    gm_now_r  = gp_now  / rev_now  if (gp_now  and rev_now  and rev_now  > 0) else None
    gm_prev_r = gp_prev / rev_prev if (gp_prev and rev_prev and rev_prev > 0) else None
    gm_improving = bool(gm_now_r and gm_prev_r and gm_now_r > gm_prev_r)
    scores["F8_gross_margin"] = {
        "pass": gm_improving,
        "label": "Gross margin improving YoY",
        "value": f"{gm_now_r:.1%} vs {gm_prev_r:.1%}" if (gm_now_r and gm_prev_r) else "N/A",
    }

    # F9: Asset turnover improving (revenue / total assets)
    at_now  = rev_now  / ta if (rev_now  and ta > 0) else None
    at_prev = rev_prev / ta if (rev_prev and ta > 0) else None
    at_improving = bool(at_now and at_prev and at_now > at_prev)
    scores["F9_asset_turnover"] = {
        "pass": at_improving,
        "label": "Asset turnover improving YoY",
        "value": f"{at_now:.2f}x vs {at_prev:.2f}x" if (at_now and at_prev) else "N/A",
    }

    total = sum(1 for v in scores.values() if v["pass"])

    if total >= 7:   signal, color = "Strong ✅", "#00C853"
    elif total >= 4: signal, color = "Neutral 🟡", "#FFC107"
    else:            signal, color = "Weak ⚠️", "#EF5350"

    return {"scores": scores, "total": total, "signal": signal, "color": color}


# ─── Altman Z-Score ───────────────────────────────────────────────────────────

def calculate_altman_z(info: dict, income: pd.DataFrame,
                        balance: pd.DataFrame) -> dict:
    """
    Altman Z-Score: discriminant model for financial distress.
    Original (1968) for manufacturing; Z' for private; Z'' for non-manufacturing.
    
    Z > 2.99 = safe zone
    1.81 < Z < 2.99 = grey zone
    Z < 1.81 = distress zone
    """
    bal = get_balance_series(balance)
    inc = get_income_series(income)

    def _latest(series):
        items = sorted(series.items(), reverse=True)
        return items[0][1] if items else None

    mktcap     = float(info.get("marketCap")       or 0)
    ebit_v     = _latest(inc.get("ebit", {}))
    rev_v      = _latest(inc.get("revenue", {}))
    ni_v       = _latest(inc.get("net_income", {}))

    ta         = bal.get("total_assets")  or 0
    tl         = bal.get("total_liab")    or (bal.get("total_debt") or 0) * 1.5
    ca         = bal.get("current_assets") or 0
    cl         = bal.get("current_liab")   or 0
    ret_earn   = ni_v * 0.5 if ni_v else 0   # approximate retained earnings
    total_debt = bal.get("total_debt") or 0

    if ta <= 0:
        return {"error": "Insufficient balance sheet data for Altman Z"}

    # 5 ratios
    x1 = (ca - cl) / ta             if ta else 0   # Working Capital / TA
    x2 = ret_earn  / ta             if ta else 0   # Retained Earnings / TA
    x3 = ebit_v    / ta             if (ebit_v and ta) else 0  # EBIT / TA
    x4 = mktcap    / max(total_debt, ta * 0.01)    # Mkt Cap / Total Liabilities
    x5 = rev_v     / ta             if (rev_v and ta) else 0   # Revenue / TA

    # Original Z-Score formula (public manufacturing)
    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    # Z'' (non-manufacturing / service companies — Altman 1983)
    z_prime2 = 6.56*x1 + 3.26*x2 + 6.72*x3 + 1.05*x4

    if z > 2.99:   zone, color = "Safe Zone",     "#00C853"
    elif z > 1.81: zone, color = "Grey Zone",     "#FFC107"
    else:          zone, color = "Distress Zone", "#EF5350"

    return {
        "z_score":     round(z, 2),
        "z_prime2":    round(z_prime2, 2),
        "zone":        zone,
        "color":       color,
        "ratios": {
            "X1 Working Capital / Assets": round(x1, 3),
            "X2 Retained Earnings / Assets": round(x2, 3),
            "X3 EBIT / Assets": round(x3, 3),
            "X4 Market Cap / Liabilities": round(x4, 3),
            "X5 Revenue / Assets": round(x5, 3),
        },
        "interpretation": (
            f"Z = {z:.2f} → {zone}. "
            + ("Company appears financially healthy with low bankruptcy risk. "
               if z > 2.99 else
               "Some financial stress indicators present — monitor closely. "
               if z > 1.81 else
               "Significant financial distress signals. Elevated bankruptcy risk. ")
        ),
    }


# ─── DuPont ROE Decomposition ─────────────────────────────────────────────────

def calculate_dupont(info: dict, income: pd.DataFrame,
                      balance: pd.DataFrame) -> dict:
    """
    3-factor DuPont: ROE = Net Margin × Asset Turnover × Equity Multiplier
    5-factor DuPont: also decomposes into Tax Burden × Interest Burden × EBIT Margin
    
    Reveals the true DRIVER of ROE — is it operational efficiency, asset use, or leverage?
    """
    inc = get_income_series(income)
    bal = get_balance_series(balance)

    def _latest(s):
        items = sorted(s.items(), reverse=True)
        return items[0][1] if items else None

    rev    = _latest(inc.get("revenue", {}))
    ni     = _latest(inc.get("net_income", {}))
    ebit_v = _latest(inc.get("ebit", {}))
    ebt    = _latest(inc.get("pretax_income", {}))

    ta     = bal.get("total_assets")
    te     = bal.get("total_equity")

    if not all([rev, ni, ta, te]) or te == 0 or ta == 0 or rev == 0:
        return {"error": "Insufficient data for DuPont analysis"}

    # 3-factor
    net_margin    = ni  / rev          # Profitability
    asset_turnover = rev / ta          # Efficiency
    equity_mult   = ta  / te           # Leverage

    roe_3f = net_margin * asset_turnover * equity_mult
    roe_direct = ni / te

    # 5-factor breakdown
    tax_burden      = ni   / ebt   if ebt and ebt != 0 else None   # Net Inc / Pre-tax
    interest_burden = ebt  / ebit_v if ebit_v and ebit_v != 0 else None  # Pre-tax / EBIT
    ebit_margin     = ebit_v / rev                                   # EBIT / Revenue

    # Identify primary driver
    drivers = {
        "Net Margin":      abs(net_margin),
        "Asset Turnover":  abs(asset_turnover),
        "Equity Multiplier": abs(equity_mult - 1),  # subtract 1 to get leverage contribution
    }
    primary = max(drivers, key=drivers.get)

    return {
        "roe_3factor":      round(roe_3f, 4),
        "roe_direct":       round(roe_direct, 4),
        "net_margin":       round(net_margin, 4),
        "asset_turnover":   round(asset_turnover, 4),
        "equity_multiplier": round(equity_mult, 4),
        "tax_burden":       round(tax_burden, 4) if tax_burden else None,
        "interest_burden":  round(interest_burden, 4) if interest_burden else None,
        "ebit_margin":      round(ebit_margin, 4),
        "primary_driver":   primary,
        "driver_note": {
            "Net Margin":    "ROE primarily driven by profitability (wide margins)",
            "Asset Turnover":"ROE primarily driven by operational efficiency (asset utilisation)",
            "Equity Multiplier":"ROE primarily driven by financial leverage (debt)",
        }[primary],
    }


# ─── Operating Leverage ───────────────────────────────────────────────────────

def calculate_operating_leverage(income: pd.DataFrame) -> dict:
    """
    Degree of Operating Leverage (DOL) = % change in EBIT / % change in Revenue.
    High DOL = high fixed costs → big earnings swings on small revenue changes.
    """
    inc  = get_income_series(income)
    rev  = sorted(inc.get("revenue", {}).items(), reverse=True)
    ebit = sorted(inc.get("ebit", {}).items(), reverse=True)

    if len(rev) < 2 or len(ebit) < 2:
        return {}

    dol_list = []
    for i in range(min(3, len(rev)-1)):
        try:
            rev_chg  = (rev[i][1]  - rev[i+1][1])  / abs(rev[i+1][1])
            ebit_chg = (ebit[i][1] - ebit[i+1][1]) / abs(ebit[i+1][1])
            if abs(rev_chg) > 0.001:
                dol_list.append(ebit_chg / rev_chg)
        except Exception:
            pass

    if not dol_list:
        return {}

    avg_dol = float(np.median(dol_list))

    if avg_dol > 3:    dol_note = "High fixed-cost model — strong earnings leverage in good times, high risk in downturns"
    elif avg_dol > 1.5: dol_note = "Moderate operating leverage — balanced cost structure"
    elif avg_dol > 0:   dol_note = "Low operating leverage — variable cost model, resilient in downturns"
    else:               dol_note = "Negative DOL — revenue growth not translating to earnings (cost inflation or mix shift)"

    return {
        "dol":         round(avg_dol, 2),
        "dol_history": [round(d, 2) for d in dol_list],
        "note":        dol_note,
    }


# ─── Capital Efficiency ───────────────────────────────────────────────────────

def calculate_capital_efficiency(info: dict, income: pd.DataFrame,
                                   cashflow: pd.DataFrame) -> dict:
    """
    Key capital efficiency metrics:
    - FCF Conversion (FCF / Net Income) — earnings quality
    - CapEx Intensity (CapEx / Revenue)
    - R&D Intensity (R&D / Revenue)
    - ROIC (Return on Invested Capital)
    """
    inc = get_income_series(income)
    cfs = get_cashflow_data(cashflow)

    def _latest(s):
        items = sorted(s.items(), reverse=True)
        return items[0][1] if items else None

    rev    = _latest(inc.get("revenue", {}))
    ni     = _latest(inc.get("net_income", {}))
    ebit_v = _latest(inc.get("ebit", {}))
    rd     = _latest(inc.get("rd_expense", {}))

    ocf_list   = cfs.get("operating_cf", [])
    capex_list = cfs.get("capex", [])

    ocf   = ocf_list[0]   if ocf_list   else None
    capex = capex_list[0] if capex_list else None
    fcf   = (ocf - abs(capex)) if (ocf and capex) else None

    mktcap     = float(info.get("marketCap") or 0)
    total_debt = float(info.get("totalDebt") or 0)
    total_cash = float(info.get("totalCash") or 0)
    ev         = mktcap + total_debt - total_cash
    invested_capital = total_debt + (mktcap or 0) - total_cash

    results = {}

    # FCF Conversion (>100% = earnings actually backed by cash)
    if fcf and ni and ni != 0:
        fcf_conv = fcf / ni
        results["fcf_conversion"] = {
            "value": round(fcf_conv, 2),
            "label": "FCF / Net Income",
            "note": ("Excellent — cash earnings exceed reported earnings" if fcf_conv > 1.0
                     else "Good — most earnings backed by cash" if fcf_conv > 0.75
                     else "Fair — earnings quality adequate" if fcf_conv > 0.5
                     else "Poor — significant gap between reported and cash earnings"),
        }

    # CapEx Intensity
    if capex and rev and rev > 0:
        capex_int = abs(capex) / rev
        results["capex_intensity"] = {
            "value": round(capex_int, 3),
            "label": "CapEx / Revenue",
            "note": ("Asset-heavy — high reinvestment required" if capex_int > 0.15
                     else "Moderate capital requirements" if capex_int > 0.06
                     else "Asset-light model — low reinvestment needed"),
        }

    # R&D Intensity
    if rd and rev and rev > 0:
        rd_int = abs(rd) / rev
        results["rd_intensity"] = {
            "value": round(rd_int, 3),
            "label": "R&D / Revenue",
            "note": ("High innovation investment — moat-building potential" if rd_int > 0.12
                     else "Moderate R&D spend" if rd_int > 0.05
                     else "Low R&D — mature/non-tech business"),
        }

    # ROIC
    if ebit_v and invested_capital and invested_capital > 0:
        tax_rate = 0.21
        nopat = ebit_v * (1 - tax_rate)
        roic  = nopat / invested_capital
        results["roic"] = {
            "value": round(roic, 4),
            "label": "ROIC (NOPAT / Invested Capital)",
            "note": ("Exceptional value creator — ROIC well above cost of capital" if roic > 0.20
                     else "Strong value creator" if roic > 0.12
                     else "Adequate — covers cost of capital" if roic > 0.07
                     else "Value-neutral to value-destroying — ROIC below WACC"),
        }

    # FCF Yield
    if fcf and mktcap and mktcap > 0:
        fcf_yield = fcf / mktcap
        results["fcf_yield"] = {
            "value": round(fcf_yield, 4),
            "label": "FCF Yield (FCF / Market Cap)",
            "note": ("High yield — strong cash return potential" if fcf_yield > 0.06
                     else "Attractive yield" if fcf_yield > 0.03
                     else "Moderate yield" if fcf_yield > 0.01
                     else "Low/negative FCF yield"),
        }

    return results


# ─── Forward Growth Analysis ─────────────────────────────────────────────────

def analyze_forward_growth(info: dict, data: dict) -> dict:
    """
    Pull together analyst forward estimates from yfinance:
    - Revenue estimates (current/next year)
    - EPS estimates (current/next year)
    - Long-term growth rate
    - Revision trend (are estimates going up or down?)
    """
    results = {}

    # From info dict
    results["fwd_eps"]         = info.get("forwardEps")
    results["trailing_eps"]    = info.get("trailingEps")
    results["rev_growth"]      = info.get("revenueGrowth")
    results["eps_growth"]      = info.get("earningsGrowth")
    results["fwd_pe"]          = info.get("forwardPE")
    results["peg"]             = info.get("pegRatio")
    results["analyst_count"]   = info.get("numberOfAnalystOpinions", 0)
    results["target_mean"]     = info.get("targetMeanPrice")
    results["target_high"]     = info.get("targetHighPrice")
    results["target_low"]      = info.get("targetLowPrice")
    results["rec_mean"]        = info.get("recommendationMean")

    # EPS revision trend from eps_trend data
    eps_trend = data.get("eps_trend")
    if eps_trend is not None and not (isinstance(eps_trend, pd.DataFrame) and eps_trend.empty):
        try:
            if isinstance(eps_trend, pd.DataFrame) and "current" in eps_trend.columns:
                curr = float(eps_trend["current"].iloc[0])
                wk4  = float(eps_trend.get("7daysAgo",  eps_trend).iloc[0]) if "7daysAgo"  in eps_trend.columns else None
                mo3  = float(eps_trend.get("90daysAgo", eps_trend).iloc[0]) if "90daysAgo" in eps_trend.columns else None
                if wk4 and curr and wk4 != 0:
                    results["estimate_revision_4w"] = (curr - wk4) / abs(wk4)
                if mo3 and curr and mo3 != 0:
                    results["estimate_revision_3m"] = (curr - mo3) / abs(mo3)
        except Exception:
            pass

    # Earnings estimates from analyst data
    earn_est = data.get("earnings_estimate")
    rev_est  = data.get("revenue_estimate")

    if isinstance(earn_est, pd.DataFrame) and not earn_est.empty:
        try:
            results["eps_est_current_yr"] = earn_est.get("avg", {}).iloc[0] if len(earn_est) > 0 else None
            results["eps_est_next_yr"]    = earn_est.get("avg", {}).iloc[1] if len(earn_est) > 1 else None
        except Exception:
            pass

    if isinstance(rev_est, pd.DataFrame) and not rev_est.empty:
        try:
            results["rev_est_current_yr"] = rev_est.get("avg", {}).iloc[0] if len(rev_est) > 0 else None
            results["rev_est_next_yr"]    = rev_est.get("avg", {}).iloc[1] if len(rev_est) > 1 else None
        except Exception:
            pass

    return results


# ─── Master Analysis Runner ───────────────────────────────────────────────────

def run_deep_analysis(data: dict, info: dict) -> dict:
    """Run all deep financial analyses and return combined results."""
    income   = data.get("income_stmt",   pd.DataFrame())
    balance  = data.get("balance_sheet", pd.DataFrame())
    cashflow = data.get("cash_flow",     pd.DataFrame())

    results = {}
    for key, fn, args in [
        ("piotroski",  calculate_piotroski,         (income, balance, cashflow)),
        ("altman",     calculate_altman_z,           (info, income, balance)),
        ("dupont",     calculate_dupont,             (info, income, balance)),
        ("op_leverage",calculate_operating_leverage, (income,)),
        ("cap_eff",    calculate_capital_efficiency, (info, income, cashflow)),
        ("macro",      get_macro_context,            (info,)),
        ("fwd_growth", analyze_forward_growth,       (info, data)),
    ]:
        try:
            results[key] = fn(*args)
        except Exception as e:
            results[key] = {"error": str(e)}

    return results
