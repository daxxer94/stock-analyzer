"""
scoring.py - Composite scoring & signal generation.

Weights:
  Fundamental  40%
  Valuation    25%
  Technical    20%
  Sentiment    15%

Each sub-score 0–10. Output: composite 0–10 + Buy/Hold/Sell signal.
"""


def _s(v, lo, hi, pts):
    """Map a value to points using breakpoint table [(threshold, points)]."""
    if v is None:
        return None
    for threshold, p in pts:
        if v >= threshold:
            return p
    return pts[-1][1]


def score_fundamental(info: dict, cashflows: dict) -> tuple[float, list]:
    total, max_pts, msgs = 0, 0, []

    def add(pts, max_p, emoji, text):
        nonlocal total, max_pts
        total += pts; max_pts += max_p
        msgs.append((emoji, text))

    # Revenue Growth
    rg = info.get("revenueGrowth")
    if rg is not None:
        if rg > .20:   add(2, 2, "✅", f"Revenue growth {rg:.1%} — strong")
        elif rg > .08: add(1.5, 2, "✅", f"Revenue growth {rg:.1%} — good")
        elif rg > 0:   add(1, 2, "🟡", f"Revenue growth {rg:.1%} — moderate")
        elif rg > -.05:add(0.5, 2, "🟡", f"Revenue growth {rg:.1%} — slight decline")
        else:           add(0, 2, "🔴", f"Revenue growth {rg:.1%} — declining")

    # Earnings Growth
    eg = info.get("earningsGrowth")
    if eg is not None:
        if eg > .20:   add(2, 2, "✅", f"Earnings growth {eg:.1%} — strong")
        elif eg > .08: add(1.5, 2, "✅", f"Earnings growth {eg:.1%} — good")
        elif eg > 0:   add(1, 2, "🟡", f"Earnings growth {eg:.1%} — moderate")
        else:           add(0, 2, "🔴", f"Earnings growth {eg:.1%} — declining")

    # Net Margin
    nm = info.get("profitMargins")
    if nm is not None:
        if nm > .25:   add(2, 2, "✅", f"Net margin {nm:.1%} — excellent")
        elif nm > .12: add(1.5, 2, "✅", f"Net margin {nm:.1%} — good")
        elif nm > .05: add(1, 2, "🟡", f"Net margin {nm:.1%} — adequate")
        elif nm > 0:   add(0.5, 2, "🟡", f"Net margin {nm:.1%} — thin")
        else:           add(0, 2, "🔴", f"Net margin {nm:.1%} — unprofitable")

    # ROE
    roe = info.get("returnOnEquity")
    if roe is not None:
        if roe > .25:  add(2, 2, "✅", f"ROE {roe:.1%} — exceptional")
        elif roe > .15:add(1.5, 2, "✅", f"ROE {roe:.1%} — strong")
        elif roe > .08:add(1, 2, "🟡", f"ROE {roe:.1%} — adequate")
        elif roe > 0:  add(0.5, 2, "🟡", f"ROE {roe:.1%} — low")
        else:           add(0, 2, "🔴", f"ROE {roe:.1%} — negative")

    # Debt/Equity (yfinance returns as %, e.g. 120 = 1.2x)
    de = info.get("debtToEquity")
    if de is not None:
        de_norm = de / 100  # normalize
        if de_norm < .20:   add(2, 2, "✅", f"Debt/Equity {de_norm:.2f}x — very low leverage")
        elif de_norm < .50: add(1.5, 2, "✅", f"Debt/Equity {de_norm:.2f}x — conservative")
        elif de_norm < 1.0: add(1, 2, "🟡", f"Debt/Equity {de_norm:.2f}x — moderate leverage")
        elif de_norm < 2.0: add(0.5, 2, "🔴", f"Debt/Equity {de_norm:.2f}x — high leverage")
        else:                add(0, 2, "🔴", f"Debt/Equity {de_norm:.2f}x — very high leverage")

    # FCF Yield
    fcf_list = cashflows.get("fcf", [])
    mktcap = info.get("marketCap") or 0
    if fcf_list and mktcap > 0:
        fcf = fcf_list[0]
        yield_ = fcf / mktcap
        if fcf > 0 and yield_ > .06: add(2, 2, "✅", f"FCF yield {yield_:.1%} — strong cash generation")
        elif fcf > 0 and yield_ > .02: add(1.5, 2, "✅", f"FCF yield {yield_:.1%} — positive FCF")
        elif fcf > 0:                  add(1, 2, "🟡", f"FCF yield {yield_:.1%} — low but positive")
        else:                          add(0, 2, "🔴", "Negative Free Cash Flow")

    # Current Ratio
    cr = info.get("currentRatio")
    if cr is not None:
        if cr > 2:     add(2, 2, "✅", f"Current ratio {cr:.2f} — strong liquidity")
        elif cr > 1.5: add(1.5, 2, "✅", f"Current ratio {cr:.2f} — good liquidity")
        elif cr > 1.0: add(1, 2, "🟡", f"Current ratio {cr:.2f} — adequate")
        else:           add(0, 2, "🔴", f"Current ratio {cr:.2f} — liquidity concern")

    normalised = (total / max_pts * 10) if max_pts > 0 else 5.0
    return max(0.0, min(10.0, normalised)), msgs


def score_valuation(info: dict, dcf_results: dict) -> tuple[float, list]:
    total, max_pts, msgs = 0, 0, []

    def add(pts, max_p, emoji, text):
        nonlocal total, max_pts
        total += pts; max_pts += max_p
        msgs.append((emoji, text))

    pe = info.get("trailingPE")
    if pe and pe > 0:
        if pe < 12:    add(2, 2, "✅", f"P/E {pe:.1f}x — low / value territory")
        elif pe < 20:  add(1.5, 2, "✅", f"P/E {pe:.1f}x — reasonable")
        elif pe < 30:  add(1, 2, "🟡", f"P/E {pe:.1f}x — elevated")
        elif pe < 50:  add(0.5, 2, "🔴", f"P/E {pe:.1f}x — high")
        else:           add(0, 2, "🔴", f"P/E {pe:.1f}x — very high")

    fpe = info.get("forwardPE")
    if fpe and fpe > 0:
        if fpe < 15:   add(2, 2, "✅", f"Forward P/E {fpe:.1f}x — attractive")
        elif fpe < 22: add(1.5, 2, "✅", f"Forward P/E {fpe:.1f}x — fair")
        elif fpe < 30: add(1, 2, "🟡", f"Forward P/E {fpe:.1f}x — stretched")
        else:           add(0.5, 2, "🔴", f"Forward P/E {fpe:.1f}x — expensive")

    peg = info.get("pegRatio")
    if peg and peg > 0:
        if peg < 1:    add(2, 2, "✅", f"PEG {peg:.2f} — growth at a discount")
        elif peg < 1.5:add(1.5, 2, "✅", f"PEG {peg:.2f} — GARP zone")
        elif peg < 2:  add(1, 2, "🟡", f"PEG {peg:.2f} — moderately priced")
        else:           add(0.5, 2, "🔴", f"PEG {peg:.2f} — expensive vs growth")

    ev_eb = info.get("enterpriseToEbitda")
    if ev_eb and ev_eb > 0:
        if ev_eb < 8:   add(2, 2, "✅", f"EV/EBITDA {ev_eb:.1f}x — cheap")
        elif ev_eb < 14:add(1.5, 2, "✅", f"EV/EBITDA {ev_eb:.1f}x — fair")
        elif ev_eb < 22:add(1, 2, "🟡", f"EV/EBITDA {ev_eb:.1f}x — elevated")
        else:            add(0.5, 2, "🔴", f"EV/EBITDA {ev_eb:.1f}x — expensive")

    # DCF upside (use first valid model)
    cp = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    if dcf_results and cp > 0:
        for m in ["wacc", "capm", "two_stage"]:
            mod = dcf_results.get(m, {})
            iv = mod.get("intrinsic_value")
            if iv and iv > 0 and not mod.get("error"):
                upside = (iv - cp) / cp
                if upside > .30:   add(2, 2, "✅", f"DCF intrinsic value ${iv:.2f} — {upside:.0%} upside")
                elif upside > .10: add(1.5, 2, "✅", f"DCF intrinsic value ${iv:.2f} — {upside:.0%} upside")
                elif upside > 0:   add(1, 2, "🟡", f"DCF intrinsic value ${iv:.2f} — slight upside {upside:.0%}")
                else:               add(0.5, 2, "🔴", f"DCF intrinsic value ${iv:.2f} — {abs(upside):.0%} premium to fair value")
                break

    normalised = (total / max_pts * 10) if max_pts > 0 else 5.0
    return max(0.0, min(10.0, normalised)), msgs


def score_sentiment(sentiment: dict) -> tuple[float, list]:
    msgs = []
    score = 5.0  # default neutral

    rec = sentiment.get("rec_mean")
    if rec is not None:
        # yfinance scale: 1=Strong Buy … 5=Strong Sell
        if rec <= 1.5:   score = 9.0; msgs.append(("✅", f"Analyst consensus: Strong Buy ({rec:.1f})"))
        elif rec <= 2.0: score = 7.5; msgs.append(("✅", f"Analyst consensus: Buy ({rec:.1f})"))
        elif rec <= 2.5: score = 6.5; msgs.append(("✅", f"Analyst consensus: Buy/Hold ({rec:.1f})"))
        elif rec <= 3.0: score = 5.0; msgs.append(("🟡", f"Analyst consensus: Hold ({rec:.1f})"))
        elif rec <= 3.5: score = 3.5; msgs.append(("🔴", f"Analyst consensus: Hold/Sell ({rec:.1f})"))
        elif rec <= 4.0: score = 2.5; msgs.append(("🔴", f"Analyst consensus: Sell ({rec:.1f})"))
        else:             score = 1.5; msgs.append(("🔴", f"Analyst consensus: Strong Sell ({rec:.1f})"))

    upside = sentiment.get("target_upside")
    if upside is not None:
        if upside > .25:   msgs.append(("✅", f"Price target upside {upside:.1%}"))
        elif upside > .10: msgs.append(("✅", f"Price target upside {upside:.1%}"))
        elif upside > 0:   msgs.append(("🟡", f"Price target upside {upside:.1%}"))
        else:               msgs.append(("🔴", f"Price target downside {abs(upside):.1%}"))

    avg_s = sentiment.get("avg_surprise")
    beats = sentiment.get("beat_count", 0)
    misses = sentiment.get("miss_count", 0)
    if avg_s is not None:
        if avg_s > 5:   msgs.append(("✅", f"Avg earnings surprise +{avg_s:.1f}% ({beats} beats / {misses} misses)"))
        elif avg_s > 0: msgs.append(("🟡", f"Avg earnings surprise +{avg_s:.1f}%"))
        else:            msgs.append(("🔴", f"Avg earnings surprise {avg_s:.1f}% (more misses than beats)"))

    return max(0.0, min(10.0, score)), msgs


def calculate_composite(fund_score: float, val_score: float,
                        tech_score: float, sent_score: float) -> dict:
    """Compute weighted composite score and derive Buy/Hold/Sell signal."""
    weights = {"fundamental": .40, "valuation": .25, "technical": .20, "sentiment": .15}
    scores  = {"fundamental": fund_score, "valuation": val_score,
                "technical": tech_score,  "sentiment": sent_score}
    composite = sum(scores[k] * weights[k] for k in weights)

    if composite >= 7.5:
        signal, color = "STRONG BUY",  "#00C853"
    elif composite >= 6.5:
        signal, color = "BUY",          "#4CAF50"
    elif composite >= 5.5:
        signal, color = "WEAK BUY",     "#8BC34A"
    elif composite >= 4.5:
        signal, color = "HOLD",         "#FFC107"
    elif composite >= 3.5:
        signal, color = "WEAK SELL",    "#FF9800"
    elif composite >= 2.5:
        signal, color = "SELL",         "#F44336"
    else:
        signal, color = "STRONG SELL",  "#B71C1C"

    return {
        "composite":  round(composite, 2),
        "scores":     scores,
        "weights":    weights,
        "signal":     signal,
        "color":      color,
    }
