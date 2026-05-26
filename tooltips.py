"""
tooltips.py — Metric definitions, formulas, and CSS hover tooltips.

Definitions are written based on standard financial analysis principles
(consistent with how sites like Investopedia and Yahoo Finance define metrics).
"""

# ─── Global CSS (inject once at app startup) ──────────────────────────────────
TOOLTIP_CSS = """
<style>
/* ── Tooltip container ─────────────────────────────────── */
.tt-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.tt-icon {
  display: inline-block;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #3a3f5c;
  color: #a0aec0;
  font-size: 9px;
  font-weight: 800;
  text-align: center;
  line-height: 14px;
  cursor: help;
  flex-shrink: 0;
  vertical-align: middle;
}
.tt-box {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  z-index: 9999;
  left: 20px;
  top: -8px;
  width: 310px;
  background: #1a1d2e;
  border: 1px solid #3a3f5c;
  border-radius: 10px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  transition: opacity 0.15s ease, visibility 0.15s ease;
  pointer-events: none;
}
.tt-wrap:hover .tt-box,
.tt-wrap:focus-within .tt-box {
  visibility: visible;
  opacity: 1;
}
/* box positioning variants */
.tt-box.tt-left  { left: auto; right: 20px; }
.tt-box.tt-up    { top: auto; bottom: 20px; }

/* ── Tooltip internals ─────────────────────────────────── */
.tt-name    { font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 5px; }
.tt-def     { font-size: 12px; color: #a0aec0; line-height: 1.5; margin-bottom: 6px; }
.tt-formula { font-size: 11px; font-family: monospace;
              background: #0e1117; border-radius: 5px;
              padding: 5px 8px; color: #81e6d9; margin-bottom: 6px; }
.tt-range   { font-size: 11px; color: #68d391; margin-bottom: 4px; }
.tt-warn    { font-size: 11px; color: #fc8181; }
.tt-src     { font-size: 10px; color: #4a5568; margin-top: 5px; border-top: 1px solid #2d3748; padding-top: 4px; }

/* ── Metric row ────────────────────────────────────────── */
.mrow {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  border-bottom: 1px solid #1e2130;
  font-size: 13px;
}
.mrow-label { color: #94a3b8; }
.mrow-val   { color: #e2e8f0; font-weight: 600; }
.mrow-beat  { color: #68d391; font-weight: 700; }
.mrow-miss  { color: #fc8181; font-weight: 700; }
.mrow-line  { color: #fbd38d; font-weight: 600; }

/* ── Forecast badge ────────────────────────────────────── */
.fc-badge {
  display: inline-block;
  font-size: 10px; font-weight: 700;
  padding: 2px 7px; border-radius: 10px;
  margin-left: 6px; vertical-align: middle;
}
.fc-beat   { background: rgba(104,211,145,.15); color: #68d391; border: 1px solid #68d391; }
.fc-miss   { background: rgba(252,129,129,.12); color: #fc8181; border: 1px solid #fc8181; }
.fc-inline { background: rgba(251,211,141,.12); color: #fbd38d; border: 1px solid #fbd38d; }
.fc-na     { background: rgba(160,174,192,.1);  color: #a0aec0; border: 1px solid #4a5568; }

/* ── Earnings date card ───────────────────────────────── */
.earn-card {
  background: #1a1d2e; border: 1px solid #3a3f5c; border-radius: 8px;
  padding: 10px 14px; margin: 6px 0; font-size: 13px;
}
.earn-date  { color: #a0aec0; font-size: 11px; margin-bottom: 3px; }
.earn-act   { color: #68d391; font-weight: 700; }
.earn-est   { color: #a0aec0; }
.earn-beat  { color: #68d391; }
.earn-miss  { color: #fc8181; }

/* ── Mobile overrides ─────────────────────────────────── */
@media (max-width: 640px) {
  .tt-box { width: 260px; font-size: 11px; }
  .mrow   { font-size: 12px; }
}
</style>
"""


# ─── Metric Definitions ───────────────────────────────────────────────────────
METRICS: dict = {

    # ── Valuation Ratios ──────────────────────────────────────────────────────
    "pe_trailing": {
        "name": "P/E Ratio (Trailing 12M)",
        "definition": "How much investors pay for each dollar of earnings over the past 12 months. One of the most widely used valuation metrics.",
        "formula": "P/E = Current Price ÷ EPS (last 12 months)",
        "good": "10–20× value / 20–35× growth / 35×+ speculative",
        "warn": "Negative P/E means the company is losing money.",
        "source": "Yahoo Finance · Investopedia",
    },
    "pe_forward": {
        "name": "Forward P/E",
        "definition": "Like trailing P/E but uses next 12 months' estimated earnings. More useful for fast-growing companies since it reflects future profitability expectations.",
        "formula": "Forward P/E = Current Price ÷ Estimated EPS (next 12M)",
        "good": "Lower forward P/E vs trailing P/E = earnings growth expected",
        "warn": "Based on analyst estimates which can be wrong.",
        "source": "Yahoo Finance · Investopedia",
    },
    "peg": {
        "name": "PEG Ratio",
        "definition": "Adjusts the P/E ratio for earnings growth rate. Helps identify if a high P/E stock is justified by its growth rate.",
        "formula": "PEG = P/E ÷ Annual EPS Growth Rate (%)",
        "good": "< 1 = undervalued relative to growth · 1–2 = fairly valued · > 2 = expensive",
        "warn": "Not useful for companies with negative or zero growth.",
        "source": "Investopedia",
    },
    "ev_ebitda": {
        "name": "EV / EBITDA",
        "definition": "Enterprise Value divided by Earnings Before Interest, Tax, Depreciation & Amortisation. Popular for comparing companies with different debt structures.",
        "formula": "EV/EBITDA = (Market Cap + Debt − Cash) ÷ EBITDA",
        "good": "< 10× cheap · 10–15× fair · > 20× expensive (sector-dependent)",
        "warn": "CapEx-heavy industries often trade at lower EV/EBITDA.",
        "source": "Yahoo Finance · Investopedia",
    },
    "ev_revenue": {
        "name": "EV / Revenue",
        "definition": "Compares a company's total enterprise value to its annual revenue. Useful for unprofitable growth companies where EBITDA is negative.",
        "formula": "EV/Revenue = Enterprise Value ÷ Total Revenue (TTM)",
        "good": "< 2× cheap · 2–5× typical SaaS/tech · > 10× high-growth premium",
        "warn": "High EV/Revenue is only justified if the business has high gross margins.",
        "source": "Yahoo Finance · Investopedia",
    },
    "price_book": {
        "name": "Price / Book (P/B)",
        "definition": "Compares stock price to the net asset value per share. Widely used for financial stocks. Below 1× may indicate undervaluation or balance sheet issues.",
        "formula": "P/B = Stock Price ÷ Book Value per Share",
        "good": "< 1× deep value · 1–3× reasonable · > 5× growth premium",
        "warn": "Book value can be misleading for asset-light businesses (tech/services).",
        "source": "Yahoo Finance · Investopedia",
    },
    "price_sales": {
        "name": "Price / Sales (P/S)",
        "definition": "Market cap divided by annual revenue. Useful when earnings are negative. Reflects how much investors pay per dollar of revenue.",
        "formula": "P/S = Market Cap ÷ Annual Revenue",
        "good": "< 1× deep value · 1–3× reasonable · > 10× high-growth premium",
        "warn": "P/S ignores profitability — a low-margin company with high P/S is risky.",
        "source": "Yahoo Finance · Investopedia",
    },

    # ── Profitability ─────────────────────────────────────────────────────────
    "gross_margin": {
        "name": "Gross Margin",
        "definition": "Percentage of revenue remaining after subtracting cost of goods sold. Higher margins indicate pricing power and competitive advantage.",
        "formula": "Gross Margin = (Revenue − COGS) ÷ Revenue × 100",
        "good": "> 60% software/pharma · > 35% industrials · > 20% retail",
        "warn": "Declining gross margin over multiple years is a red flag.",
        "source": "Yahoo Finance",
    },
    "operating_margin": {
        "name": "Operating Margin",
        "definition": "Profitability after operating expenses but before interest and taxes. Shows how efficiently management runs the core business.",
        "formula": "Operating Margin = Operating Income ÷ Revenue × 100",
        "good": "> 20% excellent · 10–20% good · < 5% thin",
        "warn": "Compare within industry — software margins differ from retail.",
        "source": "Yahoo Finance · Investopedia",
    },
    "net_margin": {
        "name": "Net Profit Margin",
        "definition": "The final bottom-line profit as a percentage of revenue, after all expenses including taxes and interest.",
        "formula": "Net Margin = Net Income ÷ Revenue × 100",
        "good": "> 20% excellent · 10–20% strong · 5–10% adequate · < 5% thin",
        "warn": "Net margin can be distorted by one-time charges or tax events.",
        "source": "Yahoo Finance · Investopedia",
    },
    "roe": {
        "name": "Return on Equity (ROE)",
        "definition": "How much profit the company generates for each dollar of shareholders' equity. Warren Buffett's preferred metric for quality businesses.",
        "formula": "ROE = Net Income ÷ Shareholders' Equity × 100",
        "good": "> 20% excellent · 15–20% strong · 8–15% adequate · < 8% weak",
        "warn": "Very high ROE can result from excessive debt (leverage), not operational excellence.",
        "source": "Investopedia",
    },
    "roa": {
        "name": "Return on Assets (ROA)",
        "definition": "Net income as a percentage of total assets. Measures how efficiently management uses assets to generate profit.",
        "formula": "ROA = Net Income ÷ Total Assets × 100",
        "good": "> 10% excellent · 5–10% good · < 2% low (typical for banks)",
        "warn": "Capital-intensive industries (utilities, mining) typically have low ROA.",
        "source": "Investopedia",
    },

    # ── Growth ────────────────────────────────────────────────────────────────
    "revenue_growth": {
        "name": "Revenue Growth (YoY)",
        "definition": "Year-over-year percentage increase in total revenue. The top-line growth driver for most valuation models.",
        "formula": "Rev Growth = (Current Revenue − Prior Revenue) ÷ Prior Revenue × 100",
        "good": "> 20% high growth · 8–20% solid · 0–8% slow · < 0% declining",
        "warn": "Acquisitions can inflate reported growth — check organic growth separately.",
        "source": "Yahoo Finance",
    },
    "earnings_growth": {
        "name": "Earnings Growth (YoY)",
        "definition": "Year-over-year percentage increase in net income or EPS. Sustained earnings growth is the primary long-term driver of share price appreciation.",
        "formula": "EPS Growth = (EPS Current − EPS Prior) ÷ |EPS Prior| × 100",
        "good": "> 20% high growth · 10–20% solid · 0–10% slow",
        "warn": "Earnings can be manipulated through buybacks or accounting choices.",
        "source": "Yahoo Finance",
    },

    # ── Financial Health ──────────────────────────────────────────────────────
    "debt_equity": {
        "name": "Debt / Equity Ratio",
        "definition": "Total debt as a multiple of shareholders' equity. Measures financial leverage and solvency risk.",
        "formula": "D/E = Total Debt ÷ Shareholders' Equity",
        "good": "< 0.5× conservative · 0.5–1.5× moderate · > 2× highly leveraged",
        "warn": "High D/E during rising interest rates significantly increases financial risk.",
        "source": "Yahoo Finance · Investopedia",
    },
    "current_ratio": {
        "name": "Current Ratio",
        "definition": "Ability to pay short-term obligations using current assets. A basic test of short-term liquidity.",
        "formula": "Current Ratio = Current Assets ÷ Current Liabilities",
        "good": "> 2× strong · 1.5–2× good · 1–1.5× adequate · < 1 = potential shortfall",
        "warn": "Very high current ratio may indicate idle cash or poor working capital management.",
        "source": "Investopedia",
    },
    "beta": {
        "name": "Beta (Market Sensitivity)",
        "definition": "Measures a stock's price volatility relative to the overall market (S&P 500 = 1.0). Used to calculate required rate of return in CAPM.",
        "formula": "β = Cov(Stock, Market) ÷ Var(Market) — calculated over 5 years monthly",
        "good": "< 0.8 defensive · 0.8–1.2 market-like · > 1.5 high-volatility",
        "warn": "Beta is backward-looking and may not predict future volatility.",
        "source": "Yahoo Finance",
    },
    "dividend_yield": {
        "name": "Dividend Yield",
        "definition": "Annual dividend payment as a percentage of current share price. Important for income investors.",
        "formula": "Dividend Yield = Annual DPS ÷ Current Stock Price × 100",
        "good": "2–4% sustainable income · > 6% may be unsustainable (check payout ratio)",
        "warn": "A very high yield often signals dividend risk — verify the payout ratio.",
        "source": "Yahoo Finance · Investopedia",
    },

    # ── DCF Model Specific ────────────────────────────────────────────────────
    "wacc": {
        "name": "WACC — Weighted Average Cost of Capital",
        "definition": "The blended required return on capital from both equity and debt investors, weighted by capital structure. Used as the discount rate in enterprise DCF models.",
        "formula": "WACC = (E/V × Ke) + (D/V × Kd × (1−tax))\nKe = Rfr + β × MRP · Kd = Interest/Debt",
        "good": "Typically 7–12% for large caps · higher for small/risky firms",
        "warn": "WACC is highly sensitive to beta and capital structure assumptions.",
        "source": "Investopedia · CFA Institute",
    },
    "capm": {
        "name": "CAPM — Capital Asset Pricing Model",
        "definition": "Model that determines the expected return on equity based on its systematic risk (beta). Used as the equity discount rate when ignoring debt.",
        "formula": "Ke = Rfr + β × MRP\nMRP (market risk premium) = 5.5% (historical avg)",
        "good": "Typically 8–15% for equities",
        "warn": "CAPM assumes beta fully captures risk — may understate risk for small/illiquid stocks.",
        "source": "Investopedia · CFA Institute",
    },
    "fcf": {
        "name": "Free Cash Flow (FCF)",
        "definition": "Cash generated by operations after capital expenditures. The actual cash available to pay debt, dividends, buybacks, or reinvest in growth.",
        "formula": "FCF = Operating Cash Flow − Capital Expenditures",
        "good": "Positive FCF = self-funding business · FCF yield > 5% = attractive",
        "warn": "Negative FCF is normal for early-stage or heavily investing companies.",
        "source": "Yahoo Finance · Investopedia",
    },
    "terminal_value": {
        "name": "Terminal Value (DCF)",
        "definition": "The present value of all cash flows beyond the explicit forecast period, assuming constant perpetual growth. Typically the largest component of a DCF.",
        "formula": "TV = FCF_n × (1+g) ÷ (r−g)\ng = terminal growth rate (~2.5%) · r = discount rate",
        "good": "TV/EV ratio < 75% is considered conservative",
        "warn": "Very sensitive to terminal growth rate — small changes have large impact.",
        "source": "Investopedia · Damodaran",
    },

    # ── Technical ─────────────────────────────────────────────────────────────
    "rsi": {
        "name": "RSI — Relative Strength Index",
        "definition": "Momentum oscillator measuring the speed and magnitude of price changes on a 0–100 scale. Developed by J. Welles Wilder in 1978.",
        "formula": "RSI = 100 − 100/(1 + RS)\nRS = Avg Gain (14d) ÷ Avg Loss (14d)",
        "good": "> 70 overbought (potential pullback) · < 30 oversold (potential bounce)",
        "warn": "In strong trends, RSI can stay overbought/oversold for extended periods.",
        "source": "Investopedia",
    },
    "macd": {
        "name": "MACD — Moving Average Convergence Divergence",
        "definition": "Trend-following momentum indicator showing the relationship between two exponential moving averages. Used to identify trend changes and momentum.",
        "formula": "MACD = EMA(12) − EMA(26)\nSignal Line = EMA(9) of MACD",
        "good": "MACD above signal = bullish · below = bearish · histogram increasing = momentum",
        "warn": "MACD can give false signals in choppy/sideways markets.",
        "source": "Investopedia",
    },
    "bollinger": {
        "name": "Bollinger Bands",
        "definition": "Volatility bands placed above and below a 20-day simple moving average. Prices outside the bands are statistically unusual and may revert.",
        "formula": "Upper/Lower = SMA(20) ± 2 × StdDev(20)\n%B = (Price − Lower) ÷ (Upper − Lower)",
        "good": "%B > 1 overbought · %B < 0 oversold · Squeeze = low vol before breakout",
        "warn": "Bollinger Bands don't predict direction — only volatility levels.",
        "source": "Investopedia",
    },
    "obv": {
        "name": "OBV — On-Balance Volume",
        "definition": "Cumulative volume indicator that adds volume on up days and subtracts on down days. Used to detect institutional buying/selling before price moves.",
        "formula": "OBV_n = OBV_(n-1) + Volume (if close > prior) or − Volume (if close < prior)",
        "good": "Rising OBV + rising price = strong trend · Rising OBV + falling price = bullish divergence",
        "warn": "Large single-day volume events can skew the signal.",
        "source": "Investopedia",
    },
    "sma": {
        "name": "SMA / EMA — Moving Averages",
        "definition": "SMA = simple (equal weights). EMA = exponential (more weight on recent prices). Used to identify trend direction and dynamic support/resistance.",
        "formula": "SMA(n) = Sum(Close, n) ÷ n\nGolden Cross = SMA50 > SMA200 (bullish)\nDeath Cross = SMA50 < SMA200 (bearish)",
        "good": "Price > 200-SMA = long-term uptrend · Golden Cross historically bullish",
        "warn": "Moving averages are lagging indicators — they confirm trends, not predict them.",
        "source": "Investopedia",
    },

    # ── Sentiment ─────────────────────────────────────────────────────────────
    "earnings_surprise": {
        "name": "Earnings Surprise",
        "definition": "The percentage by which actual reported EPS differs from the consensus analyst estimate. Positive surprises typically cause short-term price jumps.",
        "formula": "Surprise % = (Actual EPS − Estimated EPS) ÷ |Estimated EPS| × 100",
        "good": "Consistent beats (>0%) with large surprises (>5%) signal strong execution",
        "warn": "Companies sometimes 'guide down' to make beats easier — check estimate revision trend.",
        "source": "Yahoo Finance · Investopedia",
    },
    "analyst_consensus": {
        "name": "Analyst Consensus Rating",
        "definition": "Average recommendation from covering analysts on a 1–5 scale (1=Strong Buy, 5=Strong Sell). Yahoo Finance aggregates ratings from major brokerages.",
        "formula": "Mean Score = Sum(ratings) ÷ Number of Analysts\n1.0–1.5 Strong Buy · 2.0–2.5 Buy · 3.0 Hold · 4.0+ Sell",
        "good": "≤ 2.0 broadly bullish · ≥ 4.0 broadly bearish",
        "warn": "Analyst ratings have conflicts of interest (investment banking relationships).",
        "source": "Yahoo Finance",
    },
    "short_interest": {
        "name": "Short Interest / Short Ratio",
        "definition": "Percentage of float sold short (Short % Float) and days-to-cover (Short Ratio). High short interest can signal bearish sentiment OR potential for a short squeeze.",
        "formula": "Short % Float = Shares Short ÷ Float\nShort Ratio = Shares Short ÷ Avg Daily Volume",
        "good": "< 5% normal · 5–15% elevated · > 20% high short interest",
        "warn": "High short interest alone doesn't mean sell — short squeezes can cause rapid price rises.",
        "source": "Yahoo Finance · Investopedia",
    },
}


def tooltip_html(metric_key: str, label: str, value: str,
                 forecast_status: str = "", position: str = "") -> str:
    """
    Render a metric row with an inline hover tooltip.

    forecast_status: 'beat' | 'miss' | 'inline' | '' (no badge)
    position: '' | 'left' | 'up'  (tooltip placement)
    """
    m = METRICS.get(metric_key, {})
    if not m:
        # No definition available — render plain row
        return f"""
        <div class='mrow'>
          <span class='mrow-label'>{label}</span>
          <span class='mrow-val'>{value}</span>
        </div>"""

    box_class = f"tt-box{' tt-'+position if position else ''}"
    name    = m.get("name", label)
    defn    = m.get("definition", "")
    formula = m.get("formula", "")
    good    = m.get("good", "")
    warn    = m.get("warn", "")
    src     = m.get("source", "")

    formula_html = f"<div class='tt-formula'>{formula.replace(chr(10), '<br>')}</div>" if formula else ""
    good_html    = f"<div class='tt-range'>✅ {good}</div>" if good else ""
    warn_html    = f"<div class='tt-warn'>⚠️ {warn}</div>" if warn else ""
    src_html     = f"<div class='tt-src'>Source: {src}</div>" if src else ""

    # Forecast badge
    badge_map = {
        "beat":   ("<span class='fc-badge fc-beat'>BEAT</span>",   "mrow-beat"),
        "miss":   ("<span class='fc-badge fc-miss'>MISS</span>",   "mrow-miss"),
        "inline": ("<span class='fc-badge fc-inline'>IN LINE</span>", "mrow-line"),
    }
    badge_html, val_class = badge_map.get(forecast_status, ("", "mrow-val"))

    return f"""
    <div class='mrow'>
      <span class='mrow-label'>
        <span class='tt-wrap'>
          {label}&nbsp;<span class='tt-icon'>i</span>
          <div class='{box_class}'>
            <div class='tt-name'>{name}</div>
            <div class='tt-def'>{defn}</div>
            {formula_html}
            {good_html}
            {warn_html}
            {src_html}
          </div>
        </span>
      </span>
      <span class='{val_class}'>{value}{badge_html}</span>
    </div>"""


def section_header(title: str) -> str:
    return f"<div class='section-header' style='font-size:16px;font-weight:700;margin:14px 0 6px;color:#e2e8f0'>{title}</div>"
