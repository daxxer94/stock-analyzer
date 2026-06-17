"""
tooltips.py — Metric definitions and CSS hover tooltips.

Every METRICS key matches exactly what is shown in the UI.
Definitions align with Investopedia and Yahoo Finance standards.
"""

TOOLTIP_CSS = """
<style>
/* ── Ticker name tooltip ──────────────────────────────────────────────────── */
.tk-wrap {
  position: relative;
  display: inline-block;
  cursor: help;
  border-bottom: 1px dotted rgba(148,163,184,0.4);
}
.tk-wrap .tk-tip {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  background: #1e2433;
  color: #e2e8f0;
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid #2d3748;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  pointer-events: none;
  z-index: 9999;
  transition: opacity 0.15s ease;
  font-family: 'Inter', sans-serif;
}
.tk-wrap .tk-tip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 4px solid transparent;
  border-top-color: #2d3748;
}
.tk-wrap:hover .tk-tip {
  visibility: visible;
  opacity: 1;
}
/* Light theme */
@media (prefers-color-scheme: light) {
  .tk-wrap .tk-tip {
    background: #1e2433;
    color: #f1f5f9;
  }
}

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
  width: 320px;
  background: #1a1d2e;
  border: 1px solid #3a3f5c;
  border-radius: 10px;
  padding: 12px 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.55);
  transition: opacity 0.15s ease, visibility 0.15s ease;
  pointer-events: none;
}
.tt-wrap:hover .tt-box,
.tt-wrap:focus-within .tt-box {
  visibility: visible;
  opacity: 1;
}
.tt-box.tt-left { left: auto; right: 20px; }
.tt-box.tt-up   { top: auto; bottom: 20px; }
.tt-name    { font-size: 13px; font-weight: 700; color: #e2e8f0; margin-bottom: 5px; }
.tt-def     { font-size: 12px; color: #a0aec0; line-height: 1.55; margin-bottom: 6px; }
.tt-formula { font-size: 11px; font-family: monospace;
              background: #0e1117; border-radius: 5px;
              padding: 5px 8px; color: #81e6d9; margin-bottom: 6px; }
.tt-range   { font-size: 11px; color: #68d391; margin-bottom: 4px; }
.tt-warn    { font-size: 11px; color: #fc8181; }
.tt-src     { font-size: 10px; color: #4a5568; margin-top: 5px;
              border-top: 1px solid #2d3748; padding-top: 4px; }
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
.earn-card {
  background: #1a1d2e; border: 1px solid #3a3f5c; border-radius: 8px;
  padding: 10px 14px; margin: 6px 0; font-size: 13px;
}
.earn-date { color: #a0aec0; font-size: 11px; margin-bottom: 3px; }
.earn-act  { color: #68d391; font-weight: 700; }
.earn-est  { color: #a0aec0; }
.earn-beat { color: #68d391; }
.earn-miss { color: #fc8181; }
@media (max-width: 640px) {
  .tt-box { width: 260px; font-size: 11px; }
  .mrow   { font-size: 12px; }
}
</style>
"""


METRICS: dict = {

    # ── VALUATION RATIOS ─────────────────────────────────────────────────────

    "pe_trailing": {
        "name": "P/E Ratio — Trailing 12 Months",
        "definition": (
            "Price-to-Earnings ratio based on actual earnings reported over "
            "the last 12 months. Tells you how many dollars investors are paying "
            "for each dollar of profit the company has already generated."
        ),
        "formula": "P/E = Current Share Price ÷ EPS (trailing 12M)",
        "good": "< 15× value territory · 15–25× fair · 25–40× growth premium · > 40× speculative",
        "warn": (
            "A negative P/E means the company is loss-making. "
            "Always compare P/E within the same sector — tech trades higher than utilities."
        ),
        "source": "Yahoo Finance · Investopedia — Price-to-Earnings Ratio",
    },

    "pe_forward": {
        "name": "Forward P/E Ratio",
        "definition": (
            "Like trailing P/E but uses the next 12 months' consensus earnings estimate "
            "instead of historical earnings. Shows what the market is paying for future profits. "
            "Also used here for analyst price targets — the displayed value depends on context."
        ),
        "formula": "Forward P/E = Current Share Price ÷ Estimated EPS (next 12M)",
        "good": (
            "Forward P/E < Trailing P/E → earnings expected to grow. "
            "< 15× attractive · 15–25× fair · > 30× expensive unless high-growth"
        ),
        "warn": (
            "Based entirely on analyst estimates which are often wrong. "
            "Check how estimates have been revised over the last 90 days."
        ),
        "source": "Yahoo Finance · Investopedia — Forward P/E",
    },

    "peg": {
        "name": "PEG Ratio — Price/Earnings to Growth",
        "definition": (
            "Adjusts the P/E ratio for the company's expected earnings growth rate. "
            "A P/E of 30× is very different for a 5% grower vs a 30% grower — "
            "PEG captures that difference. Popularised by Peter Lynch."
        ),
        "formula": "PEG = Trailing P/E ÷ Annual EPS Growth Rate (%)",
        "good": (
            "< 1.0 → growth available at a discount (Peter Lynch's 'buy' signal) "
            "· 1.0–2.0 → fairly valued for growth "
            "· > 2.0 → expensive relative to growth rate"
        ),
        "warn": (
            "Not meaningful for companies with negative or near-zero earnings growth. "
            "Growth rate estimate source varies — check what period is used."
        ),
        "source": "Investopedia — PEG Ratio",
    },

    "ev_ebitda": {
        "name": "EV / EBITDA",
        "definition": (
            "Enterprise Value divided by Earnings Before Interest, Tax, Depreciation "
            "and Amortisation. Preferred over P/E for comparing companies with different "
            "debt levels or tax situations, as it strips out financing and accounting choices."
        ),
        "formula": (
            "EV = Market Cap + Total Debt − Cash & Equivalents\n"
            "EV/EBITDA = Enterprise Value ÷ EBITDA (trailing 12M)"
        ),
        "good": (
            "< 8× cheap · 8–14× reasonable · 14–22× elevated · > 22× expensive. "
            "Benchmarks vary widely: tech ~20×, industrials ~10×, utilities ~12×"
        ),
        "warn": (
            "EBITDA ignores capital expenditure — compare with EV/EBIT or FCF yield "
            "for capital-intensive businesses like manufacturing or telecoms."
        ),
        "source": "Yahoo Finance · Investopedia — EV/EBITDA",
    },

    "ev_revenue": {
        "name": "EV / Revenue (Price-to-Sales)",
        "definition": (
            "Enterprise Value divided by total annual revenue. Used when EBITDA is "
            "negative (early-stage, high-growth, or loss-making companies). "
            "Reflects how much investors pay per dollar of top-line revenue."
        ),
        "formula": "EV/Revenue = Enterprise Value ÷ Total Revenue (trailing 12M)",
        "good": (
            "< 1× deep value · 1–3× reasonable · 3–8× typical high-margin SaaS/tech "
            "· > 10× only justified with very high gross margins and rapid growth"
        ),
        "warn": (
            "A low EV/Revenue is meaningless if gross margins are also low. "
            "A retailer at 0.3× may be more expensive than a SaaS firm at 8× on a margin-adjusted basis."
        ),
        "source": "Yahoo Finance · Investopedia — EV/Revenue",
    },

    "price_book": {
        "name": "Price / Book Value (P/B)",
        "definition": (
            "Compares the stock price to the company's net asset value per share "
            "(total assets minus total liabilities). Widely used for banks and "
            "financial companies where assets are the business."
        ),
        "formula": (
            "Book Value per Share = (Total Assets − Total Liabilities) ÷ Shares Outstanding\n"
            "P/B = Current Share Price ÷ Book Value per Share"
        ),
        "good": (
            "< 1× trading below book value (potential deep value or distress) "
            "· 1–3× reasonable · > 5× growth/intangible premium"
        ),
        "warn": (
            "Book value is unreliable for asset-light businesses (software, brands, "
            "professional services) where intellectual property isn't on the balance sheet."
        ),
        "source": "Yahoo Finance · Investopedia — Price-to-Book Ratio",
    },

    "price_sales": {
        "name": "Price / Sales (P/S Ratio)",
        "definition": (
            "Market capitalisation divided by annual revenue. Unlike P/E, it works "
            "even when a company has no earnings, making it useful for early-stage "
            "or currently unprofitable businesses."
        ),
        "formula": "P/S = Market Capitalisation ÷ Total Revenue (trailing 12M)",
        "good": (
            "< 1× deep value · 1–3× reasonable for most sectors "
            "· 3–10× typical for high-margin tech · > 10× requires very high growth justification"
        ),
        "warn": (
            "P/S completely ignores profitability. "
            "A company with 5% net margins at 10× P/S is far more expensive than "
            "it looks versus one with 40% margins at the same multiple."
        ),
        "source": "Yahoo Finance · Investopedia — Price-to-Sales",
    },

    # ── PROFITABILITY ────────────────────────────────────────────────────────

    "gross_margin": {
        "name": "Gross Profit Margin",
        "definition": (
            "Revenue remaining after subtracting the direct cost of producing goods "
            "or services (COGS). The first line of profitability — shows pricing power "
            "and unit economics before overheads like sales, R&D, or admin."
        ),
        "formula": "Gross Margin = (Revenue − Cost of Goods Sold) ÷ Revenue × 100%",
        "good": (
            "> 70% software / pharma / luxury goods "
            "· 40–70% branded consumer / medical devices "
            "· 20–40% industrials / retail "
            "· < 20% commodities / low-margin retail"
        ),
        "warn": (
            "Declining gross margin over consecutive years suggests rising input costs, "
            "pricing pressure from competitors, or product mix deterioration."
        ),
        "source": "Yahoo Finance · Investopedia — Gross Margin",
    },

    "operating_margin": {
        "name": "Operating Profit Margin (EBIT Margin)",
        "definition": (
            "Profit from core business operations as a percentage of revenue, "
            "after deducting COGS and operating expenses (R&D, sales, admin) "
            "but before interest expense and income taxes. "
            "The best measure of management's operational efficiency."
        ),
        "formula": "Operating Margin = Operating Income (EBIT) ÷ Revenue × 100%",
        "good": (
            "> 25% excellent operational leverage "
            "· 15–25% strong · 8–15% adequate · < 5% thin / at-risk"
        ),
        "warn": (
            "Compare only within the same industry. "
            "A 10% operating margin is excellent for a grocer but poor for a software company."
        ),
        "source": "Yahoo Finance · Investopedia — Operating Margin",
    },

    "net_margin": {
        "name": "Net Profit Margin",
        "definition": (
            "The final bottom-line profit as a percentage of revenue after ALL expenses: "
            "COGS, operating costs, interest on debt, and income taxes. "
            "Also shown here with BEAT/MISS badge relative to analyst expectations."
        ),
        "formula": "Net Margin = Net Income ÷ Total Revenue × 100%",
        "good": (
            "> 20% excellent · 10–20% strong · 5–10% adequate · 0–5% thin. "
            "Negative = currently loss-making"
        ),
        "warn": (
            "One-time items (asset sales, tax credits, restructuring charges) can "
            "make a single year's net margin misleading. Look at the 3–5 year trend."
        ),
        "source": "Yahoo Finance · Investopedia — Net Profit Margin",
    },

    "roe": {
        "name": "Return on Equity (ROE)",
        "definition": (
            "Net income generated for each dollar of shareholders' equity. "
            "One of Warren Buffett's core metrics for identifying quality businesses — "
            "a consistently high ROE over 10+ years indicates durable competitive advantage."
        ),
        "formula": "ROE = Net Income ÷ Average Shareholders' Equity × 100%",
        "good": (
            "> 20% excellent (Buffett threshold) "
            "· 15–20% strong · 8–15% adequate · < 8% weak"
        ),
        "warn": (
            "ROE can be artificially inflated by high debt (leverage) or share buybacks "
            "that reduce equity. Always check alongside Debt/Equity ratio."
        ),
        "source": "Investopedia — Return on Equity",
    },

    "roa": {
        "name": "Return on Assets (ROA)",
        "definition": (
            "Net income as a percentage of total assets. Measures how efficiently "
            "management converts the asset base into profit. Less susceptible to "
            "leverage distortion than ROE."
        ),
        "formula": "ROA = Net Income ÷ Average Total Assets × 100%",
        "good": (
            "> 10% excellent · 5–10% good · 2–5% average "
            "· < 2% typical for asset-heavy industries (banks, utilities, real estate)"
        ),
        "warn": (
            "Capital-intensive businesses (airlines, manufacturers, miners) will always "
            "have lower ROA than asset-light ones (software, consulting). "
            "Compare within the same sector."
        ),
        "source": "Investopedia — Return on Assets",
    },

    # ── GROWTH ───────────────────────────────────────────────────────────────

    "revenue_growth": {
        "name": "Revenue Growth (Year-over-Year)",
        "definition": (
            "Percentage change in total revenue compared to the same period a year ago. "
            "Top-line growth is the starting point for all financial projections — "
            "you cannot grow earnings long-term without growing revenue."
        ),
        "formula": "Revenue Growth = (Revenue This Year − Revenue Last Year) ÷ Revenue Last Year × 100%",
        "good": (
            "> 20% high-growth · 10–20% solid · 5–10% moderate "
            "· 0–5% slow · < 0% declining"
        ),
        "warn": (
            "Acquisitions inflate reported growth — check organic growth. "
            "Currency fluctuations can also distort reported growth for multinationals."
        ),
        "source": "Yahoo Finance · Investopedia — Revenue Growth",
    },

    "earnings_growth": {
        "name": "Earnings Per Share Growth (YoY)",
        "definition": (
            "Year-over-year percentage change in earnings per share. "
            "Sustained EPS growth — driven by revenue growth and/or margin expansion — "
            "is the primary long-term driver of share price appreciation."
        ),
        "formula": "EPS Growth = (EPS This Year − EPS Last Year) ÷ |EPS Last Year| × 100%",
        "good": "> 20% high growth · 10–20% solid · 0–10% slow · < 0% declining",
        "warn": (
            "EPS can be boosted by share buybacks without any improvement in the underlying business. "
            "Check revenue growth alongside earnings growth for a full picture."
        ),
        "source": "Yahoo Finance · Investopedia — EPS Growth",
    },

    # ── FINANCIAL HEALTH ─────────────────────────────────────────────────────

    "debt_equity": {
        "name": "Debt / Equity Ratio (D/E)",
        "definition": (
            "Total interest-bearing debt divided by shareholders' equity. "
            "Measures financial leverage — how much of the company is funded by debt "
            "versus equity. Higher leverage amplifies both gains and losses."
        ),
        "formula": (
            "D/E = Total Debt ÷ Total Shareholders' Equity\n"
            "Note: Yahoo Finance reports this as a percentage (e.g. 120 = 1.2×) — "
            "this tool converts it to a ratio automatically."
        ),
        "good": (
            "< 0.5× conservative / financially strong "
            "· 0.5–1.5× moderate leverage "
            "· 1.5–3× elevated — monitor interest coverage "
            "· > 3× highly leveraged"
        ),
        "warn": (
            "Some industries (banks, utilities, real estate) routinely carry high D/E "
            "as part of their business model. A bank at 8× D/E is not comparable to "
            "an industrial company at the same ratio."
        ),
        "source": "Yahoo Finance · Investopedia — Debt-to-Equity Ratio",
    },

    "current_ratio": {
        "name": "Current Ratio",
        "definition": (
            "Current assets divided by current liabilities. Measures the company's "
            "ability to pay all short-term obligations (due within 12 months) "
            "using assets that can be converted to cash within 12 months."
        ),
        "formula": "Current Ratio = Current Assets ÷ Current Liabilities",
        "good": (
            "> 2.0× strong liquidity buffer "
            "· 1.5–2.0× healthy "
            "· 1.0–1.5× adequate but watch closely "
            "· < 1.0× current liabilities exceed current assets — liquidity risk"
        ),
        "warn": (
            "A very high current ratio (> 4×) may indicate the company holds too much "
            "idle cash or has slow-moving inventory — not always positive. "
            "Retailers and subscription businesses can operate well below 1.0."
        ),
        "source": "Investopedia — Current Ratio",
    },

    "beta": {
        "name": "Beta (Systematic Risk / Market Sensitivity)",
        "definition": (
            "Measures how much a stock moves relative to the broader market (S&P 500 = 1.0). "
            "A beta of 1.5 means the stock tends to move 50% more than the market "
            "in both directions. Also used here as a label for 52-week high/low data "
            "in the sentiment section."
        ),
        "formula": (
            "β = Covariance(Stock Returns, Market Returns) ÷ Variance(Market Returns)\n"
            "Calculated using monthly returns over the past 5 years vs S&P 500"
        ),
        "good": (
            "< 0.7 defensive / low volatility "
            "· 0.7–1.2 roughly market-like "
            "· 1.2–1.8 growth / cyclical "
            "· > 2.0 highly volatile"
        ),
        "warn": (
            "Beta is backward-looking — a stock's historical volatility may not "
            "predict future behaviour. New companies or those undergoing transformation "
            "may have unreliable beta readings."
        ),
        "source": "Yahoo Finance · Investopedia — Beta",
    },

    "dividend_yield": {
        "name": "Dividend Yield",
        "definition": (
            "Annual dividends paid per share as a percentage of the current share price. "
            "Represents the income return on investment from dividends alone, "
            "before any capital gains or losses."
        ),
        "formula": "Dividend Yield = Annual Dividend per Share ÷ Current Share Price × 100%",
        "good": (
            "1–2% low but growing (reinvestment focus) "
            "· 2–4% solid income yield "
            "· 4–6% high yield — check payout ratio for sustainability "
            "· > 6% potentially unsustainable — verify with free cash flow"
        ),
        "warn": (
            "A high yield can be a warning sign if caused by a falling share price. "
            "Always check the payout ratio (dividends ÷ earnings) — above 80% raises "
            "questions about long-term sustainability."
        ),
        "source": "Yahoo Finance · Investopedia — Dividend Yield",
    },

    # ── DCF MODEL SPECIFIC ────────────────────────────────────────────────────

    "wacc": {
        "name": "WACC — Weighted Average Cost of Capital",
        "definition": (
            "The minimum return a company must earn on its total invested capital "
            "to satisfy all its investors — both equity holders and debt holders. "
            "Used as the discount rate in enterprise DCF models. "
            "A lower WACC produces a higher intrinsic value."
        ),
        "formula": (
            "WACC = (E/V × Ke) + (D/V × Kd × (1 − Tax Rate))\n"
            "E = Market Cap  ·  D = Total Debt  ·  V = E + D\n"
            "Ke = Cost of Equity (CAPM)  ·  Kd = Pre-tax Cost of Debt"
        ),
        "good": (
            "7–10% typical large-cap · 10–14% mid-cap / higher risk "
            "· > 15% small-cap / emerging market"
        ),
        "warn": (
            "WACC is sensitive to beta (which changes over time), the market risk premium "
            "assumed (this tool uses 5.5%), and the cost of debt calculation. "
            "A 1% change in WACC can move the intrinsic value by 15–25%."
        ),
        "source": "Investopedia — WACC · CFA Institute",
    },

    "capm": {
        "name": "CAPM — Capital Asset Pricing Model (Cost of Equity)",
        "definition": (
            "Model that calculates the expected return an equity investor requires, "
            "based on the stock's systematic risk (beta). Used as the discount rate "
            "in equity-only DCF models and as the cost of equity component in WACC. "
            "Also used here to display the Risk-Free Rate."
        ),
        "formula": (
            "Ke = Risk-Free Rate + β × Market Risk Premium\n"
            "Risk-Free Rate = US 10Y Treasury yield (fetched live)\n"
            "Market Risk Premium = 5.5% (long-run historical average)"
        ),
        "good": (
            "With Rfr = 4.5% and β = 1.0: Ke = 4.5% + 1.0 × 5.5% = 10.0%\n"
            "Higher beta → higher required return → lower intrinsic value"
        ),
        "warn": (
            "CAPM is a single-factor model — it only captures market risk (beta), "
            "not size, value, or liquidity factors. It may understate the required "
            "return for small or illiquid companies."
        ),
        "source": "Investopedia — CAPM · CFA Institute",
    },

    "fcf": {
        "name": "Free Cash Flow (FCF)",
        "definition": (
            "Cash generated from the company's core operations after spending on "
            "capital expenditures (maintenance and growth of physical assets). "
            "FCF is the cash actually available to return to shareholders via "
            "dividends and buybacks, or to pay down debt."
        ),
        "formula": (
            "FCF = Operating Cash Flow − Capital Expenditures\n"
            "FCF Yield = FCF ÷ Market Capitalisation × 100%"
        ),
        "good": (
            "Positive FCF = company funds itself without external capital "
            "· FCF Yield > 5% = strong cash return potential "
            "· FCF > Net Income = high earnings quality"
        ),
        "warn": (
            "Negative FCF is normal and even desirable for fast-growing companies "
            "investing heavily in their future. Context matters: "
            "a startup at −FCF is very different from a mature company."
        ),
        "source": "Yahoo Finance · Investopedia — Free Cash Flow",
    },

    "terminal_value": {
        "name": "Terminal Value (Gordon Growth Model)",
        "definition": (
            "The estimated value of all cash flows beyond the explicit forecast period "
            "(typically years 6–10 onward), assuming the business grows at a constant "
            "perpetual rate forever. Usually the single largest component of a DCF valuation."
        ),
        "formula": (
            "Terminal Value = FCF_final × (1 + g) ÷ (r − g)\n"
            "g = perpetual growth rate (this tool uses 2.5% ≈ long-run nominal GDP)\n"
            "r = discount rate (WACC or CAPM)"
        ),
        "good": (
            "TV as % of Total EV < 65% = well-anchored valuation "
            "· 65–80% = significant dependence on long-term assumptions "
            "· > 80% = treat with extra caution"
        ),
        "warn": (
            "The terminal value is extremely sensitive to the perpetual growth rate. "
            "Changing g from 2.5% to 3.0% can increase intrinsic value by 10–20%. "
            "Gordon Growth Model assumes r > g — if not, the formula breaks down."
        ),
        "source": "Investopedia — Terminal Value · Damodaran (NYU)",
    },

    # ── TECHNICAL INDICATORS ─────────────────────────────────────────────────

    "rsi": {
        "name": "RSI — Relative Strength Index (14-period)",
        "definition": (
            "Momentum oscillator that measures the speed and magnitude of recent price "
            "changes on a scale of 0 to 100. Developed by J. Welles Wilder Jr. in 1978. "
            "Used to identify overbought and oversold conditions."
        ),
        "formula": (
            "RS = Average Gain over 14 periods ÷ Average Loss over 14 periods\n"
            "RSI = 100 − (100 ÷ (1 + RS))"
        ),
        "good": (
            "< 30 → oversold (potential buying opportunity) "
            "· 30–50 → bearish/weak momentum "
            "· 50–70 → bullish/strong momentum "
            "· > 70 → overbought (potential pullback)"
        ),
        "warn": (
            "In a strong uptrend, RSI can stay above 70 for weeks or months — "
            "overbought alone is not a sell signal. "
            "Best used in combination with trend and support/resistance analysis."
        ),
        "source": "Investopedia — RSI Indicator",
    },

    "macd": {
        "name": "MACD — Moving Average Convergence Divergence",
        "definition": (
            "Trend-following momentum indicator that shows the relationship between "
            "two exponential moving averages (EMA). The MACD line crossing above the "
            "signal line is a bullish signal; crossing below is bearish."
        ),
        "formula": (
            "MACD Line = EMA(12 periods) − EMA(26 periods)\n"
            "Signal Line = EMA(9 periods) of MACD Line\n"
            "Histogram = MACD Line − Signal Line"
        ),
        "good": (
            "MACD above Signal Line = bullish momentum "
            "· MACD above zero = positive territory "
            "· Rising histogram = momentum building"
        ),
        "warn": (
            "MACD is a lagging indicator — signals come after a trend starts. "
            "It can produce many false signals in choppy, sideways markets. "
            "More reliable on daily/weekly charts than intraday."
        ),
        "source": "Investopedia — MACD Indicator",
    },

    "bollinger": {
        "name": "Bollinger Bands (20-period, 2 standard deviations)",
        "definition": (
            "Volatility envelope around a 20-day simple moving average. "
            "The bands widen during high volatility and contract during low volatility. "
            "Developed by John Bollinger in the 1980s."
        ),
        "formula": (
            "Middle Band = SMA(20)\n"
            "Upper Band = SMA(20) + 2 × Standard Deviation(20)\n"
            "Lower Band = SMA(20) − 2 × Standard Deviation(20)\n"
            "%B = (Price − Lower Band) ÷ (Upper − Lower Band)"
        ),
        "good": (
            "%B > 1.0 → price above upper band (overbought caution) "
            "· %B < 0.0 → price below lower band (oversold / bounce potential) "
            "· Squeeze (narrow bands) → low volatility often precedes a breakout"
        ),
        "warn": (
            "Bollinger Bands indicate volatility level but do NOT predict direction. "
            "A price touching the upper band in an uptrend can be a sign of strength, "
            "not necessarily a reversal."
        ),
        "source": "Investopedia — Bollinger Bands",
    },

    "obv": {
        "name": "OBV — On-Balance Volume",
        "definition": (
            "Running total that adds volume on days the price closes up and "
            "subtracts volume on days the price closes down. Based on the theory "
            "that volume precedes price — institutional accumulation shows up in "
            "OBV before it appears in the price."
        ),
        "formula": (
            "If Close > Prior Close: OBV = OBV_prior + Volume\n"
            "If Close < Prior Close: OBV = OBV_prior − Volume\n"
            "If Close = Prior Close: OBV = OBV_prior"
        ),
        "good": (
            "OBV trending up with price → confirmed uptrend (strong) "
            "· OBV rising while price falls → bullish divergence (potential reversal up) "
            "· OBV falling while price rises → bearish divergence (distribution warning)"
        ),
        "warn": (
            "OBV is most useful for detecting divergences from price. "
            "A single large-volume day (earnings, index addition) can distort the indicator "
            "for weeks. The absolute level of OBV is less important than its trend."
        ),
        "source": "Investopedia — On-Balance Volume",
    },

    "sma": {
        "name": "Moving Averages — SMA and EMA",
        "definition": (
            "SMA (Simple Moving Average): equal-weight average of closing prices over N periods. "
            "EMA (Exponential Moving Average): gives more weight to recent prices, "
            "reacts faster to price changes. "
            "Used to identify trend direction and dynamic support/resistance levels."
        ),
        "formula": (
            "SMA(N) = Sum of closing prices over N periods ÷ N\n"
            "EMA applies a multiplier: k = 2 ÷ (N + 1) to weight recent prices more\n"
            "Golden Cross: SMA50 crosses above SMA200 → bullish trend signal\n"
            "Death Cross:  SMA50 crosses below SMA200 → bearish trend signal"
        ),
        "good": (
            "Price > 200-SMA = long-term uptrend (bullish regime) "
            "· Price > 50-SMA = medium-term bullish "
            "· Golden Cross = historically strong bullish signal"
        ),
        "warn": (
            "Moving averages are lagging — they confirm trends after they start, "
            "not predict them. In sideways markets they generate many false crossover signals."
        ),
        "source": "Investopedia — Simple Moving Average · EMA",
    },

    # ── SENTIMENT & ANALYST DATA ──────────────────────────────────────────────

    "analyst_consensus": {
        "name": "Analyst Consensus Rating",
        "definition": (
            "Average recommendation from all analysts covering the stock, "
            "on Yahoo Finance's 1–5 scale. Aggregated from ratings published "
            "by investment banks and independent research firms. "
            "Also used here to display analyst price target data."
        ),
        "formula": (
            "Mean Rating = Sum of all analyst ratings ÷ Number of analysts\n"
            "Scale: 1.0 = Strong Buy · 2.0 = Buy · 3.0 = Hold "
            "· 4.0 = Sell · 5.0 = Strong Sell"
        ),
        "good": (
            "≤ 1.8 strong bullish consensus "
            "· 1.8–2.5 broadly bullish "
            "· 2.5–3.5 mixed / hold "
            "· ≥ 3.5 broadly bearish"
        ),
        "warn": (
            "Analyst ratings are subject to conflicts of interest — "
            "banks that underwrite deals for a company may be reluctant to issue sell ratings. "
            "The direction of estimate revisions is often more informative than the absolute rating."
        ),
        "source": "Yahoo Finance — Analyst Recommendations",
    },

    "earnings_surprise": {
        "name": "Earnings Surprise",
        "definition": (
            "The percentage difference between actual reported EPS and "
            "the consensus analyst estimate at the time of reporting. "
            "Positive surprises (beats) tend to cause short-term price increases; "
            "negative surprises (misses) often cause sharp drops."
        ),
        "formula": (
            "Surprise % = (Actual EPS − Estimated EPS) ÷ |Estimated EPS| × 100%\n"
            "Positive = beat consensus · Negative = missed consensus"
        ),
        "good": (
            "Consistent beats over 6–8 quarters signal strong execution and "
            "conservative guidance. Average surprise > 5% is excellent."
        ),
        "warn": (
            "Companies sometimes guide analyst estimates down before the quarter ends "
            "to make a beat easier ('sandbagging'). "
            "Check if estimates were revised down in the weeks before the report."
        ),
        "source": "Yahoo Finance · Investopedia — Earnings Surprise",
    },

    "short_interest": {
        "name": "Short Interest & Short Ratio",
        "definition": (
            "Short % of Float: the percentage of tradeable shares currently sold short "
            "(borrowed and sold, betting on a price decline). "
            "Short Ratio (Days to Cover): how many average trading days it would take "
            "all short sellers to buy back their positions."
        ),
        "formula": (
            "Short % of Float = Shares Sold Short ÷ Float (tradeable shares) × 100%\n"
            "Short Ratio = Shares Sold Short ÷ Average Daily Trading Volume"
        ),
        "good": (
            "< 3% normal · 3–8% slightly elevated "
            "· 8–20% high short interest (bearish sentiment) "
            "· > 20% very high — also potential short squeeze fuel"
        ),
        "warn": (
            "High short interest is ambiguous: it signals bearish sentiment "
            "but also means there is a large pool of buyers who must cover if the "
            "price rises (short squeeze potential). "
            "GameStop (2021) is the classic example."
        ),
        "source": "Yahoo Finance · Investopedia — Short Interest",
    },
}


def tooltip_html(metric_key: str, label: str, value: str,
                 forecast_status: str = "", position: str = "") -> str:
    """
    Render a metric row with an inline hover tooltip.

    forecast_status: 'beat' | 'miss' | 'inline' | ''
    position: '' | 'left' | 'up'
    """
    m = METRICS.get(metric_key, {})
    if not m:
        return f"""
        <div class='mrow'>
          <span class='mrow-label'>{label}</span>
          <span class='mrow-val'>{value}</span>
        </div>"""

    box_class  = f"tt-box{' tt-' + position if position else ''}"
    name       = m.get("name", label)
    defn       = m.get("definition", "")
    formula    = m.get("formula", "")
    good       = m.get("good", "")
    warn       = m.get("warn", "")
    src        = m.get("source", "")

    formula_html = (f"<div class='tt-formula'>{formula.replace(chr(10), '<br>')}</div>"
                    if formula else "")
    good_html    = f"<div class='tt-range'>✅ {good}</div>"    if good   else ""
    warn_html    = f"<div class='tt-warn'>⚠️ {warn}</div>"    if warn   else ""
    src_html     = f"<div class='tt-src'>Source: {src}</div>" if src    else ""

    badge_map = {
        "beat":   ("<span class='fc-badge fc-beat'>BEAT</span>",      "mrow-beat"),
        "miss":   ("<span class='fc-badge fc-miss'>MISS</span>",      "mrow-miss"),
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
    return (f"<div class='section-header' style='font-size:16px;font-weight:700;"
            f"margin:14px 0 6px;color:#e2e8f0'>{title}</div>")


def _lookup_ticker_name(sym: str) -> str:
    """Best-effort full-name lookup from the search module's TICKER_NAMES dict."""
    try:
        from search import TICKER_NAMES
        return TICKER_NAMES.get(sym.upper(), "")
    except Exception:
        return ""


def ticker_tooltip(sym: str, name: str = "", style: str = "") -> str:
    """
    Wrap a ticker symbol in a hover tooltip showing the full company name.
    If no name is passed, falls back to a TICKER_NAMES lookup so the tooltip
    works everywhere — even when the caller doesn't have the name handy.

    Uses BOTH a CSS tooltip (styled) and a native title attribute (always works,
    even when CSS is clipped by a parent's overflow:hidden, e.g. inside columns).
    """
    if not name or name == sym:
        name = _lookup_ticker_name(sym)

    safe_name = (name or "").replace('"', "&quot;").replace("'", "&#39;")
    title_attr = f' title="{safe_name}"' if safe_name else ""

    if not safe_name:
        return f"<span style='{style}'{title_attr}>{sym}</span>"

    return (
        f"<span class='tk-wrap'{title_attr}>"
        f"<span style='{style}'>{sym}</span>"
        f"<span class='tk-tip'>{safe_name}</span>"
        f"</span>"
    )
