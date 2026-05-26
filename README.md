# 📈 Stock Analyzer

A comprehensive equity analysis tool built with Python and Streamlit.
Analyze up to 5 stocks at a time — all data from Yahoo Finance, no API keys required.

---

## Features

| Module | What it does |
|--------|-------------|
| **4 DCF Models** | WACC, CAPM, Fixed Rate, Two-Stage FCF — shown side-by-side |
| **Technical Analysis** | SMA/EMA, RSI, MACD, Bollinger Bands, OBV, Support/Resistance — each with ✅/🔴 signal feedback |
| **Peer Comparison** | Auto-detects competitors from sector/industry; manual override supported |
| **Composite Score** | Weighted 0–10 score → STRONG BUY / BUY / HOLD / SELL / STRONG SELL |
| **Sentiment** | Analyst consensus, price targets, earnings surprise history |
| **European stocks** | Supports .AS (Amsterdam), .L (London), .DE (Frankfurt), .PA (Paris), etc. |

---

## Installation

### 1. Make sure you have Python 3.10+

```bash
python --version
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or if you prefer a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate.bat       # Windows

pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. Enter up to **5 tickers** in the left sidebar
   - US stocks: `AAPL`, `MSFT`, `NVDA`
   - Amsterdam: `ASML.AS`, `HEIA.AS`, `PHIA.AS`
   - London: `SHEL.L`, `BP.L`, `AZN.L`
   - Frankfurt: `SAP.DE`, `SIE.DE`
2. Adjust the **Fixed Discount Rate** slider (used in Model 3 DCF)
3. Optionally override peer tickers per stock
4. Click **🔍 Analyze**

---

## Composite Scoring Weights

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Fundamental | 40% | Revenue/earnings growth, margins, ROE, FCF, leverage |
| Valuation | 25% | P/E, Forward P/E, PEG, EV/EBITDA, DCF upside |
| Technical | 20% | All 6 indicator groups combined |
| Sentiment | 15% | Analyst consensus, price target upside, earnings beats |

**Signal thresholds** (0–10 composite):
- ≥ 7.5 → 🟢 STRONG BUY
- ≥ 6.5 → 🟢 BUY
- ≥ 5.5 → 🟡 WEAK BUY
- ≥ 4.5 → 🟡 HOLD
- ≥ 3.5 → 🟠 WEAK SELL
- ≥ 2.5 → 🔴 SELL
- < 2.5 → 🔴 STRONG SELL

---

## DCF Models

| Model | Discount Rate | Best for |
|-------|--------------|----------|
| **WACC DCF** | Calculated WACC (equity + debt blended) | Companies with significant debt |
| **CAPM DCF** | Risk-free rate + β × 5.5% MRP | Pure equity analysis |
| **Fixed Rate DCF** | Your chosen rate (sidebar slider) | Sensitivity analysis |
| **Two-Stage FCF** | CAPM rate with fading growth stages | High-growth companies |

---

## Files

```
stock_analyzer/
├── app.py                  ← Main Streamlit UI
├── data.py                 ← yfinance fetching + caching
├── peers.py                ← Industry peer mapping (60+ industries)
├── fundamental.py          ← Financial statements + 4 DCF models
├── technical.py            ← All indicators + signal generation
├── valuation_sentiment.py  ← Relative valuation + analyst sentiment
├── scoring.py              ← Composite 0–10 scoring engine
└── requirements.txt
```

---

## Notes

- **Data is cached for 1 hour** — re-running within an hour won't re-fetch
- **DCF shows "not applicable"** for stocks with persistent negative FCF
- **Peer detection** covers ~60 industries; override manually for niche sectors
- Results are for **informational purposes only**, not financial advice
