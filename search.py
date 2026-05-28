"""
search.py — Stock name search and ticker lookup.

Strategy:
  1. yfinance Search API (works when network available)
  2. Curated name→ticker map for ~500 well-known global stocks
  3. Partial match fallback on the curated list
"""
import streamlit as st
import yfinance as yf
import re
from typing import List

# ─── Curated ticker → name map (major global stocks) ─────────────────────────
TICKER_NAMES = {
    # ── US Large Cap ─────────────────────────────────────────────────────────
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "GOOG":  "Alphabet Inc. (Google, Class C)",
    "AMZN":  "Amazon.com Inc.",
    "NVDA":  "NVIDIA Corporation",
    "META":  "Meta Platforms Inc.",
    "TSLA":  "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc.",
    "LLY":   "Eli Lilly and Company",
    "JPM":   "JPMorgan Chase & Co.",
    "V":     "Visa Inc.",
    "MA":    "Mastercard Incorporated",
    "UNH":   "UnitedHealth Group Inc.",
    "XOM":   "ExxonMobil Corporation",
    "JNJ":   "Johnson & Johnson",
    "WMT":   "Walmart Inc.",
    "AVGO":  "Broadcom Inc.",
    "PG":    "Procter & Gamble Co.",
    "HD":    "Home Depot Inc.",
    "CVX":   "Chevron Corporation",
    "MRK":   "Merck & Co. Inc.",
    "ABBV":  "AbbVie Inc.",
    "COST":  "Costco Wholesale Corporation",
    "AMD":   "Advanced Micro Devices Inc.",
    "NFLX":  "Netflix Inc.",
    "CRM":   "Salesforce Inc.",
    "BAC":   "Bank of America Corporation",
    "PEP":   "PepsiCo Inc.",
    "KO":    "Coca-Cola Company",
    "TMO":   "Thermo Fisher Scientific Inc.",
    "CSCO":  "Cisco Systems Inc.",
    "ACN":   "Accenture plc",
    "ORCL":  "Oracle Corporation",
    "MCD":   "McDonald's Corporation",
    "ABT":   "Abbott Laboratories",
    "ADBE":  "Adobe Inc.",
    "INTC":  "Intel Corporation",
    "QCOM":  "QUALCOMM Incorporated",
    "WFC":   "Wells Fargo & Company",
    "CAT":   "Caterpillar Inc.",
    "IBM":   "International Business Machines",
    "GE":    "GE Aerospace",
    "NOW":   "ServiceNow Inc.",
    "INTU":  "Intuit Inc.",
    "TXN":   "Texas Instruments Inc.",
    "AMGN":  "Amgen Inc.",
    "MS":    "Morgan Stanley",
    "GS":    "Goldman Sachs Group Inc.",
    "RTX":   "Raytheon Technologies",
    "HON":   "Honeywell International Inc.",
    "BA":    "Boeing Company",
    "SPGI":  "S&P Global Inc.",
    "BLK":   "BlackRock Inc.",
    "DE":    "Deere & Company",
    "SBUX":  "Starbucks Corporation",
    "AXP":   "American Express Company",
    "PFE":   "Pfizer Inc.",
    "BMY":   "Bristol-Myers Squibb Company",
    "GILD":  "Gilead Sciences Inc.",
    "VRTX":  "Vertex Pharmaceuticals Inc.",
    "REGN":  "Regeneron Pharmaceuticals Inc.",
    "MRNA":  "Moderna Inc.",
    "ISRG":  "Intuitive Surgical Inc.",
    "SYK":   "Stryker Corporation",
    "MDT":   "Medtronic plc",
    "BSX":   "Boston Scientific Corporation",
    "LMT":   "Lockheed Martin Corporation",
    "NOC":   "Northrop Grumman Corporation",
    "GD":    "General Dynamics Corporation",
    "UPS":   "United Parcel Service Inc.",
    "FDX":   "FedEx Corporation",
    "UNP":   "Union Pacific Corporation",
    "NEE":   "NextEra Energy Inc.",
    "DUK":   "Duke Energy Corporation",
    "SO":    "Southern Company",
    "D":     "Dominion Energy Inc.",
    "AMT":   "American Tower Corporation",
    "CCI":   "Crown Castle Inc.",
    "PLD":   "Prologis Inc.",
    "EQIX":  "Equinix Inc.",
    "DLR":   "Digital Realty Trust Inc.",
    "SPG":   "Simon Property Group Inc.",
    "O":     "Realty Income Corporation",
    "CME":   "CME Group Inc.",
    "ICE":   "Intercontinental Exchange Inc.",
    "MCO":   "Moody's Corporation",
    "MSCI":  "MSCI Inc.",
    "SCHW":  "Charles Schwab Corporation",
    "PYPL":  "PayPal Holdings Inc.",
    "SQ":    "Block Inc.",
    "SHOP":  "Shopify Inc.",
    "SNOW":  "Snowflake Inc.",
    "PLTR":  "Palantir Technologies Inc.",
    "DDOG":  "Datadog Inc.",
    "CRWD":  "CrowdStrike Holdings Inc.",
    "ZS":    "Zscaler Inc.",
    "NET":   "Cloudflare Inc.",
    "OKTA":  "Okta Inc.",
    "MU":    "Micron Technology Inc.",
    "AMAT":  "Applied Materials Inc.",
    "LRCX":  "Lam Research Corporation",
    "KLAC":  "KLA Corporation",
    "MRVL":  "Marvell Technology Inc.",
    "ON":    "ON Semiconductor Corporation",
    "ENPH":  "Enphase Energy Inc.",
    "FSLR":  "First Solar Inc.",
    "F":     "Ford Motor Company",
    "GM":    "General Motors Company",
    "RIVN":  "Rivian Automotive Inc.",
    "NIO":   "NIO Inc.",
    "LI":    "Li Auto Inc.",
    "XPEV":  "XPeng Inc.",
    "CMG":   "Chipotle Mexican Grill Inc.",
    "LULU":  "Lululemon Athletica Inc.",
    "NKE":   "Nike Inc.",
    "TGT":   "Target Corporation",
    "COST":  "Costco Wholesale Corporation",
    "DG":    "Dollar General Corporation",
    "DLTR":  "Dollar Tree Inc.",
    "TJX":   "TJX Companies Inc.",
    "LOW":   "Lowe's Companies Inc.",
    "ORLY":  "O'Reilly Automotive Inc.",
    "AZO":   "AutoZone Inc.",
    "RH":    "RH (Restoration Hardware)",
    "ULTA":  "Ulta Beauty Inc.",
    # ── European ─────────────────────────────────────────────────────────────
    "ASML.AS":   "ASML Holding N.V.",
    "PHIA.AS":   "Philips N.V.",
    "HEIA.AS":   "Heineken N.V.",
    "INGA.AS":   "ING Groep N.V.",
    "AD.AS":     "Ahold Delhaize N.V.",
    "RAND.AS":   "Randstad N.V.",
    "NXPI.AS":   "NXP Semiconductors N.V.",
    "BESI.AS":   "BE Semiconductor Industries N.V.",
    "UNA.AS":    "Unilever PLC (Amsterdam)",
    "SHEL.L":    "Shell PLC",
    "BP.L":      "BP PLC",
    "AZN.L":     "AstraZeneca PLC",
    "GSK.L":     "GSK PLC",
    "HSBA.L":    "HSBC Holdings PLC",
    "LLOY.L":    "Lloyds Banking Group PLC",
    "BARC.L":    "Barclays PLC",
    "RIO.L":     "Rio Tinto PLC",
    "BHP.L":     "BHP Group PLC",
    "VOD.L":     "Vodafone Group PLC",
    "DGE.L":     "Diageo PLC",
    "ULVR.L":    "Unilever PLC",
    "BATS.L":    "British American Tobacco PLC",
    "REL.L":     "RELX PLC",
    "NG.L":      "National Grid PLC",
    "SSE.L":     "SSE PLC",
    "SAP.DE":    "SAP SE",
    "SIE.DE":    "Siemens AG",
    "ALV.DE":    "Allianz SE",
    "MUV2.DE":   "Munich Re",
    "DTE.DE":    "Deutsche Telekom AG",
    "BMW.DE":    "BMW AG",
    "VOW3.DE":   "Volkswagen AG",
    "MBG.DE":    "Mercedes-Benz Group AG",
    "BAYN.DE":   "Bayer AG",
    "BASF.DE":   "BASF SE",
    "DBK.DE":    "Deutsche Bank AG",
    "ADS.DE":    "adidas AG",
    "IFX.DE":    "Infineon Technologies AG",
    "RWE.DE":    "RWE AG",
    "EON.DE":    "E.ON SE",
    "HEN3.DE":   "Henkel AG",
    "BAS.DE":    "BASF SE",
    "AI.PA":     "Air Liquide SA",
    "OR.PA":     "L'Oréal SA",
    "MC.PA":     "LVMH Moët Hennessy",
    "TTE.PA":    "TotalEnergies SE",
    "SAN.PA":    "Sanofi SA",
    "BNP.PA":    "BNP Paribas SA",
    "ACA.PA":    "Crédit Agricole SA",
    "KER.PA":    "Kering SA",
    "CAP.PA":    "Capgemini SE",
    "DAN.PA":    "Danone SA",
    "SGO.PA":    "Saint-Gobain SA",
    "NOVN.SW":   "Novartis AG",
    "ROG.SW":    "Roche Holding AG",
    "NESN.SW":   "Nestlé SA",
    "ABBN.SW":   "ABB Ltd.",
    "ZURN.SW":   "Zurich Insurance Group AG",
    "UBSG.SW":   "UBS Group AG",
    "CSGN.SW":   "Credit Suisse Group AG",
    "LONN.SW":   "Lonza Group AG",
    "GIVN.SW":   "Givaudan SA",
    "SOON.SW":   "Sonova Holding AG",
    "ITX.MC":    "Industria de Diseño Textil (Inditex/Zara)",
    "SAN.MC":    "Banco Santander SA",
    "BBVA.MC":   "BBVA SA",
    "IBE.MC":    "Iberdrola SA",
    "REP.MC":    "Repsol SA",
    "TEF.MC":    "Telefónica SA",
    "ENEL.MI":   "Enel SpA",
    "ENI.MI":    "Eni SpA",
    "UCG.MI":    "UniCredit SpA",
    "ISP.MI":    "Intesa Sanpaolo SpA",
    "STM.MI":    "STMicroelectronics N.V.",
    # ── Nordic ───────────────────────────────────────────────────────────────
    "VOLV-B.ST": "Volvo AB",
    "ERIC-B.ST": "Ericsson",
    "HM-B.ST":   "H&M Group",
    "ATCO-A.ST": "Atlas Copco AB",
    "SAND.ST":   "Sandvik AB",
    "EQNR.OL":   "Equinor ASA",
    "DNB.OL":    "DNB Bank ASA",
    "TEL.OL":    "Telenor ASA",
    # ── Japan ────────────────────────────────────────────────────────────────
    "7203.T":  "Toyota Motor Corporation",
    "6758.T":  "Sony Group Corporation",
    "9432.T":  "NTT (Nippon Telegraph)",
    "8306.T":  "Mitsubishi UFJ Financial Group",
    "9984.T":  "SoftBank Group Corp.",
    "6861.T":  "Keyence Corporation",
    "7974.T":  "Nintendo Co. Ltd.",
    "4519.T":  "Chugai Pharmaceutical",
    "6367.T":  "Daikin Industries",
    "8035.T":  "Tokyo Electron Ltd.",
    "6857.T":  "Advantest Corporation",
    "6920.T":  "Lasertec Corporation",
    "7267.T":  "Honda Motor Co. Ltd.",
    "7201.T":  "Nissan Motor Co. Ltd.",
    "6501.T":  "Hitachi Ltd.",
    "6702.T":  "Fujitsu Ltd.",
    "6752.T":  "Panasonic Holdings",
    "4503.T":  "Astellas Pharma Inc.",
    "4568.T":  "Daiichi Sankyo Co. Ltd.",
    "4523.T":  "Eisai Co. Ltd.",
    # ── South Korea ──────────────────────────────────────────────────────────
    "005930.KS": "Samsung Electronics Co. Ltd.",
    "000660.KS": "SK Hynix Inc.",
    "035420.KS": "NAVER Corporation",
    "005380.KS": "Hyundai Motor Company",
    "000270.KS": "Kia Corporation",
    "068270.KS": "Celltrion Inc.",
    "207940.KS": "Samsung Biologics Co. Ltd.",
    "051910.KS": "LG Chem Ltd.",
    "105560.KS": "KB Financial Group Inc.",
    "055550.KS": "Shinhan Financial Group",
    "035720.KS": "Kakao Corp.",
    "003550.KS": "LG Corp.",
    # ── Hong Kong / China ────────────────────────────────────────────────────
    "0700.HK":  "Tencent Holdings Ltd.",
    "9988.HK":  "Alibaba Group Holding Ltd.",
    "0005.HK":  "HSBC Holdings PLC",
    "0939.HK":  "China Construction Bank",
    "1398.HK":  "ICBC",
    "3988.HK":  "Bank of China",
    "2318.HK":  "Ping An Insurance",
    "9618.HK":  "JD.com Inc.",
    "0388.HK":  "Hong Kong Exchanges (HKEX)",
    "0883.HK":  "CNOOC Ltd.",
    "0386.HK":  "China Petroleum & Chemical (Sinopec)",
    "0857.HK":  "PetroChina Co. Ltd.",
    "1810.HK":  "Xiaomi Corporation",
    "9999.HK":  "NetEase Inc.",
    "BABA":     "Alibaba Group Holding (ADR)",
    "JD":       "JD.com Inc. (ADR)",
    "BIDU":     "Baidu Inc. (ADR)",
    "TCEHY":    "Tencent Holdings (OTC ADR)",
    "PDD":      "PDD Holdings Inc.",
    "NIO":      "NIO Inc. (ADR)",
    # ── Canada ───────────────────────────────────────────────────────────────
    "SHOP.TO":  "Shopify Inc.",
    "RY.TO":    "Royal Bank of Canada",
    "TD.TO":    "Toronto-Dominion Bank",
    "BNS.TO":   "Bank of Nova Scotia",
    "BMO.TO":   "Bank of Montreal",
    "CM.TO":    "Canadian Imperial Bank (CIBC)",
    "CNR.TO":   "Canadian National Railway",
    "CP.TO":    "Canadian Pacific Kansas City",
    "ENB.TO":   "Enbridge Inc.",
    "SU.TO":    "Suncor Energy Inc.",
    "CNQ.TO":   "Canadian Natural Resources",
    "TRP.TO":   "TC Energy Corporation",
    "OTEX.TO":  "Open Text Corporation",
    "MFC.TO":   "Manulife Financial Corporation",
    "SLF.TO":   "Sun Life Financial Inc.",
    "L.TO":     "Loblaw Companies Ltd.",
    "WN.TO":    "George Weston Ltd.",
    "ATD.TO":   "Alimentation Couche-Tard",
    "T.TO":     "TELUS Corporation",
    "BCE.TO":   "BCE Inc.",
    "ABX.TO":   "Barrick Gold Corporation",
    "AEM.TO":   "Agnico Eagle Mines Ltd.",
    "FNV.TO":   "Franco-Nevada Corporation",
    "TRI.TO":   "Thomson Reuters Corporation",
    "WSP.TO":   "WSP Global Inc.",
    "CAE.TO":   "CAE Inc.",
    "CCO.TO":   "Cameco Corporation",
    "TECK-B.TO":"Teck Resources Ltd.",
    # ── Australia ────────────────────────────────────────────────────────────
    "BHP.AX":  "BHP Group Ltd.",
    "CBA.AX":  "Commonwealth Bank of Australia",
    "CSL.AX":  "CSL Limited",
    "NAB.AX":  "National Australia Bank",
    "WBC.AX":  "Westpac Banking Corporation",
    "ANZ.AX":  "ANZ Group Holdings",
    "WES.AX":  "Wesfarmers Ltd.",
    "WOW.AX":  "Woolworths Group Ltd.",
    "RIO.AX":  "Rio Tinto Ltd.",
    "FMG.AX":  "Fortescue Ltd.",
    "MQG.AX":  "Macquarie Group Ltd.",
    "NCM.AX":  "Newmont Corporation (ASX)",
    # ── India (ADRs / common) ────────────────────────────────────────────────
    "INFY":    "Infosys Ltd. (ADR)",
    "WIT":     "Wipro Ltd. (ADR)",
    "HDB":     "HDFC Bank Ltd. (ADR)",
    "IBN":     "ICICI Bank Ltd. (ADR)",
    "RDY":     "Dr. Reddy's Laboratories (ADR)",
    "TTM":     "Tata Motors Ltd. (ADR)",
}

# Build reverse map: lowercase name words → ticker
_NAME_TO_TICKERS: dict = {}
for _ticker, _name in TICKER_NAMES.items():
    _key = _name.lower()
    _words = re.sub(r'[^a-z0-9 ]', '', _key).split()
    for _word in _words:
        if len(_word) >= 3:
            _NAME_TO_TICKERS.setdefault(_word, []).append(_ticker)


@st.cache_data(ttl=300, show_spinner=False)
def search_ticker(query: str, max_results: int = 8) -> list:
    """
    Search for tickers by company name or ticker symbol.
    Returns list of {symbol, name, exchange, type} dicts.
    """
    query = query.strip()
    if not query or len(query) < 2:
        return []

    results = []

    # Step 1: yfinance Search API
    try:
        s = yf.Search(query, max_results=max_results + 5, news_count=0)
        for q in s.quotes:
            sym  = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or ""
            exc  = q.get("exchange", "")
            qt   = q.get("quoteType", "")
            if sym and qt in ["EQUITY", "ETF"] and name:
                results.append({
                    "symbol":   sym,
                    "name":     name,
                    "exchange": exc,
                    "type":     qt,
                })
    except Exception:
        pass

    # Step 2: Curated map lookup
    q_up    = query.upper()
    q_lower = query.lower()

    # Exact ticker match
    if q_up in TICKER_NAMES and not any(r["symbol"] == q_up for r in results):
        results.insert(0, {"symbol": q_up, "name": TICKER_NAMES[q_up], "exchange": "", "type": "EQUITY"})

    # Name word search
    words = re.sub(r'[^a-z0-9 ]', '', q_lower).split()
    if words:
        scores: dict = {}
        for word in words:
            for t in _NAME_TO_TICKERS.get(word, []):
                scores[t] = scores.get(t, 0) + 1
        # Sort by match score descending
        matched = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        for t in matched[:6]:
            if not any(r["symbol"] == t for r in results):
                results.append({"symbol": t, "name": TICKER_NAMES[t], "exchange": "", "type": "EQUITY"})

    # Deduplicate preserving order
    seen = set()
    unique = []
    for r in results:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            unique.append(r)

    return unique[:max_results]


def get_ticker_display_name(ticker: str, info: dict = None) -> str:
    """Get a human-readable name for a ticker."""
    if info:
        name = info.get("shortName") or info.get("longName")
        if name:
            return name
    return TICKER_NAMES.get(ticker.upper(), ticker)
