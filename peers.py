"""
peers.py — Comprehensive international peer discovery.

Coverage: US (small→large cap), Canada (.TO), UK (.L), Germany (.DE/.F),
Netherlands (.AS), France (.PA), Switzerland (.SW), Italy (.MI), Spain (.MC),
Japan (.T), South Korea (.KS), Hong Kong (.HK), China ADRs / A-shares.

Strategy:
  1. Exact industry match → hardcoded international list
  2. Supplemented by yf.Search for dynamic discovery
  3. Validate & rank by market-cap proximity to target
  4. Return 5–10 peers
"""
import time
import yfinance as yf
from typing import List, Dict


# ─── Simple TTL cache ─────────────────────────────────────────────────────────
import time as _time
_MOD_CACHE: dict = {}

def _ttl(key, ttl_secs, fn):
    now = _time.time()
    if key in _MOD_CACHE:
        val, ts = _MOD_CACHE[key]
        if now - ts < ttl_secs:
            return val
    result = fn()
    _MOD_CACHE[key] = (result, now)
    return result


# ─── Comprehensive International Peer Map ─────────────────────────────────────
# Format: "yfinance industry string": [US+international tickers]
INDUSTRY_PEERS: Dict[str, List[str]] = {

    # ── SEMICONDUCTORS ────────────────────────────────────────────────────────
    "Semiconductors": [
        # US
        "NVDA","AMD","INTC","QCOM","AVGO","MU","TXN","MRVL","AMAT","KLAC","LRCX","MPWR","ON","SWKS","QRVO",
        # Europe
        "ASML.AS","STM.MI","IFX.DE","AMS.SW","NXPI.AS","BESI.AS","AKER.OL",
        # Japan
        "6723.T","6857.T","8035.T","6503.T","6645.T","6963.T",      # Renesas, Advantest, Tokyo Electron, Mitsubishi Elec, OMRON, ROHM
        # Korea
        "005930.KS","000660.KS","058470.KS","093700.KS",             # Samsung, SK Hynix, Huijai, NHN
        # Taiwan/HK/China
        "TSM","UMC","ASX","VIS","HIMX","CAMT","SMIC.HK","688981.SS",
        # Canada
        "GSY.TO","POET.V",
    ],

    "Semiconductor Equipment": [
        "AMAT","LRCX","KLAC","ONTO","UCTT","CCMP","ACMR","ICHR","MKSI",
        "ASML.AS","BESI.AS","COHU","FORM","PLAB",
        "6857.T","7735.T","6920.T",   # Advantest, Screen HD, Lasertec
        "058470.KS",
        "HWM",
    ],

    # ── SOFTWARE ──────────────────────────────────────────────────────────────
    "Software—Application": [
        # US
        "MSFT","CRM","ADBE","INTU","NOW","WDAY","TEAM","DDOG","ZM","VEEV","HUBS","BILL","TYL","PCTY","PAYC",
        # Europe
        "SAP.DE","DASSAULT.PA","SDXOF","TEMN.SW","NEMETSCHEK.DE","ASKOB.OL",
        # Japan
        "3994.T","4307.T","3659.T","9682.T",    # Money Forward, Nomura Research, Nexon, Data Arts
        # Canada
        "OTEX.TO","DSG.TO","KXS.TO","ENGH.TO",
        # Asia
        "BABA","TCEHY","9988.HK","700.HK",
    ],

    "Software—Infrastructure": [
        "MSFT","ORCL","PANW","CRWD","ZS","FTNT","NET","OKTA","S","CYLC","TENB","QUAL","RPD","VRNS",
        "DDOG","SHR","ESTC","SUMO",
        "AVAST.L","SOPHOS","NCC.L","DARKTRACE.L",
        "SAP.DE","SOFTWARE-AG.DE",
        "CAR.TO","CNDX.TO",
        "1347.HK",
    ],

    "Internet Content & Information": [
        # US
        "GOOGL","META","SNAP","PINS","IAC","YELP","CARGP","RDDT","BRZE","PUBM","MGNI","TTGT",
        # Europe
        "REL.L","SCHIBSTED-A.OL","RIGHTMOVE.L","AUTOTRADER.L","IMRG.L",
        # Japan/Korea
        "4689.T","3697.T","035420.KS","067160.KS",   # Z Holdings, Ubic, NAVER, ANK MPC
        # China
        "BIDU","TCEHY","NTES","SOHU","BABA","PDD","JD",
        # Canada
        "SHOP.TO","BBTV.V",
    ],

    "Internet Retail": [
        "AMZN","BABA","JD","EBAY","ETSY","W","CHWY","MELI","OZON","FTCH","RVLV","POSH","REAL",
        "ASOS.L","BOOHOO.L","OCDO.L","THG.L",
        "3038.T","2432.T",
        "035420.KS","069080.KS",
        "9618.HK","1688.HK","9999.HK",
        "SHOP.TO",
    ],

    "Computer Hardware": [
        "AAPL","DELL","HPQ","HPE","NTAP","PSTG","SMCI","LOGI","WDC","STX","PLTR",
        "LOGITECH.SW","ATOS.PA","BULL.PA",
        "6702.T","6752.T","6753.T",   # Fujitsu, Panasonic, Sharp
        "009150.KS","051910.KS",      # Samsung SDI, LG Chem
        "LENOVO.HK","1801.HK",
        "BB.TO","CGI.TO",
    ],

    "Information Technology Services": [
        "IBM","ACN","INFY","WIT","CTSH","DXC","EPAM","GLOB","HCLT","KFRC","MMS","PRFT","EXLS","CACI",
        "CAPGEMINI.PA","ATOS.PA","SOPRA.PA","CGI.TO",
        "INFOSYS.NS","TCS.NS","WIPRO.NS",
        "TCS.L","HCLTECH.L",
        "4307.T","9432.T",
        "057540.KS",
    ],

    # ── HEALTHCARE ────────────────────────────────────────────────────────────
    "Drug Manufacturers—General": [
        # US
        "JNJ","PFE","MRK","ABBV","LLY","BMY","AMGN","BIIB","REGN","GILD",
        # Europe
        "AZN.L","GSK.L","NVO.DE","RHHBY","NOVN.SW","SAN.PA","BAY.DE","UCB.BR","HLMA.L","HIKMA.L",
        # Japan
        "4503.T","4519.T","4568.T","4523.T",   # Astellas, Chugai, Daiichi Sankyo, Eisai
        # Korea
        "068270.KS","207940.KS",               # Celltrion, Samsung Biologics
        # China
        "1093.HK","2269.HK","ZLAB","BNT",
        # Canada
        "NVAX","BDSI.TO",
    ],

    "Biotechnology": [
        "AMGN","GILD","REGN","VRTX","BIIB","MRNA","BNTX","ILMN","ALNY","SGEN","ARWR","BLUE","EDIT","BEAM",
        "NVO.DE","EVOTEC.DE","4SC.DE","MORPHOSYS.DE",
        "OXB.L","AUTOLUS.L","4AW.L","SYNBIO.L",
        "CBMG","ZLAB","LEGN","BGNE","HUTCHMED.HK",
        "4507.T","4565.T",
        "068270.KS","032640.KS",
        "ABCM.TO","NXJ.TO",
    ],

    "Medical Devices": [
        "MDT","ABT","SYK","BSX","EW","ZBH","ISRG","DXCM","HOLX","NVCR","INSP","AXNX","SWAV","NARI",
        "PHIA.AS","PHG","FRESENIUS.DE","DRAEGERWERK.DE",
        "SN.L","SXS.L","IMD.L",
        "4543.T","7733.T","6869.T",   # Terumo, Olympus, Sysmex
        "051900.KS","00393.HK",
        "COHR","BEAT","PROS",
        "HLS.TO","NVX.TO",
    ],

    "Health Care Plans": [
        "UNH","CVS","CI","HUM","ELV","MOH","CNC","OSCR","CLOV","HWC",
        "BUPA.L","AXA.PA","GENERALI.MI",
        "8002.T","4338.T",
        "000100.KS",
        "PING.HK","TAIKANG.HK",
        "SUN.TO",
    ],

    # ── FINANCIAL SERVICES ────────────────────────────────────────────────────
    "Banks—Diversified": [
        "JPM","BAC","WFC","C","USB","PNC","TFC","SCHW","MS","GS",
        "HSBC","BARC.L","LLOY.L","NWG.L","STAN.L",
        "BNP.PA","ACA.PA","SGE.PA","SOCGEN.PA",
        "DBK.DE","CBK.DE",
        "SAN.MC","BBVA.MC",
        "ISP.MI","UCG.MI",
        "UBS.SW","CSGN.SW",
        "8306.T","8316.T","8411.T",   # MUFG, Sumitomo Mitsui, Mizuho
        "105560.KS","055550.KS",      # KB Financial, Shinhan
        "0005.HK","0939.HK","1398.HK","3988.HK",  # HSBC HK, CCB, ICBC, BoC
        "TD.TO","RY.TO","BNS.TO","BMO.TO","CM.TO",
    ],

    "Banks—Regional": [
        "USB","PNC","TFC","FITB","KEY","RF","HBAN","CFG","ZION","SIVB",
        "WBS","PACW","GBCI","IBOC","FFIN","BOH","EWBC","HOPE",
        "VLY","FNB","SNV","UCBI","CBTX","FFBC",
        "ECN.TO","LB.TO","NA.TO","CWB.TO",
    ],

    "Insurance—Life": [
        "MET","PRU","AFL","LNC","UNM","PFG","GL","RLI","HIG","CNO",
        "AVIVA.L","LGEN.L","PRU.L","STJ.L",
        "AXA.PA","SOGECAP.PA",
        "MUV2.DE","R+V.DE",
        "G.MI","CNP.PA",
        "8750.T","8725.T","8697.T",   # Dai-ichi Life, MS&AD, Japan Exchange
        "032830.KS","088350.KS",
        "1299.HK","2318.HK",
        "GWO.TO","MFC.TO","SLF.TO","IAG.TO",
    ],

    "Insurance—Property & Casualty": [
        "BRK-B","AIG","ALL","TRV","CB","PGR","HIG","MKL","CINF","WRB",
        "AXA.PA","MUV2.DE","ZURICH.SW","SLHN.SW",
        "ADM.L","RSA.L","AGEAS.BR",
        "8253.T","8630.T","8766.T",
        "000810.KS",
        "2328.HK","2601.HK",
        "IFC.TO","FFH.TO","EMP-A.TO",
    ],

    "Asset Management": [
        "BLK","SCHW","MS","GS","IVZ","BEN","TROW","AMG","APAM","VCNX","STEP","ARES","BX","KKR","APO",
        "3711.T","8698.T",
        "086790.KS",
        "2388.HK","1788.HK",
        "BROOKFIELD.TO","BAM.TO","X.TO",
        "AMUNDI.PA","IGG.MI",
        "MNKS.L","ABDN.L",
    ],

    "Credit Services": [
        "V","MA","AXP","DFS","SYF","COF","SQ","PYPL","FOUR","AFRM","LPRO","OMF","ALLY",
        "WPG.L","WU","GPN","EVERI","CASS","PRAA",
        "8253.T","8591.T",
        "034220.KS",
        "0388.HK","3323.HK",
        "NFI.TO","CPX.TO",
    ],

    # ── CONSUMER CYCLICAL ─────────────────────────────────────────────────────
    "Auto Manufacturers": [
        "TSLA","F","GM","TM","HMC","STLA","RIVN","LCID","NIO","LI","XPEV",
        "VOW.DE","BMW.DE","MBG.DE","DAI.DE",
        "RNO.PA","PEUGEOT.PA",
        "RACE.MI","FCA.MI",
        "VOLCAR-B.ST","VOLVO.ST",
        "7201.T","7202.T","7203.T","7267.T",  # Nissan, Isuzu, Toyota, Honda
        "005380.KS","000270.KS","012330.KS",  # Hyundai, Kia, Hyundai Mobis
        "0175.HK","1958.HK","2238.HK",        # Geely, BAIC, GAC
        "MGA.TO",
    ],

    "Auto Parts": [
        "APTV","MGA","BWA","LEA","VC","DAN","GNTX","MTOR","ALSN","MODV","ADNT",
        "Continental.DE","SCHAEFFLER.DE","ZF.DE","HELLA.DE","BROSE.DE",
        "VALEO.PA","FAURECIA.PA",
        "7276.T","7296.T","6902.T","7269.T",  # Aisin, FCC, DENSO, Suzuki
        "011210.KS","000150.KS",              # Hyundai WIA, Doosan
        "489.HK","1316.HK",
        "MRE.TO",
    ],

    "Specialty Retail": [
        "HD","LOW","ORLY","AZO","TSCO","BBY","TJX","ULTA","RH","WSM","FIVE","BOOT","HIBB",
        "MKS.L","NEXT.L","DUNELM.L","HALFORDS.L",
        "ADEO.PA","BRICO.FR",
        "BAS.DE","HORNBACH.DE",
        "9843.T","3382.T","7512.T",   # Nitori, Seven & I, AEON
        "139480.KS","004170.KS",
        "0291.HK","6808.HK",
        "CT-A.TO","DOLLARAMA.TO","REITMANS.TO",
    ],

    "Apparel Retail": [
        "NKE","LULU","GPS","ANF","URBN","AEO","PVH","RL","CPRI","HBI","SFIX","RENT","REAL","ONON",
        "NEXT.L","M&S.L","JD.L","WPP.L",
        "ITX.MC","ZAR.MC",  # Inditex (Zara), Zalando
        "ZAL.DE",
        "ADIDAS.DE","PUMA.DE",
        "KERING.PA","LVMH.PA","HER.PA",
        "8048.T","3865.T",
        "004560.KS","084790.KS",
        "0551.HK","0992.HK",
        "GIL.TO","REITMANS-A.TO",
    ],

    "Restaurants": [
        "MCD","SBUX","YUM","QSR","DRI","CMG","DPZ","SHAK","TXRH","WING","LOCO","JACK","FAT",
        "EAT","WEN","RRGB","PLAY","CAKE","BJRI",
        "0290.HK","6862.HK","1829.HK",
        "2702.T","7829.T","3543.T",   # McDonald's Japan, SAINT MARC, Colowide
        "ARQ.TO","ESH.TO",
    ],

    "Hotels & Motels": [
        "MAR","HLT","H","IHG","WH","APLE","RHP","SHO","RLJ","SOND","VCSA",
        "WH.L","IHG.L","WHITBREAD.L",
        "AC.PA",
        "9766.T","9726.T","3003.T",
        "0045.HK","1882.HK",
        "HOT.TO","INN.TO",
    ],

    # ── CONSUMER DEFENSIVE ───────────────────────────────────────────────────
    "Discount Stores": [
        "WMT","TGT","COST","DG","DLTR","BJ","GO","OLLI","FIVE","TJX",
        "NEXT.L","MKS.L","ALDI.DE","METRO.DE",
        "CARREFOUR.PA","CASINO.PA",
        "JERONIMO.MC",
        "ESSELUNGA.MI","AUCHAN.FR",
        "3382.T","8028.T","2651.T",
        "0288.HK","1838.HK","0345.HK",
        "L.TO","DOLLARAMA.TO","ATD.TO","EMP-A.TO",
    ],

    "Beverages—Alcoholic": [
        "BUD","TAP","SAM","STZ","BF-B","DBRV","ABEV","BREW","EAST","CRAFT","COORS",
        "ABIBB.BR","DEO","REL.L",
        "PERNOD.PA","REMY.PA",
        "HEINEKEN.AS","KGF.AS",
        "2330.T","2503.T","2502.T",
        "BUD-ADR","AICAF",
        "WN.TO","ADW-A.TO",
    ],

    "Beverages—Non-Alcoholic": [
        "KO","PEP","MNST","CELH","FIZZ","KDP","COKE","NRGV","PRMW","REED",
        "BRITVIC.L","AG BARR.L","FEVER-TREE.L",
        "DANONE.PA",
        "NESTL.SW",
        "2587.T","2579.T",
        "0322.HK","0345.HK",
        "COTT.TO",
    ],

    "Packaged Foods": [
        "MDLZ","GIS","K","CPB","CAG","HRL","SJM","MKC","LANC","INGR","JJSF","SMPL","NOMD",
        "UNILEVER.L","RKT.L","ABF.L","TREATT.L",
        "DANONE.PA","BN.PA",
        "NESTL.SW",
        "BEIERSDORF.DE","HENKEL.DE",
        "BARILLA.IT",
        "2897.T","2264.T",
        "3687.KS","004370.KS",
        "0220.HK","1968.HK",
        "MFI.TO","MAPLE.TO",
    ],

    "Household & Personal Products": [
        "PG","CL","KMB","CHD","EL","COTY","REVL","ZEUS","SPB","HHC","CLX","GPC",
        "UNILEVER.L","RB.L","HALEON.L",
        "BEIERSDORF.DE","HENKEL.DE","SYMRISE.DE",
        "LOREAL.PA","CLARINS.FR",
        "GIVAUDAN.SW","IFF.SW",
        "4452.T","7974.T",
        "090430.KS","161890.KS",
        "0351.HK","0303.HK",
        "CL.TO",
    ],

    # ── ENERGY ────────────────────────────────────────────────────────────────
    "Oil & Gas Integrated": [
        "XOM","CVX","COP","EOG","OXY","PXD","DVN","MRO","APA","FANG",
        "BP.L","SHEL.L","RDSa.L",
        "TTE.PA",
        "ENI.MI",
        "EQNR.OL",
        "REP.MC",
        "OMV.VI",
        "5020.T","5019.T",   # ENEOS, Idemitsu
        "096770.KS","010950.KS",
        "0883.HK","0386.HK","0857.HK",   # CNOOC, Sinopec, PetroChina
        "SU.TO","CNQ.TO","IMO.TO","HSE.TO","MEG.TO",
    ],

    "Oil & Gas E&P": [
        "COP","EOG","PXD","DVN","FANG","MRO","APA","OVV","HES","CHK","RRC","CNX","CTRA","AR",
        "TOTAL.PA",
        "TLW.L","PGS.OL","AKA.OL",
        "SU.TO","CNQ.TO","OVV.TO","BTE.TO","TVE.TO","ARX.TO","PEY.TO",
        "CNOOC.HK",
    ],

    "Oil & Gas Midstream": [
        "ET","EPD","MMP","KMI","WMB","OKE","MPLX","PAA","TRGP","DT","CAPL","AM","USAC",
        "PBA.TO","PPL.TO","IPL.TO","TRP.TO","ENB.TO","KEY.TO",
        "FLEX-LNG.OL","VOPAK.AS",
    ],

    "Solar": [
        "ENPH","SEDG","FSLR","RUN","ARRY","SPWR","MAXN","NOVA","CSIQ","JKS","DQ","SOL",
        "ORSTED.CO","VESTAS.CO",
        "BEP.TO","BLX.TO",
        "0968.HK","0916.HK","1798.HK",
        "SOLARIA.MC",
        "SMA-SOLAR.DE","NORDEX.DE",
    ],

    "Utilities—Regulated Electric": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","XEL","ETR","FE","CNP","AES","PPL","ES","WEC",
        "SSE.L","PENNON.L","CENTRICA.L","NGAS.L",
        "ENGIE.PA","EDF.PA",
        "RWE.DE","EON.DE","INNOGY.DE",
        "ENEL.MI","A2A.MI",
        "IBE.MC","REE.MC",
        "9501.T","9502.T","9503.T",
        "015760.KS",
        "2638.HK","0006.HK",
        "FTS.TO","EMERA.TO","EMA.TO","H.TO","AQN.TO",
    ],

    # ── INDUSTRIALS ───────────────────────────────────────────────────────────
    "Aerospace & Defense": [
        "BA","RTX","LMT","NOC","GD","HEI","TDG","HII","LHX","AXON","KTOS","MRCY","HXL","HEICO",
        "BAE.L","ROLLS-ROYCE.L","COBHAM.L","MEGGITT.L","QQ.L",
        "AIR.PA","SAFR.PA","THALESGROUP.PA",
        "DIEHL.DE","HENSOLDT.DE","MTU.DE",
        "LEONARDO.MI",
        "7011.T","7013.T","6952.T",
        "047810.KS","047050.KS",
        "0000.HK",
        "CAE.TO","MDF.TO","CHC.TO",
    ],

    "Airlines": [
        "DAL","UAL","AAL","LUV","ALK","JBLU","SAVE","HA","SKYW","MESA",
        "RYAAY","IAG.L","EZJ.L","WIZZ.L",
        "AFR.PA","AIRFRANCE.PA",
        "LHA.DE",
        "WIZZ.L",
        "9201.T","9202.T",
        "003490.KS","020560.KS",
        "0293.HK","0670.HK","1055.HK",
        "AC.TO","CHR.TO",
    ],

    "Railroads": [
        "UNP","CSX","NSC","CP","CNI","WAB","GWR","BNSF",
        "DB.DE",
        "FIRSTGROUP.L","NATIONAL-EXPRESS.L",
        "9020.T","9022.T","9021.T",
        "117930.KS",
        "0525.HK",
        "CP.TO","CNR.TO",
    ],

    "Trucking": [
        "UPS","FDX","ODFL","SAIA","JBHT","XPO","CHRW","WERN","TFII","HUBG","KNX","USX","MRTN",
        "DPWL.DE","DBSCHENKER.DE",
        "DHL.DE","GEODIS.FR",
        "WINCANTON.L","DX.L","CLIPPER.L",
        "9064.T","9065.T","9003.T",
        "TFII.TO","TFX.TO","MTL.TO",
    ],

    "Specialty Industrial Machinery": [
        "HON","EMR","ROP","ITW","PH","IR","AME","DOV","GEV","IEX","GNRC","CFX","NDSN","TTC","MSA",
        "SIEMENS.DE","KNORR-BREMSE.DE","DUERR.DE","KOENIG.DE",
        "LEGRAND.PA","SCHNEIDER-ELEC.PA",
        "ABB.SW",
        "6301.T","6326.T","6645.T","7741.T",   # Komatsu, Kubota, Omron, Hoya
        "012450.KS","006400.KS","00287.KS",
        "0941.HK","1171.HK",
        "SNC.TO","ATS.TO","WSP.TO","STLC.TO",
    ],

    "Engineering & Construction": [
        "FLR","PWR","J","STRL","MTZ","MYR","PRIM","EME","WLDN","MYRG","TTEK","EXLS",
        "BALFOUR-BEATTY.L","KIER.L","MORGAN-SINDALL.L","COSTAIN.L",
        "VINCI.PA","EIFFAGE.PA","BOUYGUES.PA",
        "HOCHTIEF.DE","BILFINGER.DE",
        "SALINI.IT","WEBUILD.MI",
        "1800.HK","1963.HK","3311.HK",
        "WSP.TO","AECOM.TO","STN.TO",
    ],

    "Waste Management": [
        "WM","RSG","CWST","SRCL","GFL","USCL","NVRI","HPIL","AQUA","MERC",
        "BIFFA.L","RENEWI.L","CLEAN.L",
        "VEOLIA.PA","SUEZ.PA",
        "REMONDIS.DE",
        "GFL.TO","RBA.TO","CLEAN.TO",
    ],

    "Farm & Heavy Construction Machinery": [
        "CAT","DE","AGCO","CNH","TEX","OSK","MTW","TWIN","BCPC","CNHI",
        "VOLVO.ST","SANDVIK.ST","EPIROC.ST","ATLAS-COPCO.ST",
        "WABCO.DE","LIEBHERR.DE","CLAAS.DE",
        "6301.T","6326.T",
        "012450.KS","241590.KS",
        "0825.HK","2202.HK",
        "CAT.TO","WAJAX.TO",
    ],

    # ── BASIC MATERIALS ───────────────────────────────────────────────────────
    "Steel": [
        "NUE","STLD","X","RS","CLF","WOR","CMC","ZEUS","MTL","CRS","KALU","ATI",
        "SSAB-A.ST","OUTOKUMPU.HE","APERAM.AS",
        "ARCELORMITTAL.AS","MT",
        "THYSSENKRUPP.DE","SALZGITTER.DE",
        "5401.T","5411.T","5444.T",
        "POSCO","PKX",
        "0323.HK","0347.HK","0697.HK",
        "STLC.TO","PKX.TO",
    ],

    "Gold": [
        "NEM","GOLD","AEM","KGC","AGI","WPM","FNV","SAND","OR","K",
        "ABX.TO","ABX","AG","CG.TO","G.TO","KL.TO","OGC.TO","MAG.TO","EDV.TO","TGZ.TO",
        "NST.AX","OGC.AX","SAR.AX","NCM.AX","EVN.AX",
        "GFI","HAR","SGL.DE",
    ],

    "Chemicals": [
        "LIN","APD","DD","DOW","LYB","EMN","HUN","OLN","TROX","AVNT","CC","KRO","OLIN",
        "LINDE.DE","BASF.DE","BAYER.DE","COVESTRO.DE","LANXESS.DE",
        "AIR-LIQUIDE.PA","ARKEMA.PA","SOLVAY.BR",
        "CLARIANT.SW","LONZA.SW",
        "4004.T","4021.T","4042.T","4631.T",
        "051910.KS","011790.KS","009830.KS",
        "0303.HK","0004.HK",
        "METHANEX.TO","CHEMTRADE.TO","CF.TO",
    ],

    "Specialty Chemicals": [
        "SHW","PPG","ECL","IFF","FMC","ALB","LTHM","AVNT","OLIN","RPM","H.B. Fuller","GCP","KALU",
        "CRODA.L","ELEMENTIS.L","SYNTHOMER.L","QUAKER-HOUGHTON.L",
        "ARKEMA.PA","ROQUETTE.FR",
        "WACKER.DE","EVONIK.DE",
        "EMS-CHEMIE.SW",
        "7970.T","6988.T","4188.T",
        "051910.KS","096770.KS",
        "3983.HK","0215.HK",
        "METHANEX.TO","CCL-B.TO",
    ],

    "Aluminum": [
        "AA","CENX","KALU","NHYDY","NHYDY","CSTM",
        "NORSK-HYDRO.OL",
        "ALCAN.MC",
        "5706.T","5707.T",
        "RIO","VALE",
        "AAC.TO","AEM.TO",
    ],

    "Copper": [
        "FCX","SCCO","TECK","HBM","CS","NEXA","CATO",
        "ANTOFAGASTA.L","HOCHSCHILD.L","KAZ-MINERALS.L",
        "CODELCO",
        "TECK-B.TO","CS.TO","FM.TO","HBM.TO","ACO-X.TO",
        "0358.HK","1208.HK",
        "5713.T",
    ],

    # ── REAL ESTATE ───────────────────────────────────────────────────────────
    "REIT—Specialty": [
        "AMT","CCI","EQIX","SBAC","DLR","IRM","UNIT","CONE","QTS","COLD","IIPR",
        "CELL-A.SW","SWISS-PRIME.SW",
        "LAND.L","SEGRO.L","DERWENT.L",
        "UNIBAIL.AS","WDP.BR","MONTEA.BR",
        "CA-IMMOBILIEN.AT",
        "8951.T","8952.T","8984.T",
        "288.HK","405.HK","2778.HK",
        "AP.TO","SRU-U.TO","DIR-U.TO","CHP-U.TO",
    ],

    "REIT—Industrial": [
        "PLD","DRE","EGP","FR","REXR","LPT","TRNO","STAG","ILPT","COLD",
        "SEGRO.L","LON.L","TRITAX-BIG.L",
        "WDP.BR","MONTEA.BR",
        "PROLOGIS.SW",
        "8961.T","3281.T","8985.T",
        "0823.HK","0778.HK",
        "DIR-U.TO","WPT-U.TO","GRT-U.TO",
    ],

    "REIT—Retail": [
        "SPG","O","NNN","KIM","REG","BRX","MAC","CBL","PREIT","WPG",
        "UNIBAIL.AS","KLÉPIERRE.PA","MERCIALYS.PA",
        "HAMMERSON.L","INTU.L","CAPITAL-SHOPPING.L",
        "0823.HK","1972.HK",
        "RioCan.TO","REI-U.TO","SRU-U.TO","CRR-U.TO",
    ],

    "Real Estate Services": [
        "CBRE","JLL","OPEN","Z","RDFN","COMP","HOUS","EXPI","RLGY","OP",
        "RIGHTMOVE.L","ONTHEMARKET.L","ZOOPLA.L",
        "0267.HK","1109.HK","3333.HK",
        "BEIKE","KE",
        "BPY.TO","COLLIERS.TO","FSV.TO",
    ],
}

# ─── Sector Fallback ──────────────────────────────────────────────────────────
SECTOR_PEERS: Dict[str, List[str]] = {
    "Technology":            ["AAPL","MSFT","GOOGL","NVDA","META","AMD","INTC","ASML.AS","SAP.DE","TSM","6857.T","005930.KS"],
    "Healthcare":            ["JNJ","UNH","PFE","ABBV","MRK","AZN.L","NVO","ROG.SW","GSK.L","4503.T","068270.KS","1093.HK"],
    "Financial Services":    ["JPM","BAC","WFC","GS","MS","HSBC","BNP.PA","8306.T","105560.KS","0005.HK","TD.TO","RY.TO"],
    "Consumer Cyclical":     ["AMZN","TSLA","HD","MCD","NKE","VOW.DE","7203.T","005380.KS","0175.HK","SHOP.TO","SBUX","CMG"],
    "Consumer Defensive":    ["WMT","PG","KO","PEP","NESTL.SW","UNILEVER.L","DANONE.PA","2587.T","WN.TO","L.TO","MDLZ","GIS"],
    "Energy":                ["XOM","CVX","BP.L","SHEL.L","TTE.PA","EQNR.OL","5020.T","096770.KS","0883.HK","SU.TO","CNQ.TO","ENB.TO"],
    "Industrials":           ["BA","HON","RTX","SIEMENS.DE","ABB.SW","SCHNEIDER-ELEC.PA","7011.T","012450.KS","WSP.TO","VOLVO.ST"],
    "Communication Services":["GOOGL","META","DIS","NFLX","T","VZ","TMUS","4689.T","035420.KS","BIDU","TCEHY","700.HK"],
    "Basic Materials":       ["LIN","BASF.DE","LINDE.DE","RIO","BHP","NEM","5401.T","POSCO","0323.HK","TECK-B.TO","FCX"],
    "Real Estate":           ["PLD","AMT","EQIX","CCI","SEGRO.L","UNIBAIL.AS","8961.T","0823.HK","AP.TO","SRU-U.TO"],
    "Utilities":             ["NEE","DUK","SO","ENGIE.PA","RWE.DE","ENEL.MI","SSE.L","9501.T","015760.KS","0006.HK","FTS.TO"],
}


# ─── Peer selection — no network validation ──────────────────────────────────
#
# The old approach fetched yf.Ticker(t).info for every candidate to validate
# and rank by market cap — this caused 30+ API calls = 2–4 minutes of waiting.
#
# New approach: use ONLY the hardcoded curated lists (already high quality),
# skip all network calls for peer selection, limit to 6 peers maximum.
# The full metrics are fetched ONCE in fetch_peer_metrics().

def get_auto_peers(ticker: str, info: dict) -> List[str]:
    """
    Return up to 6 peer tickers from the curated lists — zero network calls.
    Matching priority: exact industry → partial industry → sector fallback.
    """
    industry = info.get("industry", "")
    sector   = info.get("sector", "")
    t_upper  = ticker.upper()

    candidates: List[str] = []

    # 1. Exact industry match
    for key, peers in INDUSTRY_PEERS.items():
        if key.lower() == industry.lower():
            candidates = [p for p in peers if p != t_upper]
            break

    # 2. Partial industry match (e.g. "Semiconductors" matches "Semiconductor Equipment")
    if not candidates:
        for key, peers in INDUSTRY_PEERS.items():
            key_words = set(key.lower().replace("—", " ").split())
            ind_words = set(industry.lower().replace("—", " ").split())
            if ind_words and ind_words & key_words:
                candidates = [p for p in peers if p != t_upper]
                break

    # 3. Sector fallback
    if not candidates:
        candidates = [p for p in SECTOR_PEERS.get(sector, []) if p != t_upper]

    # Return first 6 — enough for meaningful comparison, fast to fetch
    return candidates[:6]


def get_peers(ticker: str, info: dict, manual_override: str = "") -> List[str]:
    """
    Final peer list resolver.
    Manual override: comma-separated tickers, bypasses auto-detection.
    """
    if manual_override and manual_override.strip():
        tickers = [t.strip().upper() for t in manual_override.replace(",", " ").split() if t.strip()]
        return [t for t in tickers if t != ticker.upper()][:10]
    return get_auto_peers(ticker, info)
