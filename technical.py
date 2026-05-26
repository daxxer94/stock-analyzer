"""
technical.py - All technical indicators + signal generation.

Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, OBV, Support/Resistance.
Each indicator produces a structured signal with emoji and explanation.
"""
import pandas as pd
import numpy as np

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


# ─── Indicator Calculation ───────────────────────────────────────────────────

def calculate_indicators(hist: pd.DataFrame) -> dict:
    """Calculate all technical indicators. Returns {} on insufficient data."""
    if hist is None or hist.empty or len(hist) < 30:
        return {}

    close  = hist["Close"].astype(float)
    high   = hist["High"].astype(float)
    low    = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)

    ind = {}

    if TA_AVAILABLE:
        # Moving Averages
        ind["sma_20"]  = ta.trend.sma_indicator(close, window=20)
        ind["sma_50"]  = ta.trend.sma_indicator(close, window=50)
        ind["sma_200"] = ta.trend.sma_indicator(close, window=200) if len(close) >= 200 else pd.Series(dtype=float)
        ind["ema_12"]  = ta.trend.ema_indicator(close, window=12)
        ind["ema_26"]  = ta.trend.ema_indicator(close, window=26)
        ind["ema_50"]  = ta.trend.ema_indicator(close, window=50)

        # RSI
        ind["rsi"] = ta.momentum.rsi(close, window=14)

        # MACD
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        ind["macd"]      = macd.macd()
        ind["macd_sig"]  = macd.macd_signal()
        ind["macd_hist"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        ind["bb_upper"]  = bb.bollinger_hband()
        ind["bb_mid"]    = bb.bollinger_mavg()
        ind["bb_lower"]  = bb.bollinger_lband()
        ind["bb_width"]  = bb.bollinger_wband()
        ind["bb_pct"]    = bb.bollinger_pband()   # 0=lower, 1=upper

        # Volume / OBV
        ind["obv"]        = ta.volume.on_balance_volume(close, volume)
        ind["vol_sma_20"] = ta.trend.sma_indicator(volume, window=20)

        # ATR
        ind["atr"] = ta.volatility.average_true_range(high, low, close, window=14)

    else:
        # Pure-pandas fallback (no ta library)
        ind["sma_20"]  = close.rolling(20).mean()
        ind["sma_50"]  = close.rolling(50).mean()
        ind["sma_200"] = close.rolling(200).mean()
        ind["ema_12"]  = close.ewm(span=12, adjust=False).mean()
        ind["ema_26"]  = close.ewm(span=26, adjust=False).mean()
        ind["ema_50"]  = close.ewm(span=50, adjust=False).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        ind["rsi"] = 100 - 100 / (1 + rs)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        ind["macd"]     = ema12 - ema26
        ind["macd_sig"] = ind["macd"].ewm(span=9, adjust=False).mean()
        ind["macd_hist"]= ind["macd"] - ind["macd_sig"]
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        ind["bb_upper"] = sma20 + 2 * std20
        ind["bb_mid"]   = sma20
        ind["bb_lower"] = sma20 - 2 * std20
        ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / sma20
        ind["bb_pct"]   = (close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"])
        obv = [0]
        c = close.values
        v = volume.values
        for i in range(1, len(c)):
            if c[i] > c[i-1]:
                obv.append(obv[-1] + v[i])
            elif c[i] < c[i-1]:
                obv.append(obv[-1] - v[i])
            else:
                obv.append(obv[-1])
        ind["obv"] = pd.Series(obv, index=close.index)
        ind["vol_sma_20"] = volume.rolling(20).mean()

    # Support & Resistance via local pivot points
    sr = _find_sr(hist, window=7)
    ind.update(sr)

    return ind


def _find_sr(hist: pd.DataFrame, window: int = 7) -> dict:
    """Identify key support and resistance levels using pivot analysis."""
    highs  = hist["High"].values
    lows   = hist["Low"].values
    closes = hist["Close"].values

    resist, support = [], []

    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i - j] and highs[i] >= highs[i + j] for j in range(1, window + 1)):
            resist.append(highs[i])
        if all(lows[i] <= lows[i - j] and lows[i] <= lows[i + j] for j in range(1, window + 1)):
            support.append(lows[i])

    def cluster(levels, tol=0.015):
        if not levels:
            return []
        s = sorted(set(levels))
        out = [s[0]]
        for lv in s[1:]:
            if (lv - out[-1]) / out[-1] > tol:
                out.append(lv)
            else:
                out[-1] = (out[-1] + lv) / 2
        return out

    current = float(closes[-1])
    r_all = cluster(resist)
    s_all = cluster(support)
    r_above = sorted([r for r in r_all if r > current])
    s_below = sorted([s for s in s_all if s < current], reverse=True)

    return {
        "sr_resistance": r_above[:3],
        "sr_support":    s_below[:3],
        "sr_r_all":      r_all,
        "sr_s_all":      s_all,
    }


def _last(series) -> float | None:
    """Get the last non-NaN value from a Series or plain number."""
    if isinstance(series, pd.Series):
        clean = series.dropna()
        return float(clean.iloc[-1]) if not clean.empty else None
    if isinstance(series, (int, float)) and not np.isnan(float(series)):
        return float(series)
    return None


# ─── Signal Generation ───────────────────────────────────────────────────────

def generate_signals(ind: dict, current_price: float) -> dict:
    """
    For each indicator group, return:
      { 'score': int, 'signals': [(emoji, text), ...], 'max': int }
    Plus 'overall_score' normalised 0–10.
    """
    if not ind:
        return {"overall_score": 5.0}

    sigs = {}

    # ── 1. Moving Averages ──────────────────────────────────────────────────
    sma20  = _last(ind.get("sma_20"))
    sma50  = _last(ind.get("sma_50"))
    sma200 = _last(ind.get("sma_200"))
    ema12  = _last(ind.get("ema_12"))
    ema26  = _last(ind.get("ema_26"))

    score, msgs = 0, []

    if sma200 and current_price:
        if current_price > sma200:
            score += 2; msgs.append(("✅", f"Price ${current_price:.2f} > 200-SMA ${sma200:.2f} — long-term uptrend"))
        else:
            score -= 2; msgs.append(("🔴", f"Price ${current_price:.2f} < 200-SMA ${sma200:.2f} — long-term downtrend"))
    if sma50 and current_price:
        if current_price > sma50:
            score += 1; msgs.append(("✅", f"Price > 50-SMA ${sma50:.2f} — medium-term bullish"))
        else:
            score -= 1; msgs.append(("🔴", f"Price < 50-SMA ${sma50:.2f} — medium-term bearish"))
    if sma20 and current_price:
        if current_price > sma20:
            score += 1; msgs.append(("✅", f"Price > 20-SMA ${sma20:.2f} — short-term bullish"))
        else:
            score -= 1; msgs.append(("🔴", f"Price < 20-SMA ${sma20:.2f} — short-term bearish"))
    if sma50 and sma200:
        if sma50 > sma200:
            score += 2; msgs.append(("✅", "Golden Cross: 50-SMA > 200-SMA"))
        else:
            score -= 2; msgs.append(("🔴", "Death Cross: 50-SMA < 200-SMA"))
    if ema12 and ema26:
        if ema12 > ema26:
            score += 1; msgs.append(("✅", f"EMA 12 (${ema12:.2f}) > EMA 26 (${ema26:.2f}) — bullish EMA crossover"))
        else:
            score -= 1; msgs.append(("🔴", f"EMA 12 (${ema12:.2f}) < EMA 26 (${ema26:.2f}) — bearish EMA crossover"))

    sigs["sma_ema"] = {"score": score, "max": 7, "signals": msgs, "label": "Moving Averages"}

    # ── 2. RSI ──────────────────────────────────────────────────────────────
    rsi = _last(ind.get("rsi"))
    score, msgs = 0, []
    if rsi is not None:
        msgs.append(("ℹ️", f"RSI(14) = {rsi:.1f}"))
        if rsi >= 80:
            score -= 3; msgs.append(("🔴", f"RSI {rsi:.1f} extremely overbought (≥80) — high pullback risk"))
        elif rsi >= 70:
            score -= 2; msgs.append(("🔴", f"RSI {rsi:.1f} overbought (70–80) — potential pullback"))
        elif rsi >= 60:
            score += 1; msgs.append(("✅", f"RSI {rsi:.1f} bullish momentum (60–70)"))
        elif rsi >= 50:
            score += 1; msgs.append(("✅", f"RSI {rsi:.1f} slightly bullish (50–60)"))
        elif rsi >= 40:
            score -= 1; msgs.append(("🟡", f"RSI {rsi:.1f} slightly bearish (40–50)"))
        elif rsi >= 30:
            score -= 1; msgs.append(("🔴", f"RSI {rsi:.1f} bearish momentum (30–40)"))
        elif rsi >= 20:
            score += 2; msgs.append(("🟡", f"RSI {rsi:.1f} oversold (20–30) — potential bounce"))
        else:
            score += 3; msgs.append(("🟡", f"RSI {rsi:.1f} extremely oversold (<20) — strong bounce candidate"))
    sigs["rsi"] = {"score": score, "max": 3, "signals": msgs, "label": "RSI (14)", "value": rsi}

    # ── 3. MACD ─────────────────────────────────────────────────────────────
    macd      = _last(ind.get("macd"))
    macd_sig  = _last(ind.get("macd_sig"))
    macd_hist = _last(ind.get("macd_hist"))
    score, msgs = 0, []

    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            score += 2; msgs.append(("✅", f"MACD ({macd:.3f}) above signal ({macd_sig:.3f}) — bullish"))
        else:
            score -= 2; msgs.append(("🔴", f"MACD ({macd:.3f}) below signal ({macd_sig:.3f}) — bearish"))
        if macd > 0:
            score += 1; msgs.append(("✅", "MACD above zero — positive territory"))
        else:
            score -= 1; msgs.append(("🔴", "MACD below zero — negative territory"))

    hist_series = ind.get("macd_hist")
    if isinstance(hist_series, pd.Series) and len(hist_series.dropna()) >= 3:
        clean = hist_series.dropna()
        if clean.iloc[-1] > clean.iloc[-2]:
            score += 1; msgs.append(("✅", "MACD histogram increasing — momentum building"))
        else:
            score -= 1; msgs.append(("🔴", "MACD histogram decreasing — momentum fading"))

    sigs["macd"] = {"score": score, "max": 4, "signals": msgs, "label": "MACD"}

    # ── 4. Bollinger Bands ──────────────────────────────────────────────────
    bb_pct   = _last(ind.get("bb_pct"))
    bb_upper = _last(ind.get("bb_upper"))
    bb_lower = _last(ind.get("bb_lower"))
    bb_mid   = _last(ind.get("bb_mid"))
    bb_width = _last(ind.get("bb_width"))
    score, msgs = 0, []

    if bb_upper and bb_lower:
        msgs.append(("ℹ️", f"Bands: ${bb_lower:.2f} — ${bb_mid:.2f} — ${bb_upper:.2f}"))
    if bb_pct is not None:
        if bb_pct > 1.0:
            score -= 2; msgs.append(("🔴", f"%B={bb_pct:.2f}: Price above upper band — strongly overbought"))
        elif bb_pct > 0.8:
            score -= 1; msgs.append(("🔴", f"%B={bb_pct:.2f}: Price near upper band — overbought caution"))
        elif bb_pct < 0:
            score += 2; msgs.append(("🟡", f"%B={bb_pct:.2f}: Price below lower band — strongly oversold, potential bounce"))
        elif bb_pct < 0.2:
            score += 1; msgs.append(("🟡", f"%B={bb_pct:.2f}: Price near lower band — oversold zone"))
        else:
            msgs.append(("🟡", f"%B={bb_pct:.2f}: Price within bands — neutral"))

    width_series = ind.get("bb_width")
    if isinstance(width_series, pd.Series) and bb_width is not None:
        avg_w = width_series.dropna().tail(60).mean()
        if bb_width < avg_w * 0.7:
            msgs.append(("🟡", "BB Squeeze detected — low volatility, potential breakout ahead"))
        elif bb_width > avg_w * 1.4:
            msgs.append(("🟡", "Wide bands — high volatility environment"))

    sigs["bollinger"] = {"score": score, "max": 2, "signals": msgs, "label": "Bollinger Bands"}

    # ── 5. OBV / Volume ─────────────────────────────────────────────────────
    obv_series = ind.get("obv")
    score, msgs = 0, []
    if isinstance(obv_series, pd.Series) and len(obv_series.dropna()) >= 20:
        clean = obv_series.dropna()
        obv_recent = clean.tail(10).mean()
        obv_past   = clean.tail(30).head(10).mean()
        price_series = ind.get("sma_20")
        price_trend_up = None
        if isinstance(price_series, pd.Series) and len(price_series.dropna()) >= 11:
            pc = price_series.dropna()
            price_trend_up = pc.iloc[-1] > pc.iloc[-11]

        obv_up = obv_recent > obv_past
        if obv_up:
            score += 2; msgs.append(("✅", "OBV rising — accumulation / buying pressure confirmed"))
        else:
            score -= 2; msgs.append(("🔴", "OBV falling — distribution / selling pressure detected"))

        if price_trend_up is not None:
            if price_trend_up and not obv_up:
                score -= 2; msgs.append(("🔴", "Bearish divergence: price rising but OBV falling — reversal risk"))
            elif not price_trend_up and obv_up:
                score += 1; msgs.append(("✅", "Bullish divergence: price falling but OBV rising — potential reversal"))

    sigs["obv"] = {"score": score, "max": 4, "signals": msgs, "label": "OBV / Volume"}

    # ── 6. Support & Resistance ─────────────────────────────────────────────
    resistances = ind.get("sr_resistance", [])
    supports    = ind.get("sr_support", [])
    score, msgs = 0, []

    if resistances:
        msgs.append(("ℹ️", "Resistance levels: " + ", ".join(f"${r:.2f}" for r in resistances)))
    if supports:
        msgs.append(("ℹ️", "Support levels: " + ", ".join(f"${s:.2f}" for s in supports)))

    if current_price and resistances and supports:
        nearest_r = resistances[0]
        nearest_s = supports[0] if supports else None
        pct_to_r = (nearest_r - current_price) / current_price

        if current_price > nearest_r:
            score += 2; msgs.append(("✅", f"Breakout above resistance ${nearest_r:.2f} — bullish momentum"))
        elif pct_to_r < 0.02:
            score -= 1; msgs.append(("🟡", f"Price ${current_price:.2f} very close to resistance ${nearest_r:.2f} — watch for rejection"))
        elif pct_to_r < 0.05:
            msgs.append(("🟡", f"Approaching resistance ${nearest_r:.2f} (+{pct_to_r:.1%})"))

        if nearest_s:
            pct_to_s = (current_price - nearest_s) / current_price
            if current_price < nearest_s:
                score -= 2; msgs.append(("🔴", f"Breakdown below support ${nearest_s:.2f} — bearish signal"))
            elif pct_to_s < 0.02:
                score += 1; msgs.append(("✅", f"Price holding near support ${nearest_s:.2f} — potential bounce"))

    sigs["support_resistance"] = {"score": score, "max": 3, "signals": msgs, "label": "Support & Resistance"}

    # ── Overall Technical Score ─────────────────────────────────────────────
    raw = sum(sigs[k]["score"] for k in sigs)
    max_possible = sum(sigs[k]["max"] for k in sigs)   # positive max
    min_possible = -max_possible

    normalized = (raw - min_possible) / (max_possible - min_possible) * 10 if max_possible > 0 else 5.0
    sigs["overall_score"] = max(0.0, min(10.0, normalized))
    sigs["raw_score"]     = raw

    return sigs
