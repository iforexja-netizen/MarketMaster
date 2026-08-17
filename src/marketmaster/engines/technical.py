"""
Technical Indicators Engine — Pure compute functions for technical factors.

All functions are pure: they receive data (numpy arrays or pandas Series),
produce a result. No DB calls, no side effects.

Point-in-time: when computing indicators for date D, only data up to and
including D is used. The caller is responsible for slicing the input data.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TechnicalResult:
    """Result of a single technical indicator computation."""
    name: str
    value: Optional[float]
    raw_inputs: dict[str, float] = field(default_factory=dict)
    signal: Optional[str] = None  # 'bullish', 'bearish', 'neutral'


# ============================================================================
# Moving Averages
# ============================================================================

def sma(prices: pd.Series, period: int) -> Optional[float]:
    """Simple Moving Average."""
    if len(prices) < period:
        return None
    return float(prices.iloc[-period:].mean())


def ema(prices: pd.Series, period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])


def ema_series(prices: pd.Series, period: int) -> pd.Series:
    """Full EMA series (for MACD computation)."""
    return prices.ewm(span=period, adjust=False).mean()


# ============================================================================
# RSI (Relative Strength Index)
# ============================================================================

def rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss

    Uses Wilder's smoothing method.
    """
    if len(prices) < period + 1:
        return None

    deltas = prices.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = deltas.clip(upper=0).abs()

    # Wilder's smoothing
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi_val)


# ============================================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================================

def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Optional[tuple[float, float, float]]:
    """
    MACD line, signal line, and histogram.

    Returns (macd_line, signal_line, histogram) or None if insufficient data.
    """
    if len(prices) < slow + signal:
        return None

    ema_fast = ema_series(prices, fast)
    ema_slow = ema_series(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(histogram.iloc[-1]),
    )


# ============================================================================
# ADX (Average Directional Index) — Trend Strength
# ============================================================================

def adx(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> Optional[float]:
    """
    Average Directional Index.

    ADX > 25: strong trend
    ADX < 20: weak/no trend

    Uses Wilder's smoothing.
    """
    if len(closes) < period * 2:
        return None

    # True Range
    tr = pd.Series(index=closes.index, dtype=float)
    for i in range(1, len(closes)):
        tr.iloc[i] = max(
            highs.iloc[i] - lows.iloc[i],
            abs(highs.iloc[i] - closes.iloc[i - 1]),
            abs(lows.iloc[i] - closes.iloc[i - 1]),
        )

    # Directional Movement
    plus_dm = pd.Series(index=closes.index, dtype=float)
    minus_dm = pd.Series(index=closes.index, dtype=float)

    for i in range(1, len(closes)):
        up_move = highs.iloc[i] - highs.iloc[i - 1]
        down_move = lows.iloc[i - 1] - lows.iloc[i]

        plus_dm.iloc[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm.iloc[i] = down_move if (down_move > up_move and down_move > 0) else 0

    # Wilder's smoothing
    atr_val = tr.iloc[1:period + 1].mean()
    plus_di_arr = [plus_dm.iloc[1:period + 1].sum()]
    minus_di_arr = [minus_dm.iloc[1:period + 1].sum()]

    for i in range(period + 1, len(closes)):
        atr_val = (atr_val * (period - 1) + tr.iloc[i]) / period
        plus_di_arr.append(
            plus_di_arr[-1] * (period - 1) / period + plus_dm.iloc[i]
        )
        minus_di_arr.append(
            minus_di_arr[-1] * (period - 1) / period + minus_dm.iloc[i]
        )

    # DX
    dx_values = []
    for i in range(len(plus_di_arr)):
        if atr_val == 0:
            dx_values.append(0)
            continue
        plus_di = 100 * plus_di_arr[i] / atr_val if atr_val else 0
        minus_di = 100 * minus_di_arr[i] / atr_val if atr_val else 0
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    # ADX = Wilder's smoothed average of DX
    adx_val = np.mean(dx_values[:period])
    for i in range(period, len(dx_values)):
        adx_val = (adx_val * (period - 1) + dx_values[i]) / period

    return float(adx_val)


# ============================================================================
# ATR (Average True Range) — Volatility
# ============================================================================

def atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> Optional[float]:
    """
    Average True Range using Wilder's smoothing.

    ATR measures volatility, not direction.
    """
    if len(closes) < period + 1:
        return None

    tr_values = []
    for i in range(1, len(closes)):
        tr = max(
            highs.iloc[i] - lows.iloc[i],
            abs(highs.iloc[i] - closes.iloc[i - 1]),
            abs(lows.iloc[i] - closes.iloc[i - 1]),
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return None

    # Wilder's smoothing
    atr_val = np.mean(tr_values[:period])
    for i in range(period, len(tr_values)):
        atr_val = (atr_val * (period - 1) + tr_values[i]) / period

    return float(atr_val)


# ============================================================================
# Bollinger Bands
# ============================================================================

def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[tuple[float, float, float, float]]:
    """
    Bollinger Bands.

    Returns (upper, middle, lower, bandwidth) or None.
    bandwidth = (upper - lower) / middle
    """
    if len(prices) < period:
        return None

    slice = prices.iloc[-period:]
    middle = float(slice.mean())
    std = float(slice.std(ddof=1))

    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle if middle > 0 else 0.0

    return (upper, middle, lower, bandwidth)


# ============================================================================
# Relative Strength (vs benchmark)
# ============================================================================

def relative_strength(
    prices: pd.Series,
    benchmark: pd.Series,
    period: int = 60,
) -> Optional[float]:
    """
    Relative strength of a security vs a benchmark over a period.

    RS = (security_return / benchmark_return)
    RS > 1: outperforming
    RS < 1: underperforming
    """
    if len(prices) < period or len(benchmark) < period:
        return None

    sec_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
    bench_return = (benchmark.iloc[-1] / benchmark.iloc[-period]) - 1

    if abs(bench_return) < 1e-10:
        return None

    return float(sec_return / bench_return)


# ============================================================================
# Volume Indicators
# ============================================================================

def volume_sma(volumes: pd.Series, period: int = 20) -> Optional[float]:
    """Volume Simple Moving Average."""
    if len(volumes) < period:
        return None
    return float(volumes.iloc[-period:].mean())


def volume_ratio(volumes: pd.Series, period: int = 20) -> Optional[float]:
    """
    Current volume relative to its moving average.

    ratio > 1.5: high volume
    ratio < 0.5: low volume
    """
    if len(volumes) < period + 1:
        return None
    vol_sma = float(volumes.iloc[-period - 1:-1].mean())
    current_vol = float(volumes.iloc[-1])
    if vol_sma == 0:
        return None
    return current_vol / vol_sma


# ============================================================================
# Momentum / Rate of Change
# ============================================================================

def momentum(prices: pd.Series, period: int = 10) -> Optional[float]:
    """Rate of change: (current / past) - 1"""
    if len(prices) < period + 1:
        return None
    past = float(prices.iloc[-period - 1])
    current = float(prices.iloc[-1])
    if past == 0:
        return None
    return (current / past) - 1.0


def roc(prices: pd.Series, period: int = 12) -> Optional[float]:
    """Rate of Change percentage: ((current / past) - 1) * 100"""
    mom = momentum(prices, period)
    if mom is None:
        return None
    return mom * 100.0


# ============================================================================
# Stochastic Oscillator
# ============================================================================

def stochastic(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Optional[tuple[float, float]]:
    """
    Stochastic Oscillator.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA(%K, d_period)

    Returns (%K, %D) or None.
    """
    if len(closes) < k_period + d_period:
        return None

    k_values = []
    for i in range(k_period - 1, len(closes)):
        window_high = highs.iloc[i - k_period + 1:i + 1].max()
        window_low = lows.iloc[i - k_period + 1:i + 1].min()
        close = closes.iloc[i]

        if window_high == window_low:
            k_values.append(50.0)
        else:
            k_values.append(100.0 * (close - window_low) / (window_high - window_low))

    if len(k_values) < d_period:
        return None

    k = k_values[-1]
    d = float(np.mean(k_values[-d_period:]))

    return (float(k), d)


# ============================================================================
# Commodity Channel Index (CCI)
# ============================================================================

def cci(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 20,
) -> Optional[float]:
    """
    Commodity Channel Index.

    CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
    TP = (High + Low + Close) / 3
    """
    if len(closes) < period:
        return None

    tp = (highs + lows + closes) / 3.0
    tp_slice = tp.iloc[-period:]

    sma_tp = float(tp_slice.mean())
    mean_dev = float(np.abs(tp_slice - sma_tp).mean())

    if mean_dev == 0:
        return 0.0

    current_tp = float(tp.iloc[-1])
    return (current_tp - sma_tp) / (0.015 * mean_dev)


# ============================================================================
# Williams %R
# ============================================================================

def williams_r(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> Optional[float]:
    """
    Williams %R.

    %R = -100 * (highest_high - close) / (highest_high - lowest_low)

    Range: -100 to 0
    %R > -20: overbought
    %R < -80: oversold
    """
    if len(closes) < period:
        return None

    window_high = float(highs.iloc[-period:].max())
    window_low = float(lows.iloc[-period:].min())
    close = float(closes.iloc[-1])

    if window_high == window_low:
        return -50.0

    return -100.0 * (window_high - close) / (window_high - window_low)


# ============================================================================
# On-Balance Volume (OBV)
# ============================================================================

def obv(closes: pd.Series, volumes: pd.Series) -> Optional[float]:
    """
    On-Balance Volume — cumulative volume that adds volume on up days,
    subtracts on down days.

    Returns the current OBV level.
    """
    if len(closes) < 2 or len(volumes) < 2:
        return None

    obv_val = 0.0
    for i in range(1, len(closes)):
        if closes.iloc[i] > closes.iloc[i - 1]:
            obv_val += float(volumes.iloc[i])
        elif closes.iloc[i] < closes.iloc[i - 1]:
            obv_val -= float(volumes.iloc[i])

    return obv_val


# ============================================================================
# Comprehensive Technical Analysis
# ============================================================================

def compute_all_technical(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    volumes: pd.Series,
    benchmark_closes: Optional[pd.Series] = None,
) -> dict[str, TechnicalResult]:
    """
    Compute all technical indicators at once.

    Returns a dict of indicator_name -> TechnicalResult.
    """
    results: dict[str, TechnicalResult] = {}

    # Moving Averages
    results["sma_20"] = TechnicalResult("sma_20", sma(closes, 20), {"period": 20})
    results["sma_50"] = TechnicalResult("sma_50", sma(closes, 50), {"period": 50})
    results["sma_200"] = TechnicalResult("sma_200", sma(closes, 200), {"period": 200})
    results["ema_12"] = TechnicalResult("ema_12", ema(closes, 12), {"period": 12})
    results["ema_26"] = TechnicalResult("ema_26", ema(closes, 26), {"period": 26})

    # Trend / Momentum
    rsi_val = rsi(closes, 14)
    results["rsi_14"] = TechnicalResult(
        "rsi_14", rsi_val,
        {"period": 14},
        "oversold" if rsi_val and rsi_val < 30 else ("overbought" if rsi_val and rsi_val > 70 else "neutral"),
    )

    adx_val = adx(highs, lows, closes, 14)
    results["adx_14"] = TechnicalResult(
        "adx_14", adx_val,
        {"period": 14},
        "strong_trend" if adx_val and adx_val > 25 else ("weak_trend" if adx_val and adx_val < 20 else "neutral"),
    )

    macd_result = macd(closes)
    if macd_result:
        macd_line, signal_line, histogram = macd_result
        results["macd"] = TechnicalResult("macd", macd_line, {"fast": 12, "slow": 26})
        results["macd_signal"] = TechnicalResult("macd_signal", signal_line, {"period": 9})
        results["macd_histogram"] = TechnicalResult(
            "macd_histogram", histogram,
            {"macd": macd_line, "signal": signal_line},
            "bullish" if histogram > 0 else "bearish",
        )
    else:
        results["macd"] = TechnicalResult("macd", None, {"fast": 12, "slow": 26})
        results["macd_signal"] = TechnicalResult("macd_signal", None, {"period": 9})
        results["macd_histogram"] = TechnicalResult("macd_histogram", None, {})

    results["momentum_10"] = TechnicalResult("momentum_10", momentum(closes, 10), {"period": 10})
    results["roc_12"] = TechnicalResult("roc_12", roc(closes, 12), {"period": 12})

    # Volatility
    atr_val = atr(highs, lows, closes, 14)
    close_val = float(closes.iloc[-1]) if len(closes) > 0 else None
    atr_pct = (atr_val / close_val * 100) if (atr_val and close_val and close_val > 0) else None
    results["atr_14"] = TechnicalResult("atr_14", atr_val, {"period": 14, "atr_pct": atr_pct})

    bb_result = bollinger_bands(closes, 20, 2.0)
    if bb_result:
        upper, middle, lower, bandwidth = bb_result
        results["bollinger_upper"] = TechnicalResult("bollinger_upper", upper, {"period": 20, "std_dev": 2.0})
        results["bollinger_middle"] = TechnicalResult("bollinger_middle", middle, {"period": 20})
        results["bollinger_lower"] = TechnicalResult("bollinger_lower", lower, {"period": 20, "std_dev": 2.0})
        results["bollinger_width"] = TechnicalResult("bollinger_width", bandwidth, {"period": 20})
        # Position within bands: 0 = lower, 1 = upper
        bb_position = (close_val - lower) / (upper - lower) if (close_val and upper != lower) else None
        results["bollinger_position"] = TechnicalResult(
            "bollinger_position", bb_position,
            {"upper": upper, "lower": lower},
        )
    else:
        for name in ["bollinger_upper", "bollinger_middle", "bollinger_lower", "bollinger_width", "bollinger_position"]:
            results[name] = TechnicalResult(name, None, {})

    # Volume
    results["volume_sma_20"] = TechnicalResult("volume_sma_20", volume_sma(volumes, 20), {"period": 20})
    results["volume_ratio"] = TechnicalResult(
        "volume_ratio", volume_ratio(volumes, 20),
        {"period": 20},
        "high_volume" if volume_ratio(volumes, 20) and volume_ratio(volumes, 20) > 1.5 else "normal",
    )

    # Oscillators
    stoch_result = stochastic(highs, lows, closes)
    if stoch_result:
        k, d = stoch_result
        results["stoch_k"] = TechnicalResult("stoch_k", k, {"period": 14})
        results["stoch_d"] = TechnicalResult("stoch_d", d, {"period": 3})
    else:
        results["stoch_k"] = TechnicalResult("stoch_k", None, {"period": 14})
        results["stoch_d"] = TechnicalResult("stoch_d", None, {"period": 3})

    results["cci_20"] = TechnicalResult("cci_20", cci(highs, lows, closes, 20), {"period": 20})
    results["williams_r_14"] = TechnicalResult("williams_r_14", williams_r(highs, lows, closes, 14), {"period": 14})
    results["obv"] = TechnicalResult("obv", obv(closes, volumes), {})

    # Relative Strength
    if benchmark_closes is not None:
        rs_val = relative_strength(closes, benchmark_closes, 60)
        results["relative_strength_60"] = TechnicalResult(
            "relative_strength_60", rs_val,
            {"period": 60},
            "outperforming" if rs_val and rs_val > 1.0 else ("underperforming" if rs_val and rs_val < 1.0 else "neutral"),
        )
    else:
        results["relative_strength_60"] = TechnicalResult("relative_strength_60", None, {"period": 60})

    return results
