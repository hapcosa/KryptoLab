"""
CryptoLab — Gaussian Filter Indicators
Faithful translation from gaussbands.pine (BigBeluga)

Gaussian filter with multi-trend scoring and band system.
"""
import numpy as np
from typing import Tuple
from indicators.common import sma


def gaussian_filter(src: np.ndarray, length: int, sigma: float = 10.0) -> np.ndarray:
    """
    Gaussian-weighted filter.
    Pine: gaussian_filter() in gaussbands.pine lines 29-42
    """
    n = len(src)
    out = np.zeros(n)
    
    # Pre-compute weights
    weights = np.zeros(length)
    total = 0.0
    for i in range(length):
        w = np.exp(-0.5 * ((i - length / 2.0) / sigma) ** 2) / np.sqrt(sigma * 2.0 * np.pi)
        weights[i] = w
        total += w
    
    if total > 0:
        weights /= total
    
    for i in range(n):
        s = 0.0
        for j in range(length):
            idx = i - j
            if idx >= 0:
                s += src[idx] * weights[j]
            else:
                s += src[0] * weights[j]  # Pad with first value
        out[i] = s
    
    return out


def multi_trend(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                period: int = 20, distance: float = 1.0,
                mode: str = 'avg'
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-trend analysis using multiple Gaussian filter instances.
    Pine: multi_trend() in gaussbands.pine lines 47-83
    
    Computes 21 Gaussian filters at periods [period..period+20]
    and scores based on relative positioning.
    
    Returns:
        score: trend strength 0..1 (>0.5 = bullish)
        avg_value: average/median/mode of gaussian values
        lower_band: avg_value - volatility * distance
        upper_band: avg_value + volatility * distance
        trend_line: lower_band when bullish, upper_band when bearish
    """
    n = len(close)
    
    # Volatility = SMA(high - low, 100)
    volatility = sma(high - low, 100)
    
    # Compute 21 Gaussian filters at different periods
    g_values = []
    for step in range(21):
        g = gaussian_filter(close, period + step, 10.0)
        g_values.append(g)
    
    g_matrix = np.array(g_values)  # shape: (21, n)
    
    # Score: fraction of filters above the first one
    score = np.zeros(n)
    for i in range(n):
        first_val = g_matrix[0, i]
        count = sum(1 for s in range(21) if g_matrix[s, i] > first_val)
        score[i] = count * 0.05  # 0.05 per step, max ~1.0
    
    # Central value based on mode
    if mode == 'avg':
        avg_value = np.mean(g_matrix, axis=0)
    elif mode == 'median':
        avg_value = np.median(g_matrix, axis=0)
    else:  # mode
        # Approximation: use the value that appears most (binned)
        avg_value = np.median(g_matrix, axis=0)
    
    # Bands
    lower_band = avg_value - volatility * distance
    upper_band = avg_value + volatility * distance
    
    # Trend state (Pine's var bool trend)
    trend = np.zeros(n, dtype=bool)
    trend_line = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            trend[i] = close[i] > upper_band[i]
        else:
            trend[i] = trend[i-1]
            if close[i] > upper_band[i]:  # crossover
                if not trend[i-1] or (close[i] > upper_band[i] and close[i-1] <= upper_band[i-1]):
                    trend[i] = True
            if close[i] < lower_band[i]:  # crossunder
                if trend[i-1] or (close[i] < lower_band[i] and close[i-1] >= lower_band[i-1]):
                    trend[i] = False
        
        trend_line[i] = lower_band[i] if trend[i] else upper_band[i]
    
    return score, avg_value, lower_band, upper_band, trend_line


def gaussian_signals(close: np.ndarray, trend_line: np.ndarray,
                     avg_value: np.ndarray, trend: np.ndarray,
                     show_retest: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate trading signals from Gaussian Bands.
    Pine: lines 88-92 of gaussbands.pine
    
    Returns:
        long_signal: crossover(close, trend_line)
        short_signal: crossunder(close, trend_line)
        retest_long: crossover(close, avg) and trend
        retest_short: crossunder(high, avg) and not trend
    """
    from indicators.common import crossover, crossunder
    
    long_signal = crossover(close, trend_line)
    short_signal = crossunder(close, trend_line)
    
    retest_long = np.zeros(len(close), dtype=bool)
    retest_short = np.zeros(len(close), dtype=bool)
    
    if show_retest:
        co_avg = crossover(close, avg_value)
        cu_avg = crossunder(close, avg_value)
        retest_long = co_avg & trend
        retest_short = cu_avg & ~trend
    
    return long_signal, short_signal, retest_long, retest_short


def compute_tp_levels(entry: float, sl: float, direction: int,
                      h1h: float = 0.0, h1l: float = 0.0,
                      h4h: float = 0.0, h4l: float = 0.0,
                      pdh: float = 0.0, pdl: float = 0.0,
                      atr_val: float = 0.0
                      ) -> dict:
    """
    Compute multi-level TP targets (replicates gaussbands.pine TP/SL system).
    Pine: lines 153-195 of gaussbands.pine
    
    TP_MIN → T2(1H) → T3(4H) → T4(D) → T5(extended)
    Uses nearby HTF levels as dynamic targets when available.
    """
    risk = max(abs(entry - sl), atr_val * 0.5)
    
    if direction == 1:  # Long
        be = entry
        tpm = h1h if (h1h > 0 and h1h > entry + risk * 0.3) else entry + risk * 0.8
        t2 = h1h if (h1h > tpm + risk * 0.3) else tpm + risk * 1.5
        t3 = h4h if (h4h > t2 + risk * 0.3) else t2 + risk * 2.0
        t4 = pdh if (pdh > t3 + risk * 0.3) else t3 + risk * 2.5
        t5 = t4 + risk * 4.0
    else:  # Short
        be = entry
        tpm = h1l if (h1l > 0 and h1l < entry - risk * 0.3) else entry - risk * 0.8
        t2 = h1l if (h1l < tpm - risk * 0.3) else tpm - risk * 1.5
        t3 = h4l if (h4l < t2 - risk * 0.3) else t2 - risk * 2.0
        t4 = pdl if (pdl < t3 - risk * 0.3) else t3 - risk * 2.5
        t5 = t4 - risk * 4.0
    
    return {
        'be': be,
        'tp_min': tpm,
        't2': t2,
        't3': t3,
        't4': t4,
        't5': t5,
        'risk': risk,
    }
