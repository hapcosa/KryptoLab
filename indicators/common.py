"""
CryptoLab — Common Indicators
Standard TA functions vectorized with NumPy.
"""
import numpy as np
from typing import Tuple


def sma(src: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    n = len(src)
    out = np.full(n, np.nan)
    if period > n:
        return out
    cumsum = np.cumsum(src)
    out[period-1:] = (cumsum[period-1:] - np.concatenate([[0], cumsum[:-period]])) / period
    # Fill initial NaNs with expanding mean
    for i in range(period - 1):
        out[i] = np.mean(src[:i+1])
    return out


def ema(src: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (matches Pine's ta.ema)."""
    n = len(src)
    out = np.zeros(n)
    k = 2.0 / (period + 1.0)
    out[0] = src[0]
    for i in range(1, n):
        out[i] = k * src[i] + (1.0 - k) * out[i-1]
    return out


def ema_alpha(src: np.ndarray, alpha: float) -> np.ndarray:
    """EMA with direct alpha parameter (not period-based)."""
    n = len(src)
    out = np.zeros(n)
    out[0] = src[0]
    for i in range(1, n):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i-1]
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> np.ndarray:
    """Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
    # RMA (Wilder's smoothing) — same as Pine's ta.atr
    return rma(tr, period)


def rma(src: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Moving Average (Pine's ta.rma)."""
    n = len(src)
    out = np.zeros(n)
    alpha = 1.0 / period
    out[0] = src[0]
    for i in range(1, n):
        out[i] = alpha * src[i] + (1.0 - alpha) * out[i-1]
    return out


def highest(src: np.ndarray, period: int) -> np.ndarray:
    """Rolling highest value (Pine's ta.highest)."""
    n = len(src)
    out = np.zeros(n)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.max(src[start:i+1])
    return out


def lowest(src: np.ndarray, period: int) -> np.ndarray:
    """Rolling lowest value (Pine's ta.lowest)."""
    n = len(src)
    out = np.zeros(n)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.min(src[start:i+1])
    return out


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True when a crosses above b (Pine's ta.crossover)."""
    n = len(a)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n):
        out[i] = a[i] > b[i] and a[i-1] <= b[i-1]
    return out


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True when a crosses below b (Pine's ta.crossunder)."""
    n = len(a)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n):
        out[i] = a[i] < b[i] and a[i-1] >= b[i-1]
    return out


def pivothigh(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """Pivot high detection (Pine's ta.pivothigh)."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        pivot_idx = i - right
        is_pivot = True
        pivot_val = high[pivot_idx]
        for j in range(pivot_idx - left, pivot_idx):
            if j >= 0 and high[j] >= pivot_val:
                is_pivot = False
                break
        if is_pivot:
            for j in range(pivot_idx + 1, pivot_idx + right + 1):
                if j < n and high[j] >= pivot_val:
                    is_pivot = False
                    break
        if is_pivot:
            out[i] = pivot_val
    return out


def pivotlow(low: np.ndarray, left: int, right: int) -> np.ndarray:
    """Pivot low detection (Pine's ta.pivotlow)."""
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        pivot_idx = i - right
        is_pivot = True
        pivot_val = low[pivot_idx]
        for j in range(pivot_idx - left, pivot_idx):
            if j >= 0 and low[j] <= pivot_val:
                is_pivot = False
                break
        if is_pivot:
            for j in range(pivot_idx + 1, pivot_idx + right + 1):
                if j < n and low[j] <= pivot_val:
                    is_pivot = False
                    break
        if is_pivot:
            out[i] = pivot_val
    return out


def volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Current volume / SMA(volume, period)."""
    vol_sma = sma(volume, period)
    ratio = np.zeros_like(volume)
    mask = vol_sma > 0
    ratio[mask] = volume[mask] / vol_sma[mask]
    return ratio


def htf_resample(src: np.ndarray, timestamps: np.ndarray,
                 htf_seconds: int) -> np.ndarray:
    """
    Resample to higher timeframe (simulates Pine's request.security).
    Returns the HTF value for each bar without future leak.
    
    htf_seconds: e.g. 14400 for 4H
    """
    n = len(src)
    out = np.zeros(n)
    
    current_htf_start = 0
    current_htf_val = src[0]
    
    for i in range(n):
        htf_bar = int(timestamps[i]) // htf_seconds
        prev_htf_bar = int(timestamps[i-1]) // htf_seconds if i > 0 else htf_bar
        
        if htf_bar != prev_htf_bar:
            # New HTF bar — use the close of the PREVIOUS htf bar
            current_htf_val = src[i-1] if i > 0 else src[i]
        
        out[i] = current_htf_val
    
    return out
