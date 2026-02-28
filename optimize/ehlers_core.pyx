# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Ehlers Digital Signal Processing functions implemented in Cython.
Exact reimplementation of Pine Script v4 CyberCycle indicator.

Compile with: python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
from libc.math cimport log, fabs, fmin, fmax

ctypedef np.float64_t DTYPE_t


def cyber_cycle(np.ndarray[DTYPE_t, ndim=1] src, double alpha):
    """
    Ehlers Cyber Cycle.
    
    Pine Script equivalent:
        smooth = (src + 2*src[1] + 2*src[2] + src[3]) / 6
        if bar_index < 7:
            cycle = (src - 2*src[1] + src[2]) / 4
        else:
            cycle = (1-0.5*alpha)^2 * (smooth - 2*smooth[1] + smooth[2])
                   + 2*(1-alpha)*cycle[1] - (1-alpha)^2 * cycle[2]
    
    Parameters
    ----------
    src : ndarray
        Source series (typically hl2 = (high+low)/2)
    alpha : float
        Damping factor (0.06 to 0.6)
    
    Returns
    -------
    cycle : ndarray
        Cyber cycle values
    """
    cdef int n = src.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] smooth = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] cycle = np.zeros(n, dtype=np.float64)
    cdef double a1 = (1.0 - 0.5 * alpha) * (1.0 - 0.5 * alpha)
    cdef double a2 = 2.0 * (1.0 - alpha)
    cdef double a3 = (1.0 - alpha) * (1.0 - alpha)
    cdef int i

    for i in range(n):
        # Smooth
        if i >= 3:
            smooth[i] = (src[i] + 2.0 * src[i-1] + 2.0 * src[i-2] + src[i-3]) / 6.0
        elif i >= 2:
            smooth[i] = (src[i] + 2.0 * src[i-1] + 2.0 * src[i-2]) / 5.0
        elif i >= 1:
            smooth[i] = (src[i] + 2.0 * src[i-1]) / 3.0
        else:
            smooth[i] = src[i]

        # Cycle
        if i < 7:
            if i >= 2:
                cycle[i] = (src[i] - 2.0 * src[i-1] + src[i-2]) / 4.0
            elif i >= 1:
                cycle[i] = (src[i] - 2.0 * src[i-1]) / 4.0
            else:
                cycle[i] = 0.0
        else:
            cycle[i] = (a1 * (smooth[i] - 2.0 * smooth[i-1] + smooth[i-2])
                        + a2 * cycle[i-1]
                        - a3 * cycle[i-2])

    return cycle


def ema(np.ndarray[DTYPE_t, ndim=1] src, int length):
    """
    Exponential Moving Average (accepts series-length like Pine Script f_ema).
    
    Parameters
    ----------
    src : ndarray
        Source series
    length : int
        EMA period
    
    Returns
    -------
    result : ndarray
        EMA values
    """
    cdef int n = src.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros(n, dtype=np.float64)
    cdef double k = 2.0 / (length + 1.0)
    cdef double k_inv = 1.0 - k
    cdef int i

    result[0] = src[0]
    for i in range(1, n):
        result[i] = k * src[i] + k_inv * result[i-1]

    return result


def instantaneous_trendline(np.ndarray[DTYPE_t, ndim=1] close, double alpha):
    """
    Ehlers Instantaneous Trendline.
    
    Pine Script equivalent:
        iTrend = (alpha - alpha^2/4)*close + 0.5*alpha^2*close[1]
                - (alpha - 0.75*alpha^2)*close[2]
                + 2*(1-alpha)*iTrend[1] - (1-alpha)^2*iTrend[2]
    
    Parameters
    ----------
    close : ndarray
        Close prices
    alpha : float
        Smoothing factor
    
    Returns
    -------
    itrend : ndarray
        Instantaneous trendline values
    """
    cdef int n = close.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] itrend = np.zeros(n, dtype=np.float64)
    cdef double a2 = alpha * alpha
    cdef double c0 = alpha - a2 / 4.0
    cdef double c1 = 0.5 * a2
    cdef double c2 = alpha - 0.75 * a2
    cdef double d1 = 2.0 * (1.0 - alpha)
    cdef double d2 = (1.0 - alpha) * (1.0 - alpha)
    cdef int i

    if n > 0:
        itrend[0] = close[0]
    if n > 1:
        itrend[1] = close[1]

    for i in range(2, n):
        itrend[i] = (c0 * close[i]
                     + c1 * close[i-1]
                     - c2 * close[i-2]
                     + d1 * itrend[i-1]
                     - d2 * itrend[i-2])

    return itrend


def fisher_transform(np.ndarray[DTYPE_t, ndim=1] cycle, int length):
    """
    Fisher Transform of Cyber Cycle (smoothed).
    
    Pine Script equivalent:
        highest/lowest over length bars
        value = 2 * ((cycle - lowest) / range - 0.5), clamped to (-0.999, 0.999)
        rawFish = 0.5 * ln((1+value)/(1-value))
        fisher = 0.5 * rawFish + 0.5 * fisher[1]
    
    Parameters
    ----------
    cycle : ndarray
        Cyber cycle values
    length : int
        Lookback for highest/lowest (typically 10)
    
    Returns
    -------
    fisher : ndarray
        Fisher transform values
    """
    cdef int n = cycle.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] fisher = np.zeros(n, dtype=np.float64)
    cdef double highest, lowest, rng, value, raw_fish
    cdef int i, j

    for i in range(n):
        # Find highest and lowest in window
        highest = cycle[i]
        lowest = cycle[i]
        for j in range(max(0, i - length + 1), i + 1):
            if cycle[j] > highest:
                highest = cycle[j]
            if cycle[j] < lowest:
                lowest = cycle[j]

        rng = highest - lowest
        if rng != 0.0:
            value = 2.0 * ((cycle[i] - lowest) / rng - 0.5)
        else:
            value = 0.0

        # Clamp
        value = fmax(-0.999, fmin(0.999, value))

        # Fisher transform
        raw_fish = 0.5 * log((1.0 + value) / (1.0 - value))

        # Smooth
        if i == 0:
            fisher[i] = 0.5 * raw_fish
        else:
            fisher[i] = 0.5 * raw_fish + 0.5 * fisher[i-1]

    return fisher


def compute_mfe_mae(np.ndarray[DTYPE_t, ndim=1] high,
                    np.ndarray[DTYPE_t, ndim=1] low,
                    np.ndarray[DTYPE_t, ndim=1] close,
                    np.ndarray[long, ndim=1] signal_bars,
                    np.ndarray[long, ndim=1] signal_types,
                    int window):
    """
    Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    for each signal within a forward-looking window.
    
    This is the CRITICAL performance function - nested loop over signals × bars.
    
    Parameters
    ----------
    high : ndarray
        High prices
    low : ndarray 
        Low prices
    close : ndarray
        Close prices (entry price = close at signal bar)
    signal_bars : ndarray
        Bar indices where signals occurred
    signal_types : ndarray
        1 = BUY, -1 = SELL
    window : int
        Forward-looking window (bars)
    
    Returns
    -------
    mfe : ndarray
        Maximum Favorable Excursion (%) for each signal
    mae : ndarray
        Maximum Adverse Excursion (%) for each signal
    mfe_bar : ndarray
        Bar offset where MFE was reached
    mae_bar : ndarray
        Bar offset where MAE was reached
    """
    cdef int n_signals = signal_bars.shape[0]
    cdef int n_bars = high.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] mfe = np.zeros(n_signals, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] mae = np.zeros(n_signals, dtype=np.float64)
    cdef np.ndarray[long, ndim=1] mfe_bar_offset = np.zeros(n_signals, dtype=np.int64)
    cdef np.ndarray[long, ndim=1] mae_bar_offset = np.zeros(n_signals, dtype=np.int64)
    cdef int i, j, bar_idx, end_bar, sig_type
    cdef double entry_price, favorable, adverse, pct

    for i in range(n_signals):
        bar_idx = signal_bars[i]
        sig_type = signal_types[i]
        entry_price = close[bar_idx]

        if entry_price == 0.0:
            continue

        end_bar = min(bar_idx + window, n_bars)
        favorable = 0.0
        adverse = 0.0

        for j in range(bar_idx + 1, end_bar):
            if sig_type == 1:  # BUY signal
                # Favorable = how high did it go
                pct = (high[j] - entry_price) / entry_price * 100.0
                if pct > favorable:
                    favorable = pct
                    mfe_bar_offset[i] = j - bar_idx
                # Adverse = how low did it drop
                pct = (entry_price - low[j]) / entry_price * 100.0
                if pct > adverse:
                    adverse = pct
                    mae_bar_offset[i] = j - bar_idx
            else:  # SELL signal
                # Favorable = how low did it go (profit for short)
                pct = (entry_price - low[j]) / entry_price * 100.0
                if pct > favorable:
                    favorable = pct
                    mfe_bar_offset[i] = j - bar_idx
                # Adverse = how high did it go (loss for short)
                pct = (high[j] - entry_price) / entry_price * 100.0
                if pct > adverse:
                    adverse = pct
                    mae_bar_offset[i] = j - bar_idx

        mfe[i] = favorable
        mae[i] = adverse

    return mfe, mae, mfe_bar_offset, mae_bar_offset


def compute_early_mae(np.ndarray[DTYPE_t, ndim=1] high,
                      np.ndarray[DTYPE_t, ndim=1] low,
                      np.ndarray[DTYPE_t, ndim=1] close,
                      np.ndarray[long, ndim=1] signal_bars,
                      np.ndarray[long, ndim=1] signal_types,
                      int early_bars):
    """
    Compute MAE within the first N bars only (early adverse excursion).
    Signals that drop fast are low quality regardless of eventual MFE.
    
    Parameters
    ----------
    early_bars : int
        Number of bars to check for early adverse movement
    
    Returns
    -------
    early_mae : ndarray
        Maximum adverse excursion in first N bars (%)
    """
    cdef int n_signals = signal_bars.shape[0]
    cdef int n_bars = high.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] early_mae = np.zeros(n_signals, dtype=np.float64)
    cdef int i, j, bar_idx, end_bar, sig_type
    cdef double entry_price, adverse, pct

    for i in range(n_signals):
        bar_idx = signal_bars[i]
        sig_type = signal_types[i]
        entry_price = close[bar_idx]

        if entry_price == 0.0:
            continue

        end_bar = min(bar_idx + early_bars + 1, n_bars)
        adverse = 0.0

        for j in range(bar_idx + 1, end_bar):
            if sig_type == 1:  # BUY
                pct = (entry_price - low[j]) / entry_price * 100.0
            else:  # SELL
                pct = (high[j] - entry_price) / entry_price * 100.0
            if pct > adverse:
                adverse = pct

        early_mae[i] = adverse

    return early_mae
