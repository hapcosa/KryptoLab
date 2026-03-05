"""
CryptoLab — Incremental Ehlers CyberCycle Processor
====================================================
State-machine that processes one bar at a time, producing the exact
same numerical results as the vectorized ehlers.py indicators.

Every update() call is O(1) — no array allocations, no look-ahead.
This is structurally identical to how TradingView evaluates the
indicator on each incoming tick, making it the most faithful
representation of real-time execution.

Usage:
    from indicators.incremental_ehlers import IncrementalCyberCycle

    proc = IncrementalCyberCycle(strategy.params)
    for bar in detail_bars:
        signal = proc.update(bar['high'], bar['low'], bar['close'],
                             bar['volume'], bar['timestamp'])
        if signal is not None:
            engine.open_position(signal, bar)

Compatible alpha methods: manual, kalman, homodyne, mama, autocorrelation
"""

import math
from typing import Optional, Dict, Any

# Avoid heavy import at module level — numpy only for the
# autocorrelation circular buffer (optional).
import numpy as np

from strategies.base import Signal


# ═══════════════════════════════════════════════════════════════
#  CIRCULAR BUFFER — fixed-size, O(1) push/read
# ═══════════════════════════════════════════════════════════════

class _RingBuf:
    """Fixed-capacity ring buffer backed by a flat list."""
    __slots__ = ('_buf', '_cap', '_idx', '_count')

    def __init__(self, capacity: int, fill: float = 0.0):
        self._buf = [fill] * capacity
        self._cap = capacity
        self._idx = 0  # next write position
        self._count = 0

    def push(self, value: float):
        self._buf[self._idx % self._cap] = value
        self._idx += 1
        if self._count < self._cap:
            self._count += 1

    def ago(self, n: int) -> float:
        """Return value n steps ago (0 = most recent)."""
        if n >= self._count:
            return 0.0
        pos = (self._idx - 1 - n) % self._cap
        return self._buf[pos]

    @property
    def last(self) -> float:
        return self.ago(0)

    def min_max(self) -> tuple:
        """Return (min, max) of populated entries."""
        if self._count == 0:
            return (0.0, 0.0)
        if self._count < self._cap:
            active = self._buf[:self._count]
        else:
            active = self._buf
        return (min(active), max(active))

    @property
    def count(self) -> int:
        return self._count

    def as_list(self, n: int = 0) -> list:
        """Return last n values (most recent first). 0 = all populated."""
        if n <= 0:
            n = self._count
        n = min(n, self._count)
        return [self.ago(i) for i in range(n)]


# ═══════════════════════════════════════════════════════════════
#  ALPHA COMPUTERS (one per method, all O(1) per update)
# ═══════════════════════════════════════════════════════════════

class _ManualAlpha:
    """Constant alpha — trivial."""

    def __init__(self, alpha: float):
        self._a = alpha
        self._p = max(2.0, (2.0 / alpha) - 1.0)

    def update(self, hl2: float) -> tuple:
        return self._a, self._p


class _KalmanAlpha:
    """Kalman Innovation-Based alpha — exact match of ehlers.py kalman_alpha."""

    def __init__(self, process_noise: float, meas_noise: float,
                 alpha_fast: float, alpha_slow: float, sensitivity: float):
        self.pn = process_noise
        self.mn = meas_noise
        self.af = alpha_fast
        self.al = alpha_slow
        self.sens = sensitivity
        self.k_x = 0.0
        self.k_P = 1.0
        self.innov_ema = 0.001
        self._first = True

    def update(self, hl2: float) -> tuple:
        if self._first:
            self.k_x = hl2
            self._first = False

        # Predict
        x_pred = self.k_x
        P_pred = self.k_P + self.pn

        # Innovation
        innovation = hl2 - x_pred
        S = P_pred + self.mn
        K = P_pred / S if S > 1e-12 else 0.5

        # Update
        self.k_x = x_pred + K * innovation
        self.k_P = (1.0 - K) * P_pred

        # Innovation magnitude → alpha
        abs_innov = abs(innovation)
        self.innov_ema = 0.05 * abs_innov + 0.95 * self.innov_ema

        norm_innov = abs_innov / self.innov_ema if self.innov_ema > 1e-12 else 1.0
        ratio = (norm_innov - 1.0) * self.sens
        sigmoid = 1.0 / (1.0 + math.exp(-ratio))

        a = self.al + (self.af - self.al) * sigmoid
        a = max(self.al, min(self.af, a))
        p = max(2.0, (2.0 / a) - 1.0)
        return a, p


class _HomodyneAlpha:
    """
    Homodyne Discriminator — exact match of ehlers.py homodyne_alpha.
    Uses Hilbert Transform to estimate dominant cycle period.
    """

    def __init__(self, min_period: float = 3.0, max_period: float = 40.0):
        self.min_p = min_period
        self.max_p = max_period
        # Buffers for Hilbert (need 7 history each)
        self.smooth_buf = _RingBuf(8)
        self.det_buf = _RingBuf(8)
        self.Q1_buf = _RingBuf(8)
        self.I1_buf = _RingBuf(8)  # I1 = det[i-3]
        # Smoothed quadrature
        self.I2_prev = 0.0
        self.Q2_prev = 0.0
        self.I2_val = 0.0
        self.Q2_val = 0.0
        # Homodyne discriminator
        self.Re_prev = 0.0
        self.Im_prev = 0.0
        # Period tracking
        self.hd_period = 15.0
        self.smooth_period = 15.0
        # Source buffer for 4-bar smooth
        self.src_buf = _RingBuf(4)

    def _hilbert(self, buf: _RingBuf, adj: float) -> float:
        v0 = buf.ago(0)
        v2 = buf.ago(2)
        v4 = buf.ago(4)
        v6 = buf.ago(6)
        return (0.0962 * v0 + 0.5769 * v2 - 0.5769 * v4 - 0.0962 * v6) * adj

    def update(self, hl2: float) -> tuple:
        self.src_buf.push(hl2)

        # 4-bar weighted smooth
        s0 = self.src_buf.ago(0)
        s1 = self.src_buf.ago(1)
        s2 = self.src_buf.ago(2)
        s3 = self.src_buf.ago(3)
        sm = (4.0 * s0 + 3.0 * s1 + 2.0 * s2 + s3) / 10.0
        self.smooth_buf.push(sm)

        # Adjustment factor from previous period
        adj = 0.075 * self.hd_period + 0.54

        # Hilbert Transform components
        det = self._hilbert(self.smooth_buf, adj)
        self.det_buf.push(det)

        Q1 = self._hilbert(self.det_buf, adj)
        self.Q1_buf.push(Q1)

        I1 = self.det_buf.ago(3)  # det[i-3]
        self.I1_buf.push(I1)

        jI = self._hilbert(self.I1_buf, adj)
        jQ = self._hilbert(self.Q1_buf, adj)

        # Smooth quadrature components
        I2 = 0.2 * (I1 - jQ) + 0.8 * self.I2_prev
        Q2 = 0.2 * (Q1 + jI) + 0.8 * self.Q2_prev

        # Homodyne discriminator
        Re = 0.2 * (I2 * self.I2_prev + Q2 * self.Q2_prev) + 0.8 * self.Re_prev
        Im = 0.2 * (I2 * self.Q2_prev - Q2 * self.I2_prev) + 0.8 * self.Im_prev

        # Phase advance → period
        if abs(Im) > 1e-10 and abs(Re) > 1e-10:
            phase_adv = math.atan(Im / Re)
        else:
            phase_adv = 0.0

        raw_per = (2.0 * math.pi / phase_adv) if phase_adv > 0.001 else self.hd_period

        # Clamp rate of change
        raw_per = max(raw_per, 0.67 * self.hd_period)
        raw_per = min(raw_per, 1.5 * self.hd_period)
        raw_per = max(self.min_p, min(self.max_p, raw_per))

        self.hd_period = 0.2 * raw_per + 0.8 * self.hd_period
        self.smooth_period = 0.33 * self.hd_period + 0.67 * self.smooth_period
        self.smooth_period = max(self.min_p, min(self.max_p, self.smooth_period))

        # Store state for next bar
        self.I2_prev = I2
        self.Q2_prev = Q2
        self.I2_val = I2
        self.Q2_val = Q2
        self.Re_prev = Re
        self.Im_prev = Im

        alpha = 2.0 / (self.smooth_period + 1.0)
        return alpha, self.smooth_period


class _MamaAlpha:
    """
    MAMA — exact match of ehlers.py mama_alpha.
    Uses Hilbert Transform phase rate to adapt alpha.
    """

    def __init__(self, fast_limit: float = 0.5, slow_limit: float = 0.05):
        self.fl = fast_limit
        self.sl = slow_limit
        # Hilbert state (same as homodyne but different discriminator)
        self.src_buf = _RingBuf(4)
        self.smooth_buf = _RingBuf(8)
        self.det_buf = _RingBuf(8)
        self.Q1_buf = _RingBuf(8)
        self.I1_buf = _RingBuf(8)
        self.I2_prev = 0.0
        self.Q2_prev = 0.0
        # MAMA-specific: phase tracking
        self.prev_phase = 0.0
        self.dp_smooth = 5.0
        # Fixed adj (MAMA uses adj=1.665 fixed, not period-dependent)
        self.adj = 1.665  # 0.075 * 15.0 + 0.54

    def _hilbert(self, buf: _RingBuf) -> float:
        v0 = buf.ago(0)
        v2 = buf.ago(2)
        v4 = buf.ago(4)
        v6 = buf.ago(6)
        return (0.0962 * v0 + 0.5769 * v2 - 0.5769 * v4 - 0.0962 * v6) * self.adj

    def update(self, hl2: float) -> tuple:
        self.src_buf.push(hl2)

        s0 = self.src_buf.ago(0)
        s1 = self.src_buf.ago(1)
        s2 = self.src_buf.ago(2)
        s3 = self.src_buf.ago(3)
        sm = (4.0 * s0 + 3.0 * s1 + 2.0 * s2 + s3) / 10.0
        self.smooth_buf.push(sm)

        det = self._hilbert(self.smooth_buf)
        self.det_buf.push(det)

        Q1 = self._hilbert(self.det_buf)
        self.Q1_buf.push(Q1)

        I1 = self.det_buf.ago(3)
        self.I1_buf.push(I1)

        jI = self._hilbert(self.I1_buf)
        jQ = self._hilbert(self.Q1_buf)

        I2 = 0.2 * (I1 - jQ) + 0.8 * self.I2_prev
        Q2 = 0.2 * (Q1 + jI) + 0.8 * self.Q2_prev

        # Phase calculation with quadrant correction
        sumI = I1 + self.det_buf.ago(4)
        sumQ = Q1 + self.Q1_buf.ago(1)

        if abs(sumI) > 0.001:
            raw_phase = math.atan(abs(sumQ / sumI)) * (180.0 / math.pi)
        else:
            raw_phase = 90.0

        if sumI < 0 and sumQ > 0:
            raw_phase = 180.0 - raw_phase
        elif sumI < 0 and sumQ < 0:
            raw_phase = 180.0 + raw_phase
        elif sumI > 0 and sumQ < 0:
            raw_phase = 360.0 - raw_phase

        # Delta phase
        dp_raw = self.prev_phase - raw_phase
        if dp_raw > 180.0:
            dp_raw -= 360.0
        if dp_raw < -180.0:
            dp_raw += 360.0
        dp = max(1.0, min(60.0, dp_raw))

        self.dp_smooth = 0.33 * dp + 0.67 * self.dp_smooth
        self.dp_smooth = max(1.0, min(60.0, self.dp_smooth))

        # Alpha from phase rate
        a = self.fl / self.dp_smooth
        a = max(self.sl, min(self.fl, a))

        # Store for next bar
        self.prev_phase = raw_phase
        self.I2_prev = I2
        self.Q2_prev = Q2

        p = max(2.0, (2.0 / a) - 1.0)
        return a, p


class _AutocorrelationAlpha:
    """
    Autocorrelation Periodogram — exact match of ehlers.py autocorrelation_alpha.
    Needs a circular buffer of filtered values for the correlation search.
    This is the most expensive alpha method: O(max_period² × avg_length) per bar.
    """

    def __init__(self, min_period: int = 6, max_period: int = 48,
                 avg_length: int = 3):
        self.min_p = min_period
        self.max_p = max_period
        self.avg_len = avg_length
        self.step = max(1, (max_period - min_period) // 10)

        # HP filter coefficients
        a1 = (0.707 * 2.0 * math.pi) / max_period
        self.alpha_hp = (math.cos(a1) + math.sin(a1) - 1.0) / math.cos(a1)

        # Super smoother coefficients
        a1ss = math.exp(-1.414 * math.pi / min_period)
        b1ss = 2.0 * a1ss * math.cos(1.414 * math.pi / min_period)
        self.c2ss = b1ss
        self.c3ss = -a1ss * a1ss
        self.c1ss = 1.0 - self.c2ss - self.c3ss

        # State
        self.hp_buf = _RingBuf(3)  # high-pass (need [i], [i-1], [i-2])
        self.src_buf = _RingBuf(3)
        self.filt_buf_ring = _RingBuf(3)  # for super smoother recurrence

        # Large buffer for autocorrelation search
        buf_size = max(max_period * avg_length + 20, 250)
        self.filt_history = _RingBuf(buf_size)

        self.best_period = 15.0
        self.bar_count = 0
        self.warmup = max_period * avg_length + 10

    def update(self, hl2: float) -> tuple:
        self.bar_count += 1
        ahp = self.alpha_hp

        # High-pass filter
        s0 = hl2
        s1 = self.src_buf.ago(0) if self.src_buf.count > 0 else 0.0
        s2 = self.src_buf.ago(1) if self.src_buf.count > 1 else 0.0
        self.src_buf.push(s0)

        hp1 = self.hp_buf.ago(0) if self.hp_buf.count > 0 else 0.0
        hp2 = self.hp_buf.ago(1) if self.hp_buf.count > 1 else 0.0

        hp = ((1.0 - ahp / 2.0) ** 2 * (s0 - 2.0 * s1 + s2)
              + 2.0 * (1.0 - ahp) * hp1
              - (1.0 - ahp) ** 2 * hp2)
        self.hp_buf.push(hp)

        # Super smoother
        f1 = self.filt_buf_ring.ago(0) if self.filt_buf_ring.count > 0 else 0.0
        f2 = self.filt_buf_ring.ago(1) if self.filt_buf_ring.count > 1 else 0.0
        hp_prev = self.hp_buf.ago(1) if self.hp_buf.count > 1 else 0.0
        filt = (self.c1ss * (hp + hp_prev) / 2.0
                + self.c2ss * f1 + self.c3ss * f2)
        self.filt_buf_ring.push(filt)
        self.filt_history.push(filt)

        if self.bar_count < self.warmup:
            alpha = 2.0 / (self.best_period + 1.0)
            return alpha, self.best_period

        # Autocorrelation search
        max_corr = 0.0
        best_p = self.best_period

        for p in range(self.min_p, self.max_p + 1, self.step):
            sx = sy = sxx = syy = sxy = 0.0
            cnt = min(self.avg_len * p, 200)

            for j in range(cnt):
                x = self.filt_history.ago(j)
                y = self.filt_history.ago(j + p)
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y

            denom = (cnt * sxx - sx * sx) * (cnt * syy - sy * sy)
            corr = (cnt * sxy - sx * sy) / math.sqrt(denom) if denom > 0 else 0.0

            if corr > max_corr:
                max_corr = corr
                best_p = float(p)

        self.best_period = 0.25 * best_p + 0.75 * self.best_period
        self.best_period = max(self.min_p, min(self.max_p, self.best_period))

        alpha = 2.0 / (self.best_period + 1.0)
        return alpha, self.best_period


# ═══════════════════════════════════════════════════════════════
#  MAIN PROCESSOR: IncrementalCyberCycle
# ═══════════════════════════════════════════════════════════════

class IncrementalCyberCycle:
    """
    Complete incremental CyberCycle processor.

    Maintains all indicator state (alpha, cybercycle, trigger, iTrend,
    Fisher, ATR, volume, HTF, confidence) and produces Signal objects
    identical to the vectorized CyberCycleStrategy.generate_signal().

    Every update() call is O(1) for manual/kalman/homodyne/mama alpha
    and O(max_period²) for autocorrelation alpha.
    """

    def __init__(self, params: Dict[str, Any], detail_tf_ratio: int = 1):
        self.p = dict(params)  # shallow copy
        # Ratio of detail bars per main-TF bar (e.g. 60 for 1m detail / 1h main).
        # Used to scale min_bars throttle so it operates in main-TF bar units.
        self._detail_tf_ratio = max(1, int(detail_tf_ratio))
        self._build_alpha_computer()
        self.reset()

    # ── Setup ────────────────────────────────────────────────

    def _build_alpha_computer(self):
        method = self.p.get('alpha_method', 'kalman')
        if method == 'manual':
            self._alpha = _ManualAlpha(self.p.get('manual_alpha', 0.35))
        elif method == 'kalman':
            self._alpha = _KalmanAlpha(
                self.p.get('kal_process_noise', 0.01),
                self.p.get('kal_meas_noise', 0.5),
                self.p.get('kal_alpha_fast', 0.5),
                self.p.get('kal_alpha_slow', 0.05),
                self.p.get('kal_sensitivity', 2.0),
            )
        elif method == 'homodyne':
            self._alpha = _HomodyneAlpha(
                self.p.get('hd_min_period', 3.0),
                self.p.get('hd_max_period', 40.0),
            )
        elif method == 'mama':
            self._alpha = _MamaAlpha(
                self.p.get('mama_fast', 0.5),
                self.p.get('mama_slow', 0.05),
            )
        elif method == 'autocorrelation':
            self._alpha = _AutocorrelationAlpha(
                self.p.get('ac_min_period', 6),
                self.p.get('ac_max_period', 48),
                self.p.get('ac_avg_length', 3),
            )
        else:
            self._alpha = _ManualAlpha(0.35)

    def reset(self):
        """Reset all state for a new dataset."""
        self._build_alpha_computer()

        # ── CyberCycle state ──
        self._src_buf = _RingBuf(4)  # hl2 for smooth
        self._smooth_buf = _RingBuf(3)  # smooth[i], [i-1], [i-2]
        self._cycle_buf = _RingBuf(4)  # cycle[i], [i-1], [i-2], [i-3] for momentum

        # ── Trigger (EMA of cycle) ──
        self._trigger = 0.0
        self._trigger_k = 2.0 / (self.p.get('trigger_ema', 14) + 1.0)

        # ── iTrend ──
        self._itrend_buf = _RingBuf(3)  # it[i], [i-1], [i-2]
        self._close_buf = _RingBuf(3)  # close[i], [i-1], [i-2]
        self._it_alpha = self.p.get('itrend_alpha', 0.07)

        # ── Fisher Transform (rolling min/max of cycle over lookback) ──
        self._fisher_lookback = 10
        self._cycle_window = _RingBuf(self._fisher_lookback)
        self._fisher_prev = 0.0

        # ── ATR (Wilder's RMA) ──
        self._atr_period = 14
        self._atr_val = 0.0
        self._atr_count = 0
        self._prev_close = 0.0

        # ── Volume ratio (SMA of volume over 20 bars) ──
        self._vol_period = 20
        self._vol_buf = _RingBuf(self._vol_period)
        self._vol_sum = 0.0

        # ── HTF filter (long-period EMA proxy) ──
        self._htf_ema_src = 0.0  # EMA(hl2, 40)
        self._htf_ema_close = 0.0  # EMA(close, 40)
        self._htf_k = 2.0 / (40 + 1.0)

        # ── Crossover detection ──
        self._prev_cycle = 0.0
        self._prev_trigger = 0.0

        # ── Signal control ──
        self._bar_count = 0
        self._last_signal_bar = -9999
        self._daily_count = 0
        self._current_day = -1
        self._trigger_initialized = False  # seed trigger with first cycle value

    # ── Core update ──────────────────────────────────────────

    def update(self, high: float, low: float, close: float,
               volume: float, timestamp: int) -> Optional[Signal]:
        """
        Process one bar. Returns Signal or None.
        Exactly reproduces vectorized calculate_indicators + generate_signal.

        Parameters
        ----------
        high, low, close : float
            Bar OHLC (open not needed for CyberCycle)
        volume : float
            Bar volume
        timestamp : int
            Millisecond epoch timestamp

        Returns
        -------
        Signal or None
        """
        i = self._bar_count
        hl2 = (high + low) / 2.0

        # ─── 1. Alpha ───
        alpha_val, period_val = self._alpha.update(hl2)
        floor = self.p.get('alpha_floor', 0.0)
        if floor > 0:
            alpha_val = max(alpha_val, floor)

        # ─── 2. CyberCycle ───
        self._src_buf.push(hl2)
        s0 = self._src_buf.ago(0)
        s1 = self._src_buf.ago(1)
        s2 = self._src_buf.ago(2)
        s3 = self._src_buf.ago(3)
        smooth = (s0 + 2.0 * s1 + 2.0 * s2 + s3) / 6.0
        self._smooth_buf.push(smooth)

        if i < 7:
            cycle = (s0 - 2.0 * s1 + s2) / 4.0 if i >= 2 else 0.0
        else:
            a = alpha_val
            a1 = (1.0 - 0.5 * a) ** 2
            a2 = 2.0 * (1.0 - a)
            a3 = (1.0 - a) ** 2
            sm0 = self._smooth_buf.ago(0)
            sm1 = self._smooth_buf.ago(1)
            sm2 = self._smooth_buf.ago(2)
            c1 = self._cycle_buf.ago(0)
            c2 = self._cycle_buf.ago(1)
            cycle = a1 * (sm0 - 2.0 * sm1 + sm2) + a2 * c1 - a3 * c2

        # ─── 3. Trigger (EMA of cycle) ───
        # IMPORTANT: save prev values BEFORE updating state.
        # prev_cycle must be captured before push(); prev_trigger before EMA update.
        # Asymmetry here was causing crossover detection to always use current
        # cycle as "previous", making bull/bear_cross always False → WR=0%.
        self._prev_cycle = self._cycle_buf.ago(0)   # current cycle, pre-push = previous
        self._prev_trigger = self._trigger           # trigger before this bar's EMA update
        self._cycle_buf.push(cycle)
        k = self._trigger_k
        # Seed trigger with first cycle value — matches ema(cycle, period) in common.py
        # which initializes out[0] = src[0]. Without this, trigger starts at 0
        # and takes many bars to converge, causing false crossovers during warmup.
        if not self._trigger_initialized:
            self._trigger = cycle
            self._trigger_initialized = True
        else:
            self._trigger = k * cycle + (1.0 - k) * self._trigger

        # ─── 4. iTrend ───
        self._close_buf.push(close)
        a = self._it_alpha
        if i < 3:
            it_val = close
        else:
            c0 = self._close_buf.ago(0)
            c1_c = self._close_buf.ago(1)
            c2_c = self._close_buf.ago(2)
            t1 = self._itrend_buf.ago(0)
            t2 = self._itrend_buf.ago(1)
            it_val = ((a - a * a / 4.0) * c0
                      + 0.5 * a * a * c1_c
                      - (a - 0.75 * a * a) * c2_c
                      + 2.0 * (1.0 - a) * t1
                      - (1.0 - a) ** 2 * t2)
        self._itrend_buf.push(it_val)

        bull_trend = self._itrend_buf.ago(0) > self._itrend_buf.ago(2) if i >= 2 else False
        bear_trend = self._itrend_buf.ago(0) < self._itrend_buf.ago(2) if i >= 2 else False

        # ─── 5. Fisher Transform ───
        self._cycle_window.push(cycle)
        fL, fH = self._cycle_window.min_max()
        fR = fH - fL
        if fR != 0:
            fV = 2.0 * ((cycle - fL) / fR - 0.5)
        else:
            fV = 0.0
        fV = max(-0.999, min(0.999, fV))
        raw_fish = 0.5 * math.log((1.0 + fV) / (1.0 - fV))
        fisher_val = 0.5 * raw_fish + 0.5 * self._fisher_prev
        fish_rising = fisher_val > self._fisher_prev
        fish_falling = fisher_val < self._fisher_prev
        self._fisher_prev = fisher_val

        # ─── 6. ATR (Wilder's RMA) ───
        if i == 0:
            self._prev_close = close
            self._atr_val = high - low
        else:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
            if self._atr_count < self._atr_period:
                # Accumulate for initial SMA
                self._atr_val = (self._atr_val * self._atr_count + tr) / (self._atr_count + 1)
                self._atr_count += 1
            else:
                # Wilder's RMA
                self._atr_val = (self._atr_val * (self._atr_period - 1) + tr) / self._atr_period
        self._prev_close = close
        atr_val = self._atr_val

        # ─── 7. Volume ratio ───
        # Push volume, maintain running sum for SMA
        if self._vol_buf.count >= self._vol_period:
            oldest = self._vol_buf.ago(self._vol_period - 1)
            self._vol_sum -= oldest
        self._vol_buf.push(volume)
        self._vol_sum += volume
        vol_sma = self._vol_sum / self._vol_buf.count if self._vol_buf.count > 0 else 1.0
        vol_ratio = volume / vol_sma if vol_sma > 0 else 1.0

        vol_mult = self.p.get('volume_mult', 2.0)
        use_vol = self.p.get('use_volume', True)
        volume_ok = (not use_vol) or (vol_ratio >= vol_mult)

        # ─── 8. HTF filter (EMA proxy) ───
        hk = self._htf_k
        self._htf_ema_src = hk * hl2 + (1.0 - hk) * self._htf_ema_src
        self._htf_ema_close = hk * close + (1.0 - hk) * self._htf_ema_close
        use_htf = self.p.get('use_htf', False)
        htf_align_buy = (not use_htf) or (self._htf_ema_src > self._htf_ema_close)
        htf_align_sell = (not use_htf) or (self._htf_ema_src < self._htf_ema_close)

        # ─── 9. OB/OS zones ───
        ob_level = self.p.get('ob_level', 1.5)
        os_level = self.p.get('os_level', -1.5)
        in_ob = cycle > ob_level
        in_os = cycle < os_level

        # ─── 10. Momentum (cycle - cycle[3]) ───
        momentum3 = cycle - self._cycle_buf.ago(3) if i >= 3 else 0.0

        # ─── 11. Crossover detection ───
        prev_c = self._prev_cycle
        prev_t = self._prev_trigger
        curr_c = cycle
        curr_t = self._trigger

        bull_cross = (prev_c <= prev_t) and (curr_c > curr_t)
        bear_cross = (prev_c >= prev_t) and (curr_c < curr_t)

        self._bar_count += 1

        # ─── 12. Signal generation (mirrors generate_signal exactly) ───
        if i < 10:
            return None

        if not (bull_cross or bear_cross):
            return None

        # min_bars throttle — must be expressed in MAIN-TF bars, not detail bars.
        # Multiply by detail_tf_ratio so e.g. min_bars=24 (1h bars) becomes
        # 24*60=1440 when running on 1m detail bars.
        min_bars = self.p.get('min_bars', 24) * self._detail_tf_ratio
        if i - self._last_signal_bar < min_bars:
            return None

        is_buy = bull_cross
        use_trend = self.p.get('use_trend', True)

        # ── Confidence scoring (exact match of compute_confidence) ──
        conf = 0.0
        if is_buy:
            conf += 20.0 if bull_cross else 0.0
            conf += 15.0 if (bull_trend if use_trend else True) else 0.0
            conf += 15.0 if in_os else 0.0
            conf += 15.0 if (volume_ok if use_vol else True) else 0.0
            conf += 10.0 if fish_rising else 0.0
            conf += 10.0 if (momentum3 > 0) else 0.0
            conf += 15.0 if htf_align_buy else 0.0
        else:
            conf += 20.0 if bear_cross else 0.0
            conf += 15.0 if (bear_trend if use_trend else True) else 0.0
            conf += 15.0 if in_ob else 0.0
            conf += 15.0 if (volume_ok if use_vol else True) else 0.0
            conf += 10.0 if fish_falling else 0.0
            conf += 10.0 if (momentum3 < 0) else 0.0
            conf += 15.0 if htf_align_sell else 0.0
        conf = min(conf, 100.0)

        # Confidence filter
        conf_min = self.p.get('confidence_min', 80.0)
        if conf < conf_min:
            return None

        # Daily signal limit
        max_daily = self.p.get('max_signals_per_day', 0)
        if max_daily > 0:
            day_key = timestamp // 86400000  # ms → day
            if day_key != self._current_day:
                self._current_day = day_key
                self._daily_count = 0
            if self._daily_count >= max_daily:
                return None
            self._daily_count += 1

        self._last_signal_bar = i

        # ── Build Signal (mirrors cybercycle.py exactly) ──
        direction = 1 if is_buy else -1
        entry = close

        sl_dist = atr_val * self.p.get('sl_atr_mult', 1.5)
        sl = entry - direction * sl_dist
        risk = sl_dist

        tp1_rr = self.p.get('tp1_rr', 2.0)
        tp2_rr = self.p.get('tp2_rr', 3.0)
        tp1_size = self.p.get('tp1_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        tp1 = entry + direction * risk * tp1_rr
        tp2 = entry + direction * risk * tp2_rr

        # Break-even
        be_pct = self.p.get('be_pct', 1.5)
        be_trigger = entry * (1.0 + direction * be_pct / 100.0) if be_pct > 0 else 0.0

        # Trailing
        use_trailing = self.p.get('use_trailing', True)
        trail_pullback = self.p.get('trail_pullback_pct', 1.0)
        trailing_dist = entry * trail_pullback / 100.0 if use_trailing else 0.0

        return Signal(
            direction=direction,
            confidence=conf,
            entry_price=entry,
            sl_price=sl,
            tp_levels=[tp1, tp2],
            tp_sizes=[tp1_size, tp2_size],
            leverage=self.p.get('leverage', 3.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_dist,
            metadata={
                'close_on_signal': self.p.get('close_on_signal', True),
                'max_signals_per_day': self.p.get('max_signals_per_day', 0),
                'alpha_method': self.p.get('alpha_method', 'kalman'),
                'alpha': alpha_val,
                'period': period_val,
                'cycle': cycle,
                'fisher': fisher_val,
                'sl_dist_atr': sl_dist / atr_val if atr_val > 0 else 0,
                'tp1': tp1,
                'tp2': tp2,
                'be_pct': be_pct,
                'trail_pct': trail_pullback if use_trailing else 0,
            }
        )

    # ── Convenience ──────────────────────────────────────────

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def get_state_snapshot(self) -> dict:
        """Return current indicator values (for debugging/logging)."""
        return {
            'cycle': self._cycle_buf.ago(0),
            'trigger': self._trigger,
            'itrend': self._itrend_buf.ago(0),
            'fisher': self._fisher_prev,
            'atr': self._atr_val,
            'bar_count': self._bar_count,
        }