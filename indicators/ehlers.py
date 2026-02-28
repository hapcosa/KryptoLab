"""
CryptoLab — Ehlers Indicators
Faithful translation from cybercycle_v62.pine

References:
- John Ehlers, "Cybernetic Analysis for Stocks and Futures" (2004)
- John Ehlers, "Cycle Analytics for Traders" (2013)
- John Ehlers, "MESA and Trading Market Cycles" (2nd ed, 2001)

All functions operate on numpy arrays (vectorized where possible,
bar-by-bar loop where state is required — matching Pine's var behavior).
"""
import numpy as np
from typing import Tuple


# ═══════════════════════════════════════════════════════════════
#  UTILITY: nz() equivalent — replaces NaN with 0.0 or default
# ═══════════════════════════════════════════════════════════════

def _nz(arr: np.ndarray, idx: int, default: float = 0.0) -> float:
    """Pine's nz(): return arr[idx] if valid, else default."""
    if idx < 0 or idx >= len(arr) or np.isnan(arr[idx]):
        return default
    return arr[idx]


# ═══════════════════════════════════════════════════════════════
#  ALPHA METHOD 1: HOMODYNE DISCRIMINATOR
#  From: Ehlers "MESA and Trading Market Cycles"
#  Pine: f_homodyne() in cybercycle_v62.pine lines 391-432
# ═══════════════════════════════════════════════════════════════

def homodyne_alpha(src: np.ndarray,
                   min_period: float = 3.0,
                   max_period: float = 40.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Homodyne Discriminator — estimates dominant cycle period
    using quadrature components and phase advance.
    
    Returns:
        alpha: adaptive smoothing factor per bar
        period: estimated dominant period per bar
    """
    n = len(src)
    alpha_out = np.full(n, 0.07)
    period_out = np.full(n, 15.0)
    
    # State arrays (Pine's var)
    smooth = np.zeros(n)
    det = np.zeros(n)
    Q1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    I2 = np.zeros(n)
    Q2 = np.zeros(n)
    Re = np.zeros(n)
    Im = np.zeros(n)
    hd_period = np.full(n, 15.0)
    smooth_period = np.full(n, 15.0)
    
    for i in range(n):
        # 4-bar weighted smooth
        s0 = src[i] if i >= 0 else 0.0
        s1 = src[i-1] if i >= 1 else 0.0
        s2 = src[i-2] if i >= 2 else 0.0
        s3 = src[i-3] if i >= 3 else 0.0
        smooth[i] = (4.0 * s0 + 3.0 * s1 + 2.0 * s2 + s3) / 10.0
        
        # Adjustment factor from previous period
        prev_per = hd_period[i-1] if i >= 1 else 15.0
        adj = 0.075 * prev_per + 0.54
        
        # Hilbert Transform (bandpass filter coefficients)
        def _hilbert(arr, idx, _adj):
            """Ehlers' Hilbert Transform approximation."""
            v0 = arr[idx] if idx >= 0 else 0.0
            v2 = arr[idx-2] if idx >= 2 else 0.0
            v4 = arr[idx-4] if idx >= 4 else 0.0
            v6 = arr[idx-6] if idx >= 6 else 0.0
            return (0.0962 * v0 + 0.5769 * v2 - 0.5769 * v4 - 0.0962 * v6) * _adj
        
        det[i] = _hilbert(smooth, i, adj)
        Q1[i] = _hilbert(det, i, adj)
        I1 = det[i-3] if i >= 3 else 0.0
        
        # Compute I1 array for jI hilbert
        I1_arr = np.zeros(n)
        for k in range(min(i+1, n)):
            I1_arr[k] = det[k-3] if k >= 3 else 0.0
        
        jI[i] = _hilbert(I1_arr, i, adj)
        jQ[i] = _hilbert(Q1, i, adj)
        
        # Smooth quadrature components
        prev_I2 = I2[i-1] if i >= 1 else 0.0
        prev_Q2 = Q2[i-1] if i >= 1 else 0.0
        I2[i] = 0.2 * (I1 - jQ[i]) + 0.8 * prev_I2
        Q2[i] = 0.2 * (Q1[i] + jI[i]) + 0.8 * prev_Q2
        
        # Homodyne discriminator
        prev_Re = Re[i-1] if i >= 1 else 0.0
        prev_Im = Im[i-1] if i >= 1 else 0.0
        prev_I2_1 = I2[i-1] if i >= 1 else 0.0
        prev_Q2_1 = Q2[i-1] if i >= 1 else 0.0
        Re[i] = 0.2 * (I2[i] * prev_I2_1 + Q2[i] * prev_Q2_1) + 0.8 * prev_Re
        Im[i] = 0.2 * (I2[i] * prev_Q2_1 - Q2[i] * prev_I2_1) + 0.8 * prev_Im
        
        # Phase advance → period
        if abs(Im[i]) > 1e-10 and abs(Re[i]) > 1e-10:
            phase_adv = np.arctan(Im[i] / Re[i])
        else:
            phase_adv = 0.0
        
        raw_per = (2.0 * np.pi / phase_adv) if phase_adv > 0.001 else prev_per
        
        # Clamp rate of change
        raw_per = max(raw_per, 0.67 * prev_per)
        raw_per = min(raw_per, 1.5 * prev_per)
        raw_per = max(min_period, min(max_period, raw_per))
        
        hd_period[i] = 0.2 * raw_per + 0.8 * prev_per
        
        # Smooth period
        prev_sp = smooth_period[i-1] if i >= 1 else 15.0
        smooth_period[i] = 0.33 * hd_period[i] + 0.67 * prev_sp
        smooth_period[i] = max(min_period, min(max_period, smooth_period[i]))
        
        alpha_out[i] = 2.0 / (smooth_period[i] + 1.0)
        period_out[i] = smooth_period[i]
    
    return alpha_out, period_out


# ═══════════════════════════════════════════════════════════════
#  ALPHA METHOD 2: MAMA (Phase Rate of Change)
#  From: Ehlers "MESA and Trading Market Cycles"
#  Pine: f_mama() in cybercycle_v62.pine lines 439-500
# ═══════════════════════════════════════════════════════════════

def mama_alpha(src: np.ndarray,
               fast_limit: float = 0.5,
               slow_limit: float = 0.05
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    MAMA — Mother of Adaptive Moving Averages.
    Uses Hilbert Transform phase rate to adapt alpha.
    
    Returns:
        alpha: adaptive smoothing factor per bar
        period: estimated dominant period per bar
    """
    n = len(src)
    alpha_out = np.full(n, slow_limit)
    period_out = np.full(n, 15.0)
    
    adj = 1.665  # Fixed: 0.075 * 15.0 + 0.54
    
    smooth = np.zeros(n)
    det = np.zeros(n)
    Q1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    I2 = np.zeros(n)
    Q2 = np.zeros(n)
    phase = np.zeros(n)
    dp_smooth = np.full(n, 5.0)
    
    for i in range(n):
        s0 = src[i]
        s1 = src[i-1] if i >= 1 else 0.0
        s2 = src[i-2] if i >= 2 else 0.0
        s3 = src[i-3] if i >= 3 else 0.0
        smooth[i] = (4.0 * s0 + 3.0 * s1 + 2.0 * s2 + s3) / 10.0
        
        def _ht(arr, idx):
            v0 = arr[idx] if idx >= 0 else 0.0
            v2 = arr[idx-2] if idx >= 2 else 0.0
            v4 = arr[idx-4] if idx >= 4 else 0.0
            v6 = arr[idx-6] if idx >= 6 else 0.0
            return (0.0962 * v0 + 0.5769 * v2 - 0.5769 * v4 - 0.0962 * v6) * adj
        
        det[i] = _ht(smooth, i)
        Q1[i] = _ht(det, i)
        I1 = det[i-3] if i >= 3 else 0.0
        
        I1_arr = np.zeros(n)
        for k in range(min(i+1, n)):
            I1_arr[k] = det[k-3] if k >= 3 else 0.0
        
        jI[i] = _ht(I1_arr, i)
        jQ[i] = _ht(Q1, i)
        
        prev_I2 = I2[i-1] if i >= 1 else 0.0
        prev_Q2 = Q2[i-1] if i >= 1 else 0.0
        I2[i] = 0.2 * (I1 - jQ[i]) + 0.8 * prev_I2
        Q2[i] = 0.2 * (Q1[i] + jI[i]) + 0.8 * prev_Q2
        
        # Phase calculation with quadrant correction
        sumI = I1 + (det[i-4] if i >= 4 else 0.0)
        sumQ = Q1[i] + (Q1[i-1] if i >= 1 else 0.0)
        
        if abs(sumI) > 0.001:
            raw_phase = np.arctan(abs(sumQ / sumI)) * (180.0 / np.pi)
        else:
            raw_phase = 90.0
        
        if sumI < 0 and sumQ > 0:
            raw_phase = 180.0 - raw_phase
        elif sumI < 0 and sumQ < 0:
            raw_phase = 180.0 + raw_phase
        elif sumI > 0 and sumQ < 0:
            raw_phase = 360.0 - raw_phase
        
        phase[i] = raw_phase
        
        # Delta phase
        dp_raw = (phase[i-1] if i >= 1 else 0.0) - phase[i]
        if dp_raw > 180.0:
            dp_raw -= 360.0
        if dp_raw < -180.0:
            dp_raw += 360.0
        dp = max(1.0, min(60.0, dp_raw))
        
        prev_dps = dp_smooth[i-1] if i >= 1 else 5.0
        dp_smooth[i] = 0.33 * dp + 0.67 * prev_dps
        dp_smooth[i] = max(1.0, min(60.0, dp_smooth[i]))
        
        # Alpha from phase rate
        a = fast_limit / dp_smooth[i]
        a = max(slow_limit, min(fast_limit, a))
        
        alpha_out[i] = a
        period_out[i] = max(2.0, (2.0 / a) - 1.0)
    
    return alpha_out, period_out


# ═══════════════════════════════════════════════════════════════
#  ALPHA METHOD 3: AUTOCORRELATION PERIODOGRAM
#  From: Ehlers "Cycle Analytics for Traders"
#  Pine: f_autocorrelation() in cybercycle_v62.pine lines 507-554
# ═══════════════════════════════════════════════════════════════

def autocorrelation_alpha(src: np.ndarray,
                          min_period: int = 6,
                          max_period: int = 48,
                          avg_length: int = 3
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autocorrelation Periodogram — finds dominant cycle
    by computing autocorrelation at multiple lag periods.
    
    Returns:
        alpha: adaptive smoothing factor per bar
        period: estimated dominant period per bar
    """
    n = len(src)
    alpha_out = np.full(n, 0.07)
    period_out = np.full(n, 15.0)
    
    # High-pass filter coefficients
    a1 = (0.707 * 2.0 * np.pi) / max_period
    alpha_hp = (np.cos(a1) + np.sin(a1) - 1.0) / np.cos(a1)
    
    # Super smoother coefficients
    a1ss = np.exp(-1.414 * np.pi / min_period)
    b1ss = 2.0 * a1ss * np.cos(1.414 * np.pi / min_period)
    c2ss = b1ss
    c3ss = -a1ss * a1ss
    c1ss = 1.0 - c2ss - c3ss
    
    hp = np.zeros(n)
    filt = np.zeros(n)
    best_period = np.full(n, 15.0)
    
    step = max(1, (max_period - min_period) // 10)
    
    for i in range(n):
        # High-pass filter
        s0 = src[i]
        s1 = src[i-1] if i >= 1 else 0.0
        s2 = src[i-2] if i >= 2 else 0.0
        hp1 = hp[i-1] if i >= 1 else 0.0
        hp2 = hp[i-2] if i >= 2 else 0.0
        
        hp[i] = ((1.0 - alpha_hp / 2.0) ** 2 * (s0 - 2.0 * s1 + s2)
                 + 2.0 * (1.0 - alpha_hp) * hp1
                 - (1.0 - alpha_hp) ** 2 * hp2)
        
        # Super smoother
        f1 = filt[i-1] if i >= 1 else 0.0
        f2 = filt[i-2] if i >= 2 else 0.0
        filt[i] = (c1ss * (hp[i] + (hp[i-1] if i >= 1 else 0.0)) / 2.0
                   + c2ss * f1 + c3ss * f2)
        
        # Autocorrelation search
        if i < max_period * avg_length + 10:
            best_period[i] = best_period[i-1] if i >= 1 else 15.0
            alpha_out[i] = 2.0 / (best_period[i] + 1.0)
            period_out[i] = best_period[i]
            continue
        
        max_corr = 0.0
        best_p = best_period[i-1] if i >= 1 else 15.0
        
        for p in range(min_period, max_period + 1, step):
            sx = sy = sxx = syy = sxy = 0.0
            cnt = min(avg_length * p, 200)
            
            for j in range(cnt):
                if i - j < 0 or i - j - p < 0:
                    break
                x = filt[i - j]
                y = filt[i - j - p]
                sx += x
                sy += y
                sxx += x * x
                syy += y * y
                sxy += x * y
            
            denom = (cnt * sxx - sx * sx) * (cnt * syy - sy * sy)
            corr = (cnt * sxy - sx * sy) / np.sqrt(denom) if denom > 0 else 0.0
            
            if corr > max_corr:
                max_corr = corr
                best_p = float(p)
        
        prev_bp = best_period[i-1] if i >= 1 else 15.0
        best_period[i] = 0.25 * best_p + 0.75 * prev_bp
        best_period[i] = max(min_period, min(max_period, best_period[i]))
        
        alpha_out[i] = 2.0 / (best_period[i] + 1.0)
        period_out[i] = best_period[i]
    
    return alpha_out, period_out


# ═══════════════════════════════════════════════════════════════
#  ALPHA METHOD 4: KALMAN INNOVATION-BASED
#  Custom implementation using Kalman filter innovation
#  Pine: f_kalman() in cybercycle_v62.pine lines 561-587
# ═══════════════════════════════════════════════════════════════

def kalman_alpha(src: np.ndarray,
                 process_noise: float = 0.01,
                 meas_noise: float = 0.5,
                 alpha_fast: float = 0.5,
                 alpha_slow: float = 0.05,
                 sensitivity: float = 2.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kalman Innovation-Based Alpha.
    Uses Kalman filter innovation (prediction error) magnitude
    to adapt alpha via a sigmoid mapping.
    
    Returns:
        alpha: adaptive smoothing factor per bar
        period: estimated dominant period per bar
    """
    n = len(src)
    alpha_out = np.full(n, alpha_slow)
    period_out = np.full(n, 15.0)
    
    # Kalman state
    k_x = src[0] if n > 0 else 0.0
    k_P = 1.0
    innov_ema = 0.001
    
    for i in range(n):
        # Predict
        x_pred = k_x
        P_pred = k_P + process_noise
        
        # Innovation
        innovation = src[i] - x_pred
        S = P_pred + meas_noise
        K = P_pred / S if S > 1e-12 else 0.5
        
        # Update
        k_x = x_pred + K * innovation
        k_P = (1.0 - K) * P_pred
        
        # Innovation magnitude → alpha
        abs_innov = abs(innovation)
        innov_ema = 0.05 * abs_innov + 0.95 * innov_ema
        
        norm_innov = abs_innov / innov_ema if innov_ema > 1e-12 else 1.0
        ratio = (norm_innov - 1.0) * sensitivity
        sigmoid = 1.0 / (1.0 + np.exp(-ratio))
        
        a = alpha_slow + (alpha_fast - alpha_slow) * sigmoid
        a = max(alpha_slow, min(alpha_fast, a))
        
        alpha_out[i] = a
        period_out[i] = max(2.0, (2.0 / a) - 1.0)
    
    return alpha_out, period_out


# ═══════════════════════════════════════════════════════════════
#  ADAPTIVE CYBERCYCLE OSCILLATOR
#  From: Ehlers "Cybernetic Analysis for Stocks and Futures"
#  Pine: lines 633-644 of cybercycle_v62.pine
# ═══════════════════════════════════════════════════════════════

def cybercycle(src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Ehlers Adaptive CyberCycle oscillator.
    
    smooth = (src + 2*src[1] + 2*src[2] + src[3]) / 6
    cycle  = a1*(smooth - 2*smooth[1] + smooth[2]) + a2*cycle[1] - a3*cycle[2]
    
    where a1 = (1 - 0.5*alpha)^2
          a2 = 2*(1 - alpha)
          a3 = (1 - alpha)^2
    
    First 7 bars use simplified formula.
    """
    n = len(src)
    smooth = np.zeros(n)
    cycle = np.zeros(n)
    
    for i in range(n):
        s0 = src[i]
        s1 = src[i-1] if i >= 1 else 0.0
        s2 = src[i-2] if i >= 2 else 0.0
        s3 = src[i-3] if i >= 3 else 0.0
        smooth[i] = (s0 + 2.0 * s1 + 2.0 * s2 + s3) / 6.0
        
        if i < 7:
            # Simple differencing for startup
            cycle[i] = (s0 - 2.0 * s1 + s2) / 4.0
        else:
            a = alpha[i]
            a1 = (1.0 - 0.5 * a) ** 2
            a2 = 2.0 * (1.0 - a)
            a3 = (1.0 - a) ** 2
            
            sm0 = smooth[i]
            sm1 = smooth[i-1] if i >= 1 else 0.0
            sm2 = smooth[i-2] if i >= 2 else 0.0
            c1 = cycle[i-1] if i >= 1 else 0.0
            c2 = cycle[i-2] if i >= 2 else 0.0
            
            cycle[i] = a1 * (sm0 - 2.0 * sm1 + sm2) + a2 * c1 - a3 * c2
    
    return cycle


# ═══════════════════════════════════════════════════════════════
#  iTREND — Ehlers Instantaneous Trendline
#  Pine: lines 648-653 of cybercycle_v62.pine
# ═══════════════════════════════════════════════════════════════

def itrend(close: np.ndarray, alpha: float = 0.07) -> np.ndarray:
    """
    Ehlers Instantaneous Trendline.
    
    iTrend = (a - a²/4)*close + 0.5*a²*close[1] - (a - 0.75*a²)*close[2]
             + 2*(1-a)*iTrend[1] - (1-a)²*iTrend[2]
    """
    n = len(close)
    it = np.zeros(n)
    a = alpha
    
    # Startup: use close directly
    for i in range(min(3, n)):
        it[i] = close[i]
    
    for i in range(3, n):
        c0 = close[i]
        c1 = close[i-1]
        c2 = close[i-2]
        t1 = it[i-1]
        t2 = it[i-2]
        
        it[i] = ((a - a*a/4.0) * c0
                 + 0.5 * a*a * c1
                 - (a - 0.75 * a*a) * c2
                 + 2.0 * (1.0 - a) * t1
                 - (1.0 - a) ** 2 * t2)
    
    return it


# ═══════════════════════════════════════════════════════════════
#  FISHER TRANSFORM (on CyberCycle values)
#  Pine: lines 660-668 of cybercycle_v62.pine
# ═══════════════════════════════════════════════════════════════

def fisher_transform(cycle: np.ndarray, lookback: int = 10) -> np.ndarray:
    """
    Fisher Transform applied to CyberCycle oscillator.
    Normalizes cycle to -1..+1, then applies inverse Fisher.
    """
    n = len(cycle)
    fisher = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - lookback + 1)
        window = cycle[start:i+1]
        
        fH = np.max(window) if len(window) > 0 else 0.0
        fL = np.min(window) if len(window) > 0 else 0.0
        fR = fH - fL
        
        if fR != 0:
            fV = 2.0 * ((cycle[i] - fL) / fR - 0.5)
        else:
            fV = 0.0
        
        fV = max(-0.999, min(0.999, fV))
        raw = 0.5 * np.log((1.0 + fV) / (1.0 - fV))
        
        prev = fisher[i-1] if i >= 1 else 0.0
        fisher[i] = 0.5 * raw + 0.5 * prev
    
    return fisher


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE SCORE (replicates Pine's signal scoring)
#  Pine: lines 706-725 of cybercycle_v62.pine
# ═══════════════════════════════════════════════════════════════

def compute_confidence(is_buy: bool,
                       bull_cross: bool, bear_cross: bool,
                       bull_trend: bool, bear_trend: bool,
                       in_ob: bool, in_os: bool,
                       volume_ok: bool,
                       fish_rising: bool, fish_falling: bool,
                       momentum_positive: bool, momentum_negative: bool,
                       htf_align_buy: bool, htf_align_sell: bool,
                       use_trend: bool, use_vol: bool) -> float:
    """
    Replicates the confidence scoring system from cybercycle_v62.pine.
    Each component contributes a fixed amount to the total score (max 100).
    """
    conf = 0.0
    
    if is_buy:
        conf += 20.0 if bull_cross else 0.0
        conf += 15.0 if (bull_trend if use_trend else True) else 0.0
        conf += 15.0 if in_os else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_rising else 0.0
        conf += 10.0 if momentum_positive else 0.0
        conf += 15.0 if htf_align_buy else 0.0
    else:
        conf += 20.0 if bear_cross else 0.0
        conf += 15.0 if (bear_trend if use_trend else True) else 0.0
        conf += 15.0 if in_ob else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_falling else 0.0
        conf += 10.0 if momentum_negative else 0.0
        conf += 15.0 if htf_align_sell else 0.0
    
    return min(conf, 100.0)
