"""
CryptoLab — CyberCycle v6.2 Strategy (Pine-Faithful)
=====================================================
FIXED VERSION: 1:1 match with Pine Script "Ehlers CyberCycle Obrero"

Changes vs previous cybercyclev3.py:
─────────────────────────────────────────────────────────────────
 1. CONFIDENCE SCORING — Exact Pine weights:
      Cross=20, iTrend=15, OB/OS=15, Volume=15, Fisher=10,
      Momentum=10, HTF=15 → Max=100
    OLD (broken): Cross=20, iTrend=25, Fisher=15, OB/OS=15,
      HTF=15, Momentum=10, Volume=10 → Max=110 (!)

 2. HARD GATES — Pine applies these as AND conditions OUTSIDE
    confidence, not just as score components:
      • use_trend: bullTrend (buy) / bearTrend (sell)
      • use_htf:   htfAlignBuy / htfAlignSell
    These are now hard gates in generate_signal, matching:
      bool buySignal = bullCross and buyConf >= minConf
                       and barFilter
                       and (eUseTrend ? bullTrend : true)
                       and htfAlignBuy

 3. HTF FILTER — Real 4H data instead of EMA(40) proxy:
      Pine: request.security("240", hl2) vs request.security("240", ema(close,10))
      Python now: resample 1h→4h OHLCV, compute hl2_4h vs ema(close_4h, 10),
      then map back to 1h bars.

 4. VOLUME FILTER — Now matches Pine's behavior:
      Pine: volumeOK feeds into confidence score AND is also
      implicitly part of the hard gate through the confidence sum.
      The weights now match exactly so the same confidence threshold
      produces the same filtering effect.
─────────────────────────────────────────────────────────────────

Ehlers Adaptive CyberCycle with:
- 4 alpha methods (Homodyne/MAMA/Autocorrelation/Kalman) + Manual
- iTrend filter
- Fisher Transform
- Volume filter
- HTF 4H filter (real resampled data)
- Confidence scoring system (0-100) — exact Pine match
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    homodyne_alpha, mama_alpha, autocorrelation_alpha,
    kalman_alpha, cybercycle, itrend, fisher_transform,
    compute_confidence
)
from indicators.common import ema, sma, atr, crossover, crossunder, volume_ratio


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE SCORING — Exact Pine Script match
# ═══════════════════════════════════════════════════════════════
#
#  Pine Script lines (buyConf/sellConf):
#    bullCross                         ? 20 : 0
#    (eUseTrend ? bullTrend : true)    ? 15 : 0
#    inOS                              ? 15 : 0
#    (eUseVol ? volumeOK : true)       ? 15 : 0
#    fishRising                        ? 10 : 0
#    momentum3 > 0                     ? 10 : 0
#    htfAlignBuy                       ? 15 : 0
#    ─────────────────────────────────────
#    Max: 100
#
#  NOTE: In Pine, the trend and volume terms inside the confidence
#  scoring use the ternary pattern: (use_X ? condition : true) ? pts : 0
#  This means when use_trend=false, iTrend always contributes 15 pts.
#  When use_trend=true, it only contributes if trend is aligned.
#  Same for volume.
#
#  SEPARATELY, Pine applies hard gates AFTER confidence:
#    buySignal = bullCross AND conf >= min AND barFilter
#                AND (use_trend ? bullTrend : true)
#                AND htfAlignBuy
#  So trend and HTF are BOTH in confidence AND as hard gates.
# ═══════════════════════════════════════════════════════════════

def compute_confidence_pine(
    is_buy: bool,
    bull_cross: bool, bear_cross: bool,
    bull_trend: bool, bear_trend: bool,
    in_ob: bool, in_os: bool,
    volume_ok: bool,
    fish_rising: bool, fish_falling: bool,
    momentum_positive: bool, momentum_negative: bool,
    htf_align_buy: bool, htf_align_sell: bool,
    use_trend: bool, use_vol: bool,
) -> float:
    """
    Exact replica of Pine Script confidence scoring.

    Weights (max 100):
        Cross signal:    20
        iTrend:          15  (ternary: use_trend ? aligned : true)
        OB/OS zone:      15
        Volume:          15  (ternary: use_vol ? volumeOK : true)
        Fisher:          10
        Momentum:        10
        HTF:             15
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


# ═══════════════════════════════════════════════════════════════
#  HTF RESAMPLING — Build real 4H bars from 1H data
# ═══════════════════════════════════════════════════════════════

def resample_to_htf(timestamps, opens, highs, lows, closes, volumes,
                    src_tf_seconds=3600, htf_tf_seconds=14400):
    """
    Resample lower-timeframe OHLCV to higher-timeframe bars and map
    the HTF values back to each LTF bar.

    This replicates TradingView's request.security() with
    lookahead=barmerge.lookahead_off — each 1h bar sees the LAST
    COMPLETED 4h bar's values, not the current in-progress one.

    Returns:
        htf_hl2:    array[n] — hl2 of the last completed 4H bar
        htf_ema10:  array[n] — EMA(close, 10) of 4H bars, mapped to 1H
    """
    n = len(timestamps)
    ratio = htf_tf_seconds // src_tf_seconds  # e.g., 4 for 1h→4h

    if ratio <= 1:
        # Same or higher TF — just compute directly
        htf_hl2 = (highs + lows) / 2.0
        htf_ema_close = ema(closes, 10)
        return htf_hl2, htf_ema_close

    # ── Step 1: Build 4H bars from 1H bars ──
    # Group by HTF bar boundaries using timestamp
    htf_period_ms = htf_tf_seconds * 1000
    bar_group = (timestamps // htf_period_ms).astype(np.int64)

    # Find unique groups and their boundaries
    unique_groups = np.unique(bar_group)
    n_htf = len(unique_groups)

    htf_open = np.zeros(n_htf)
    htf_high = np.zeros(n_htf)
    htf_low = np.zeros(n_htf)
    htf_close = np.zeros(n_htf)
    htf_vol = np.zeros(n_htf)
    htf_ts = np.zeros(n_htf)

    # Map each 1h bar to its HTF group index
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    bar_htf_idx = np.array([group_to_idx[g] for g in bar_group])

    for gi, group in enumerate(unique_groups):
        mask = bar_group == group
        idx_in_group = np.where(mask)[0]
        htf_open[gi] = opens[idx_in_group[0]]
        htf_high[gi] = np.max(highs[idx_in_group])
        htf_low[gi] = np.min(lows[idx_in_group])
        htf_close[gi] = closes[idx_in_group[-1]]
        htf_vol[gi] = np.sum(volumes[idx_in_group])
        htf_ts[gi] = timestamps[idx_in_group[0]]

    # ── Step 2: Compute HTF indicators ──
    htf_hl2_bars = (htf_high + htf_low) / 2.0
    htf_ema10_bars = ema(htf_close, 10)

    # ── Step 3: Map back to 1H with lookahead_off ──
    # Each 1h bar sees the LAST COMPLETED 4H bar, not the current one.
    # A 4H bar is "completed" when the next 4H group starts.
    # For bars within the same 4H group, they see the PREVIOUS 4H bar.
    result_hl2 = np.zeros(n)
    result_ema10 = np.zeros(n)

    for i in range(n):
        htf_idx = bar_htf_idx[i]

        # Check if this is the last 1h bar in its 4H group
        # (i.e., next bar belongs to next group, or this is the last bar)
        is_last_in_group = (i == n - 1) or (bar_htf_idx[i + 1] != htf_idx)

        if is_last_in_group:
            # Last bar of the 4H candle: this bar completes the 4H bar,
            # so it CAN see the current 4H bar (Pine's behavior at bar close)
            result_hl2[i] = htf_hl2_bars[htf_idx]
            result_ema10[i] = htf_ema10_bars[htf_idx]
        else:
            # Mid-group: see the PREVIOUS completed 4H bar
            if htf_idx > 0:
                result_hl2[i] = htf_hl2_bars[htf_idx - 1]
                result_ema10[i] = htf_ema10_bars[htf_idx - 1]
            else:
                # First 4H group, no previous — use current as fallback
                result_hl2[i] = htf_hl2_bars[0]
                result_ema10[i] = htf_ema10_bars[0]

    return result_hl2, result_ema10


class CyberCycleStrategyhtf(IStrategy):

    def name(self) -> str:
        return "CyberCycle v6.2"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # Alpha method
            ParamDef('alpha_method', 'categorical', 'manual',
                     options=['kalman', 'manual']),
            ParamDef('manual_alpha', 'float', 0.42, 0.05, 0.80, 0.01),
            ParamDef('alpha_floor', 'float', 0.0, 0.0, 0.50, 0.01),

            # Kalman params
            ParamDef('kal_process_noise', 'float', 0.01, 0.001, 0.2, 0.005),
            ParamDef('kal_meas_noise', 'float', 0.5, 0.05, 3.0, 0.1),
            ParamDef('kal_alpha_fast', 'float', 0.5, 0.2, 0.8, 0.05),
            ParamDef('kal_alpha_slow', 'float', 0.05, 0.01, 0.2, 0.01),
            ParamDef('kal_sensitivity', 'float', 2.0, 0.5, 5.0, 0.5),

            # Signal params
            ParamDef('itrend_alpha', 'float', 0.09, 0.01, 0.30, 0.01),
            ParamDef('trigger_ema', 'int', 9, 3, 30),
            ParamDef('min_bars', 'int', 16, 12, 50),
            ParamDef('confidence_min', 'float', 75.0, 30.0, 95.0, 5.0),
            ParamDef('ob_level', 'float', 1.5, 0.3, 3.0, 0.1),
            ParamDef('os_level', 'float', -1.5, -3.0, -0.3, 0.1),

            # Filters
            ParamDef('use_trend', 'bool', True),
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 1.5, 0.5, 5.0, 0.1),
            ParamDef('use_htf', 'bool', False),

            # ═══════════════════════════════════════════════════════
            #  SL/TP MODE SELECTOR
            # ═══════════════════════════════════════════════════════
            ParamDef('sltp_type', 'categorical', 'sltp_fixed',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── Risk params: ATR mode (slatr_tprr) ──
            ParamDef('leverage',    'float', 20.0, 5.0, 45.0, 5.0),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5,  4.0, 0.1),
            ParamDef('tp1_rr',   'float', 2.0, 0.5,  5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1,  0.9, 0.05),
            ParamDef('tp2_rr',   'float', 3.0, 1.0, 10.0, 0.25),

            # ── Risk params: FIXED mode (sltp_fixed) ──
            ParamDef('sl_fixed_pct',   'float', 2.0, 0.3, 5.0, 0.1),
            ParamDef('tp1_fixed_pct',  'float', 1.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.35, 0.1, 0.9, 0.05),
            ParamDef('tp2_fixed_pct',  'float', 2.0, 1.0, 10.0, 0.5),

            # ── Break-even ──
            ParamDef('be_pct', 'float', 1.0, 0.8, 4.5, 0.1),

            # ── Trailing stop ──
            ParamDef('use_trailing', 'bool', True),
            ParamDef('trail_activate_pct', 'float', 2.5, 0.6, 6.0, 0.25),
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.5,  3.0, 0.10),

            # ── Signal control ──
            ParamDef('close_on_signal', 'bool', True),
            ParamDef('max_signals_per_day', 'int', 0, 0, 10),
        ]

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS — Pine-faithful with real HTF resampling
    # ─────────────────────────────────────────────────────────────

    def calculate_indicators(self, data: dict) -> dict:
        """
        Calculate all CyberCycle indicators.

        HTF filter now uses REAL 4H resampled data instead of
        EMA(40) proxy. This matches Pine's request.security("240", ...).
        """
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'manual')

        # ── Alpha computation (only selected method) ──
        if method == 'homodyne':
            alpha, period = homodyne_alpha(
                src,
                self.get_param('hd_min_period', 3.0),
                self.get_param('hd_max_period', 40.0))
        elif method == 'mama':
            alpha, period = mama_alpha(
                src,
                self.get_param('mama_fast', 0.5),
                self.get_param('mama_slow', 0.05))
        elif method == 'autocorrelation':
            alpha, period = autocorrelation_alpha(
                src,
                self.get_param('ac_min_period', 6),
                self.get_param('ac_max_period', 48),
                self.get_param('ac_avg_length', 3))
        elif method == 'kalman':
            alpha, period = kalman_alpha(
                src,
                self.get_param('kal_process_noise', 0.01),
                self.get_param('kal_meas_noise', 0.5),
                self.get_param('kal_alpha_fast', 0.5),
                self.get_param('kal_alpha_slow', 0.05),
                self.get_param('kal_sensitivity', 2.0))
        else:  # manual
            manual_a = self.get_param('manual_alpha', 0.35)
            alpha = np.full(n, manual_a)
            period = np.full(n, (2.0 / manual_a) - 1.0)

        # Apply alpha floor
        floor = self.get_param('alpha_floor', 0.0)
        if floor > 0:
            alpha = np.maximum(alpha, floor)

        # CyberCycle oscillator
        cycle = cybercycle(src, alpha)

        # Trigger (EMA of cycle)
        trigger = ema(cycle, self.get_param('trigger_ema', 14))

        # iTrend
        it = itrend(close, self.get_param('itrend_alpha', 0.07))
        bull_trend = np.zeros(n, dtype=bool)
        bear_trend = np.zeros(n, dtype=bool)
        for i in range(2, n):
            bull_trend[i] = it[i] > it[i - 2]
            bear_trend[i] = it[i] < it[i - 2]

        # Fisher Transform
        fisher = fisher_transform(cycle, 10)
        fish_rising = np.zeros(n, dtype=bool)
        fish_falling = np.zeros(n, dtype=bool)
        for i in range(1, n):
            fish_rising[i] = fisher[i] > fisher[i - 1]
            fish_falling[i] = fisher[i] < fisher[i - 1]

        # Volume filter — Pine: volRatio = volume / sma(volume, 20)
        #                 Pine: volumeOK = not eUseVol or volRatio >= eVolMult
        vol_ratio = volume_ratio(vol, 20)
        vol_mult = self.get_param('volume_mult', 2.0)
        use_vol = self.get_param('use_volume', True)
        volume_ok = (~use_vol) | (vol_ratio >= vol_mult)

        # Cross signals
        bull_cross = crossover(cycle, trigger)
        bear_cross = crossunder(cycle, trigger)

        # OB/OS zones
        ob_level = self.get_param('ob_level', 1.5)
        os_level = self.get_param('os_level', -1.5)
        in_ob = cycle > ob_level
        in_os = cycle < os_level

        # Momentum
        momentum3 = np.zeros(n)
        for i in range(3, n):
            momentum3[i] = cycle[i] - cycle[i - 3]

        # ATR for SL/TP
        atr_vals = atr(data['high'], data['low'], close, 14)

        # ══════════════════════════════════════════════════════════
        #  HTF FILTER — Real 4H resampled data
        # ══════════════════════════════════════════════════════════
        #  Pine:
        #    float htfSrc = request.security(syminfo.tickerid, "240", hl2, ...)
        #    float htfCC  = request.security(syminfo.tickerid, "240",
        #                                     ta.ema(close, 10), ...)
        #    bool htfBullish = not eUseHTF or htfSrc > htfCC
        #    bool htfBearish = not eUseHTF or htfSrc < htfCC
        # ══════════════════════════════════════════════════════════
        use_htf = self.get_param('use_htf', True)

        if use_htf:
            htf_hl2, htf_ema10 = resample_to_htf(
                data['timestamp'],
                data['open'], data['high'], data['low'],
                data['close'], data['volume'],
                src_tf_seconds=3600,    # 1h
                htf_tf_seconds=14400,   # 4h
            )
            htf_bull = htf_hl2 > htf_ema10
            htf_bear = htf_hl2 < htf_ema10
        else:
            htf_bull = np.ones(n, dtype=bool)
            htf_bear = np.ones(n, dtype=bool)

        return {
            'cycle': cycle,
            'trigger': trigger,
            'alpha': alpha,
            'period': period,
            'itrend': it,
            'bull_trend': bull_trend,
            'bear_trend': bear_trend,
            'fisher': fisher,
            'fish_rising': fish_rising,
            'fish_falling': fish_falling,
            'vol_ratio': vol_ratio,
            'volume_ok': volume_ok,
            'bull_cross': bull_cross,
            'bear_cross': bear_cross,
            'in_ob': in_ob,
            'in_os': in_os,
            'momentum3': momentum3,
            'atr': atr_vals,
            'htf_align_buy': htf_bull,
            'htf_align_sell': htf_bear,
        }

    # ─────────────────────────────────────────────────────────────
    #  SL/TP CALCULATION — DUAL MODE
    # ─────────────────────────────────────────────────────────────

    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        """SL = entry ∓ ATR × mult, TP = entry ± risk × R:R"""
        sl_dist = atr_val * self.get_param('sl_atr_mult', 1.5)
        sl = entry - direction * sl_dist
        risk = sl_dist

        tp1_rr = self.get_param('tp1_rr', 2.0)
        tp2_rr = self.get_param('tp2_rr', 3.0)
        tp1_size = self.get_param('tp1_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        tp1 = entry + direction * risk * tp1_rr
        tp2 = entry + direction * risk * tp2_rr

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'slatr_tprr',
        }

    def _compute_sltp_fixed(self, entry: float, direction: int) -> dict:
        """SL and TP as fixed % of entry price."""
        sl_pct = self.get_param('sl_fixed_pct', 2.5) / 100.0
        tp1_pct = self.get_param('tp1_fixed_pct', 3.0) / 100.0
        tp2_pct = self.get_param('tp2_fixed_pct', 4.5) / 100.0
        tp1_size = self.get_param('tp1_fixed_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        sl = entry * (1.0 - direction * sl_pct)
        tp1 = entry * (1.0 + direction * tp1_pct)
        tp2 = entry * (1.0 + direction * tp2_pct)

        sl_dist = abs(entry - sl)
        risk = sl_dist

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'risk': risk,
            'mode': 'sltp_fixed',
        }

    # ─────────────────────────────────────────────────────────────
    #  SIGNAL GENERATION — Exact Pine Script match
    # ─────────────────────────────────────────────────────────────
    #
    #  Pine Script signal logic:
    #
    #    bool buySignal  = bullCross
    #                      and buyConf  >= minConf
    #                      and barFilter
    #                      and (eUseTrend ? bullTrend : true)
    #                      and htfAlignBuy
    #
    #    bool sellSignal = bearCross
    #                      and sellConf >= minConf
    #                      and barFilter
    #                      and (eUseTrend ? bearTrend : true)
    #                      and htfAlignSell
    #
    #  Note: bullCross is already required for buyConf to be > 0
    #  (cross=20 is the base), but it's explicit in Pine as an AND.
    #  The trend and HTF are HARD GATES separate from confidence.
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate CyberCycle signal at bar idx — exact Pine match."""
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        # Pine: if not (bullCross or bearCross) → no signal possible
        if not (bull_cross or bear_cross):
            return None

        # ── Bar filter (Pine: bar_index - lastSigBar >= eMinBars) ──
        min_bars = self.get_param('min_bars', 24)
        if idx - self._last_signal_bar < min_bars:
            return None

        use_trend = self.get_param('use_trend', True)
        use_vol = self.get_param('use_volume', True)
        use_htf = self.get_param('use_htf', True)

        is_buy = bull_cross

        # ── Confidence scoring (exact Pine weights) ──
        conf = compute_confidence_pine(
            is_buy=is_buy,
            bull_cross=bull_cross,
            bear_cross=bear_cross,
            bull_trend=ind['bull_trend'][idx],
            bear_trend=ind['bear_trend'][idx],
            in_ob=ind['in_ob'][idx],
            in_os=ind['in_os'][idx],
            volume_ok=ind['volume_ok'][idx],
            fish_rising=ind['fish_rising'][idx],
            fish_falling=ind['fish_falling'][idx],
            momentum_positive=ind['momentum3'][idx] > 0,
            momentum_negative=ind['momentum3'][idx] < 0,
            htf_align_buy=ind['htf_align_buy'][idx],
            htf_align_sell=ind['htf_align_sell'][idx],
            use_trend=use_trend,
            use_vol=use_vol,
        )

        # ── Confidence gate ──
        confidence_min = self.get_param('confidence_min', 80.0)
        if conf < confidence_min:
            return None

        # ══════════════════════════════════════════════════════════
        #  HARD GATES — These are AND conditions in Pine,
        #  separate from the confidence score.
        # ══════════════════════════════════════════════════════════

        # Pine: and (eUseTrend ? bullTrend : true)
        if use_trend:
            if is_buy and not ind['bull_trend'][idx]:
                return None
            if not is_buy and not ind['bear_trend'][idx]:
                return None

        # Pine: and htfAlignBuy  (which already includes "not eUseHTF or ...")
        # htf_align_buy/sell already encode use_htf logic in calculate_indicators
        if is_buy and not ind['htf_align_buy'][idx]:
            return None
        if not is_buy and not ind['htf_align_sell'][idx]:
            return None

        # ── Signal accepted ──
        self._last_signal_bar = idx

        direction = 1 if is_buy else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]
        if atr_val <= 0:
            return None

        # ── SL/TP ──
        sltp_type = self.get_param('sltp_type', 'slatr_tprr')
        if sltp_type == 'sltp_fixed':
            sltp = self._compute_sltp_fixed(entry, direction)
        else:
            sltp = self._compute_sltp_atr_rr(entry, direction, atr_val)

        sl = sltp['sl']
        tp_levels = sltp['tp_levels']
        tp_sizes = sltp['tp_sizes']
        sl_dist = sltp['sl_dist']

        # ── Break-even ──
        be_pct = self.get_param('be_pct', 1.5)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ──
        use_trailing = self.get_param('use_trailing', True)
        if use_trailing:
            trail_activate_pct = self.get_param('trail_activate_pct', 2.5)
            trail_pullback_pct = self.get_param('trail_pullback_pct', 1.0)
            trailing_distance = entry * (trail_pullback_pct / 100.0)

            trail_activation_price = entry + direction * entry * (trail_activate_pct / 100.0)
            if be_pct > 0.0:
                dist_be = abs(be_trigger - entry)
                dist_trail = abs(trail_activation_price - entry)
                be_trigger = be_trigger if dist_be <= dist_trail else trail_activation_price
            else:
                be_trigger = trail_activation_price
        else:
            trailing_distance = 0.0

        return Signal(
            direction=direction,
            confidence=conf,
            entry_price=entry,
            sl_price=sl,
            tp_levels=tp_levels,
            tp_sizes=tp_sizes,
            leverage=self.get_param('leverage', 15.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_distance,
            metadata={
                'close_on_signal': self.get_param('close_on_signal', True),
                'max_signals_per_day': self.get_param('max_signals_per_day', 0),
                'alpha_method': self.get_param('alpha_method'),
                'alpha': ind['alpha'][idx],
                'period': ind['period'][idx],
                'cycle': ind['cycle'][idx],
                'fisher': ind['fisher'][idx],
                'sltp_mode': sltp_type,
                'sl_dist_atr': sl_dist / atr_val if atr_val > 0 else 0,
                'tp1': tp_levels[0] if tp_levels else 0,
                'tp2': tp_levels[1] if len(tp_levels) > 1 else 0,
                'be_pct': be_pct,
                'trail_pct': self.get_param('trail_pullback_pct', 1.0) if use_trailing else 0,
                # ── Datos para intrabar entry ──
                'intrabar_entry': True,
                'signal_type': 'bull_cross' if is_buy else 'bear_cross',
                'cycle_val': ind['cycle'][idx],
                'trigger_val': ind['trigger'][idx],
                'cycle_prev': ind['cycle'][idx - 1] if idx > 0 else 0,
                'trigger_prev': ind['trigger'][idx - 1] if idx > 0 else 0,
            }
        )

    def create_incremental_processor(self, detail_tf_ratio: int = 60):
        """Create incremental processor for IntrabarBacktestEngine."""
        from indicators.incremental_ehlers import IncrementalCyberCycleV3
        return IncrementalCyberCycleV3(self.params, detail_tf_ratio=detail_tf_ratio)