"""
CryptoLab — CyberCycle NoHTF (Pine-Parity)

Maximum parity with "Ehlers Cyber Cycle SIGNALS" PineScript.

═══════════════════════════════════════════════════════════════
 ORIGEN: Ehlers Cyber Cycle SIGNALS (overlay, sin HTF)
═══════════════════════════════════════════════════════════════

 CONFIDENCE WEIGHTS — Pine-exact (f_confidence):
    Cross signal:  25  — bullCross / bearCross
    iTrend:        20  — trend alignment
    OB/OS zone:    20  — oversold (buy) / overbought (sell)
    Volume:        15  — volume confirmation
    Fisher:        10  — Fisher transform direction
    Momentum:      10  — 3-bar cycle momentum
    ───────────────────
    Total max:    100  (SIN HTF)

 SIGNAL LOGIC — Pine-exact:
    buySignal = bullCross
                AND buyConf >= minConf
                AND barFilter
                AND ((useTrendFilter AND bullTrend) OR NOT useTrendFilter)

 DIFERENCIAS vs cybercyclev3.py (Pine-Parity con HTF):
    - Sin HTF filter (ni en confidence ni como hard gate)
    - Pesos de confianza distintos: 25/20/20/15/10/10 vs 20/15/15/15/10/10/15
    - Cross vale 25 pts (vs 20 en v6.2 con HTF)
    - iTrend vale 20 pts (vs 15 en v6.2 con HTF)
    - OB/OS vale 20 pts (vs 15 en v6.2 con HTF)

 RISK MANAGEMENT — idéntico a cybercyclev3.py:
    - Dual SL/TP: slatr_tprr | sltp_fixed
    - Break-even con be_pct
    - Trailing stop con activación + pullback
    - close_on_signal, max_signals_per_day

 ALPHA METHODS — solo Kalman + Manual (los que pasan gates)
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    kalman_alpha, cybercycle, itrend, fisher_transform,
)
from indicators.common import (
    ema, sma, atr, crossover, crossunder, volume_ratio,
)


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE — Pine-exact weights (no HTF)
#
#  Pine f_confidence(isBuy):
#    bullCross                                          → 25
#    (useTrendFilter and bullTrend) or not useTrendFilter → 20
#    inOversold                                          → 20
#    volumeOK                                            → 15
#    fisher > fisherPrev                                 → 10
#    momentum > 0                                        → 10
#                                                        ────
#                                                 total: 100
# ═══════════════════════════════════════════════════════════════

def compute_confidence_nohtf(
    is_buy: bool,
    bull_cross: bool, bear_cross: bool,
    bull_trend: bool, bear_trend: bool,
    in_ob: bool, in_os: bool,
    volume_ok: bool,
    fish_rising: bool, fish_falling: bool,
    momentum_positive: bool, momentum_negative: bool,
    use_trend: bool, use_vol: bool,
) -> float:
    """
    Confidence scoring — identical to "Ehlers Cyber Cycle SIGNALS" Pine.

    NO HTF component. Weights redistributed:
        Cross signal:  25  — cycle × trigger crossover detected
        iTrend:        20  — trend alignment
        OB/OS zone:    20  — oversold (buy) / overbought (sell)
        Volume:        15  — volume confirmation
        Fisher:        10  — Fisher transform direction
        Momentum:      10  — 3-bar cycle momentum
        ───────────────────
        Total max:    100
    """
    conf = 0.0

    if is_buy:
        conf += 25.0 if bull_cross else 0.0
        conf += 20.0 if (bull_trend if use_trend else True) else 0.0
        conf += 20.0 if in_os else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_rising else 0.0
        conf += 10.0 if momentum_positive else 0.0
    else:
        conf += 25.0 if bear_cross else 0.0
        conf += 20.0 if (bear_trend if use_trend else True) else 0.0
        conf += 20.0 if in_ob else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_falling else 0.0
        conf += 10.0 if momentum_negative else 0.0

    return min(conf, 100.0)


class CyberCycleNoHTFStrategy(IStrategy):

    def name(self) -> str:
        return "CyberCycle NoHTF"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # ── Alpha method (solo Kalman + Manual) ──────────────
            ParamDef('alpha_method', 'categorical', 'manual',
                     options=['kalman', 'manual']),
            ParamDef('manual_alpha', 'float', 0.42, 0.05, 0.80, 0.01),
            ParamDef('alpha_floor', 'float', 0.0, 0.0, 0.50, 0.01),

            # ── Kalman params ────────────────────────────────────
            ParamDef('kal_process_noise', 'float', 0.01, 0.001, 0.2, 0.005),
            ParamDef('kal_meas_noise', 'float', 0.5, 0.05, 3.0, 0.1),
            ParamDef('kal_alpha_fast', 'float', 0.5, 0.2, 0.8, 0.05),
            ParamDef('kal_alpha_slow', 'float', 0.05, 0.01, 0.2, 0.01),
            ParamDef('kal_sensitivity', 'float', 2.0, 0.5, 5.0, 0.5),

            # ── Signal params ────────────────────────────────────
            ParamDef('itrend_alpha', 'float', 0.09, 0.01, 0.30, 0.01),
            ParamDef('trigger_ema', 'int', 9, 3, 30),
            ParamDef('min_bars', 'int', 16, 5, 50),
            ParamDef('confidence_min', 'float', 75.0, 30.0, 95.0, 5.0),
            ParamDef('ob_level', 'float', 1.5, 0.3, 3.0, 0.1),
            ParamDef('os_level', 'float', -1.5, -3.0, -0.3, 0.1),

            # ── Filters ──────────────────────────────────────────
            ParamDef('use_trend', 'bool', True),
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 1.5, 0.5, 5.0, 0.1),

            # ── SL/TP mode ───────────────────────────────────────
            ParamDef('sltp_type', 'categorical', 'sltp_fixed',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── Risk params: ATR mode (slatr_tprr) ──────────────
            ParamDef('leverage', 'float', 20.0, 5.0, 40.0, 5.0),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5, 4.0, 0.1),
            ParamDef('tp1_rr', 'float', 2.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1, 0.9, 0.05),
            ParamDef('tp2_rr', 'float', 3.0, 1.0, 10.0, 0.25),

            # ── Risk params: FIXED mode (sltp_fixed) ────────────
            ParamDef('sl_fixed_pct', 'float', 2.0, 0.3, 5.0, 0.1),
            ParamDef('tp1_fixed_pct', 'float', 2.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.35, 0.1, 0.9, 0.05),
            ParamDef('tp2_fixed_pct', 'float', 3.0, 1.0, 10.0, 0.5),

            # ── Break-even ───────────────────────────────────────
            ParamDef('be_pct', 'float', 1.5, 0.0, 4.5, 0.1),

            # ── Trailing stop ────────────────────────────────────
            ParamDef('use_trailing', 'bool', True),
            ParamDef('trail_activate_pct', 'float', 2.5, 0.6, 6.0, 0.25),
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.5, 3.0, 0.10),

            # ── Signal control ───────────────────────────────────
            ParamDef('close_on_signal', 'bool', True),
            ParamDef('max_signals_per_day', 'int', 0, 0, 10),
        ]

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────────

    def calculate_indicators(self, data: dict) -> dict:
        """
        Calculate all CyberCycle indicators — sin HTF.

        Solo computa Kalman o Manual alpha.
        """
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'kalman')

        # ── Alpha: solo el seleccionado ─────────────────────────
        if method == 'kalman':
            alpha, period = kalman_alpha(
                src,
                self.get_param('kal_process_noise', 0.01),
                self.get_param('kal_meas_noise', 0.5),
                self.get_param('kal_alpha_fast', 0.5),
                self.get_param('kal_alpha_slow', 0.05),
                self.get_param('kal_sensitivity', 2.0),
            )
        else:  # manual
            manual_a = self.get_param('manual_alpha', 0.07)
            alpha = np.full(n, manual_a)
            period = np.full(n, (2.0 / manual_a) - 1.0)

        # Apply alpha floor
        floor = self.get_param('alpha_floor', 0.0)
        if floor > 0:
            alpha = np.maximum(alpha, floor)

        # ── CyberCycle oscillator ───────────────────────────────
        cycle = cybercycle(src, alpha)

        # ── Trigger (EMA of cycle) ──────────────────────────────
        trigger = ema(cycle, self.get_param('trigger_ema', 9))

        # ── iTrend (Pine usa close, no hl2) ─────────────────────
        it = itrend(close, self.get_param('itrend_alpha', 0.07))
        bull_trend = np.zeros(n, dtype=bool)
        bear_trend = np.zeros(n, dtype=bool)
        for i in range(2, n):
            bull_trend[i] = it[i] > it[i - 2]
            bear_trend[i] = it[i] < it[i - 2]

        # ── Fisher Transform ────────────────────────────────────
        fisher = fisher_transform(cycle, 10)
        fish_rising = np.zeros(n, dtype=bool)
        fish_falling = np.zeros(n, dtype=bool)
        for i in range(1, n):
            fish_rising[i] = fisher[i] > fisher[i - 1]
            fish_falling[i] = fisher[i] < fisher[i - 1]

        # ── Volume filter ───────────────────────────────────────
        vol_rat = volume_ratio(vol, 20)
        vol_mult = self.get_param('volume_mult', 1.2)
        use_vol_flag = self.get_param('use_volume', True)
        volume_ok = (~use_vol_flag) | (vol_rat >= vol_mult)

        # ── Cross signals ───────────────────────────────────────
        bull_cross = crossover(cycle, trigger)
        bear_cross = crossunder(cycle, trigger)

        # ── OB/OS zones ─────────────────────────────────────────
        ob_level = self.get_param('ob_level', 1.5)
        os_level = self.get_param('os_level', -1.5)
        in_ob = cycle > ob_level
        in_os = cycle < os_level

        # ── Momentum 3 bars ─────────────────────────────────────
        momentum3 = np.zeros(n)
        for i in range(3, n):
            momentum3[i] = cycle[i] - cycle[i - 3]

        # ── ATR for SL/TP ───────────────────────────────────────
        atr_vals = atr(data['high'], data['low'], close, 14)

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
            'vol_ratio': vol_rat,
            'volume_ok': volume_ok,
            'bull_cross': bull_cross,
            'bear_cross': bear_cross,
            'in_ob': in_ob,
            'in_os': in_os,
            'momentum3': momentum3,
            'atr': atr_vals,
            # Diagnostics
            'alpha_kl': alpha if method == 'kalman' else np.full(n, 0.0),
        }

    # ─────────────────────────────────────────────────────────────
    #  SL/TP — ATR + R:R mode
    # ─────────────────────────────────────────────────────────────

    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        """SL = entry ∓ ATR × mult, TP = entry ± risk × R:R."""
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

    # ─────────────────────────────────────────────────────────────
    #  SL/TP — Fixed % mode
    # ─────────────────────────────────────────────────────────────

    def _compute_sltp_fixed(self, entry: float, direction: int) -> dict:
        """SL = entry × (1 ∓ sl_pct/100), TP = entry × (1 ± tp_pct/100)."""
        sl_pct = self.get_param('sl_fixed_pct', 2.0) / 100.0
        tp1_pct = self.get_param('tp1_fixed_pct', 1.0) / 100.0
        tp2_pct = self.get_param('tp2_fixed_pct', 2.0) / 100.0
        tp1_size = self.get_param('tp1_fixed_size', 0.35)
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
    #  SIGNAL GENERATION — Pine-parity (no HTF)
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """
        Generate CyberCycle signal at bar idx.

        Pine-exact logic (Ehlers Cyber Cycle SIGNALS):
          buySignal = bullCross
                      AND buyConf >= minConf
                      AND barFilter
                      AND ((useTrendFilter AND bullTrend) OR NOT useTrendFilter)

        No HTF filter anywhere.
        """
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        if not (bull_cross or bear_cross):
            return None

        # ── Bar filter (min_bars between signals) ───────────────
        min_bars = self.get_param('min_bars', 5)
        if idx - self._last_signal_bar < min_bars:
            return None

        use_trend = self.get_param('use_trend', True)
        use_vol = self.get_param('use_volume', True)
        is_buy = bull_cross

        # ═════════════════════════════════════════════════════════
        #  CONFIDENCE — Pine-exact weights (25/20/20/15/10/10)
        #  No HTF component at all.
        # ═════════════════════════════════════════════════════════
        conf = compute_confidence_nohtf(
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
            use_trend=use_trend,
            use_vol=use_vol,
        )

        # ── Confidence gate ─────────────────────────────────────
        confidence_min = self.get_param('confidence_min', 75.0)
        if conf < confidence_min:
            return None

        # ═════════════════════════════════════════════════════════
        #  HARD TREND FILTER — post-confidence, como Pine
        #
        #  Pine: AND ((useTrendFilter AND bullTrend) OR NOT useTrendFilter)
        #  Separado del confidence scoring — una señal con conf=100
        #  se rechaza si el trend no está alineado.
        # ═════════════════════════════════════════════════════════
        if use_trend:
            if is_buy and not ind['bull_trend'][idx]:
                return None
            if not is_buy and not ind['bear_trend'][idx]:
                return None

        # ── Daily signal cap ────────────────────────────────────
        max_daily = self.get_param('max_signals_per_day', 0)
        if max_daily > 0:
            if hasattr(self, '_daily_count') and hasattr(self, '_current_day'):
                ts = data.get('timestamp', None)
                if ts is not None and idx < len(ts):
                    day = int(ts[idx]) // 86400000
                else:
                    day = idx // 24
                if day != self._current_day:
                    self._current_day = day
                    self._daily_count = 0
                if self._daily_count >= max_daily:
                    return None
                self._daily_count += 1
            else:
                self._current_day = -1
                self._daily_count = 0

        # ── Signal accepted ─────────────────────────────────────
        self._last_signal_bar = idx

        direction = 1 if is_buy else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]

        # ── SL/TP — dual mode ───────────────────────────────────
        sltp_type = self.get_param('sltp_type', 'sltp_fixed')

        if sltp_type == 'sltp_fixed':
            sltp = self._compute_sltp_fixed(entry, direction)
        else:
            if atr_val <= 0:
                return None
            sltp = self._compute_sltp_atr_rr(entry, direction, atr_val)

        sl = sltp['sl']
        tp_levels = sltp['tp_levels']
        tp_sizes = sltp['tp_sizes']
        sl_dist = sltp['sl_dist']

        # ── Break-even ──────────────────────────────────────────
        be_pct = self.get_param('be_pct', 1.0)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ───────────────────────────────────────
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
            leverage=self.get_param('leverage', 20.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_distance,
            metadata={
                'close_on_signal': self.get_param('close_on_signal', True),
                'max_signals_per_day': max_daily,
                'alpha_method': self.get_param('alpha_method', 'kalman'),
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
                'partial_bar': False,
            }
        )

    # ─────────────────────────────────────────────────────────────
    #  INCREMENTAL PROCESSOR (para intrabar execution)
    # ─────────────────────────────────────────────────────────────

    def create_incremental_processor(self, detail_tf_ratio: int = 1):
        """
        Create incremental processor for intrabar execution.
        Uses IncrementalCyberCycleV2 (v6.3 confidence without HTF).
        Its weights (20/25/20/20/10/5) differ slightly from this
        strategy's Pine weights (25/20/20/15/10/10).
        For maximum parity, use --no-intrabar (bar-close mode).
        """
        from indicators.incremental_ehlers import IncrementalCyberCycleV3

        full_params = self.default_params()
        full_params.update(self.params)
        return IncrementalCyberCycleV3(full_params, detail_tf_ratio=detail_tf_ratio)

    def default_params(self) -> dict:
        """Return default values for all parameters."""
        return {pd.name: pd.default for pd in self.parameter_defs()}

    def get_all_params(self) -> dict:
        """Return all params with defaults filled in."""
        result = self.default_params()
        result.update(self.params)
        return result