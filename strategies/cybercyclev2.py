"""
CryptoLab — CyberCycle v6.2 Strategy (Legacy — sin cycle_strength)
Version antigua para comparar con v6.3/v7.

Ehlers Adaptive CyberCycle with:
- 4 alpha methods (Homodyne/MAMA/Autocorrelation/Kalman) + Manual
- iTrend filter
- Fisher Transform
- Volume filter
- Confidence scoring system (0-100)

Diferencia con v7 (cybercycle.py):
- No usa cycle_strength
- Sin HTF filter (eliminado v6.2 → v6.3)
- Soporta sltp_type: 'slatr_tprr' | 'sltp_fixed'
- Nombre: "CyberCycle v6.3"
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
#  CONFIDENCE v6.3 — sin HTF, sin cycle_strength
# ═══════════════════════════════════════════════════════════════

def compute_confidence_v62(
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
    Confidence scoring v6.3 — sin HTF, sin cycle_strength.

    Redistribución de los 15 pts del HTF eliminado:
        Cross signal:    20  — señal base (cycle × trigger cruce)
        iTrend:          25  — alineación de tendencia  (+5 vs v6.2)
        OB/OS zone:      20  — zona de sobreventa/sobrecompra (+5 vs v6.2)
        Volume:          15  — confirmación de volumen
        Fisher:          10  — dirección Fisher transform
        Momentum:        10  — momentum 3 barras del ciclo
        ──────────────────────
        Total max:      100
    """
    conf = 0.0

    if is_buy:
        conf += 20.0 if bull_cross else 0.0
        conf += 25.0 if (bull_trend if use_trend else True) else 0.0
        conf += 20.0 if in_os else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_rising else 0.0
        conf += 10.0 if momentum_positive else 0.0
    else:
        conf += 20.0 if bear_cross else 0.0
        conf += 25.0 if (bear_trend if use_trend else True) else 0.0
        conf += 20.0 if in_ob else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_falling else 0.0
        conf += 10.0 if momentum_negative else 0.0

    return min(conf, 100.0)


class CyberCycleStrategyv2(IStrategy):

    def name(self) -> str:
        return "CyberCycle v6.3"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # Alpha method
            ParamDef('alpha_method', 'categorical', 'manual',
                     options=['kalman', 'manual']),
            ParamDef('manual_alpha', 'float', 0.35, 0.05, 0.80, 0.01),
            ParamDef('alpha_floor', 'float', 0.0, 0.0, 0.50, 0.01),

            # Kalman params
            ParamDef('kal_process_noise', 'float', 0.01, 0.001, 0.2, 0.005),
            ParamDef('kal_meas_noise', 'float', 0.5, 0.05, 3.0, 0.1),
            ParamDef('kal_alpha_fast', 'float', 0.5, 0.2, 0.8, 0.05),
            ParamDef('kal_alpha_slow', 'float', 0.05, 0.01, 0.2, 0.01),
            ParamDef('kal_sensitivity', 'float', 2.0, 0.5, 5.0, 0.5),

            # Signal params
            ParamDef('itrend_alpha', 'float', 0.07, 0.01, 0.30, 0.01),
            ParamDef('trigger_ema', 'int', 14, 3, 30),
            ParamDef('min_bars', 'int', 24, 12, 50),
            ParamDef('confidence_min', 'float', 80.0, 30.0, 90.0, 5.0),
            ParamDef('ob_level', 'float', 1.5, 0.3, 3.0, 0.1),
            ParamDef('os_level', 'float', -1.5, -3.0, -0.3, 0.1),

            # Filters (HTF eliminado)
            ParamDef('use_trend', 'bool', True),
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 2.0, 0.5, 5.0, 0.1),

            # ═══════════════════════════════════════════════════════════
            #  SL/TP MODE SELECTOR
            # ═══════════════════════════════════════════════════════════
            #  slatr_tprr  → SL basado en ATR, TP basado en Risk:Reward
            #  sltp_fixed  → SL y TP como porcentaje fijo del entry price
            # ───────────────────────────────────────────────────────────
            ParamDef('sltp_type', 'categorical', 'sltp_fixed',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── Risk params: ATR mode (slatr_tprr) ──────────────────
            ParamDef('leverage',    'float', 15.0, 6.0, 40.0, 2.0),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5, 4.0, 0.1),
            ParamDef('tp1_rr',      'float', 2.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_size',    'float', 0.6, 0.1, 0.9, 0.05),
            ParamDef('tp2_rr',      'float', 3.0, 1.0, 10.0, 0.25),

            # ── Risk params: FIXED mode (sltp_fixed) ────────────────
            #  SL y TP expresados como % del precio de entrada.
            #  Ej: sl_fixed_pct=1.5 → SL a -1.5% del entry
            #      tp1_fixed_pct=2.0 → TP1 a +2.0% del entry
            #      tp2_fixed_pct=4.0 → TP2 a +4.0% del entry
            # ────────────────────────────────────────────────────────
            ParamDef('sl_fixed_pct',   'float', 2.5, 0.3, 5.0, 0.1),
            ParamDef('tp1_fixed_pct',  'float', 3.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.7, 0.1, 0.9, 0.05),
            ParamDef('tp2_fixed_pct',  'float', 4.5, 1.0, 10.0, 0.5),

            # ── Break-even ────────────────────────────────────────────
            ParamDef('be_pct', 'float', 1.5, 0.0, 2.5, 0.1),

            # ── Trailing stop ─────────────────────────────────────────
            ParamDef('use_trailing', 'bool', True),
            ParamDef('trail_activate_pct', 'float', 2.5, 0.0, 5.0, 0.25),
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.1, 2.0, 0.10),

            # ── Signal control ────────────────────────────────────────
            ParamDef('close_on_signal', 'bool', True),
            ParamDef('max_signals_per_day', 'int', 0, 0, 10),
        ]

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────────
    def calculate_indicators(self, data: dict) -> dict:
        """Calculate all CyberCycle v6.3 indicators."""
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'manual')

        # ── Solo computar el alpha seleccionado (4x menos CPU) ──
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

        # Volume filter
        vol_ratio = volume_ratio(vol, 20)
        vol_mult = self.get_param('volume_mult', 2.0)
        volume_ok = ~self.get_param('use_volume', True) | (vol_ratio >= vol_mult)

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

        # ATR
        atr14 = atr(data['high'], data['low'], close, 14)

        return {
            'alpha': alpha,
            'period': period,
            'cycle': cycle,
            'trigger': trigger,
            'itrend': it,
            'fisher': fisher,
            'bull_cross': bull_cross,
            'bear_cross': bear_cross,
            'bull_trend': bull_trend,
            'bear_trend': bear_trend,
            'fish_rising': fish_rising,
            'fish_falling': fish_falling,
            'volume_ok': volume_ok,
            'in_ob': in_ob,
            'in_os': in_os,
            'momentum3': momentum3,
            'atr': atr14,
        }
    # ─────────────────────────────────────────────────────────────
    #  SL/TP CALCULATION — DUAL MODE
    # ─────────────────────────────────────────────────────────────

    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        """
        Modo ATR + Risk:Reward.
        SL = entry ∓ ATR × mult
        TP = entry ± riesgo × R:R
        """
        sl_dist = atr_val * self.get_param('sl_atr_mult', 1.5)
        sl = entry - direction * sl_dist

        tp1_rr   = self.get_param('tp1_rr', 2.0)
        tp2_rr   = self.get_param('tp2_rr', 3.0)
        tp1_size = self.get_param('tp1_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        tp1 = entry + direction * sl_dist * tp1_rr
        tp2 = entry + direction * sl_dist * tp2_rr

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'mode': 'slatr_tprr',
        }

    def _compute_sltp_fixed(self, entry: float, direction: int) -> dict:
        """
        Modo fijo: SL y TP como porcentaje del precio de entrada.
        SL = entry × (1 ∓ sl_pct/100)
        TP = entry × (1 ± tp_pct/100)
        """
        sl_pct  = self.get_param('sl_fixed_pct',   2.5) / 100.0
        tp1_pct = self.get_param('tp1_fixed_pct',  3.0) / 100.0
        tp2_pct = self.get_param('tp2_fixed_pct',  4.5) / 100.0
        tp1_size = self.get_param('tp1_fixed_size', 0.7)
        tp2_size = round(1.0 - tp1_size, 8)

        sl  = entry * (1.0 - direction * sl_pct)
        tp1 = entry * (1.0 + direction * tp1_pct)
        tp2 = entry * (1.0 + direction * tp2_pct)

        sl_dist = abs(entry - sl)

        return {
            'sl': sl,
            'tp_levels': [tp1, tp2],
            'tp_sizes': [tp1_size, tp2_size],
            'sl_dist': sl_dist,
            'mode': 'sltp_fixed',
        }

    # ─────────────────────────────────────────────────────────────
    #  SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate CyberCycle v6.3 signal at bar idx."""
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        if not (bull_cross or bear_cross):
            return None

        # Bar filter
        min_bars = self.get_param('min_bars', 24)
        if idx - self._last_signal_bar < min_bars:
            return None

        use_trend = self.get_param('use_trend', True)
        use_vol   = self.get_param('use_volume', True)

        is_buy = bull_cross

        # ── Confidence v6.3 (sin HTF, sin cycle_strength) ───────
        conf = compute_confidence_v62(
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

        confidence_min = self.get_param('confidence_min', 80.0)
        if conf < confidence_min:
            return None

        # ── Signal accepted ──────────────────────────────────────
        self._last_signal_bar = idx

        direction = 1 if is_buy else -1
        entry     = data['close'][idx]
        atr_val   = ind['atr'][idx]

        # ── SL/TP — dual mode ────────────────────────────────────
        sltp_type = self.get_param('sltp_type', 'sltp_fixed')

        if sltp_type == 'sltp_fixed':
            sltp = self._compute_sltp_fixed(entry, direction)
        else:
            if atr_val <= 0:
                return None
            sltp = self._compute_sltp_atr_rr(entry, direction, atr_val)

        sl        = sltp['sl']
        tp_levels = sltp['tp_levels']
        tp_sizes  = sltp['tp_sizes']
        sl_dist   = sltp['sl_dist']

        # ── Break-even ───────────────────────────────────────────
        be_pct = self.get_param('be_pct', 1.5)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ────────────────────────────────────────
        use_trailing = self.get_param('use_trailing', True)
        if use_trailing:
            trail_activate_pct = self.get_param('trail_activate_pct', 2.5)
            trail_pullback_pct = self.get_param('trail_pullback_pct', 1.0)
            trailing_distance  = entry * (trail_pullback_pct / 100.0)

            trail_activation_price = entry + direction * entry * (trail_activate_pct / 100.0)
            if be_pct > 0.0:
                dist_be    = abs(be_trigger - entry)
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
        from indicators.incremental_ehlers import IncrementalCyberCycleV2
        return IncrementalCyberCycleV2(self.params, detail_tf_ratio=detail_tf_ratio)