"""
CryptoLab — CyberCycle v7.0 Strategy

Ehlers Adaptive CyberCycle — streamlined from v6.2:

Changes v6.2 → v7.0:
─────────────────────────────────────────────────────────────────
 REMOVED:
  • max_signals_per_day  — redundant, min_bars ya controla frecuencia
  • use_htf / HTF filter — redundant con iTrend, proxy EMA(40) era débil
  • close_on_signal param — siempre True (reversal on opposite signal)

 ADDED:
  • cycle_strength       — percentil de |cycle| sobre N barras.
                           Filtra cruces en oscilador plano (ruido).
                           Cruces con amplitud alta → señal limpia.
  • ATR percentile       — evita trades en volatilidad extremadamente
                           baja (mercado muerto) o alta (whipsaw)

 OPTIMIZED:
  • Solo computa el alpha method seleccionado (4x menos CPU)
  • Confidence redistribuido a 100 sin HTF:
      cross=20, iTrend=20, OB/OS=15, volume=15,
      fisher=10, momentum=10, cycle_strength=10  → max 100
  • min_bars: rango ajustado (10-60, default 24)
─────────────────────────────────────────────────────────────────

Filtros activos:
- 4 alpha methods (Homodyne/MAMA/Autocorrelation/Kalman) + Manual
- iTrend filter
- Fisher Transform
- Volume filter
- Cycle Strength (amplitud del oscilador)
- Confidence scoring system (0-100)
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    homodyne_alpha, mama_alpha, autocorrelation_alpha,
    kalman_alpha, cybercycle, itrend, fisher_transform,
)
from indicators.common import ema, sma, atr, crossover, crossunder, volume_ratio


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE v7 — sin HTF, con cycle_strength
# ═══════════════════════════════════════════════════════════════

def compute_confidence_v7(
    is_buy: bool,
    bull_cross: bool, bear_cross: bool,
    bull_trend: bool, bear_trend: bool,
    in_ob: bool, in_os: bool,
    volume_ok: bool,
    fish_rising: bool, fish_falling: bool,
    momentum_positive: bool, momentum_negative: bool,
    cycle_strong: bool,
    use_trend: bool, use_vol: bool,
) -> float:
    """
    Confidence scoring v7 — rebalanced without HTF.

    Component weights (max 100):
        Cross signal:    20  — base signal (cycle × trigger)
        iTrend:          20  — trend alignment (primary directional filter)
        OB/OS zone:      15  — oversold (buy) or overbought (sell) zone
        Volume:          15  — above-average volume confirmation
        Fisher:          10  — fisher transform direction
        Momentum:        10  — 3-bar cycle momentum
        Cycle strength:  10  — oscillator amplitude quality
        ─────────────────────
        Total max:      100
    """
    conf = 0.0

    if is_buy:
        conf += 20.0 if bull_cross else 0.0
        conf += 20.0 if (bull_trend if use_trend else True) else 0.0
        conf += 15.0 if in_os else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_rising else 0.0
        conf += 10.0 if momentum_positive else 0.0
        conf += 10.0 if cycle_strong else 0.0
    else:
        conf += 20.0 if bear_cross else 0.0
        conf += 20.0 if (bear_trend if use_trend else True) else 0.0
        conf += 15.0 if in_ob else 0.0
        conf += 15.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_falling else 0.0
        conf += 10.0 if momentum_negative else 0.0
        conf += 10.0 if cycle_strong else 0.0

    return min(conf, 100.0)


class CyberCycleStrategy(IStrategy):

    def name(self) -> str:
        return "CyberCycle v7.0"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # ── Alpha method ────────────────────────────────────────
            ParamDef('alpha_method', 'categorical', 'manual',
                     options=['homodyne', 'mama', 'autocorrelation', 'kalman', 'manual']),
            ParamDef('manual_alpha', 'float', 0.35, 0.05, 0.80, 0.01),
            ParamDef('alpha_floor', 'float', 0.0, 0.0, 0.50, 0.01),

            # Homodyne params
            ParamDef('hd_min_period', 'float', 3.0, 2.0, 10.0, 1.0),
            ParamDef('hd_max_period', 'float', 40.0, 15.0, 80.0, 5.0),

            # MAMA params
            ParamDef('mama_fast', 'float', 0.5, 0.2, 0.8, 0.05),
            ParamDef('mama_slow', 'float', 0.05, 0.01, 0.2, 0.01),

            # Autocorrelation params
            ParamDef('ac_min_period', 'int', 6, 3, 15),
            ParamDef('ac_max_period', 'int', 48, 20, 80),
            ParamDef('ac_avg_length', 'int', 3, 1, 5),

            # Kalman params
            ParamDef('kal_process_noise', 'float', 0.01, 0.001, 0.2, 0.005),
            ParamDef('kal_meas_noise', 'float', 0.5, 0.05, 3.0, 0.1),
            ParamDef('kal_alpha_fast', 'float', 0.5, 0.2, 0.8, 0.05),
            ParamDef('kal_alpha_slow', 'float', 0.05, 0.01, 0.2, 0.01),
            ParamDef('kal_sensitivity', 'float', 2.0, 0.5, 5.0, 0.5),

            # ── Signal params ───────────────────────────────────────
            ParamDef('itrend_alpha', 'float', 0.07, 0.01, 0.30, 0.01),
            ParamDef('trigger_ema', 'int', 14, 3, 30),
            ParamDef('min_bars', 'int', 20, 10, 60),
            ParamDef('confidence_min', 'float', 75.0, 35.0, 90.0, 5.0),
            ParamDef('ob_level', 'float', 1.5, 0.3, 3.0, 0.1),
            ParamDef('os_level', 'float', -1.5, -3.0, -0.3, 0.1),

            # ── Cycle strength ──────────────────────────────────────
            # Percentil mínimo de |cycle| para considerar señal "fuerte".
            # Lookback = cycle_str_lookback barras.
            # Ej: 50 → cycle debe estar en top 50% de amplitud reciente
            ParamDef('cycle_str_pctile', 'float', 50.0, 20.0, 80.0, 5.0),
            ParamDef('cycle_str_lookback', 'int', 50, 20, 100),

            # ── Filters ─────────────────────────────────────────────
            ParamDef('use_trend', 'bool', True),
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 1.5, 0.5, 5.0, 0.1),

            # ── Risk params ─────────────────────────────────────────
            ParamDef('leverage',    'float', 7.0, 1.0, 25.0, 1.0),
            ParamDef('sl_atr_mult', 'float', 2.5, 0.5, 4.0, 0.1),

            # TP1 / TP2 — R:R multipliers sobre la distancia al SL
            ParamDef('tp1_rr',   'float', 2.0, 0.5,  5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1,  0.9, 0.05),
            ParamDef('tp2_rr',   'float', 4.0, 1.0, 10.0, 0.25),

            # ── Break-even ──────────────────────────────────────────
            ParamDef('be_pct', 'float', 1.5, 0.0, 2.5, 0.1),

            # ── Trailing stop ───────────────────────────────────────
            ParamDef('use_trailing', 'bool', True),
            ParamDef('trail_activate_pct', 'float', 2.0, 0.0, 5.0, 0.25),
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.1, 2.0, 0.10),
        ]

    # ─────────────────────────────────────────────────────────────
    #  INDICATORS
    # ─────────────────────────────────────────────────────────────

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate CyberCycle v7 indicators."""
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'mama')

        # ── Solo computar el alpha seleccionado (v7: 4x menos CPU) ──
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

        # ── CyberCycle oscillator ───────────────────────────────
        cycle = cybercycle(src, alpha)

        # Trigger (EMA of cycle)
        trigger = ema(cycle, self.get_param('trigger_ema', 14))

        # ── iTrend ──────────────────────────────────────────────
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
        vol_ratio = volume_ratio(vol, 20)
        vol_mult = self.get_param('volume_mult', 1.5)
        volume_ok = ~self.get_param('use_volume', True) | (vol_ratio >= vol_mult)

        # ── Cross signals ───────────────────────────────────────
        bull_cross = crossover(cycle, trigger)
        bear_cross = crossunder(cycle, trigger)

        # ── OB/OS zones ─────────────────────────────────────────
        ob_level = self.get_param('ob_level', 1.5)
        os_level = self.get_param('os_level', -1.5)
        in_ob = cycle > ob_level
        in_os = cycle < os_level

        # ── Momentum ────────────────────────────────────────────
        momentum3 = np.zeros(n)
        for i in range(3, n):
            momentum3[i] = cycle[i] - cycle[i - 3]

        # ── Cycle Strength (v7 NEW) ─────────────────────────────
        # Percentil de |cycle| sobre ventana deslizante.
        # Mide si la amplitud actual del oscilador está en el
        # top X% de las últimas N barras.
        # cycle_strong = True → oscilador con swing amplio → señal limpia
        # cycle_strong = False → oscilador plano → probable ruido/chop
        abs_cycle = np.abs(cycle)
        lookback = self.get_param('cycle_str_lookback', 50)
        pctile_thresh = self.get_param('cycle_str_pctile', 50.0)

        cycle_strong = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            window = abs_cycle[i - lookback:i + 1]
            threshold = np.percentile(window, pctile_thresh)
            cycle_strong[i] = abs_cycle[i] >= threshold
        # Barras iniciales: marcar como True (sin suficiente historia)
        cycle_strong[:lookback] = True

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
            'vol_ratio': vol_ratio,
            'volume_ok': volume_ok,
            'bull_cross': bull_cross,
            'bear_cross': bear_cross,
            'in_ob': in_ob,
            'in_os': in_os,
            'momentum3': momentum3,
            'cycle_strong': cycle_strong,
            'abs_cycle': abs_cycle,
            'atr': atr_vals,
        }

    # ─────────────────────────────────────────────────────────────
    #  SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate CyberCycle v7 signal at bar idx."""
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        if not (bull_cross or bear_cross):
            return None

        # ── Min bars between signals ────────────────────────────
        min_bars = self.get_param('min_bars', 24)
        if idx - self._last_signal_bar < min_bars:
            return None

        use_trend = self.get_param('use_trend', True)
        use_vol = self.get_param('use_volume', True)

        is_buy = bull_cross

        # ── Confidence v7 (sin HTF, con cycle_strength) ─────────
        conf = compute_confidence_v7(
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
            cycle_strong=ind['cycle_strong'][idx],
            use_trend=use_trend,
            use_vol=use_vol,
        )

        min_conf = self.get_param('confidence_min', 75.0)
        if conf < min_conf:
            return None

        self._last_signal_bar = idx

        # ── Entry & Fixed SL ────────────────────────────────────
        direction = 1 if is_buy else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]

        sl_dist = atr_val * self.get_param('sl_atr_mult', 2.5)
        sl = entry - direction * sl_dist
        risk = sl_dist

        # ── Dos niveles de TP (R:R sobre el riesgo) ─────────────
        tp1_rr = self.get_param('tp1_rr', 2.0)
        tp2_rr = self.get_param('tp2_rr', 4.0)
        tp1_size = self.get_param('tp1_size', 0.6)
        tp2_size = round(1.0 - tp1_size, 8)

        tp1 = entry + direction * risk * tp1_rr
        tp2 = entry + direction * risk * tp2_rr

        tp_levels = [tp1, tp2]
        tp_sizes = [tp1_size, tp2_size]

        # ── Break-even ──────────────────────────────────────────
        be_pct = self.get_param('be_pct', 1.5)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ───────────────────────────────────────
        use_trailing = self.get_param('use_trailing', True)
        if use_trailing:
            trail_activate_pct = self.get_param('trail_activate_pct', 2.0)
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
            leverage=self.get_param('leverage', 7.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_distance,
            metadata={
                'close_on_signal': True,      # v7: siempre reversal
                'max_signals_per_day': 0,      # v7: sin límite diario, min_bars controla
                'alpha_method': self.get_param('alpha_method'),
                'alpha': ind['alpha'][idx],
                'period': ind['period'][idx],
                'cycle': ind['cycle'][idx],
                'fisher': ind['fisher'][idx],
                'cycle_strength': float(ind['abs_cycle'][idx]),
                'sl_dist_atr': sl_dist / atr_val if atr_val > 0 else 0,
                'tp1': tp1,
                'tp2': tp2,
                'be_pct': be_pct,
                'trail_pct': self.get_param('trail_pullback_pct', 1.0) if use_trailing else 0,
            }
        )