"""
CryptoLab — CyberCycle v6.2 Strategy (con HTF real)

Ehlers Adaptive CyberCycle with:
- 4 alpha methods (Homodyne/MAMA/Autocorrelation/Kalman) + Manual
- iTrend filter
- Fisher Transform
- Volume filter
- HTF filter (REAL resample, no EMA proxy)
- Confidence scoring system (0-100)

Diferencia con v7 (cybercycle.py):
- No usa cycle_strength
- Nombre: "CyberCycle v6.2"

Historial:
- v6.2 original: HTF era EMA(40) proxy (INCORRECTO)
- v6.2 fix:      HTF usa htf_resample() real con timestamps
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    homodyne_alpha, mama_alpha, autocorrelation_alpha,
    kalman_alpha, cybercycle, itrend, fisher_transform,
    compute_confidence
)
from indicators.common import (
    ema, sma, atr, crossover, crossunder, volume_ratio,
    htf_resample, get_htf_seconds,       # ← NUEVO: HTF real
)


class CyberCycleStrategyhtfcc(IStrategy):

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
            ParamDef('use_htf', 'bool', True),
            # ── HTF timeframe selector (NUEVO) ──────────────────
            # 'auto': 1h→4h, 4h→1d, etc. (mapeo automático)
            # O seleccionar manualmente: '4h', '1d', '1w'
            ParamDef('htf_timeframe', 'categorical', 'auto',
                     options=['auto', '1h', '4h', '1d', '1w']),

            # ═════════════════════════════════════════════════════════
            #  SL/TP MODE SELECTOR
            # ═════════════════════════════════════════════════════════
            ParamDef('sltp_type', 'categorical', 'sltp_fixed',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── Risk params: ATR mode (slatr_tprr) ──────────────────
            ParamDef('leverage',    'float', 20.0, 5.0, 40.0, 5.0),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5,  4.0, 0.1),

            ParamDef('tp1_rr',   'float', 2.0, 0.5,  5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1,  0.9, 0.05),
            ParamDef('tp2_rr',   'float', 3.0, 1.0, 10.0, 0.25),

            # ── Risk params: FIXED mode (sltp_fixed) ────────────────
            ParamDef('sl_fixed_pct',  'float', 2.5, 0.5,  5.0,  0.1),
            ParamDef('tp1_fixed_pct', 'float', 3.0, 0.5,  8.0,  0.25),
            ParamDef('tp2_fixed_pct', 'float', 4.5, 1.0, 15.0,  0.25),
            ParamDef('tp1_fixed_size','float', 0.6, 0.3,  0.9,  0.05),

            # ── Break-even & trailing ──────────────────────────────
            ParamDef('be_pct', 'float', 1.5, 0.0, 5.0, 0.25),
            ParamDef('use_trailing', 'bool', True),
            ParamDef('trail_activate_pct', 'float', 2.5, 0.5, 8.0, 0.25),
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.2, 3.0, 0.1),

            # Control
            ParamDef('close_on_signal', 'bool', True),
            ParamDef('max_signals_per_day', 'int', 0, 0, 5),
        ]

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate all indicators for vectorized backtesting."""
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'manual')

        # ── Compute selected alpha method ───────────────────────
        # Pre-compute all 4 for diagnostics
        a_hd, p_hd = homodyne_alpha(src, 3.0, 40.0)
        a_ma, p_ma = mama_alpha(src, 0.5, 0.05)
        a_ac, p_ac = autocorrelation_alpha(src, 6, 48, 3)
        a_kl, p_kl = kalman_alpha(
            src,
            self.get_param('kal_process_noise', 0.01),
            self.get_param('kal_meas_noise', 0.5),
            self.get_param('kal_alpha_fast', 0.5),
            self.get_param('kal_alpha_slow', 0.05),
            self.get_param('kal_sensitivity', 2.0))

        alpha_map = {
            'homodyne': (a_hd, p_hd),
            'mama': (a_ma, p_ma),
            'autocorrelation': (a_ac, p_ac),
            'kalman': (a_kl, p_kl),
        }

        if method == 'manual':
            manual_a = self.get_param('manual_alpha', 0.42)
            alpha = np.full(n, manual_a)
            period = np.full(n, (2.0 / manual_a) - 1.0)
        else:
            alpha, period = alpha_map.get(method, (a_ma, p_ma))

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
            bull_trend[i] = it[i] > it[i-2]
            bear_trend[i] = it[i] < it[i-2]

        # Fisher Transform
        fisher = fisher_transform(cycle, 10)
        fish_rising = np.zeros(n, dtype=bool)
        fish_falling = np.zeros(n, dtype=bool)
        for i in range(1, n):
            fish_rising[i] = fisher[i] > fisher[i-1]
            fish_falling[i] = fisher[i] < fisher[i-1]

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
            momentum3[i] = cycle[i] - cycle[i-3]

        # ATR for SL/TP
        atr_vals = atr(data['high'], data['low'], close, 14)

        # ════════════════════════════════════════════════════════
        #  HTF FILTER — RESAMPLEO REAL (reemplaza EMA proxy)
        # ════════════════════════════════════════════════════════
        # Antes: htf_src = ema(src, 40) ← INCORRECTO, no es HTF
        # Ahora: htf_resample() con timestamps reales
        # ────────────────────────────────────────────────────────
        use_htf = self.get_param('use_htf', True)

        if use_htf and 'timestamp' in data:
            htf_tf = self.get_param('htf_timeframe', 'auto')
            # Obtener el TF base desde el data dict (inyectado por engine)
            base_tf = data.get('_timeframe', '1h')
            htf_seconds = get_htf_seconds(base_tf, htf_tf)

            timestamps = data['timestamp']

            # Resamplear hl2 y close al HTF (sin future leak)
            # htf_resample propaga el valor del cierre de la barra HTF
            # anterior a todas las barras del TF actual dentro de esa
            # ventana. Equivale a request.security() de Pine Script.
            htf_src_vals = htf_resample(src, timestamps, htf_seconds)
            htf_close_vals = htf_resample(close, timestamps, htf_seconds)

            # Bull HTF = hl2 del HTF > close del HTF (tendencia alcista)
            htf_bull = htf_src_vals > htf_close_vals
            htf_bear = htf_src_vals < htf_close_vals
        elif use_htf:
            # Fallback si no hay timestamps: usar EMA como proxy
            # (backward compat con datos sin timestamp)
            htf_src_proxy = ema(src, 40)
            htf_cc_proxy = ema(close, 40)
            htf_bull = htf_src_proxy > htf_cc_proxy
            htf_bear = htf_src_proxy < htf_cc_proxy
        else:
            # HTF desactivado: todo pasa
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
            # All alphas for diagnostics
            'alpha_hd': a_hd, 'alpha_ma': a_ma,
            'alpha_ac': a_ac, 'alpha_kl': a_kl,
        }

    # ─────────────────────────────────────────────────────────────
    #  SL/TP CALCULATION — DUAL MODE (copiado de v7.1)
    # ─────────────────────────────────────────────────────────────
    def _compute_sltp_atr_rr(self, entry: float, direction: int,
                              atr_val: float) -> dict:
        """
        Modo ATR + Risk:Reward (original v6.2).
        SL = entry ∓ ATR × mult
        TP = entry ± risk × R:R
        """
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
        """
        Modo fijo: SL y TP como porcentaje del precio de entrada.
        SL = entry × (1 ∓ sl_pct/100)
        TP = entry × (1 ± tp_pct/100)
        """
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

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate CyberCycle signal at bar idx."""
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
        use_vol = self.get_param('use_volume', True)

        is_buy = bull_cross

        # ── Confidence scoring (sin cycle_strength) ──
        # Cada componente suma puntos. confidence_min filtra.
        conf = 0.0

        # Base: cross detected = 20 pts
        conf += 20.0

        # iTrend alignment: +25 pts
        if is_buy and ind['bull_trend'][idx]:
            conf += 25.0
        elif not is_buy and ind['bear_trend'][idx]:
            conf += 25.0

        # Fisher alignment: +15 pts
        if is_buy and ind['fish_rising'][idx]:
            conf += 15.0
        elif not is_buy and ind['fish_falling'][idx]:
            conf += 15.0

        # OB/OS zone: +15 pts
        if is_buy and ind['in_os'][idx]:
            conf += 15.0
        elif not is_buy and ind['in_ob'][idx]:
            conf += 15.0

        # HTF alignment: +15 pts
        if is_buy and ind['htf_align_buy'][idx]:
            conf += 15.0
        elif not is_buy and ind['htf_align_sell'][idx]:
            conf += 15.0

        # Momentum alignment: +10 pts
        if is_buy and ind['momentum3'][idx] > 0:
            conf += 10.0
        elif not is_buy and ind['momentum3'][idx] < 0:
            conf += 10.0

        # Volume: already filtered via volume_ok, but add pts
        if ind['volume_ok'][idx]:
            conf += 10.0

        # ── Confidence gate ──
        confidence_min = self.get_param('confidence_min', 80.0)
        if conf < confidence_min:
            return None

        # ── Signal accepted ──
        self._last_signal_bar = idx

        direction = 1 if is_buy else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]
        if atr_val <= 0:
            return None

        # ── SL/TP — dual mode ──────────────────────────────────
        sltp_type = self.get_param('sltp_type', 'slatr_tprr')

        if sltp_type == 'sltp_fixed':
            sltp = self._compute_sltp_fixed(entry, direction)
        else:
            sltp = self._compute_sltp_atr_rr(entry, direction, atr_val)

        sl = sltp['sl']
        tp_levels = sltp['tp_levels']
        tp_sizes = sltp['tp_sizes']
        sl_dist = sltp['sl_dist']

        # ── Break-even ─────────────────────────────────────────
        be_pct = self.get_param('be_pct', 1.5)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ─────────────────────────────────────
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
                'htf_timeframe': self.get_param('htf_timeframe', 'auto'),
                'htf_align': 'buy' if (is_buy and ind['htf_align_buy'][idx]) else
                             ('sell' if (not is_buy and ind['htf_align_sell'][idx]) else 'none'),
            }
        )

    def create_incremental_processor(self, detail_tf_ratio: int = 60):
        """Create incremental processor for IntrabarBacktestEngine."""
        from indicators.incremental_ehlers import IncrementalCyberCycleV2
        return IncrementalCyberCycleV2(self.params, detail_tf_ratio=detail_tf_ratio)