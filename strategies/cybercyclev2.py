"""
CryptoLab — CyberCycle v6.3 Strategy
Based on v6.2 — HTF removed, confidence rebalanced, sltp_type added, intrabar support.

Ehlers Adaptive CyberCycle with:
- 5 alpha methods (Homodyne/MAMA/Autocorrelation/Kalman/Manual)
- iTrend filter
- Fisher Transform
- Volume filter
- Confidence scoring v6.3 (100pts, HTF redistributed)
- Dual SL/TP mode: slatr_tprr | sltp_fixed
- IntrabarBacktestEngine compatible
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    homodyne_alpha, mama_alpha, autocorrelation_alpha,
    kalman_alpha, cybercycle, itrend, fisher_transform,
)
from indicators.incremental_ehlers import IncrementalCyberCycle
from indicators.common import ema, sma, atr, crossover, crossunder, volume_ratio



# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE v6.3 — HTF eliminado, 15 pts redistribuidos
#
#  v6.2 (con HTF, max 105):
#    Cross=20, iTrend=20, OB/OS=15, Volume=15,
#    Fisher=10, Momentum=10, HTF=15
#
#  v6.3 (sin HTF, max 100):
#    Cross=20, iTrend=25(+5), OB/OS=20(+5), Volume=20(+5),
#    Fisher=10, Momentum=5(-5)
#
#  Lógica: iTrend/OB/OS/Volume son los filtros de dirección más
#  sólidos — absorben el rol del HTF. Momentum es el más débil
#  y cede 5 pts.
# ═══════════════════════════════════════════════════════════════

def compute_confidence_v63(
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
    Confidence scoring v6.3 — sin HTF, redistribuido a 100 pts.

        Cross signal:  20  (sin cambio)
        iTrend:        25  (+5 desde HTF)
        OB/OS zone:    20  (+5 desde HTF)
        Volume:        20  (+5 desde HTF)
        Fisher:        10  (sin cambio)
        Momentum:       5  (-5, el más débil)
        ─────────────
        Total max:    100
    """
    conf = 0.0
    if is_buy:
        conf += 20.0 if bull_cross else 0.0
        conf += 25.0 if (bull_trend if use_trend else True) else 0.0
        conf += 20.0 if in_os else 0.0
        conf += 20.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_rising else 0.0
        conf +=  5.0 if momentum_positive else 0.0
    else:
        conf += 20.0 if bear_cross else 0.0
        conf += 25.0 if (bear_trend if use_trend else True) else 0.0
        conf += 20.0 if in_ob else 0.0
        conf += 20.0 if (volume_ok if use_vol else True) else 0.0
        conf += 10.0 if fish_falling else 0.0
        conf +=  5.0 if momentum_negative else 0.0
    return min(conf, 100.0)

class CyberCycleStrategyv2(IStrategy):

    def name(self) -> str:
        return "CyberCycle v6.3"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # Alpha method
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

            # Signal params
            ParamDef('itrend_alpha', 'float', 0.07, 0.01, 0.30, 0.01),
            ParamDef('trigger_ema', 'int', 14, 3, 30),
            ParamDef('min_bars', 'int', 24, 5, 50),
            ParamDef('confidence_min', 'float', 80.0, 30.0, 95.0, 5.0),
            ParamDef('ob_level', 'float', 1.5, 0.3, 3.0, 0.1),
            ParamDef('os_level', 'float', -1.5, -3.0, -0.3, 0.1),

            # Filters
            ParamDef('use_trend', 'bool', True),
            ParamDef('use_volume', 'bool', True),
            ParamDef('volume_mult', 'float', 2.0, 0.5, 5.0, 0.1),
            # ── Risk params ──────────────────────────────────────────
            ParamDef('leverage',    'float', 15.0, 10.0, 25.0, 50.0),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5,  4.0, 0.1),

            # TP1 / TP2 — R:R multipliers sobre la distancia al SL
            # Ej: tp1_rr=1.5 → TP1 en entry ± risk*1.5
            ParamDef('tp1_rr',   'float', 2.0, 0.5,  5.0, 0.25),
            ParamDef('tp1_size', 'float', 0.6, 0.1,  0.9, 0.05),   # fraccion cerrada en TP1
            ParamDef('tp2_rr',   'float', 3.0, 1.0, 10.0, 0.25),
            # TP2 cierra el resto (1 - tp1_size)

            # ── Break-even ────────────────────────────────────────────
            # BE se activa cuando el precio mueve be_pct% a nuestro favor desde entrada
            # be_pct=0 → desactivado
            ParamDef('be_pct', 'float', 1.5, 0.0, 2.5, 0.1),

            # ── Trailing stop ─────────────────────────────────────────
            ParamDef('use_trailing', 'bool', True),
            # Trailing se activa al mover trail_activate_pct% a favor
            ParamDef('trail_activate_pct', 'float', 2.5, 0.0, 5.0, 0.25),
            # Una vez activo: SL sigue al mejor precio con retroceso de trail_pullback_pct%
            ParamDef('trail_pullback_pct', 'float', 1.0, 0.1,  2.0, 0.10),

            # ═════════════════════════════════════════════════════════
            #  SL/TP MODE SELECTOR
            # ═════════════════════════════════════════════════════════
            #  slatr_tprr  → SL basado en ATR, TP basado en Risk:Reward
            #  sltp_fixed  → SL y TP como porcentaje fijo del entry price
            # ─────────────────────────────────────────────────────────
            ParamDef('sltp_type', 'categorical', 'slatr_tprr',
                     options=['slatr_tprr', 'sltp_fixed']),

            # ── Risk params: FIXED mode (sltp_fixed) ────────────────
            ParamDef('sl_fixed_pct',   'float', 2.5, 0.3,  5.0, 0.1),
            ParamDef('tp1_fixed_pct',  'float', 3.0, 0.5,  8.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.7, 0.1,  0.9, 0.05),
            ParamDef('tp2_fixed_pct',  'float', 4.5, 1.0, 15.0, 0.5),
        ]

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate all CyberCycle indicators."""
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'mama')

        # Compute ALL alpha methods (Pine calls all each bar for state)
        a_hd, p_hd = homodyne_alpha(
            src,
            self.get_param('hd_min_period', 3.0),
            self.get_param('hd_max_period', 40.0)
        )
        a_ma, p_ma = mama_alpha(
            src,
            self.get_param('mama_fast', 0.5),
            self.get_param('mama_slow', 0.05)
        )
        a_ac, p_ac = autocorrelation_alpha(
            src,
            self.get_param('ac_min_period', 6),
            self.get_param('ac_max_period', 48),
            self.get_param('ac_avg_length', 3)
        )
        a_kl, p_kl = kalman_alpha(
            src,
            self.get_param('kal_process_noise', 0.01),
            self.get_param('kal_meas_noise', 0.5),
            self.get_param('kal_alpha_fast', 0.5),
            self.get_param('kal_alpha_slow', 0.05),
            self.get_param('kal_sensitivity', 2.0)
        )

        # Select active alpha
        alpha_map = {
            'homodyne': (a_hd, p_hd),
            'mama': (a_ma, p_ma),
            'autocorrelation': (a_ac, p_ac),
            'kalman': (a_kl, p_kl),
        }

        if method == 'manual':
            manual_a = self.get_param('manual_alpha', 0.35)
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
            # All alphas for diagnostics
            'alpha_hd': a_hd, 'alpha_ma': a_ma,
            'alpha_ac': a_ac, 'alpha_kl': a_kl,
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

        # Compute confidence
        conf = compute_confidence_v63(
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

        min_conf = self.get_param('confidence_min', 80.0)
        if conf < min_conf:
            return None

        # Trend filter
        if use_trend:
            if is_buy and not ind['bull_trend'][idx]:
                return None
            if not is_buy and not ind['bear_trend'][idx]:
                return None

        self._last_signal_bar = idx

        # ── Entry ────────────────────────────────────────────────────
        direction = 1 if is_buy else -1
        entry    = data['close'][idx]
        atr_val  = ind['atr'][idx]

        # ── SL/TP — dual mode ─────────────────────────────────────────
        sltp_type = self.get_param('sltp_type', 'slatr_tprr')

        if sltp_type == 'sltp_fixed':
            sl_pct   = self.get_param('sl_fixed_pct',   2.5) / 100.0
            tp1_pct  = self.get_param('tp1_fixed_pct',  3.0) / 100.0
            tp2_pct  = self.get_param('tp2_fixed_pct',  4.5) / 100.0
            tp1_size = self.get_param('tp1_fixed_size', 0.7)
            sl       = entry * (1.0 - direction * sl_pct)
            tp1      = entry * (1.0 + direction * tp1_pct)
            tp2      = entry * (1.0 + direction * tp2_pct)
            sl_dist  = abs(entry - sl)
        else:  # slatr_tprr
            sl_dist  = atr_val * self.get_param('sl_atr_mult', 1.5)
            sl       = entry - direction * sl_dist
            tp1_size = self.get_param('tp1_size', 0.6)
            tp1      = entry + direction * sl_dist * self.get_param('tp1_rr', 2.0)
            tp2      = entry + direction * sl_dist * self.get_param('tp2_rr', 3.0)

        tp2_size  = round(1.0 - tp1_size, 8)
        tp_levels = [tp1, tp2]
        tp_sizes  = [tp1_size, tp2_size]
        risk      = sl_dist

        # ── Break-even ───────────────────────────────────────────────
        # El motor mueve el SL a entry cuando el precio alcanza be_trigger.
        # be_pct=0 → desactivado (be_trigger=0 en el engine = sin BE)
        be_pct = self.get_param('be_pct', 0.5)
        if be_pct > 0.0:
            be_trigger = entry + direction * entry * (be_pct / 100.0)
        else:
            be_trigger = 0.0

        # ── Trailing stop ────────────────────────────────────────────
        # trail_activate_pct: precio debe moverse X% para activar trailing
        # trail_pullback_pct: una vez activo, el SL queda X% por debajo del mejor precio
        use_trailing = self.get_param('use_trailing', True)
        if use_trailing:
            trail_activate_pct  = self.get_param('trail_activate_pct', 1.0)
            trail_pullback_pct  = self.get_param('trail_pullback_pct', 0.5)

            # trailing_distance se expresa en puntos de precio (absoluto)
            # El engine usa: new_sl = trailing_high - trailing_distance (longs)
            #                          trailing_high + trailing_distance (shorts)
            # Aquí lo calculamos como % del precio de entrada
            trailing_distance = entry * (trail_pullback_pct / 100.0)

            # Sobreescribir be_trigger con el nivel de activacion del trailing
            # si trail_activate_pct genera un nivel más cercano que be_pct
            trail_activation_price = entry + direction * entry * (trail_activate_pct / 100.0)
            # Usamos el más conservador como be_trigger (el que está más cerca de entry)
            if be_pct > 0.0:
                # Ambos activos: usar el que esté más cerca de la entrada
                dist_be    = abs(be_trigger - entry)
                dist_trail = abs(trail_activation_price - entry)
                be_trigger = be_trigger if dist_be <= dist_trail else trail_activation_price
            else:
                # Solo trailing: usar trail_activation como trigger de BE tambien
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
            leverage=self.get_param('leverage', 3.0),
            be_trigger=be_trigger,
            trailing=use_trailing,
            trailing_distance=trailing_distance,
            metadata={
                'alpha_method':  self.get_param('alpha_method'),
                'alpha':         ind['alpha'][idx],
                'period':        ind['period'][idx],
                'cycle':         ind['cycle'][idx],
                'fisher':        ind['fisher'][idx],
                'sltp_mode':     sltp_type,
                'sl_dist_atr':   sl_dist / atr_val if atr_val > 0 else 0,
                'tp1':           tp1,
                'tp2':           tp2,
                'be_pct':        be_pct,
                'trail_pct':     self.get_param('trail_pullback_pct', 0.5) if use_trailing else 0,
            }
        )

    def create_incremental_processor(self, detail_tf_ratio: int = 1):
        """Create incremental processor for intrabar execution.

        Enables IntrabarBacktestEngine to fire signals at the 1m bar
        where the crossover first occurs, not at 1h close.

        Args:
            detail_tf_ratio: detail bars per main-TF bar (e.g. 60 for 1m/1h).
        """
        full_params = self.default_params()
        full_params.update(self.params)
        return IncrementalCyberCycle(full_params, detail_tf_ratio=detail_tf_ratio)