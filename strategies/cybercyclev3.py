"""
CryptoLab — CyberCycle v6.2 Pine-Parity Strategy

Maximum parity with cybercycle_v62.pine (TradingView).

═══════════════════════════════════════════════════════════════
 CAMBIOS vs cybercyclev3.py original:
═══════════════════════════════════════════════════════════════

 1. ELIMINADOS Homodyne, MAMA, Autocorrelation
    → Solo Kalman + Manual (los únicos que pasan gates)
    → Reduce espacio de búsqueda de 35 a ~22 params efectivos
    → 4x menos CPU en calculate_indicators()

 2. HTF FILTER CORREGIDO — usa htf_resample() real
    → Antes: ema(src, 40) vs ema(close, 40) — proxy falso
    → Ahora: htf_resample(hl2, ts, 14400) vs htf_resample(ema10, ts, 14400)
    → Replica Pine: request.security("240", hl2) vs request.security("240", ema(close,10))
    → Requiere timestamps en data['timestamp'] (epoch ms)

 3. CONFIDENCE WEIGHTS = Pine EXACTO (20/15/15/15/10/10/15)
    → Cross: 20, iTrend: 15, OB/OS: 15, Volume: 15,
      Fisher: 10, Momentum: 10, HTF: 15 = 100
    → Antes v3 tenía pesos redistribuidos (25/15/15/15/10/10 = 110)

 4. HARD TREND FILTER post-confidence (como Pine)
    → Pine: buySignal = bullCross AND buyConf >= minConf AND barFilter
            AND (eUseTrend ? bullTrend : true) AND htfAlignBuy
    → Es un AND separado, NO parte del confidence scoring
    → Antes: trend solo sumaba pts, no era gate obligatorio

 5. ENTRY PRICE = hl2 en histórico (como Pine)
    → Pine captura srcCyber (=hl2) en barstate.ishistory, close en realtime
    → Para backtest histórico, hl2 es la paridad correcta

═══════════════════════════════════════════════════════════════

Filtros activos:
- 2 alpha methods: Kalman + Manual
- iTrend filter
- Fisher Transform
- Volume filter
- HTF 4H filter (htf_resample real)
- Confidence scoring system (0-100) — pesos idénticos a Pine
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.ehlers import (
    kalman_alpha, cybercycle, itrend, fisher_transform,
)
from indicators.common import (
    ema, sma, atr, crossover, crossunder, volume_ratio, htf_resample,
)


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE — Pine-exact weights (lines 706-725 of .pine)
#
#  buyConf:
#    bullCross                        → 20
#    (eUseTrend ? bullTrend : true)   → 15
#    inOS                             → 15
#    (eUseVol ? volumeOK : true)      → 15
#    fishRising                       → 10
#    momentum3 > 0                    → 10
#    htfAlignBuy                      → 15
#                                     ────
#                              total: 100
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
    Confidence scoring — identical to cybercycle_v62.pine.

    Each component contributes a fixed amount:
        Cross signal:  20  — cycle × trigger crossover detected
        iTrend:        15  — trend alignment
        OB/OS zone:    15  — oversold (buy) / overbought (sell)
        Volume:        15  — volume confirmation
        Fisher:        10  — Fisher transform direction
        Momentum:      10  — 3-bar cycle momentum
        HTF:           15  — higher timeframe alignment
        ───────────────────
        Total max:    100
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


class CyberCycleStrategyv3(IStrategy):

    def name(self) -> str:
        return "CyberCycle v6.2 Pine-Parity"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # ── Alpha method (solo Kalman + Manual) ──────────────
            ParamDef('alpha_method', 'categorical', 'kalman',
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
            ParamDef('use_htf', 'bool', True),

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
            ParamDef('tp1_fixed_pct', 'float', 1.0, 0.5, 5.0, 0.25),
            ParamDef('tp1_fixed_size', 'float', 0.35, 0.1, 0.9, 0.05),
            ParamDef('tp2_fixed_pct', 'float', 2.0, 1.0, 10.0, 0.5),

            # ── Break-even ───────────────────────────────────────
            ParamDef('be_pct', 'float', 1.0, 0.0, 4.5, 0.1),

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
        Calculate all CyberCycle indicators.

        Solo computa el alpha method seleccionado (Kalman o Manual).
        HTF filter usa htf_resample() real con timestamps.
        """
        src = data['hl2']
        close = data['close']
        vol = data['volume']
        n = len(src)

        method = self.get_param('alpha_method', 'kalman')

        # ── Alpha: solo el seleccionado (no los 4) ──────────────
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
            manual_a = self.get_param('manual_alpha', 0.42)
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

        # ── iTrend ──────────────────────────────────────────────
        it = itrend(close, self.get_param('itrend_alpha', 0.09))
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
        vol_mult = self.get_param('volume_mult', 1.5)
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

        # ════════════════════════════════════════════════════════
        #  HTF FILTER — Pine-parity con htf_resample()
        #
        #  Pine:
        #    htfSrc = request.security(syminfo.tickerid, "240", hl2)
        #    htfCC  = request.security(syminfo.tickerid, "240", ta.ema(close,10))
        #    htfBullish = htfSrc > htfCC
        #
        #  Python equivalente:
        #    1. Resample hl2 al cierre del bar 4H anterior
        #    2. Resample ema(close,10) al cierre del bar 4H anterior
        #    3. Comparar sin future leak
        # ════════════════════════════════════════════════════════
        use_htf = self.get_param('use_htf', True)

        if use_htf and 'timestamp' in data and data['timestamp'] is not None:
            ts = data['timestamp']

            # Pine: request.security("240", hl2, lookahead=off)
            # → valor de hl2 al cierre del último bar 4H completado
            htf_src = htf_resample(src, ts, 14400)

            # Pine: request.security("240", ta.ema(close, 10), lookahead=off)
            # → EMA(close,10) evaluado en el TF actual, resampleado a 4H
            ema10_close = ema(close, 10)
            htf_cc = htf_resample(ema10_close, ts, 14400)

            htf_align_buy = htf_src > htf_cc
            htf_align_sell = htf_src < htf_cc
        elif use_htf:
            # Fallback si no hay timestamps: proxy EMA larga (menos preciso)
            htf_src_fb = ema(src, 40)
            htf_cc_fb = ema(close, 40)
            htf_align_buy = htf_src_fb > htf_cc_fb
            htf_align_sell = htf_src_fb < htf_cc_fb
        else:
            # HTF desactivado → siempre true
            htf_align_buy = np.ones(n, dtype=bool)
            htf_align_sell = np.ones(n, dtype=bool)

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
            'htf_align_buy': htf_align_buy,
            'htf_align_sell': htf_align_sell,
            # Diagnostics — solo Kalman + Manual
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
    #  SIGNAL GENERATION — Pine-parity
    # ─────────────────────────────────────────────────────────────

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """
        Generate CyberCycle signal at bar idx.

        Pine-exact logic:
          buySignal = bullCross
                      AND buyConf >= minConf
                      AND barFilter
                      AND (eUseTrend ? bullTrend : true)
                      AND htfAlignBuy
        """
        if idx < 10:
            return None

        bull_cross = ind['bull_cross'][idx]
        bear_cross = ind['bear_cross'][idx]

        if not (bull_cross or bear_cross):
            return None

        # ── Bar filter (min_bars between signals) ───────────────
        min_bars = self.get_param('min_bars', 16)
        if idx - self._last_signal_bar < min_bars:
            return None

        use_trend = self.get_param('use_trend', True)
        use_vol = self.get_param('use_volume', True)
        is_buy = bull_cross

        # ═════════════════════════════════════════════════════════
        #  CONFIDENCE — Pine-exact weights (20/15/15/15/10/10/15)
        # ═════════════════════════════════════════════════════════
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

        # ── Confidence gate ─────────────────────────────────────
        confidence_min = self.get_param('confidence_min', 75.0)
        if conf < confidence_min:
            return None

        # ═════════════════════════════════════════════════════════
        #  HARD TREND FILTER — post-confidence, como Pine
        #
        #  Pine: AND (eUseTrend ? bullTrend : true) AND htfAlignBuy
        #  Esto es un gate separado del confidence scoring.
        #  Una señal puede tener conf=100 pero ser rechazada
        #  si el trend no está alineado.
        # ═════════════════════════════════════════════════════════
        if use_trend:
            if is_buy and not ind['bull_trend'][idx]:
                return None
            if not is_buy and not ind['bear_trend'][idx]:
                return None

        # ── Hard HTF filter — post-confidence, como Pine ────────
        use_htf = self.get_param('use_htf', True)
        if use_htf:
            if is_buy and not ind['htf_align_buy'][idx]:
                return None
            if not is_buy and not ind['htf_align_sell'][idx]:
                return None

        # ── Daily signal cap ────────────────────────────────────
        max_daily = self.get_param('max_signals_per_day', 0)
        if max_daily > 0:
            if hasattr(self, '_daily_count') and hasattr(self, '_current_day'):
                ts = data.get('timestamp', None)
                if ts is not None and idx < len(ts):
                    day = int(ts[idx]) // 86400000
                else:
                    day = idx // 24  # fallback: assume 1h bars
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

        # ═════════════════════════════════════════════════════════
        #  ENTRY PRICE = hl2 (Pine-parity para backtest histórico)
        #
        #  Pine: capturedBuyPrice = barstate.ishistory ? srcCyber : close
        #  srcCyber = hl2
        #  En backtest todo es "history", así que entry = hl2
        # ═════════════════════════════════════════════════════════
        entry = data['hl2'][idx]

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
                'entry_source': 'hl2',
                'htf_align': bool(ind['htf_align_buy'][idx] if is_buy else ind['htf_align_sell'][idx]),
            }
        )

    # ─────────────────────────────────────────────────────────────
    #  INCREMENTAL PROCESSOR (para intrabar execution)
    # ─────────────────────────────────────────────────────────────

    def create_incremental_processor(self, detail_tf_ratio: int = 1):
        """
        Create incremental processor for intrabar execution.
        Uses IncrementalCyberCycleV2 (v6.3 confidence) which is the
        closest match to this strategy's confidence weights.

        Note: The incremental processor uses its own confidence weights
        (v6.3: 20/25/20/20/10/5 without HTF) which differ from Pine.
        For maximum Pine parity, use --no-intrabar (bar-close mode).
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