"""
CryptoLab — Smart Money Concepts Strategy
Translation of smcoscillator.pine (v7.3)

Oscillator-based SMC strategy:
- Position scoring via BOS/CHoCH trend detection (60% weight)
- OB proximity scoring (40% weight)
- FVG confluence filter
- Multi-level confidence: L1 (weak) / L2 (medium) / L3 (strong)
- BSL/SSL liquidity-aware TP targets
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.structure import (
    detect_structure, detect_fvg, detect_order_blocks,
    liquidity_levels, smc_signals, StructureBreak
)
from indicators.common import atr, ema, sma


class SmartMoneyStrategy(IStrategy):

    def name(self) -> str:
        return "Smart Money Concepts"

    def parameter_defs(self) -> List[ParamDef]:
        return [
            # Structure detection
            ParamDef('swing_length', 'int', 5, 2, 15),
            ParamDef('ob_lookback', 'int', 5, 2, 20),
            ParamDef('use_sweeps', 'bool', True),

            # Oscillator fusion weights (Pine: wA=0.6, wB=0.4)
            ParamDef('weight_position', 'float', 0.60, 0.0, 1.0, 0.05),
            ParamDef('weight_ob_prox', 'float', 0.40, 0.0, 1.0, 0.05),

            # Signal levels
            ParamDef('signal_mode', 'categorical', 'L2',
                     options=['L1', 'L2', 'L3']),
            ParamDef('l2_threshold', 'float', 60.0, 30.0, 90.0, 5.0),
            ParamDef('l3_threshold', 'float', 80.0, 50.0, 95.0, 5.0),

            # HTF filter
            ParamDef('use_htf_filter', 'bool', True),
            ParamDef('htf_ema_period', 'int', 50, 20, 100),

            # FVG filter
            ParamDef('use_fvg_filter', 'bool', True),
            ParamDef('fvg_lookback', 'int', 20, 5, 50),

            # Risk management
            ParamDef('leverage', 'float', 3.0, 1.0, 25.0, 0.5),
            ParamDef('sl_atr_mult', 'float', 1.5, 0.5, 4.0, 0.1),
            ParamDef('min_bars', 'int', 10, 3, 50),
            ParamDef('use_trailing', 'bool', True),
            ParamDef('be_at_tp1', 'bool', True),

            # TP structure
            ParamDef('tp1_rr', 'float', 1.0, 0.5, 3.0, 0.25),
            ParamDef('tp2_rr', 'float', 2.0, 1.0, 5.0, 0.25),
            ParamDef('tp3_rr', 'float', 3.5, 2.0, 8.0, 0.5),
            ParamDef('tp1_pct', 'float', 0.40, 0.2, 0.6, 0.05),
            ParamDef('tp2_pct', 'float', 0.35, 0.2, 0.5, 0.05),
            ParamDef('tp3_pct', 'float', 0.25, 0.1, 0.4, 0.05),
        ]

    def calculate_indicators(self, data: dict) -> dict:
        """Calculate all SMC indicators: structure, FVGs, OBs, liquidity."""
        high = data['high']
        low = data['low']
        close = data['close']
        open_ = data['open']
        volume = data['volume']
        n = len(close)

        swing_len = self.get_param('swing_length', 5)
        ob_look = self.get_param('ob_lookback', 5)

        # --- Core SMC detection ---
        breaks, trend = detect_structure(high, low, close, swing_len)
        fvgs = detect_fvg(high, low, close, open_)
        obs = detect_order_blocks(high, low, close, open_, volume, ob_look)
        bsl, ssl = liquidity_levels(high, low, swing_len)

        # --- Position score (Layer A): normalized trend position ---
        # Score based on where close is relative to recent swing structure
        pos_score = np.zeros(n)
        lookback = swing_len * 4
        for i in range(lookback, n):
            seg_high = np.max(high[i - lookback:i + 1])
            seg_low = np.min(low[i - lookback:i + 1])
            rng = seg_high - seg_low
            if rng > 0:
                raw = (close[i] - seg_low) / rng  # 0..1
                # Map to -1..+1 centered
                pos_score[i] = (raw - 0.5) * 2.0
            # Adjust by trend direction
            if trend[i] == 1:
                pos_score[i] = min(1.0, pos_score[i] + 0.2)
            elif trend[i] == -1:
                pos_score[i] = max(-1.0, pos_score[i] - 0.2)

        # --- OB Proximity score (Layer B) ---
        ob_prox_score = np.zeros(n)
        for i in range(n):
            # Find nearest unmitigated OBs
            nearest_bull_dist = float('inf')
            nearest_bear_dist = float('inf')

            for ob in obs:
                if ob.mitigated or ob.idx > i:
                    continue
                if ob.direction == 1:  # Bullish OB below
                    dist = (close[i] - ob.top) / close[i] if close[i] > 0 else 1.0
                    if 0 < dist < nearest_bull_dist:
                        nearest_bull_dist = dist
                elif ob.direction == -1:  # Bearish OB above
                    dist = (ob.bottom - close[i]) / close[i] if close[i] > 0 else 1.0
                    if 0 < dist < nearest_bear_dist:
                        nearest_bear_dist = dist

            # Score: close to bullish OB → positive, close to bearish OB → negative
            # Proximity within 2% is strong signal
            bull_prox = max(0, 1.0 - nearest_bull_dist / 0.02) if nearest_bull_dist < float('inf') else 0
            bear_prox = max(0, 1.0 - nearest_bear_dist / 0.02) if nearest_bear_dist < float('inf') else 0
            ob_prox_score[i] = bull_prox - bear_prox

        # --- Fused oscillator (Pine: osc = wA * pos + wB * ob_prox) ---
        wA = self.get_param('weight_position', 0.60)
        wB = self.get_param('weight_ob_prox', 0.40)
        oscillator = wA * pos_score + wB * ob_prox_score
        # Normalize to 0..100 scale
        osc_norm = (oscillator + 1.0) * 50.0  # maps [-1,+1] -> [0, 100]
        osc_norm = np.clip(osc_norm, 0, 100)

        # --- HTF trend filter (proxy: long-period EMA) ---
        htf_period = self.get_param('htf_ema_period', 50)
        htf_ema = ema(close, htf_period)
        htf_bull = close > htf_ema
        htf_bear = close < htf_ema

        # --- FVG confluence ---
        fvg_bull_zones = np.zeros(n, dtype=bool)
        fvg_bear_zones = np.zeros(n, dtype=bool)
        fvg_lookback = self.get_param('fvg_lookback', 20)

        for fvg in fvgs:
            if fvg.mitigated:
                continue
            end_idx = min(fvg.idx + fvg_lookback, n)
            for i in range(fvg.idx, end_idx):
                if fvg.direction == 1:  # Bullish FVG — price entering from above
                    if low[i] <= fvg.top and low[i] >= fvg.bottom:
                        fvg_bull_zones[i] = True
                else:  # Bearish FVG — price entering from below
                    if high[i] >= fvg.bottom and high[i] <= fvg.top:
                        fvg_bear_zones[i] = True

        # --- Signal generation arrays ---
        # CHoCH signals (strongest — reversal)
        choch_long = np.zeros(n, dtype=bool)
        choch_short = np.zeros(n, dtype=bool)
        # BOS signals (continuation)
        bos_long = np.zeros(n, dtype=bool)
        bos_short = np.zeros(n, dtype=bool)

        for brk in breaks:
            if brk.confirmed and brk.idx < n:
                if brk.break_type == 'CHoCH':
                    if brk.direction == 1:
                        choch_long[brk.idx] = True
                    else:
                        choch_short[brk.idx] = True
                elif brk.break_type == 'BOS':
                    if brk.direction == 1:
                        bos_long[brk.idx] = True
                    else:
                        bos_short[brk.idx] = True

        # ATR for SL/TP
        atr_vals = atr(high, low, close, 14)

        return {
            'trend': trend,
            'oscillator': osc_norm,
            'pos_score': pos_score,
            'ob_prox_score': ob_prox_score,
            'choch_long': choch_long,
            'choch_short': choch_short,
            'bos_long': bos_long,
            'bos_short': bos_short,
            'htf_bull': htf_bull,
            'htf_bear': htf_bear,
            'fvg_bull': fvg_bull_zones,
            'fvg_bear': fvg_bear_zones,
            'bsl': bsl,
            'ssl': ssl,
            'atr': atr_vals,
            'breaks': breaks,
            'fvgs': fvgs,
            'order_blocks': obs,
        }

    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """
        Generate SMC signal at bar idx.

        Signal levels (from smcoscillator.pine):
        - L1: BOS only (weak, trend continuation)
        - L2: CHoCH OR BOS + oscillator above threshold (medium)
        - L3: CHoCH + oscillator above threshold + FVG confluence (strong)
        """
        if idx < 30:
            return None

        min_bars = self.get_param('min_bars', 10)
        if idx - self._last_signal_bar < min_bars:
            return None

        mode = self.get_param('signal_mode', 'L2')
        l2_thresh = self.get_param('l2_threshold', 60.0)
        l3_thresh = self.get_param('l3_threshold', 80.0)
        use_htf = self.get_param('use_htf_filter', True)
        use_fvg = self.get_param('use_fvg_filter', True)

        osc = ind['oscillator'][idx]
        is_long = False
        is_short = False
        confidence = 0.0

        # === LONG SIGNALS ===
        has_choch_long = ind['choch_long'][idx]
        has_bos_long = ind['bos_long'][idx]
        htf_ok_long = (not use_htf) or ind['htf_bull'][idx]
        fvg_ok_long = (not use_fvg) or ind['fvg_bull'][idx]

        if mode == 'L1':
            if (has_bos_long or has_choch_long) and htf_ok_long:
                is_long = True
                confidence = osc
        elif mode == 'L2':
            if has_choch_long and osc >= l2_thresh and htf_ok_long:
                is_long = True
                confidence = osc
            elif has_bos_long and osc >= l2_thresh and htf_ok_long:
                is_long = True
                confidence = osc * 0.85  # BOS is weaker than CHoCH
        elif mode == 'L3':
            if has_choch_long and osc >= l3_thresh and htf_ok_long and fvg_ok_long:
                is_long = True
                confidence = osc

        # === SHORT SIGNALS ===
        has_choch_short = ind['choch_short'][idx]
        has_bos_short = ind['bos_short'][idx]
        htf_ok_short = (not use_htf) or ind['htf_bear'][idx]
        fvg_ok_short = (not use_fvg) or ind['fvg_bear'][idx]

        if not is_long:
            if mode == 'L1':
                if (has_bos_short or has_choch_short) and htf_ok_short:
                    is_short = True
                    confidence = 100.0 - osc
            elif mode == 'L2':
                if has_choch_short and osc <= (100 - l2_thresh) and htf_ok_short:
                    is_short = True
                    confidence = 100.0 - osc
                elif has_bos_short and osc <= (100 - l2_thresh) and htf_ok_short:
                    is_short = True
                    confidence = (100.0 - osc) * 0.85
            elif mode == 'L3':
                if has_choch_short and osc <= (100 - l3_thresh) and htf_ok_short and fvg_ok_short:
                    is_short = True
                    confidence = 100.0 - osc

        if not (is_long or is_short):
            return None

        self._last_signal_bar = idx

        # --- Entry, SL, TP ---
        direction = 1 if is_long else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]
        sl_mult = self.get_param('sl_atr_mult', 1.5)

        sl = entry - direction * atr_val * sl_mult
        risk = abs(entry - sl)

        # Use BSL/SSL as dynamic TP targets when available
        bsl_val = ind['bsl'][idx]
        ssl_val = ind['ssl'][idx]

        tp1_rr = self.get_param('tp1_rr', 1.0)
        tp2_rr = self.get_param('tp2_rr', 2.0)
        tp3_rr = self.get_param('tp3_rr', 3.5)

        if direction == 1:
            tp1 = entry + risk * tp1_rr
            tp2 = entry + risk * tp2_rr
            # Use BSL as T3 target if it's reasonable
            if not np.isnan(bsl_val) and bsl_val > tp2:
                tp3 = bsl_val
            else:
                tp3 = entry + risk * tp3_rr
        else:
            tp1 = entry - risk * tp1_rr
            tp2 = entry - risk * tp2_rr
            if not np.isnan(ssl_val) and ssl_val < tp2:
                tp3 = ssl_val
            else:
                tp3 = entry - risk * tp3_rr

        tp_levels = [tp1, tp2, tp3]
        tp_sizes = [
            self.get_param('tp1_pct', 0.40),
            self.get_param('tp2_pct', 0.35),
            self.get_param('tp3_pct', 0.25),
        ]
        total = sum(tp_sizes)
        tp_sizes = [s / total for s in tp_sizes]

        be_trigger = tp1 if self.get_param('be_at_tp1', True) else 0.0

        return Signal(
            direction=direction,
            confidence=confidence,
            entry_price=entry,
            sl_price=sl,
            tp_levels=tp_levels,
            tp_sizes=tp_sizes,
            leverage=self.get_param('leverage', 3.0),
            be_trigger=be_trigger,
            trailing=self.get_param('use_trailing', True),
            trailing_distance=risk * 0.5,
            metadata={
                'oscillator': osc,
                'is_choch': has_choch_long if is_long else has_choch_short,
                'trend': int(ind['trend'][idx]),
                'signal_level': mode,
            }
        )