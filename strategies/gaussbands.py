"""
CryptoLab — Gaussian Bands Strategy
Faithful translation of gaussbands.pine (BigBeluga)

Multi-trend Gaussian filter with:
- 21 overlapping Gaussian filters for trend scoring
- Dynamic bands based on volatility
- Multi-level TP system (TP_MIN → T2 → T3 → T4 → T5)
- BSL/SSL liquidity-aware targets
- Progressive trailing stop
"""
import numpy as np
from typing import Optional, List, Dict, Any

from strategies.base import IStrategy, Signal, ParamDef
from indicators.gaussian import multi_trend, gaussian_signals, compute_tp_levels
from indicators.common import atr, highest, lowest, pivothigh, pivotlow


class GaussianBandsStrategy(IStrategy):
    
    def name(self) -> str:
        return "Gaussian Bands"
    
    def parameter_defs(self) -> List[ParamDef]:
        return [
            # Gaussian params
            ParamDef('length', 'int', 20, 5, 50),
            ParamDef('mode', 'categorical', 'avg', options=['avg', 'median', 'mode']),
            ParamDef('distance', 'float', 1.0, 0.5, 3.0, 0.1),
            ParamDef('show_retest', 'bool', False),
            
            # Target params
            ParamDef('pivot_length', 'int', 5, 2, 15),
            ParamDef('ob_length', 'int', 10, 3, 20),
            ParamDef('atr_length', 'int', 14, 7, 21),
            ParamDef('atr_sl_buffer', 'float', 0.3, 0.1, 1.0, 0.1),
            
            # TP distribution
            ParamDef('tp_min_pct', 'float', 0.30, 0.1, 0.5, 0.05),
            ParamDef('tp_t2_pct', 'float', 0.25, 0.1, 0.4, 0.05),
            ParamDef('tp_t3_pct', 'float', 0.20, 0.05, 0.3, 0.05),
            ParamDef('tp_t4_pct', 'float', 0.15, 0.05, 0.2, 0.05),
            ParamDef('tp_t5_pct', 'float', 0.10, 0.05, 0.2, 0.05),
            
            # Risk
            ParamDef('leverage', 'float', 3.0, 1.0, 25.0, 0.5),
            ParamDef('use_trailing', 'bool', True),
            ParamDef('be_at_tpmin', 'bool', True),
        ]
    
    def calculate_indicators(self, data: dict) -> dict:
        """Calculate Gaussian Bands indicators."""
        close = data['close']
        high = data['high']
        low = data['low']
        n = len(close)
        
        length = self.get_param('length', 20)
        mode = self.get_param('mode', 'avg')
        distance = self.get_param('distance', 1.0)
        
        # Multi-trend analysis
        score, avg_value, lower_band, upper_band, trend_line = multi_trend(
            close, high, low, length, distance, mode
        )
        
        # Trend state
        trend = np.zeros(n, dtype=bool)
        for i in range(1, n):
            trend[i] = trend[i-1]
            if close[i] > upper_band[i] and close[i-1] <= upper_band[i-1]:
                trend[i] = True
            if close[i] < lower_band[i] and close[i-1] >= lower_band[i-1]:
                trend[i] = False
        
        # Signals
        long_sig, short_sig, retest_long, retest_short = gaussian_signals(
            close, trend_line, avg_value, trend,
            self.get_param('show_retest', False)
        )
        
        # ATR
        atr_vals = atr(high, low, close, self.get_param('atr_length', 14))
        
        # OB high/low for SL
        ob_len = self.get_param('ob_length', 10)
        ob_high = np.zeros(n)
        ob_low = np.zeros(n)
        for i in range(ob_len + 1, n):
            ob_high[i] = np.max(high[i-ob_len-1:i])
            ob_low[i] = np.min(low[i-ob_len-1:i])
        
        # BSL / SSL
        piv_len = self.get_param('pivot_length', 5)
        ph = pivothigh(high, piv_len, piv_len)
        pl = pivotlow(low, piv_len, piv_len)
        
        # Track nearest BSL/SSL
        bsl = np.full(n, np.nan)
        ssl = np.full(n, np.nan)
        bsl_levels = []
        ssl_levels = []
        
        for i in range(n):
            if not np.isnan(ph[i]):
                bsl_levels.append(ph[i])
                if len(bsl_levels) > 40:
                    bsl_levels.pop(0)
            if not np.isnan(pl[i]):
                ssl_levels.append(pl[i])
                if len(ssl_levels) > 40:
                    ssl_levels.pop(0)
            
            bsl_levels = [l for l in bsl_levels if high[i] < l]
            ssl_levels = [l for l in ssl_levels if low[i] > l]
            
            above = [l for l in bsl_levels if l > high[i]]
            if above:
                bsl[i] = min(above)
            below = [l for l in ssl_levels if l < low[i]]
            if below:
                ssl[i] = max(below)
        
        return {
            'score': score,
            'avg_value': avg_value,
            'lower_band': lower_band,
            'upper_band': upper_band,
            'trend_line': trend_line,
            'trend': trend,
            'long_signal': long_sig,
            'short_signal': short_sig,
            'retest_long': retest_long,
            'retest_short': retest_short,
            'atr': atr_vals,
            'ob_high': ob_high,
            'ob_low': ob_low,
            'bsl': bsl,
            'ssl': ssl,
        }
    
    def generate_signal(self, ind: dict, idx: int,
                        data: dict) -> Optional[Signal]:
        """Generate Gaussian Bands signal at bar idx."""
        if idx < 25:
            return None
        
        is_long = ind['long_signal'][idx]
        is_short = ind['short_signal'][idx]
        
        if not (is_long or is_short):
            return None
        
        direction = 1 if is_long else -1
        entry = data['close'][idx]
        atr_val = ind['atr'][idx]
        buf = self.get_param('atr_sl_buffer', 0.3)
        
        # SL based on OB (replicates Pine)
        if direction == 1:
            sl = ind['ob_low'][idx] - atr_val * buf
        else:
            sl = ind['ob_high'][idx] + atr_val * buf
        
        # Ensure SL makes sense
        if direction == 1 and sl >= entry:
            sl = entry - atr_val * 1.5
        if direction == -1 and sl <= entry:
            sl = entry + atr_val * 1.5
        
        # Multi-level TP (replicates gaussbands.pine TP system)
        bsl_val = ind['bsl'][idx] if not np.isnan(ind['bsl'][idx]) else 0.0
        ssl_val = ind['ssl'][idx] if not np.isnan(ind['ssl'][idx]) else 0.0
        
        tp_data = compute_tp_levels(
            entry=entry, sl=sl, direction=direction,
            h1h=bsl_val, h1l=ssl_val,
            atr_val=atr_val
        )
        
        tp_levels = [
            tp_data['tp_min'],
            tp_data['t2'],
            tp_data['t3'],
            tp_data['t4'],
            tp_data['t5'],
        ]
        
        tp_sizes = [
            self.get_param('tp_min_pct', 0.30),
            self.get_param('tp_t2_pct', 0.25),
            self.get_param('tp_t3_pct', 0.20),
            self.get_param('tp_t4_pct', 0.15),
            self.get_param('tp_t5_pct', 0.10),
        ]
        
        # Normalize sizes
        total = sum(tp_sizes)
        tp_sizes = [s / total for s in tp_sizes]
        
        be_trigger = tp_data['tp_min'] if self.get_param('be_at_tpmin', True) else 0.0
        
        return Signal(
            direction=direction,
            confidence=ind['score'][idx] * 100,
            entry_price=entry,
            sl_price=sl,
            tp_levels=tp_levels,
            tp_sizes=tp_sizes,
            leverage=self.get_param('leverage', 3.0),
            be_trigger=be_trigger,
            trailing=self.get_param('use_trailing', True),
            trailing_distance=tp_data['risk'] * 0.5,
            metadata={
                'score': ind['score'][idx],
                'risk': tp_data['risk'],
                'tp_data': tp_data,
            }
        )
