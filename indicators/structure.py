"""
CryptoLab — Smart Money Concepts Structure Detection
Translation of key SMC concepts from smartmoney.pine (BigBeluga)

Implements:
- Break of Structure (BOS) / Change of Character (CHoCH)
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Swing Failure Patterns (SFP)
- Liquidity Grabs
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from indicators.common import highest, lowest


@dataclass
class StructurePoint:
    """A swing high or swing low."""
    idx: int
    price: float
    is_high: bool
    timestamp: float = 0.0


@dataclass
class StructureBreak:
    """BOS or CHoCH event."""
    idx: int
    price: float
    break_type: str      # 'BOS' or 'CHoCH'
    direction: int       # 1 = bullish, -1 = bearish
    swing_point: StructurePoint = None
    confirmed: bool = False


@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance zone)."""
    idx: int
    top: float
    bottom: float
    direction: int       # 1 = bullish FVG, -1 = bearish FVG
    mitigated: bool = False
    mitigation_idx: int = -1


@dataclass
class OrderBlock:
    """Volumetric Order Block."""
    idx: int
    top: float
    bottom: float
    direction: int       # 1 = bullish OB, -1 = bearish OB
    strength: float = 0.0  # Liquidation percentage
    mitigated: bool = False


# ═══════════════════════════════════════════════════════════════
#  SWING POINT DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_swing_points(high: np.ndarray, low: np.ndarray,
                        swing_length: int = 5
                        ) -> Tuple[List[StructurePoint], List[StructurePoint]]:
    """
    Detect swing highs and swing lows.
    A swing high requires `swing_length` bars on each side with lower highs.
    """
    n = len(high)
    swing_highs = []
    swing_lows = []
    
    for i in range(swing_length, n - swing_length):
        # Swing High
        is_sh = True
        for j in range(1, swing_length + 1):
            if high[i - j] >= high[i] or high[i + j] >= high[i]:
                is_sh = False
                break
        if is_sh:
            swing_highs.append(StructurePoint(idx=i, price=high[i], is_high=True))
        
        # Swing Low
        is_sl = True
        for j in range(1, swing_length + 1):
            if low[i - j] <= low[i] or low[i + j] <= low[i]:
                is_sl = False
                break
        if is_sl:
            swing_lows.append(StructurePoint(idx=i, price=low[i], is_high=False))
    
    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════
#  MARKET STRUCTURE — BOS & CHoCH
# ═══════════════════════════════════════════════════════════════

def detect_structure(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     swing_length: int = 5,
                     internal_length: int = 5,
                     swing_limit: int = 50
                     ) -> Tuple[List[StructureBreak], np.ndarray]:
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character).
    
    BOS: price breaks a swing point in the direction of the current trend.
    CHoCH: price breaks a swing point AGAINST the current trend → reversal signal.
    
    Returns:
        breaks: list of StructureBreak events
        trend: array of trend direction per bar (1=bull, -1=bear, 0=neutral)
    """
    n = len(high)
    breaks = []
    trend = np.zeros(n, dtype=int)
    
    swing_highs, swing_lows = detect_swing_points(high, low, swing_length)
    
    if not swing_highs or not swing_lows:
        return breaks, trend
    
    # Track current trend and key levels
    current_trend = 0  # 0=neutral, 1=bullish, -1=bearish
    last_significant_high = swing_highs[0].price if swing_highs else high[0]
    last_significant_low = swing_lows[0].price if swing_lows else low[0]
    last_high_idx = swing_highs[0].idx if swing_highs else 0
    last_low_idx = swing_lows[0].idx if swing_lows else 0
    
    # Combine and sort all swing points by index
    all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.idx)
    
    for sp in all_swings:
        i = sp.idx
        if i >= n:
            break
        
        if sp.is_high:
            # Check if price breaks above the last significant high
            if close[i] > last_significant_high and i > last_high_idx:
                if current_trend == -1:
                    # Was bearish, now breaking high → CHoCH (bullish reversal)
                    breaks.append(StructureBreak(
                        idx=i, price=close[i],
                        break_type='CHoCH', direction=1,
                        swing_point=sp, confirmed=True
                    ))
                else:
                    # Continuation → BOS (bullish)
                    breaks.append(StructureBreak(
                        idx=i, price=close[i],
                        break_type='BOS', direction=1,
                        swing_point=sp, confirmed=True
                    ))
                current_trend = 1
            
            last_significant_high = max(sp.price, last_significant_high)
            last_high_idx = i
        
        else:
            # Check if price breaks below the last significant low
            if close[i] < last_significant_low and i > last_low_idx:
                if current_trend == 1:
                    # Was bullish, now breaking low → CHoCH (bearish reversal)
                    breaks.append(StructureBreak(
                        idx=i, price=close[i],
                        break_type='CHoCH', direction=-1,
                        swing_point=sp, confirmed=True
                    ))
                else:
                    # Continuation → BOS (bearish)
                    breaks.append(StructureBreak(
                        idx=i, price=close[i],
                        break_type='BOS', direction=-1,
                        swing_point=sp, confirmed=True
                    ))
                current_trend = -1
            
            last_significant_low = min(sp.price, last_significant_low)
            last_low_idx = i
    
    # Fill trend array
    current = 0
    break_idx = 0
    for i in range(n):
        while break_idx < len(breaks) and breaks[break_idx].idx <= i:
            current = breaks[break_idx].direction
            break_idx += 1
        trend[i] = current
    
    return breaks, trend


# ═══════════════════════════════════════════════════════════════
#  FAIR VALUE GAPS (FVG)
# ═══════════════════════════════════════════════════════════════

def detect_fvg(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               open_: np.ndarray
               ) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (imbalances in price).
    
    Bullish FVG: gap between bar[2].high and bar[0].low (bar[1] body doesn't cover it)
    Bearish FVG: gap between bar[0].high and bar[2].low
    """
    n = len(high)
    fvgs = []
    
    for i in range(2, n):
        # Bullish FVG: low[i] > high[i-2] — gap up
        if low[i] > high[i-2]:
            fvgs.append(FairValueGap(
                idx=i-1,
                top=low[i],
                bottom=high[i-2],
                direction=1,
            ))
        
        # Bearish FVG: high[i] < low[i-2] — gap down
        if high[i] < low[i-2]:
            fvgs.append(FairValueGap(
                idx=i-1,
                top=low[i-2],
                bottom=high[i],
                direction=-1,
            ))
    
    # Check mitigation
    for fvg in fvgs:
        for i in range(fvg.idx + 2, n):
            if fvg.direction == 1:
                # Bullish FVG mitigated when price drops into it
                if low[i] <= fvg.top:
                    fvg.mitigated = True
                    fvg.mitigation_idx = i
                    break
            else:
                # Bearish FVG mitigated when price rises into it
                if high[i] >= fvg.bottom:
                    fvg.mitigated = True
                    fvg.mitigation_idx = i
                    break
    
    return fvgs


# ═══════════════════════════════════════════════════════════════
#  ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════

def detect_order_blocks(high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, open_: np.ndarray,
                        volume: np.ndarray,
                        ob_length: int = 10,
                        max_blocks: int = 5
                        ) -> List[OrderBlock]:
    """
    Detect Order Blocks — last opposing candle before a strong move.
    
    Bullish OB: last bearish candle before strong bullish move
    Bearish OB: last bullish candle before strong bearish move
    """
    n = len(high)
    blocks = []
    
    avg_vol = np.zeros(n)
    for i in range(n):
        start = max(0, i - 20)
        avg_vol[i] = np.mean(volume[start:i+1]) if i > 0 else volume[0]
    
    for i in range(ob_length + 1, n):
        # Strong bullish move — look for preceding bearish candle
        if close[i] > high[i-1] and close[i] > highest(high[:i], ob_length)[i-1]:
            # Find last bearish candle
            for j in range(i-1, max(0, i-ob_length-1), -1):
                if close[j] < open_[j]:  # Bearish candle
                    strength = volume[j] / avg_vol[j] if avg_vol[j] > 0 else 1.0
                    blocks.append(OrderBlock(
                        idx=j,
                        top=max(close[j], open_[j]),
                        bottom=min(close[j], open_[j]),
                        direction=1,
                        strength=strength,
                    ))
                    break
        
        # Strong bearish move — look for preceding bullish candle
        if close[i] < low[i-1] and close[i] < lowest(low[:i], ob_length)[i-1]:
            for j in range(i-1, max(0, i-ob_length-1), -1):
                if close[j] > open_[j]:  # Bullish candle
                    strength = volume[j] / avg_vol[j] if avg_vol[j] > 0 else 1.0
                    blocks.append(OrderBlock(
                        idx=j,
                        top=max(close[j], open_[j]),
                        bottom=min(close[j], open_[j]),
                        direction=-1,
                        strength=strength,
                    ))
                    break
    
    # Check mitigation
    for ob in blocks:
        for i in range(ob.idx + 1, n):
            if ob.direction == 1:
                if low[i] <= ob.bottom:
                    ob.mitigated = True
                    break
            else:
                if high[i] >= ob.top:
                    ob.mitigated = True
                    break
    
    return blocks


# ═══════════════════════════════════════════════════════════════
#  LIQUIDITY LEVELS (BSL / SSL)
# ═══════════════════════════════════════════════════════════════

def liquidity_levels(high: np.ndarray, low: np.ndarray,
                     pivot_length: int = 5,
                     max_levels: int = 40
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL) levels.
    Pine: BSL/SSL from gaussbands.pine lines 100-140
    
    Returns:
        bsl: nearest BSL above current price per bar
        ssl: nearest SSL below current price per bar
    """
    from indicators.common import pivothigh, pivotlow
    
    n = len(high)
    ph = pivothigh(high, pivot_length, pivot_length)
    pl = pivotlow(low, pivot_length, pivot_length)
    
    bsl_out = np.full(n, np.nan)
    ssl_out = np.full(n, np.nan)
    
    bsl_levels = []
    ssl_levels = []
    
    for i in range(n):
        # Add new pivot highs to BSL pool
        if not np.isnan(ph[i]):
            bsl_levels.append(ph[i])
            if len(bsl_levels) > max_levels:
                bsl_levels.pop(0)
        
        # Add new pivot lows to SSL pool
        if not np.isnan(pl[i]):
            ssl_levels.append(pl[i])
            if len(ssl_levels) > max_levels:
                ssl_levels.pop(0)
        
        # Remove swept levels
        bsl_levels = [lvl for lvl in bsl_levels if high[i] < lvl]
        ssl_levels = [lvl for lvl in ssl_levels if low[i] > lvl]
        
        # Find nearest BSL above
        above = [lvl for lvl in bsl_levels if lvl > high[i]]
        if above:
            bsl_out[i] = min(above)
        
        # Find nearest SSL below
        below = [lvl for lvl in ssl_levels if lvl < low[i]]
        if below:
            ssl_out[i] = max(below)
    
    return bsl_out, ssl_out


# ═══════════════════════════════════════════════════════════════
#  UNIFIED SMC SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════

def smc_signals(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, open_: np.ndarray,
                volume: np.ndarray,
                swing_length: int = 5,
                ob_length: int = 10
                ) -> dict:
    """
    Generate complete SMC analysis.
    Returns dict with all structure info and trading signals.
    """
    n = len(high)
    
    # Structure
    breaks, trend = detect_structure(high, low, close, swing_length)
    
    # FVGs
    fvgs = detect_fvg(high, low, close, open_)
    
    # Order blocks
    obs = detect_order_blocks(high, low, close, open_, volume, ob_length)
    
    # Liquidity
    bsl, ssl = liquidity_levels(high, low)
    
    # Generate signals from structure breaks
    long_signals = np.zeros(n, dtype=bool)
    short_signals = np.zeros(n, dtype=bool)
    
    for brk in breaks:
        if brk.break_type == 'CHoCH' and brk.confirmed:
            if brk.direction == 1:
                long_signals[brk.idx] = True
            else:
                short_signals[brk.idx] = True
        elif brk.break_type == 'BOS' and brk.confirmed:
            if brk.direction == 1:
                long_signals[brk.idx] = True
            else:
                short_signals[brk.idx] = True
    
    return {
        'breaks': breaks,
        'trend': trend,
        'fvgs': fvgs,
        'order_blocks': obs,
        'bsl': bsl,
        'ssl': ssl,
        'long_signal': long_signals,
        'short_signal': short_signals,
    }
