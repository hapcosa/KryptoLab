"""
CryptoLab — Market Regime Detector
Identifies market regimes for adaptive strategy selection.

Regime types:
  1. Trend-Up:    sustained upward movement, low-to-medium volatility
  2. Trend-Down:  sustained downward movement, low-to-medium volatility
  3. Ranging:     sideways consolidation, low volatility
  4. High-Vol Up: sharp upward moves, high volatility (breakout/rally)
  5. High-Vol Down: sharp downward moves, high volatility (crash/panic)

Methods:
  A. Volatility-Trend Classification (rule-based, fast)
  B. Rolling Feature Clustering (KMeans on rolling stats)
  C. Hidden Markov Model (Gaussian HMM via hmmlearn if available)

Each bar is assigned a regime label. Strategies can then:
  - Only trade in compatible regimes
  - Switch parameters by regime
  - Size positions by regime confidence
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import IntEnum

from indicators.common import ema, atr, sma


class Regime(IntEnum):
    """Market regime labels."""
    TREND_UP = 1
    TREND_DOWN = 2
    RANGING = 3
    HIGH_VOL_UP = 4
    HIGH_VOL_DOWN = 5


REGIME_NAMES = {
    Regime.TREND_UP: "Trend Up",
    Regime.TREND_DOWN: "Trend Down",
    Regime.RANGING: "Ranging",
    Regime.HIGH_VOL_UP: "High-Vol Up",
    Regime.HIGH_VOL_DOWN: "High-Vol Down",
}


@dataclass
class RegimeSegment:
    """A contiguous segment of a single regime."""
    regime: int
    start_idx: int
    end_idx: int
    duration: int
    avg_return: float = 0.0
    volatility: float = 0.0
    direction: float = 0.0


@dataclass
class RegimeResult:
    """Complete regime detection result."""
    labels: np.ndarray            # Regime label per bar
    probabilities: Optional[np.ndarray] = None  # Regime probability per bar (n, n_regimes)
    segments: List[RegimeSegment] = field(default_factory=list)
    regime_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    method: str = ""
    n_regimes: int = 5
    features: Optional[Dict[str, np.ndarray]] = None


# ═══════════════════════════════════════════════════════════════
#  METHOD A: Volatility-Trend Classification (Rule-Based)
# ═══════════════════════════════════════════════════════════════

def classify_vt(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                trend_period: int = 50,
                vol_period: int = 20,
                vol_lookback: int = 100,
                trend_threshold: float = 0.002,
                vol_percentile: float = 70.0
                ) -> RegimeResult:
    """
    Volatility-Trend regime classification (rule-based).

    Logic:
      1. Trend = sign of EMA slope (EMA(close, trend_period))
      2. Volatility = ATR(vol_period) / close
      3. High-vol = normalized ATR > percentile threshold

    Regime assignment:
      - High-vol + Up slope   → HIGH_VOL_UP
      - High-vol + Down slope → HIGH_VOL_DOWN
      - Low-vol + Up slope    → TREND_UP
      - Low-vol + Down slope  → TREND_DOWN
      - Low-vol + Flat slope  → RANGING
    """
    n = len(close)
    labels = np.full(n, Regime.RANGING, dtype=int)

    # Trend: EMA slope
    ema_line = ema(close, trend_period)
    slope = np.zeros(n)
    for i in range(1, n):
        slope[i] = (ema_line[i] - ema_line[i - 1]) / close[i] if close[i] > 0 else 0

    # Smoothed slope
    slope_smooth = ema(slope, 10)

    # Volatility: normalized ATR
    atr_vals = atr(high, low, close, vol_period)
    norm_vol = np.zeros(n)
    for i in range(n):
        norm_vol[i] = atr_vals[i] / close[i] if close[i] > 0 else 0

    # Rolling vol percentile threshold
    vol_threshold = np.zeros(n)
    for i in range(vol_lookback, n):
        window = norm_vol[max(0, i - vol_lookback):i + 1]
        vol_threshold[i] = np.percentile(window, vol_percentile)
    # Fill early bars
    if vol_lookback < n:
        vol_threshold[:vol_lookback] = vol_threshold[vol_lookback]

    # Classify
    for i in range(n):
        is_high_vol = norm_vol[i] > vol_threshold[i]
        is_up = slope_smooth[i] > trend_threshold
        is_down = slope_smooth[i] < -trend_threshold

        if is_high_vol:
            labels[i] = Regime.HIGH_VOL_UP if is_up else (
                Regime.HIGH_VOL_DOWN if is_down else Regime.RANGING)
        else:
            if is_up:
                labels[i] = Regime.TREND_UP
            elif is_down:
                labels[i] = Regime.TREND_DOWN
            else:
                labels[i] = Regime.RANGING

    # Build segments and stats
    features = {
        'ema': ema_line,
        'slope': slope_smooth,
        'norm_vol': norm_vol,
        'vol_threshold': vol_threshold,
    }

    segments = _build_segments(labels, close)
    stats = _compute_regime_stats(labels, close, n)

    return RegimeResult(
        labels=labels,
        segments=segments,
        regime_stats=stats,
        method='volatility_trend',
        features=features,
    )


# ═══════════════════════════════════════════════════════════════
#  METHOD B: Rolling Feature Clustering (KMeans)
# ═══════════════════════════════════════════════════════════════

def classify_cluster(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     volume: np.ndarray,
                     n_regimes: int = 4,
                     window: int = 20,
                     seed: int = 42
                     ) -> RegimeResult:
    """
    Cluster-based regime detection using rolling features + KMeans.

    Features per bar (computed over rolling window):
      - Return (log)
      - Volatility (std of returns)
      - Volume ratio (vs mean)
      - Range ratio (high-low / close)
      - Trend strength (abs slope of linear fit)

    KMeans clusters these feature vectors into n_regimes groups.
    Cluster labels are post-mapped to Regime enum by centroid characteristics.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # Fallback to VT method
        return classify_vt(close, high, low)

    n = len(close)

    # Compute rolling features
    log_returns = np.zeros(n)
    for i in range(1, n):
        log_returns[i] = np.log(close[i] / close[i - 1]) if close[i - 1] > 0 else 0

    features = np.zeros((n, 5))
    for i in range(window, n):
        w_rets = log_returns[i - window + 1:i + 1]
        w_close = close[i - window + 1:i + 1]
        w_vol = volume[i - window + 1:i + 1]
        w_range = (high[i - window + 1:i + 1] - low[i - window + 1:i + 1])

        features[i, 0] = np.mean(w_rets)                          # Return
        features[i, 1] = np.std(w_rets) if len(w_rets) > 1 else 0  # Volatility
        features[i, 2] = (volume[i] / np.mean(w_vol)
                          if np.mean(w_vol) > 0 else 1.0)          # Vol ratio
        features[i, 3] = np.mean(w_range / np.maximum(w_close, 1e-12))  # Range
        features[i, 4] = abs(np.polyfit(range(window), w_close, 1)[0]
                              / np.mean(w_close)) if np.mean(w_close) > 0 else 0  # Trend

    # Scale and cluster (only valid bars)
    valid_mask = np.arange(n) >= window
    X = features[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Map clusters to regime types based on centroid characteristics
    labels = np.full(n, Regime.RANGING, dtype=int)
    labels[valid_mask] = _map_clusters_to_regimes(
        cluster_labels, kmeans.cluster_centers_, scaler)

    segments = _build_segments(labels, close)
    stats = _compute_regime_stats(labels, close, n)

    return RegimeResult(
        labels=labels,
        segments=segments,
        regime_stats=stats,
        method='cluster',
        n_regimes=n_regimes,
        features={'raw': features},
    )


def _map_clusters_to_regimes(labels: np.ndarray,
                              centroids: np.ndarray,
                              scaler) -> np.ndarray:
    """
    Map KMeans cluster IDs to Regime enum based on centroid features.
    Features: [return, volatility, vol_ratio, range, trend_strength]
    """
    # Inverse-transform centroids to original scale
    centroids_orig = scaler.inverse_transform(centroids)

    mapping = {}
    for i in range(len(centroids_orig)):
        ret = centroids_orig[i, 0]      # Mean return
        vol = centroids_orig[i, 1]      # Volatility
        trend = centroids_orig[i, 4]    # Trend strength

        # Determine regime by centroid characteristics
        high_vol = vol > np.median(centroids_orig[:, 1])
        positive_ret = ret > 0.0001
        negative_ret = ret < -0.0001

        if high_vol and positive_ret:
            mapping[i] = Regime.HIGH_VOL_UP
        elif high_vol and negative_ret:
            mapping[i] = Regime.HIGH_VOL_DOWN
        elif not high_vol and positive_ret:
            mapping[i] = Regime.TREND_UP
        elif not high_vol and negative_ret:
            mapping[i] = Regime.TREND_DOWN
        else:
            mapping[i] = Regime.RANGING

    return np.array([mapping.get(l, Regime.RANGING) for l in labels], dtype=int)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _build_segments(labels: np.ndarray, close: np.ndarray
                    ) -> List[RegimeSegment]:
    """Build contiguous regime segments."""
    n = len(labels)
    segments = []
    seg_start = 0
    current = labels[0]

    for i in range(1, n):
        if labels[i] != current or i == n - 1:
            end = i if labels[i] != current else i + 1
            duration = end - seg_start

            seg_close = close[seg_start:end]
            ret = ((seg_close[-1] - seg_close[0]) / seg_close[0] * 100
                   if len(seg_close) > 1 and seg_close[0] > 0 else 0)

            rets = np.diff(seg_close) / seg_close[:-1] if len(seg_close) > 1 else [0]
            vol = np.std(rets) if len(rets) > 1 else 0

            direction = 1 if ret > 0 else (-1 if ret < 0 else 0)

            segments.append(RegimeSegment(
                regime=int(current),
                start_idx=seg_start,
                end_idx=end,
                duration=duration,
                avg_return=ret,
                volatility=vol,
                direction=direction,
            ))

            seg_start = i
            current = labels[i]

    return segments


def _compute_regime_stats(labels: np.ndarray, close: np.ndarray,
                          n: int) -> Dict[int, Dict[str, float]]:
    """Compute summary statistics per regime."""
    stats = {}
    for regime in [1, 2, 3, 4, 5]:
        mask = labels == regime
        count = np.sum(mask)

        if count < 2:
            stats[regime] = {
                'count': int(count),
                'pct': count / n * 100,
                'avg_return': 0.0,
                'avg_vol': 0.0,
            }
            continue

        # Returns during this regime
        regime_close = close[mask]
        rets = np.diff(regime_close) / regime_close[:-1] if len(regime_close) > 1 else [0]

        stats[regime] = {
            'count': int(count),
            'pct': count / n * 100,
            'avg_return': float(np.mean(rets) * 100),
            'avg_vol': float(np.std(rets) * 100),
            'name': REGIME_NAMES.get(regime, f"Regime {regime}"),
        }

    return stats


def detect_regime(data: dict,
                  method: str = 'vt',
                  verbose: bool = True,
                  **kwargs) -> RegimeResult:
    """
    Unified regime detection interface.

    Args:
        data: OHLCV data dict
        method: 'vt' (volatility-trend), 'cluster' (KMeans)
        verbose: Print regime summary
        **kwargs: Method-specific parameters

    Returns:
        RegimeResult with labels, segments, and stats
    """
    close = data['close']
    high = data['high']
    low = data['low']

    if method == 'vt':
        result = classify_vt(close, high, low, **kwargs)
    elif method == 'cluster':
        volume = data.get('volume', np.ones_like(close))
        result = classify_cluster(close, high, low, volume, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'vt' or 'cluster'.")

    if verbose:
        print(f"\n  Regime Detection ({method})")
        print(f"  {'─' * 50}")
        for regime_id, st in result.regime_stats.items():
            if st['count'] > 0:
                name = REGIME_NAMES.get(regime_id, f"R{regime_id}")
                print(f"    {name:15s}: {st['pct']:5.1f}% of bars "
                      f"({st['count']} bars)")
        print(f"    Segments: {len(result.segments)}")

    return result


def strategy_regime_performance(strategy,
                                data: dict,
                                engine_factory: Callable = None,
                                regime_result: RegimeResult = None,
                                symbol: str = "",
                                timeframe: str = "",
                                verbose: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Analyze strategy performance broken down by regime.

    Returns dict[regime_id → {sharpe, return, win_rate, n_trades}]
    """
    import copy
    from core.engine import BacktestEngine

    if regime_result is None:
        regime_result = detect_regime(data, method='vt', verbose=False)

    if engine_factory is None:
        def engine_factory():
            return BacktestEngine(initial_capital=10000)

    # Run full backtest
    strat = copy.deepcopy(strategy)
    engine = engine_factory()
    result = engine.run(strat, data, symbol, timeframe)

    # Assign each trade to a regime based on entry bar
    labels = regime_result.labels
    regime_trades = {r: [] for r in [1, 2, 3, 4, 5]}

    for trade in result.trades:
        bar = trade.entry_bar
        if bar < len(labels):
            regime = labels[bar]
            regime_trades[regime].append(trade)

    perf = {}
    for regime_id, trades in regime_trades.items():
        n = len(trades)
        if n == 0:
            perf[regime_id] = {'n_trades': 0, 'sharpe': 0, 'return': 0, 'win_rate': 0}
            continue

        pnls = [t.net_pnl for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        wr = wins / n * 100

        if n > 1 and np.std(pnls) > 0:
            sr = np.mean(pnls) / np.std(pnls) * np.sqrt(n)
        else:
            sr = 0

        perf[regime_id] = {
            'n_trades': n,
            'sharpe': sr,
            'return': total_pnl,
            'win_rate': wr,
        }

    if verbose:
        print(f"\n  Strategy Performance by Regime")
        print(f"  {'─' * 55}")
        print(f"  {'Regime':15s} {'Trades':>7} {'WR':>7} {'PnL':>10} {'SR':>7}")
        for rid in [1, 2, 3, 4, 5]:
            p = perf.get(rid, {})
            if p.get('n_trades', 0) > 0:
                name = REGIME_NAMES.get(rid, f"R{rid}")
                print(f"  {name:15s} {p['n_trades']:>7} "
                      f"{p['win_rate']:>6.1f}% "
                      f"{p['return']:>+9.1f} "
                      f"{p['sharpe']:>6.2f}")

    return perf
