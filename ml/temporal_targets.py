"""
CryptoLab — Temporal Target Evaluation
Evaluates strategy performance against time-bucketed objectives.

Objective format: "metric:period:threshold:consistency"
Examples:
    "win_rate:weekly:60:0.8"      → WR ≥ 60% in ≥ 80% of weeks
    "profit:monthly:5:0.75"       → Profit ≥ 5% in ≥ 75% of months
    "max_drawdown:daily:-2:0.9"   → DD < 2% in ≥ 90% of days
    "sharpe:monthly:1.0:0.6"      → Sharpe ≥ 1.0 in ≥ 60% of months
    "n_trades:weekly:3:0.7"       → ≥ 3 trades in ≥ 70% of weeks

This module answers: "Is the strategy consistently profitable across time,
or does a single lucky period drive all the returns?"
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TemporalBucket:
    """Performance in a single time bucket."""
    period_start: str
    period_end: str
    n_trades: int = 0
    win_rate: float = 0.0
    profit_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    pnl: float = 0.0


@dataclass
class TargetEvaluation:
    """Evaluation of a single temporal target."""
    spec: str                          # Original spec string
    metric: str                        # 'win_rate', 'profit', etc.
    period: str                        # 'daily', 'weekly', 'monthly'
    threshold: float                   # Target value
    consistency: float                 # Required fraction of periods meeting target
    actual_consistency: float = 0.0    # Actual fraction meeting target
    n_periods: int = 0                 # Total periods evaluated
    n_passing: int = 0                 # Periods meeting target
    passed: bool = False
    buckets: List[TemporalBucket] = field(default_factory=list)


@dataclass
class TemporalResult:
    """Complete temporal target analysis."""
    evaluations: List[TargetEvaluation]
    all_passed: bool = False
    n_targets: int = 0
    n_passed: int = 0
    overall_consistency: float = 0.0   # Avg consistency across all targets
    worst_target: str = ""
    summary_stats: Dict[str, Any] = field(default_factory=dict)


def parse_target_spec(spec: str) -> Tuple[str, str, float, float]:
    """
    Parse target spec string.
    Format: "metric:period:threshold:consistency"
    """
    parts = spec.strip().split(':')
    if len(parts) != 4:
        raise ValueError(f"Invalid spec format: {spec}. "
                         f"Expected metric:period:threshold:consistency")

    metric = parts[0].lower()
    period = parts[1].lower()
    threshold = float(parts[2])
    consistency = float(parts[3])

    valid_metrics = ['win_rate', 'profit', 'max_drawdown', 'sharpe',
                     'n_trades', 'profit_factor', 'avg_pnl']
    valid_periods = ['daily', 'weekly', 'monthly']

    if metric not in valid_metrics:
        raise ValueError(f"Unknown metric: {metric}. Valid: {valid_metrics}")
    if period not in valid_periods:
        raise ValueError(f"Unknown period: {period}. Valid: {valid_periods}")
    if not 0 < consistency <= 1.0:
        raise ValueError(f"Consistency must be in (0, 1]: {consistency}")

    return metric, period, threshold, consistency


def _bucket_trades(trades: list, timestamps: np.ndarray,
                   equity_curve: np.ndarray,
                   period: str) -> List[TemporalBucket]:
    """
    Group trades and equity data into time-based buckets.
    """
    if not trades or len(equity_curve) < 2:
        return []

    # Create a DataFrame of trades with timing info
    trade_records = []
    for t in trades:
        ts = t.entry_time
        if ts > 1e12:
            ts = ts / 1000  # ms → s
        trade_records.append({
            'entry_time': ts,
            'exit_time': t.exit_time / 1000 if t.exit_time > 1e12 else t.exit_time,
            'pnl': t.net_pnl,
            'pnl_pct': t.pnl_pct,
            'direction': t.direction,
            'entry_bar': t.entry_bar,
            'exit_bar': t.exit_bar,
        })

    if not trade_records:
        return []

    df = pd.DataFrame(trade_records)

    # Convert to datetime
    try:
        df['dt'] = pd.to_datetime(df['entry_time'], unit='s')
    except Exception:
        # If timestamps are bar indices, create synthetic dates
        df['dt'] = pd.date_range('2023-01-01', periods=len(df), freq='4h')

    # Group by period
    if period == 'daily':
        df['bucket'] = df['dt'].dt.date
    elif period == 'weekly':
        df['bucket'] = df['dt'].dt.isocalendar().week.astype(str) + '-' + df['dt'].dt.year.astype(str)
    elif period == 'monthly':
        df['bucket'] = df['dt'].dt.to_period('M').astype(str)

    buckets = []
    for bucket_key, group in df.groupby('bucket'):
        n_trades = len(group)
        wins = (group['pnl'] > 0).sum()
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        total_pnl = group['pnl'].sum()
        profit_pct = group['pnl_pct'].sum()

        # Max drawdown within bucket (cumulative PnL)
        cum_pnl = group['pnl'].cumsum()
        peak = cum_pnl.cummax()
        dd_series = peak - cum_pnl
        max_dd = dd_series.max() if len(dd_series) > 0 else 0

        # Per-period Sharpe
        if n_trades > 1 and group['pnl'].std() > 0:
            sharpe = group['pnl'].mean() / group['pnl'].std()
        else:
            sharpe = 0.0

        buckets.append(TemporalBucket(
            period_start=str(bucket_key),
            period_end=str(bucket_key),
            n_trades=n_trades,
            win_rate=wr,
            profit_pct=profit_pct,
            max_drawdown=max_dd,
            sharpe=sharpe,
            pnl=total_pnl,
        ))

    return buckets


def evaluate_target(spec: str, trades: list, timestamps: np.ndarray,
                    equity_curve: np.ndarray,
                    initial_capital: float = 10000.0) -> TargetEvaluation:
    """
    Evaluate a single temporal target.

    Args:
        spec: Target spec string (e.g. "win_rate:weekly:60:0.8")
        trades: List of Trade objects from BacktestResult
        timestamps: Array of bar timestamps
        equity_curve: Equity curve array
        initial_capital: Starting capital

    Returns:
        TargetEvaluation with pass/fail and bucket details
    """
    metric, period, threshold, consistency = parse_target_spec(spec)
    buckets = _bucket_trades(trades, timestamps, equity_curve, period)

    n_periods = len(buckets)
    if n_periods == 0:
        return TargetEvaluation(
            spec=spec, metric=metric, period=period,
            threshold=threshold, consistency=consistency,
            actual_consistency=0.0, n_periods=0, n_passing=0,
            passed=False, buckets=[],
        )

    # Count periods meeting the target
    n_passing = 0
    for b in buckets:
        val = _get_bucket_metric(b, metric)
        if _meets_threshold(val, threshold, metric):
            n_passing += 1

    actual_consistency = n_passing / n_periods
    passed = actual_consistency >= consistency

    return TargetEvaluation(
        spec=spec, metric=metric, period=period,
        threshold=threshold, consistency=consistency,
        actual_consistency=actual_consistency,
        n_periods=n_periods, n_passing=n_passing,
        passed=passed, buckets=buckets,
    )


def _get_bucket_metric(bucket: TemporalBucket, metric: str) -> float:
    """Extract a specific metric from a bucket."""
    mapping = {
        'win_rate': bucket.win_rate,
        'profit': bucket.profit_pct,
        'max_drawdown': -bucket.max_drawdown,  # Negative because threshold is negative
        'sharpe': bucket.sharpe,
        'n_trades': bucket.n_trades,
        'avg_pnl': bucket.pnl / max(1, bucket.n_trades),
        'profit_factor': 0.0,  # Would need win/loss separation
    }
    return mapping.get(metric, 0.0)


def _meets_threshold(value: float, threshold: float, metric: str) -> bool:
    """Check if metric value meets the threshold."""
    if metric == 'max_drawdown':
        return value >= threshold  # threshold is negative, DD is negative
    return value >= threshold


def evaluate_targets(target_specs: List[str],
                     trades: list,
                     timestamps: np.ndarray,
                     equity_curve: np.ndarray,
                     initial_capital: float = 10000.0,
                     verbose: bool = True) -> TemporalResult:
    """
    Evaluate multiple temporal targets.

    Args:
        target_specs: List of spec strings
        trades: Trade list from BacktestResult
        timestamps: Timestamp array
        equity_curve: Equity curve
        initial_capital: Starting capital
        verbose: Print results

    Returns:
        TemporalResult with all evaluations
    """
    evaluations = []
    for spec in target_specs:
        ev = evaluate_target(spec, trades, timestamps,
                            equity_curve, initial_capital)
        evaluations.append(ev)

    n_passed = sum(1 for e in evaluations if e.passed)
    all_passed = n_passed == len(evaluations)
    avg_consistency = np.mean([e.actual_consistency for e in evaluations]) if evaluations else 0

    worst = ""
    if evaluations:
        worst_ev = min(evaluations, key=lambda e: e.actual_consistency)
        worst = worst_ev.spec

    # Summary stats per period
    summary = {}
    for ev in evaluations:
        if ev.buckets:
            metrics = [_get_bucket_metric(b, ev.metric) for b in ev.buckets]
            summary[ev.spec] = {
                'mean': np.mean(metrics),
                'std': np.std(metrics),
                'min': np.min(metrics),
                'max': np.max(metrics),
                'median': np.median(metrics),
            }

    result = TemporalResult(
        evaluations=evaluations,
        all_passed=all_passed,
        n_targets=len(evaluations),
        n_passed=n_passed,
        overall_consistency=avg_consistency,
        worst_target=worst,
        summary_stats=summary,
    )

    if verbose:
        print(f"\n  Temporal Target Analysis ({len(evaluations)} targets)")
        print(f"  {'─' * 60}")
        for ev in evaluations:
            icon = "✅" if ev.passed else "❌"
            print(f"  {icon} {ev.spec}")
            print(f"      {ev.n_passing}/{ev.n_periods} periods pass "
                  f"({ev.actual_consistency:.0%} vs {ev.consistency:.0%} required)")
        print(f"  {'─' * 60}")
        status = "✅ ALL PASSED" if all_passed else f"❌ {n_passed}/{len(evaluations)} passed"
        print(f"  {status} | Overall consistency: {avg_consistency:.0%}")

    return result


# ─── Default target sets ───

CONSERVATIVE_TARGETS = [
    "win_rate:weekly:50:0.75",
    "profit:monthly:2:0.60",
    "max_drawdown:daily:-3:0.90",
    "n_trades:weekly:2:0.70",
]

AGGRESSIVE_TARGETS = [
    "win_rate:weekly:55:0.70",
    "profit:monthly:5:0.60",
    "max_drawdown:daily:-5:0.85",
    "sharpe:monthly:0.5:0.60",
]

CONSISTENCY_TARGETS = [
    "win_rate:weekly:45:0.85",
    "profit:monthly:1:0.80",
    "max_drawdown:daily:-2:0.95",
    "n_trades:weekly:3:0.80",
]
