"""
CryptoLab — Parallel Regime & Targets
Concurrent execution for regime detection and targets evaluation.

These are lighter optimizations compared to validate/download,
but still provide measurable speedup.

Usage in cli.py:
    # Replace cmd_regime:
    from optimize.task_parallel import parallel_regime, parallel_targets

    def cmd_regime(args):
        ...
        rr, perf = parallel_regime(strategy, data, engine_factory, symbol, tf)

    def cmd_targets(args):
        ...
        tt_result = parallel_targets(
            strategy, data, engine_factory, symbol, tf,
            target_specs, capital)
"""
import numpy as np
import copy
import time
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_regime(strategy, data, engine_factory,
                    symbol="", timeframe="",
                    methods=None, verbose=True):
    """
    Run regime detection and backtest concurrently.

    The key insight: detect_regime() and engine.run() are independent.
    We can run them in parallel, then merge (assign trades → regimes).

    With ThreadPoolExecutor:
      Thread 1: detect_regime(data, method='vt')     ~50ms (NumPy releases GIL)
      Thread 2: engine.run(strategy, data)            ~200ms (NumPy releases GIL)
      Merge: assign trades to regimes                 ~1ms

    Total: ~200ms (vs ~250ms sequential) → ~1.25x speedup
    For multi-method: ~200ms (vs ~350ms) → ~1.75x speedup

    Args:
        strategy: Configured IStrategy
        data: OHLCV data dict
        engine_factory: Callable returning BacktestEngine
        symbol, timeframe: For reporting
        methods: List of detection methods to run (default: ['vt'])
        verbose: Print progress

    Returns:
        (regime_results_dict, perf_dict)
        If single method: returns (RegimeResult, perf_dict) for backward compat
    """
    from ml.regime_detector import (
        detect_regime, strategy_regime_performance,
        REGIME_NAMES
    )

    if methods is None:
        methods = ['vt']

    t0 = time.time()

    # Run detection + backtest concurrently
    with ThreadPoolExecutor(max_workers=len(methods) + 1) as executor:
        # Submit regime detection for each method
        regime_futures = {}
        for method in methods:
            future = executor.submit(
                detect_regime, data, method=method, verbose=False)
            regime_futures[future] = method

        # Submit backtest (independent of regime detection)
        strat = copy.deepcopy(strategy)
        engine = engine_factory()
        bt_future = executor.submit(
            engine.run, strat, data, symbol, timeframe)

        # Collect results
        regime_results = {}
        for future in as_completed(regime_futures):
            method = regime_futures[future]
            try:
                regime_results[method] = future.result()
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Regime detection ({method}) failed: {e}")

        try:
            bt_result = bt_future.result()
        except Exception as e:
            if verbose:
                print(f"  ⚠ Backtest failed: {e}")
            bt_result = None

    # Merge: assign trades to regimes for each method
    perf_results = {}
    if bt_result and bt_result.trades:
        for method, rr in regime_results.items():
            labels = rr.labels
            regime_trades = {r: [] for r in [1, 2, 3, 4, 5]}

            for trade in bt_result.trades:
                bar = trade.entry_bar
                if bar < len(labels):
                    regime = labels[bar]
                    regime_trades[regime].append(trade)

            perf = {}
            for regime_id, trades in regime_trades.items():
                n = len(trades)
                if n == 0:
                    perf[regime_id] = {
                        'n_trades': 0, 'sharpe': 0,
                        'return': 0, 'win_rate': 0}
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

            perf_results[method] = perf

    elapsed = time.time() - t0

    if verbose:
        for method, rr in regime_results.items():
            print(f"\n  Regime Detection ({method})")
            print(f"  {'─' * 50}")
            for regime_id, st in rr.regime_stats.items():
                if st['count'] > 0:
                    name = REGIME_NAMES.get(regime_id, f"R{regime_id}")
                    print(f"    {name:15s}: {st['pct']:5.1f}% ({st['count']} bars)")
            print(f"    Segments: {len(rr.segments)}")

        for method, perf in perf_results.items():
            print(f"\n  Strategy Performance by Regime ({method})")
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

        print(f"\n  ⚡ Completed in {elapsed:.2f}s")

    # Backward compatibility: if single method, return (rr, perf) directly
    if len(methods) == 1:
        m = methods[0]
        return regime_results.get(m), perf_results.get(m, {})

    return regime_results, perf_results


def parallel_targets(strategy, data, engine_factory,
                     symbol="", timeframe="",
                     target_specs=None, capital=10000.0,
                     verbose=True):
    """
    Optimized targets evaluation with pre-computed buckets.

    Optimization: _bucket_trades() is called once per period type
    instead of once per target spec (avoids redundant groupby ops).

    Args:
        strategy: Configured IStrategy
        data: OHLCV data dict
        engine_factory: Callable returning BacktestEngine
        symbol, timeframe: For reporting
        target_specs: List of target spec strings
        capital: Initial capital

    Returns:
        TemporalResult
    """
    from ml.temporal_targets import (
        evaluate_targets, evaluate_target,
        parse_target_spec, _bucket_trades,
        _get_bucket_metric, _meets_threshold,
        TemporalResult, TargetEvaluation,
        CONSERVATIVE_TARGETS,
    )

    if target_specs is None:
        target_specs = CONSERVATIVE_TARGETS

    t0 = time.time()

    # Run backtest
    strat = copy.deepcopy(strategy)
    engine = engine_factory()
    result = engine.run(strat, data, symbol, timeframe)

    timestamps = data.get('timestamp', np.arange(len(data['close'])))
    equity = result.equity_curve
    trades = result.trades

    if not trades:
        return TemporalResult(
            evaluations=[], all_passed=False,
            n_targets=len(target_specs), n_passed=0)

    # ── Pre-compute buckets per period (the optimization) ──
    # Instead of calling _bucket_trades() for each target,
    # we call it once per unique period type.
    periods_needed = set()
    parsed_specs = []
    for spec in target_specs:
        metric, period, threshold, consistency = parse_target_spec(spec)
        periods_needed.add(period)
        parsed_specs.append((spec, metric, period, threshold, consistency))

    bucket_cache = {}
    for period in periods_needed:
        bucket_cache[period] = _bucket_trades(trades, timestamps, equity, period)

    # ── Evaluate all targets using cached buckets ──
    evaluations = []
    for spec, metric, period, threshold, consistency in parsed_specs:
        buckets = bucket_cache[period]
        n_periods = len(buckets)

        if n_periods == 0:
            evaluations.append(TargetEvaluation(
                spec=spec, metric=metric, period=period,
                threshold=threshold, consistency=consistency,
                actual_consistency=0.0, n_periods=0, n_passing=0,
                passed=False, buckets=[]))
            continue

        n_passing = sum(
            1 for b in buckets
            if _meets_threshold(_get_bucket_metric(b, metric), threshold, metric)
        )

        actual_consistency = n_passing / n_periods
        passed = actual_consistency >= consistency

        evaluations.append(TargetEvaluation(
            spec=spec, metric=metric, period=period,
            threshold=threshold, consistency=consistency,
            actual_consistency=actual_consistency,
            n_periods=n_periods, n_passing=n_passing,
            passed=passed, buckets=buckets))

    # Build result
    n_passed = sum(1 for e in evaluations if e.passed)
    all_passed = n_passed == len(evaluations)
    avg_consistency = (np.mean([e.actual_consistency for e in evaluations])
                       if evaluations else 0)

    worst = ""
    if evaluations:
        worst_ev = min(evaluations, key=lambda e: e.actual_consistency)
        worst = worst_ev.spec

    tt_result = TemporalResult(
        evaluations=evaluations,
        all_passed=all_passed,
        n_targets=len(evaluations),
        n_passed=n_passed,
        overall_consistency=avg_consistency,
        worst_target=worst,
    )

    elapsed = time.time() - t0

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
        print(f"  ⚡ Completed in {elapsed:.2f}s")

    return tt_result