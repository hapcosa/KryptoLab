#!/usr/bin/env python3
"""
CryptoLab — Command Line Interface

Usage:
    python cli.py backtest --strategy cybercycle --symbol BTCUSDT --tf 4h
    python cli.py backtest --strategy gaussbands --symbol ETHUSDT --tf 1h --leverage 5
    python cli.py backtest --strategy smartmoney --symbol SOLUSDT --tf 4h
    python cli.py demo      (runs demo with sample data)
"""
import sys
import os
import json
import time
import copy
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from core.engine import BacktestEngine, format_result
from data.bitget_client import DataCache, generate_sample_data


def get_strategy(name: str):
    """Load strategy by name."""
    name = name.lower().replace('-', '').replace('_', '')

    if name in ('cybercycle', 'cc', 'cyber'):
        from strategies.cybercycle import CyberCycleStrategy
        return CyberCycleStrategy()
    elif name in ('gaussbands', 'gaussian', 'gb'):
        from strategies.gaussbands import GaussianBandsStrategy
        return GaussianBandsStrategy()
    elif name in ('smartmoney', 'smc', 'sm'):
        from strategies.smartmoney import SmartMoneyStrategy
        return SmartMoneyStrategy()
    else:
        raise ValueError(f"Unknown strategy: {name}. Options: cybercycle, gaussbands, smartmoney")


# ═══════════════════════════════════════════════════════════════
#  PARAMS JSON I/O
# ═══════════════════════════════════════════════════════════════

def _load_params_file(args, strategy):
    """
    Load params from JSON file if --params-file was provided.
    Supports --trial N to select which trial from the top 10.

    Usage:
        --params-file output/params_*.json              → loads #1 (best)
        --params-file output/params_*.json --trial 4    → loads #4
    """
    params_file = args.get('params_file')
    if not params_file:
        return False

    path = Path(params_file)
    if not path.exists():
        print(f"   ⚠️  Params file not found: {params_file}")
        return False

    with open(path) as f:
        loaded = json.load(f)

    trial_n = args.get('trial')  # 1-indexed, None = best

    if trial_n is not None:
        trial_n = int(trial_n)
        # Load from top_trials array
        top_trials = loaded.get('top_trials', loaded.get('top_5', []))
        if not top_trials:
            print(f"   ⚠️  No top_trials found in {path.name}")
            return False
        if trial_n < 1 or trial_n > len(top_trials):
            print(f"   ⚠️  --trial {trial_n} out of range (1-{len(top_trials)})")
            print(f"       Available trials:")
            for t in top_trials:
                r = t.get('rank', '?')
                m = t.get('metrics', t)
                sr = m.get('sharpe', 0)
                ret = m.get('return', 0)
                wr = m.get('win_rate', 0)
                dd = m.get('max_drawdown', 0)
                print(f"       #{r}: SR={sr:.2f} Ret={ret:+.1f}% WR={wr:.1f}% DD={dd:.1f}%")
            return False

        selected = top_trials[trial_n - 1]
        params = selected.get('params', {})
        metrics = selected.get('metrics', selected)  # backward compat
        src_label = f"trial #{trial_n}"
    else:
        # Default: load best_params
        if 'best_params' in loaded:
            params = loaded['best_params']
        elif 'params' in loaded:
            params = loaded['params']
        else:
            params = loaded
        metrics = loaded.get('metrics', {})
        src_label = "best"

    strategy.set_params(params)

    n = len(params)
    strategy_name = loaded.get('strategy', '?')
    print(f"   📂 Loaded {n} params from: {path.name} ({src_label})")
    if metrics:
        sr = metrics.get('sharpe', 0)
        wr = metrics.get('win_rate', 0)
        dd = metrics.get('max_drawdown', 0)
        ret = metrics.get('return', 0)
        print(f"      Origin: {strategy_name} | SR={sr:.2f} "
              f"Ret={ret:+.1f}% WR={wr:.1f}% DD={dd:.1f}%")

    # Show available trials for awareness
    top_trials = loaded.get('top_trials', [])
    if top_trials and trial_n is None and len(top_trials) > 1:
        print(f"      💡 {len(top_trials)} trials available — use --trial N to select (1-{len(top_trials)})")

    return True


def _save_params_json(result, strategy_name, symbol, tf, objective,
                       method='grid', extra=None):
    """
    Save optimization results to JSON with top 10 selectable trials.

    Usage after optimization:
        --params-file output/params_*.json              → loads #1 (best)
        --params-file output/params_*.json --trial 4    → loads #4
    """
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    params_file = output_dir / f"params_{strategy_name}_{symbol}_{tf}.json"

    best = result.best_trial if hasattr(result, 'best_trial') else None
    best_params = result.best_params if hasattr(result, 'best_params') else {}

    export = {
        'strategy': strategy_name,
        'symbol': symbol,
        'timeframe': tf,
        'objective': objective,
        'method': method,
        'best_params': best_params,
    }

    if best:
        export['metrics'] = {
            'sharpe': getattr(best, 'sharpe_ratio', 0),
            'return': getattr(best, 'total_return', 0),
            'win_rate': getattr(best, 'win_rate', 0),
            'max_drawdown': getattr(best, 'max_drawdown', 0),
            'profit_factor': getattr(best, 'profit_factor', 0),
            'trades': getattr(best, 'n_trades', 0),
        }

    # Top 10 trials (selectable via --trial N)
    trials = getattr(result, 'trials', [])
    if not trials and hasattr(result, 'hall_of_fame'):
        trials = result.hall_of_fame
    export['top_trials'] = [
        {
            'rank': i + 1,
            'params': getattr(t, 'params', {}),
            'metrics': {
                'sharpe': getattr(t, 'sharpe_ratio', 0),
                'return': getattr(t, 'total_return', 0),
                'win_rate': getattr(t, 'win_rate', 0),
                'max_drawdown': getattr(t, 'max_drawdown', 0),
                'profit_factor': getattr(t, 'profit_factor', 0),
                'trades': getattr(t, 'n_trades', 0),
            }
        }
        for i, t in enumerate(trials[:10])
    ]

    # Backward compat: keep top_5 key
    export['top_5'] = export['top_trials'][:5]

    # Extra info (param importances, monthly stats, etc.)
    if extra:
        export.update(extra)

    with open(params_file, 'w') as f:
        json.dump(export, f, indent=2, default=str)

    print(f"\n📁 Params saved: {params_file}")
    print(f"   Use: --params-file {params_file}")
    print(f"   Pick trial: --params-file {params_file} --trial N  (1-{len(export['top_trials'])})")
    return params_file


def cmd_backtest(args):
    """Run backtest."""
    from core.engine import BacktestEngine, format_result, result_to_dataframe
    from data.bitget_client import DataCache, MarketConfig, generate_sample_data

    strategy_name = args.get('strategy', 'cybercycle')
    symbol = args.get('symbol', 'BTCUSDT')
    timeframe = args.get('timeframe', '4h')
    leverage = args.get('leverage', 3.0)
    capital = args.get('capital', 1000.0)
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')

    strategy = get_strategy(strategy_name)

    # Override leverage
    strategy.set_params({'leverage': leverage})

    # Load params from JSON file (overrides defaults, leverage stays unless in file)
    _load_params_file(args, strategy)

    # Apply any extra params
    if 'params' in args:
        strategy.set_params(args['params'])

    print(f"\n⚡ CryptoLab — {strategy.name()}")
    print(f"   {symbol} | {timeframe} | {start} → {end}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")
    print()

    # Load data + detail data via centralized helper
    data, detail_info, symbol, timeframe = _load_data(args, timeframe)

    # Market config
    market = MarketConfig.detect(symbol)

    # Create engine with detail data
    engine_factory = _make_engine_factory(capital, detail_info, market)

    # Run backtest
    print("🚀 Running backtest...")
    t0 = time.time()

    engine = engine_factory()
    result = engine.run(strategy, data, symbol, timeframe)

    elapsed = time.time() - t0
    bars = len(data['close'])
    print(f"   Done in {elapsed:.2f}s ({bars / elapsed:.0f} bars/sec)")
    print()

    # Print report
    print(format_result(result))

    # Save trade log
    trade_df = result_to_dataframe(result)
    if len(trade_df) > 0:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        trade_file = output_dir / f"trades_{strategy_name}_{symbol}_{timeframe}.csv"
        trade_df.to_csv(trade_file, index=False)
        print(f"\n📁 Trade log saved: {trade_file}")

        # Save equity curve
        eq_file = output_dir / f"equity_{strategy_name}_{symbol}_{timeframe}.csv"
        eq_df = pd.DataFrame({
            'bar': range(len(result.equity_curve)),
            'equity': result.equity_curve,
            'drawdown': result.drawdown_curve,
        })
        eq_df.to_csv(eq_file, index=False)
        print(f"📁 Equity curve saved: {eq_file}")

    return result


def cmd_demo():
    """Run demo backtest with sample data."""
    print("\n" + "=" * 60)
    print("  CryptoLab Engine — Demo")
    print("=" * 60)

    results = {}

    for strat_name in ['cybercycle', 'gaussbands', 'smartmoney']:
        print(f"\n{'─' * 60}")
        print(f"  Running: {strat_name}")
        print(f"{'─' * 60}")

        result = cmd_backtest({
            'strategy': strat_name,
            'symbol': 'BTCUSDT',
            'timeframe': '4h',
            'leverage': 3.0,
            'capital': 10000.0,
            'sample': True,
        })
        results[strat_name] = result

    # Compare
    print("\n" + "=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)
    print(f"  {'Strategy':>20} {'Return':>10} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'Trades':>7}")
    print("  " + "─" * 58)
    for name, r in results.items():
        print(
            f"  {name:>20} {r.total_return:>+9.1f}% {r.sharpe_ratio:>7.2f} {r.win_rate:>7.1f}% {r.max_drawdown:>7.1f}% {r.n_trades:>7}")
    print("=" * 60)


def cmd_list_params(args):
    """List all parameters for a strategy."""
    strategy = get_strategy(args.get('strategy', 'cybercycle'))

    print(f"\n📋 Parameters for {strategy.name()}:")
    print(f"{'─' * 70}")
    print(f"  {'Name':<25} {'Type':<12} {'Default':<10} {'Range'}")
    print(f"{'─' * 70}")

    for pd in strategy.parameter_defs():
        if pd.ptype == 'categorical':
            range_str = f"options: {pd.options}"
        elif pd.ptype == 'bool':
            range_str = "True/False"
        else:
            range_str = f"[{pd.min_val} → {pd.max_val}] step={pd.step}"

        print(f"  {pd.name:<25} {pd.ptype:<12} {str(pd.default):<10} {range_str}")

    print(f"{'─' * 70}")
    print(f"  Total: {len(strategy.parameter_defs())} parameters")


def cmd_validate(args):
    """Run anti-overfitting validation pipeline (Phase 3)."""
    from core.engine import BacktestEngine
    from data.bitget_client import DataCache, generate_sample_data
    from optimize.anti_overfit import AntiOverfitPipeline

    strategy_name = args.get('strategy', 'cybercycle')
    symbol = args.get('symbol', 'BTCUSDT')
    timeframe = args.get('timeframe', '4h')
    capital = args.get('capital', 10000.0)
    leverage = args.get('leverage', None)

    strategy = get_strategy(strategy_name)

    # Override leverage if provided
    if leverage is not None:
        strategy.set_params({'leverage': leverage})

    # Load params from JSON file
    _load_params_file(args, strategy)

    print(f"\n🔬 CryptoLab — Anti-Overfitting Validation")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {timeframe}")
    if leverage:
        print(f"   Leverage: {leverage}x")
    print()

    # Load data + detail data
    data, detail_info, symbol, timeframe = _load_data(args, timeframe)

    # Engine factory WITH detail data
    engine_factory = _make_engine_factory(capital, detail_info)

    # Build a small param grid for WFA from strategy defaults
    param_grid = _build_validation_grid(strategy)

    pipeline = AntiOverfitPipeline(
        wfa_windows=4,
        mc_simulations=1000,
        fail_fast=False,
        verbose=True,
    )

    result = pipeline.run(strategy, data, engine_factory,
                          param_grid, symbol, timeframe)

    return result


def _build_validation_grid(strategy) -> dict:
    """Build a compact param grid for validation from strategy parameter defs."""
    grid = {}
    for pdef in strategy.parameter_defs():
        if pdef.ptype == 'categorical':
            grid[pdef.name] = pdef.options[:3]  # max 3 options
        elif pdef.ptype == 'float' and pdef.min_val is not None:
            # 3 values: low, default, high
            lo = pdef.min_val
            hi = pdef.max_val
            mid = pdef.default
            grid[pdef.name] = sorted(set([lo, mid, hi]))[:3]
        elif pdef.ptype == 'int' and pdef.min_val is not None:
            lo = pdef.min_val
            hi = pdef.max_val
            mid = pdef.default
            grid[pdef.name] = sorted(set([lo, mid, hi]))[:3]

    # Keep grid manageable — only use the 3 most impactful params
    # Priority: alpha_method > confidence_min > leverage (for CyberCycle)
    priority_keys = ['alpha_method', 'confidence_min', 'leverage', 'signal_mode',
                     'length', 'swing_length', 'mode']

    small_grid = {}
    for key in priority_keys:
        if key in grid:
            small_grid[key] = grid[key]
        if len(small_grid) >= 3:
            break

    return small_grid if small_grid else grid


def _load_data(args, timeframe='4h'):
    """
    Helper: load data from sample, cache, or API download.
    Also loads detail data (e.g. 5m for 4h) for intra-bar SL/TP simulation.

    Priority:
    1. --sample → synthetic data (no API needed)
    2. Cache exists → load from Parquet
    3. API credentials → auto-download from Bitget
    4. Fallback → synthetic data with warning

    Returns:
        (data, detail_info, symbol, tf)
        detail_info = {'data': dict|None, 'tf': str|None}
    """
    from data.bitget_client import DataCache, generate_sample_data
    from data.data_manager import DataManager

    use_sample = args.get('sample', False)
    no_detail = args.get('no_detail', False)
    symbol = args.get('symbol', 'BTCUSDT')
    tf = args.get('timeframe', timeframe)
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')

    # Allow user override of detail TF (e.g. --detail-tf 1m)
    detail_tf_override = args.get('detail_tf', None)

    if use_sample:
        print("📊 Generating sample data...")
        df = generate_sample_data(n_bars=5000, timeframe=tf)
    else:
        # Use DataManager: checks cache, downloads if needed
        dm = DataManager(verbose=True)

        # Check if data is cached
        cache = DataCache()
        df = cache.load(symbol, tf, start, end)

        if len(df) > 0:
            print(f"📊 Loaded from cache: {len(df):,} bars")
        else:
            # Try downloading from API
            print(f"📡 No cached data for {symbol} {tf} ({start} → {end})")
            print(f"   Attempting download from Bitget API...")
            try:
                df = dm.get_data(symbol, tf, start, end, warmup=True)
            except Exception as e:
                print(f"   ❌ Download failed: {e}")
                df = pd.DataFrame()

            if len(df) == 0:
                print("⚠ No data available. Using sample data.")
                print("   To download real data first:")
                print(f"   python cli.py download --symbol {symbol} --tf {tf} "
                      f"--start {start} --end {end}")
                df = generate_sample_data(n_bars=5000, timeframe=tf)

    cache = DataCache()
    data = cache.to_numpy(df)
    data['open'] = df['open'].values.astype(np.float64)
    print(f"   {len(df):,} bars loaded")

    # ── Load detail data for intra-bar simulation ──
    detail_info = {'data': None, 'tf': None}

    if not use_sample and not no_detail:
        dm = DataManager(verbose=True)

        # Determine detail TF: user override > default map
        if detail_tf_override:
            detail_tf = detail_tf_override
            print(f"📐 Detail TF override: {detail_tf}")
        else:
            detail_tf = dm.detail_tf_for(tf)

        if detail_tf:
            print(f"📐 Loading {detail_tf} detail data for intra-bar SL/TP simulation...")
            try:
                detail_data = dm.get_detail_data(symbol, tf, start, end)
                if detail_data is None and detail_tf_override:
                    # If override, try direct download
                    detail_data = dm.get_data_numpy(
                        symbol, detail_tf, start, end, warmup=False, validate=False)
                    if detail_data and len(detail_data.get('close', [])) > 0:
                        pass
                    else:
                        detail_data = None

                if detail_data is not None:
                    n_detail = len(detail_data['close'])
                    print(f"   ✅ {n_detail:,} detail bars loaded ({detail_tf})")
                    detail_info = {'data': detail_data, 'tf': detail_tf}
                else:
                    print(f"   ⚠ No detail data — using bar-level simulation")
            except Exception as e:
                print(f"   ⚠ Detail data unavailable: {e}")
        else:
            print(f"   ℹ No detail TF defined for {tf}")

    print()
    return data, detail_info, symbol, tf


def _make_engine_factory(capital: float = 10000.0,
                         detail_info: dict = None,
                         market_config: dict = None):
    """
    Create an engine_factory that produces BacktestEngines
    with detail data and market config pre-loaded.

    This ensures that ALL commands (validate, optimize, regime, etc.)
    use the same intra-bar simulation as backtest.

    Args:
        capital: Initial capital in USDT
        detail_info: {'data': dict|None, 'tf': str|None} from _load_data
        market_config: MarketConfig dict or None

    Returns:
        Callable that returns a configured BacktestEngine
    """
    from core.engine import BacktestEngine

    _detail = detail_info or {'data': None, 'tf': None}
    _dd = _detail.get('data')
    _dtf = _detail.get('tf')

    def factory():
        engine = BacktestEngine(
            initial_capital=capital,
            market_config=market_config,
        )
        if _dd is not None and _dtf is not None:
            engine.set_detail_data(_dd, _dtf)
        return engine

    return factory


# ─── Phase 4 Commands ───

def cmd_optimize(args):
    """Optimization with multiple methods and objectives."""
    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 10000.0)
    leverage = args.get('leverage', None)
    objective = args.get('objective', 'sharpe')
    method = args.get('method', 'grid')

    strategy = get_strategy(strategy_name)
    if leverage is not None:
        strategy.set_params({'leverage': leverage})

    # Load params from JSON as warm start baseline
    _load_params_file(args, strategy)

    # Parse --optimize-params (comma-separated list of param names)
    param_subset = None
    opt_params_str = args.get('optimize_params')
    if opt_params_str:
        param_subset = [p.strip() for p in opt_params_str.split(',')]
        valid_names = {pd.name for pd in strategy.parameter_defs()}
        invalid = [p for p in param_subset if p not in valid_names]
        if invalid:
            print(f"   ⚠️  Unknown params: {', '.join(invalid)}")
            print(f"       Valid: {', '.join(sorted(valid_names))}")
            param_subset = [p for p in param_subset if p in valid_names]

    data, detail_info, symbol, tf = _load_data(args)
    engine_factory = _make_engine_factory(capital, detail_info)

    method_labels = {
        'grid': 'Grid Search',
        'bayesian': 'Bayesian (Optuna TPE)',
        'genetic': 'Genetic Algorithm (DEAP)',
    }

    print(f"\n⚡ CryptoLab — {method_labels.get(method, method)} Optimization")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {tf}")
    print(f"   Objective: {objective} | Method: {method}")
    if leverage:
        print(f"   Leverage: {leverage}x")
    if param_subset:
        print(f"   Params: {', '.join(param_subset)} ({len(param_subset)} of {len(strategy.parameter_defs())})")
    print()

    # ── Run optimization (Ctrl+C safe) ──────────────────────────
    result = None
    optimizer = None
    extra_json = {}
    interrupted = False

    try:
        if method == 'bayesian':
            try:
                from optimize.bayesian import BayesianOptimizer, HAS_OPTUNA
                if not HAS_OPTUNA:
                    print("   ❌ Optuna not installed. Run: pip install optuna")
                    print("      Falling back to grid search.\n")
                    method = 'grid'
            except ImportError:
                print("   ❌ bayesian.py not found. Falling back to grid search.\n")
                method = 'grid'

            if method == 'bayesian':
                n_trials = int(args.get('n_trials', 100))
                optimizer = BayesianOptimizer(
                    n_trials=n_trials,
                    objective=objective,
                    verbose=True,
                )
                result = optimizer.run(
                    strategy, data, engine_factory,
                    symbol=symbol, timeframe=tf,
                    param_subset=param_subset,
                )
                if result and result.param_importances:
                    extra_json['param_importances'] = result.param_importances

        elif method == 'genetic':
            try:
                from optimize.genetic import GeneticOptimizer, HAS_DEAP
                if not HAS_DEAP:
                    print("   ❌ DEAP not installed. Run: pip install deap")
                    print("      Falling back to grid search.\n")
                    method = 'grid'
            except ImportError:
                print("   ❌ genetic.py not found. Falling back to grid search.\n")
                method = 'grid'

            if method == 'genetic':
                n_gen = int(args.get('n_generations', 30))
                pop_size = int(args.get('pop_size', 40))
                optimizer = GeneticOptimizer(
                    n_generations=n_gen,
                    population_size=pop_size,
                    objective=objective,
                    verbose=True,
                )
                result = optimizer.run(
                    strategy, data, engine_factory,
                    symbol=symbol, timeframe=tf,
                )

        if method == 'grid':
            from optimize.grid_search import GridSearchOptimizer

            if param_subset:
                param_grid = {}
                for pdef in strategy.parameter_defs():
                    if pdef.name in param_subset:
                        if pdef.ptype == 'categorical':
                            param_grid[pdef.name] = pdef.options[:4]
                        elif pdef.ptype in ('float', 'int') and pdef.min_val is not None:
                            lo, hi, mid = pdef.min_val, pdef.max_val, pdef.default
                            param_grid[pdef.name] = sorted(set([lo, mid, hi]))[:3]
                        elif pdef.ptype == 'bool':
                            param_grid[pdef.name] = [True, False]
            else:
                param_grid = _build_validation_grid(strategy)

            optimizer = GridSearchOptimizer(objective=objective, verbose=True)
            result = optimizer.run_with_validation(
                strategy, data, engine_factory,
                param_grid, symbol=symbol, timeframe=tf,
            )

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n   ⚠️  Ctrl+C — compiling partial results...\n")
        # Recover partial result from optimizer's internal state
        if result is None and optimizer is not None:
            result = getattr(optimizer, 'last_result', None)

    # ── Print results (complete OR partial) ──────────────────────
    if result is not None:
        trials = getattr(result, 'trials', None)
        if trials is None:
            trials = getattr(result, 'hall_of_fame', [])

        if trials:
            from optimize.grid_search import compute_monthly_stats

            n_shown = min(10, len(trials))
            label = "PARTIAL " if interrupted else ""
            print(f"\n{'═' * 80}")
            print(f"{label}Top {n_shown} Results ({method} / {objective})")
            print(f"{'═' * 80}")

            all_monthly = []  # For JSON export

            for i, trial in enumerate(trials[:n_shown]):
                sr = getattr(trial, 'sharpe_ratio', 0)
                ret = getattr(trial, 'total_return', 0)
                wr = getattr(trial, 'win_rate', 0)
                dd = getattr(trial, 'max_drawdown', 0)
                pf = getattr(trial, 'profit_factor', 0)
                nt = getattr(trial, 'n_trades', 0)
                params = getattr(trial, 'params', {})

                # Header for this trial
                print(f"\n  ┌─ #{i+1} {'─' * 60}")
                print(f"  │ SR={sr:.2f}  Ret={ret:+.1f}%  WR={wr:.1f}%  "
                      f"DD={dd:.1f}%  PF={pf:.2f}  Trades={nt}")

                # Key params (compact)
                key_params = {}
                for k in ['alpha_method', 'confidence_min', 'leverage',
                          'sl_atr_mult', 'tp1_rr', 'tp1_size', 'tp2_rr',
                          'use_trend', 'use_htf', 'close_on_signal',
                          'be_pct', 'trail_activate_pct', 'trail_pullback_pct']:
                    if k in params:
                        v = params[k]
                        key_params[k] = f'{v:.1f}' if isinstance(v, float) else str(v)
                print(f"  │ {key_params}")

                # Monthly breakdown
                try:
                    trial_strat = copy.deepcopy(strategy)
                    trial_strat.set_params(params)
                    trial_engine = engine_factory()
                    trial_result = trial_engine.run(trial_strat, data, symbol, tf)
                    ms = compute_monthly_stats(trial_result.trades)

                    if ms['n_months'] >= 2:
                        # Compact monthly line
                        month_parts = []
                        for m in ms['months']:
                            icon = '✅' if m['pnl_pct'] >= 0 else '❌'
                            month_parts.append(
                                f"{m['year']}-{m['month']:02d}: {m['pnl_pct']:+5.1f}% "
                                f"({m['n_trades']}t {m['win_rate']:.0f}%wr) {icon}")

                        print(f"  │")
                        print(f"  │ 📅 Monthly:")
                        # Two columns if possible
                        for j in range(0, len(month_parts), 2):
                            left = month_parts[j]
                            right = month_parts[j+1] if j+1 < len(month_parts) else ""
                            if right:
                                print(f"  │   {left:<38} {right}")
                            else:
                                print(f"  │   {left}")

                        print(f"  │   ── Avg: {ms['avg_monthly_return']:+.1f}%/mo | "
                              f"Pos: {ms['pct_positive']:.0f}% | "
                              f"mSR: {ms['monthly_sharpe']:.2f} | "
                              f"Worst: {ms['worst_month']:+.1f}%")

                        # Store for JSON
                        all_monthly.append({
                            'rank': i + 1,
                            'sharpe': sr, 'return': ret, 'win_rate': wr,
                            'max_dd': dd, 'trades': nt,
                            'monthly_sharpe': ms['monthly_sharpe'],
                            'pct_positive': ms['pct_positive'],
                            'avg_monthly': ms['avg_monthly_return'],
                            'worst_month': ms['worst_month'],
                            'best_month': ms['best_month'],
                            'months': ms['months'],
                            'params': params,
                        })

                except Exception as e:
                    print(f"  │ ⚠️  Monthly calc failed: {e}")

                print(f"  └{'─' * 65}")

            # ── Store all monthly in JSON ──
            if all_monthly:
                extra_json['top_trials_monthly'] = all_monthly
                # Best trial monthly for backward compat
                extra_json['monthly_stats'] = {
                    'avg_return': all_monthly[0].get('avg_monthly', 0),
                    'pct_positive': all_monthly[0].get('pct_positive', 0),
                    'monthly_sharpe': all_monthly[0].get('monthly_sharpe', 0),
                    'worst_month': all_monthly[0].get('worst_month', 0),
                    'best_month': all_monthly[0].get('best_month', 0),
                    'months': all_monthly[0].get('months', []),
                }

            # ── Save JSON (marks partial if interrupted) ──
            tag = '_partial' if interrupted else ''
            _save_params_json(
                result, strategy_name, symbol, tf,
                objective=objective, method=method + tag, extra=extra_json,
            )
        else:
            print("\n   ❌ No completed trials to show.")
    elif interrupted:
        print("\n   ❌ No trials completed before interruption.")

    return result



def cmd_regime(args):
    """Detect market regimes and analyze per-regime performance."""
    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 10000.0)
    leverage = args.get('leverage', None)

    strategy = get_strategy(strategy_name)
    if leverage is not None:
        strategy.set_params({'leverage': leverage})

    # Load params from JSON file
    _load_params_file(args, strategy)

    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n🔄 CryptoLab — Regime Detection & Analysis")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {tf}\n")

    engine_factory = _make_engine_factory(capital, detail_info)

    from ml.regime_detector import detect_regime, strategy_regime_performance

    # Detect regimes
    rr = detect_regime(data, method='vt', verbose=True)

    # Analyze per-regime performance
    perf = strategy_regime_performance(
        strategy, data, engine_factory, rr,
        symbol=symbol, timeframe=tf, verbose=True)

    return rr, perf


def cmd_ensemble(args):
    """Run ensemble of all 3 strategies."""
    capital = args.get('capital', 10000.0)
    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n🎯 CryptoLab — Strategy Ensemble")
    print(f"   Methods: CyberCycle + GaussBands + SmartMoney")
    print(f"   {symbol} | {tf}\n")

    engine_factory = _make_engine_factory(capital, detail_info)

    from ml.ensemble import EnsembleBuilder

    builder = EnsembleBuilder()
    builder.add('CyberCycle', get_strategy('cybercycle'))
    builder.add('GaussBands', get_strategy('gaussbands'))
    builder.add('SmartMoney', get_strategy('smartmoney'))

    result = builder.evaluate(data, engine_factory, method='confidence_vote',
                              symbol=symbol, timeframe=tf, verbose=True)
    return result


def cmd_targets(args):
    """Evaluate temporal targets for a strategy."""
    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 10000.0)
    leverage = args.get('leverage', None)
    target_preset = args.get('targets', 'conservative')

    strategy = get_strategy(strategy_name)
    if leverage is not None:
        strategy.set_params({'leverage': leverage})

    # Load params from JSON file
    _load_params_file(args, strategy)

    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n📅 CryptoLab — Temporal Target Analysis")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {tf}")
    print(f"   Targets: {target_preset}\n")

    # Create engine with detail data
    engine = _make_engine_factory(capital, detail_info)()
    result = engine.run(strategy, data, symbol, tf)

    from ml.temporal_targets import (
        evaluate_targets, CONSERVATIVE_TARGETS,
        AGGRESSIVE_TARGETS, CONSISTENCY_TARGETS,
    )

    targets_map = {
        'conservative': CONSERVATIVE_TARGETS,
        'aggressive': AGGRESSIVE_TARGETS,
        'consistency': CONSISTENCY_TARGETS,
    }
    targets = targets_map.get(target_preset, CONSERVATIVE_TARGETS)

    if target_preset not in targets_map:
        print(f"   ⚠️  Unknown preset '{target_preset}', using conservative")
        print(f"      Options: conservative, aggressive, consistency\n")

    tt_result = evaluate_targets(
        targets, result.trades,
        data.get('timestamp', np.arange(len(data['close']))),
        result.equity_curve, initial_capital=capital, verbose=True)

    return tt_result


def cmd_combinatorial(args):
    """Search for optimal strategy combinations."""
    capital = args.get('capital', 10000.0)
    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n🧬 CryptoLab — Combinatorial Strategy Search")
    print(f"   {symbol} | {tf}\n")

    engine_factory = _make_engine_factory(capital, detail_info)

    from ml.combinatorial import CombinatorialSearch, StrategyConfig

    cc = get_strategy('cybercycle')
    gb = get_strategy('gaussbands')
    smc = get_strategy('smartmoney')

    configs = [
        StrategyConfig('CC-mama', cc, {'alpha_method': 'mama', 'confidence_min': 70}),
        StrategyConfig('CC-homo', cc, {'alpha_method': 'homodyne', 'confidence_min': 60}),
        StrategyConfig('GB-default', gb, {}),
        StrategyConfig('SMC-default', smc, {}),
    ]

    cs = CombinatorialSearch(max_portfolio_size=3, verbose=True)
    result = cs.run(configs, data, engine_factory, symbol, tf)

    print(f"\n{'─' * 50}")
    print(f"Top 5 Portfolios:")
    for i, p in enumerate(result.portfolios[:5]):
        names = [c.name for c in p.configs]
        print(f"  #{i + 1}: {' + '.join(names)}")
        print(f"       SR={p.combined_sharpe:.2f} Ret={p.combined_return:+.1f}% "
              f"DD={p.combined_max_dd:.1f}% Synergy={p.synergy_score:+.2f}")

    return result


# ─── Data Management Commands ───

def cmd_download(args):
    """
    Download OHLCV candle data from Bitget API.

    Examples:
        python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-01-01
        python cli.py download --symbol ETHUSDT --tf 1h --start 2024-01-01 --end 2025-01-01
        python cli.py download --symbol TSLAUSDT --tf 4h --start 2024-06-01 --end 2025-01-01
    """
    from data.data_manager import DataManager, POPULAR_CRYPTO, POPULAR_STOCKS

    symbol = args.get('symbol', None)
    tf = args.get('timeframe', '4h')
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')
    batch = args.get('batch', None)  # 'crypto' or 'stocks' or 'all'

    dm = DataManager(verbose=True)

    if batch:
        # Batch download
        if batch == 'crypto':
            symbols = POPULAR_CRYPTO
        elif batch == 'stocks':
            symbols = POPULAR_STOCKS
        elif batch == 'all':
            symbols = POPULAR_CRYPTO + POPULAR_STOCKS
        else:
            symbols = batch.split(',')

        dm.download_batch(symbols, tf, start, end)
        return

    if symbol is None:
        print("\n❌ Specify --symbol (e.g. BTCUSDT) or --batch crypto|stocks|all")
        print("\nExamples:")
        print("  python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-01-01")
        print("  python cli.py download --symbol ETHUSDT --tf 1h --start 2024-01-01 --end 2025-06-01")
        print("  python cli.py download --batch crypto --tf 4h --start 2023-01-01 --end 2025-01-01")
        print("  python cli.py download --batch BTCUSDT,ETHUSDT,SOLUSDT --tf 4h --start 2024-01-01 --end 2025-01-01")
        print(f"\nPopular Crypto: {', '.join(POPULAR_CRYPTO[:8])}...")
        print(f"Stock Tokens:   {', '.join(POPULAR_STOCKS)}")
        return

    print(f"\n📡 CryptoLab — Data Download")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {tf}")
    print(f"   Range: {start} → {end}")
    print()

    try:
        df = dm.get_data(symbol, tf, start, end, warmup=True, validate=True)

        if len(df) > 0:
            print(f"\n✅ Download complete!")
            print(f"   {len(df):,} bars cached in data/cache/{symbol}_{tf}.parquet")
            print(f"\n   Now you can run:")
            print(
                f"   python cli.py backtest --strategy cybercycle --symbol {symbol} --tf {tf} --start {start} --end {end}")

    except Exception as e:
        print(f"\n❌ Download error: {e}")
        print("\nTips:")
        print("  1. Check your internet connection")
        print("  2. Verify the symbol exists on Bitget Futures")
        print("  3. Bitget API allows public candle access (no key needed)")
        print("  4. Rate limits: max 20 requests/sec")


def cmd_data(args):
    """
    Data management: info, list, validate, delete.

    Examples:
        python cli.py data list
        python cli.py data info --symbol BTCUSDT --tf 4h
        python cli.py data validate --symbol BTCUSDT --tf 4h
        python cli.py data delete --symbol BTCUSDT --tf 4h
    """
    from data.data_manager import DataManager

    subcommand = args.get('data_cmd', 'list')
    dm = DataManager(verbose=True)

    if subcommand == 'list':
        infos = dm.list_cached()

        if not infos:
            print("\n📂 No cached data found.")
            print("   Download data first:")
            print("   python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-01-01")
            return

        print(f"\n📂 Cached Data ({len(infos)} datasets):")
        print(f"{'─' * 60}")

        total_mb = 0
        total_bars = 0
        for info in infos:
            print(info.summary())
            total_mb += info.file_size_mb
            total_bars += info.bars

        print(f"{'─' * 60}")
        print(f"  Total: {total_bars:,} bars | {total_mb:.1f} MB")

    elif subcommand == 'info':
        symbol = args.get('symbol', 'BTCUSDT')
        tf = args.get('timeframe', '4h')
        info = dm.data_info(symbol, tf)
        print(f"\n📊 Data Info:")
        print(info.summary())

    elif subcommand == 'validate':
        symbol = args.get('symbol', 'BTCUSDT')
        tf = args.get('timeframe', '4h')

        from data.bitget_client import DataCache
        cache = DataCache()
        df = cache.load(symbol, tf)

        if len(df) == 0:
            print(f"\n⚠ No data for {symbol} {tf}")
            return

        vr = dm.validate_data(df, tf)
        print(f"\n📊 Data Validation: {symbol} {tf}")
        print(vr.summary())

    elif subcommand == 'delete':
        symbol = args.get('symbol', 'BTCUSDT')
        tf = args.get('timeframe', '4h')

        if dm.delete_cached(symbol, tf):
            print(f"\n🗑 Deleted cache for {symbol} {tf}")
        else:
            print(f"\n⚠ No cache found for {symbol} {tf}")

    elif subcommand == 'clear':
        dm.clear_cache()
        print("\n🗑 All cached data cleared.")

    else:
        print(f"\n❌ Unknown data command: {subcommand}")
        print("   Options: list, info, validate, delete, clear")


def main():
    """Main CLI entry point."""

    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    CryptoLab Engine v0.5                    ║
║     Backtesting, Validation & ML for Perp Futures           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Data Commands:                                              ║
║    download                Download candle data from Bitget  ║
║    data list               List cached datasets              ║
║    data info               Show info for a dataset           ║
║    data validate           Validate data integrity           ║
║    data delete             Delete cached data                ║
║                                                              ║
║  Core Commands:                                              ║
║    demo                    Run demo with all strategies       ║
║    backtest                Run single backtest                ║
║    params                  List strategy parameters           ║
║                                                              ║
║  Phase 3 — Validation:                                       ║
║    validate                Anti-overfitting pipeline          ║
║                                                              ║
║  Phase 4 — Optimization & ML:                                ║
║    optimize                Parameter optimization             ║
║    regime                  Regime detection + per-regime perf ║
║    ensemble                Multi-strategy ensemble backtest   ║
║    targets                 Temporal target evaluation          ║
║    portfolio               Combinatorial strategy search      ║
║                                                              ║
║  Options:                                                    ║
║    --strategy   STR        cybercycle|gaussbands|smartmoney   ║
║    --symbol     STR        BTCUSDT, ETHUSDT, TSLAUSDT, etc  ║
║    --tf         STR        1m,5m,15m,1h,4h,1d               ║
║    --leverage   FLOAT      1.0 - 125.0                       ║
║    --capital    FLOAT      Starting capital (USDT)           ║
║    --start      DATE       Start date (YYYY-MM-DD)           ║
║    --end        DATE       End date (YYYY-MM-DD)             ║
║    --sample                Use sample data (no API needed)   ║
║    --batch      STR        crypto|stocks|all|SYM1,SYM2,...   ║
║    --detail-tf  STR        Override detail TF (e.g. 1m, 5m) ║
║    --no-detail             Disable detail data loading       ║
║    --params-file PATH      Load params from JSON file        ║
║    --objective  STR        sharpe|return|calmar|composite    ║
║                           monthly|monthly_robust            ║
║    --method     STR        grid|bayesian|genetic             ║
║    --targets    STR        conservative|aggressive|consistency║
║    --n-trials   INT        Trials for bayesian (default 100) ║
║    --optimize-params STR   Params to optimize (comma-sep)    ║
║    --trial   INT          Pick trial from top 10 (1-10)     ║
║                                                              ║
║  Examples:                                                   ║
║    python cli.py backtest --strategy cybercycle --sample     ║
║    python cli.py backtest --strategy cybercycle \\             ║
║      --symbol SOLUSDT --tf 1h --leverage 10                  ║
║    python cli.py optimize --strategy cybercycle \\             ║
║      --method bayesian --objective monthly --n-trials 200    ║
║    python cli.py optimize --strategy cybercycle \\             ║
║      --method bayesian --objective composite \\                ║
║      --optimize-params confidence_min,sl_atr_mult,tp1_rr    ║
║    python cli.py backtest --params-file output/params_*.json ║
║    python cli.py validate --strategy cybercycle \\             ║
║      --params-file output/params_cybercycle_SOLUSDT_1h.json  ║
║    python cli.py validate --strategy cybercycle \\             ║
║      --params-file output/params_*.json --trial 4            ║
║    python cli.py targets --strategy cybercycle \\              ║
║      --targets aggressive --leverage 10                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
        return

    command = sys.argv[1]

    # Parse arguments
    args = {}
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--strategy':
            args['strategy'] = sys.argv[i + 1]
            i += 2
        elif arg == '--symbol':
            args['symbol'] = sys.argv[i + 1]
            i += 2
        elif arg in ('--tf', '--timeframe'):
            args['timeframe'] = sys.argv[i + 1]
            i += 2
        elif arg == '--leverage':
            args['leverage'] = float(sys.argv[i + 1])
            i += 2
        elif arg == '--capital':
            args['capital'] = float(sys.argv[i + 1])
            i += 2
        elif arg == '--start':
            args['start'] = sys.argv[i + 1]
            i += 2
        elif arg == '--end':
            args['end'] = sys.argv[i + 1]
            i += 2
        elif arg == '--batch':
            args['batch'] = sys.argv[i + 1]
            i += 2
        elif arg == '--sample':
            args['sample'] = True
            i += 1
        elif arg == '--detail-tf':
            args['detail_tf'] = sys.argv[i + 1]
            i += 2
        elif arg == '--no-detail':
            args['no_detail'] = True
            i += 1
        elif arg == '--force':
            args['force'] = True
            i += 1
        elif arg == '--params-file':
            args['params_file'] = sys.argv[i + 1]
            i += 2
        elif arg == '--objective':
            args['objective'] = sys.argv[i + 1]
            i += 2
        elif arg == '--method':
            args['method'] = sys.argv[i + 1]
            i += 2
        elif arg == '--targets':
            args['targets'] = sys.argv[i + 1]
            i += 2
        elif arg == '--n-trials':
            args['n_trials'] = int(sys.argv[i + 1])
            i += 2
        elif arg == '--optimize-params':
            args['optimize_params'] = sys.argv[i + 1]
            i += 2
        elif arg == '--trial':
            args['trial'] = int(sys.argv[i + 1])
            i += 2
        elif not arg.startswith('--'):
            # Sub-command for 'data' command
            args['data_cmd'] = arg
            i += 1
        else:
            i += 1

    commands = {
        'demo': lambda: cmd_demo(),
        'backtest': lambda: cmd_backtest(args),
        'params': lambda: cmd_list_params(args),
        'validate': lambda: cmd_validate(args),
        # Data management
        'download': lambda: cmd_download(args),
        'data': lambda: cmd_data(args),
        # Phase 4
        'optimize': lambda: cmd_optimize(args),
        'regime': lambda: cmd_regime(args),
        'ensemble': lambda: cmd_ensemble(args),
        'targets': lambda: cmd_targets(args),
        'portfolio': lambda: cmd_combinatorial(args),
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands.keys())}")


if __name__ == '__main__':
    main()