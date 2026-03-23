#!/usr/bin/env python3
"""
CryptoLab — Command Line Interface v0.7 (Parallel)

Usage:
    python cli.py backtest --strategy cybercycle --symbol BTCUSDT --tf 4h
    python cli.py backtest --strategy gaussbands --symbol ETHUSDT --tf 1h --leverage 5
    python cli.py backtest --strategy smartmoney --symbol SOLUSDT --tf 4h
    python cli.py demo      (runs demo with sample data)

Parallel:
    python cli.py optimize --method bayesian --n-jobs -1      (auto-detect cores)
    python cli.py validate --n-jobs 8                         (8 workers)
    python cli.py download --batch crypto --n-jobs 4          (4 concurrent symbols)
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
    elif name in ('cybercyclev4','ccv4','cyberv4'):
        from strategies.cybercyclev4 import CyberCycleStrategyv4
        return CyberCycleStrategyv4()
    elif name in ('cybercyclev3','ccv3','cyberv3'):
        from strategies.cybercyclev3 import CyberCycleStrategyv3
        return CyberCycleStrategyv3()
    elif name in ('cybercyclehtf','cchtf','cyberhtf'):
        from strategies.cybercyclehtf import CyberCycleStrategyhtf
        return CyberCycleStrategyhtf()
    elif name in ('cybercyclenohtf','ccnohtf','cybernohtf'):
        from strategies.cybercyclenohtf import CyberCycleNoHTFStrategy
        return CyberCycleNoHTFStrategy()
    else:
        raise ValueError(f"Unknown strategy: {name}. Options: cybercycle, gaussbands, smartmoney")


# ═══════════════════════════════════════════════════════════════
#  PARALLEL HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_n_jobs(args):
    """Get number of parallel workers from args, with auto-detect."""
    n_jobs_raw = args.get('n_jobs', 1)
    try:
        from optimize.parallel import get_n_jobs
        return get_n_jobs(n_jobs_raw)
    except ImportError:
        return max(1, n_jobs_raw) if n_jobs_raw > 0 else 1


def _build_engine_config(capital, detail_info, market_config=None, no_intrabar: bool = False):
    """Build serializable engine config for parallel workers."""
    _detail = detail_info or {'data': None, 'tf': None}
    return {
        'capital': capital,
        'market_config': market_config,
        'detail_data': _detail.get('data'),
        'detail_tf': _detail.get('tf'),
        'no_intrabar': no_intrabar,
    }


def _print_parallel_info(n_jobs):
    """Print hardware/parallel info if n_jobs > 1."""
    if n_jobs <= 1:
        return
    try:
        from optimize.parallel import detect_hardware
        hw = detect_hardware()
        cpu_str = f"{hw['cpu_name']}" if hw['cpu_name'] != 'Unknown' else ''
        gpu_str = f" | GPU: {hw['gpu_name']} ({hw['gpu_vram']})" if hw.get('gpu_name') else ''
        print(f"   ⚡ Parallel: {n_jobs} workers / {hw['cpu_count']} cores{gpu_str}")
        if cpu_str:
            print(f"   CPU: {cpu_str}")
    except Exception:
        print(f"   ⚡ Parallel: {n_jobs} workers")


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
    timeframe = args.get('timeframe', '1h')
    capital = args.get('capital', 1000.0)
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')

    strategy = get_strategy(strategy_name)

    # ── FIX: Orden correcto — JSON primero, CLI override después ──
    # 1) Cargar JSON (incluye leverage de la optimización)
    _load_params_file(args, strategy)

    # 2) CLI leverage SIEMPRE gana si fue explícitamente pasado
    if 'leverage' in args:
        strategy.set_params({'leverage': args['leverage']})

    # 3) Extra params al final
    if 'params' in args:
        strategy.set_params(args['params'])

    # 4) Leer leverage EFECTIVO de la estrategia para display
    leverage = strategy.get_param('leverage', 3.0)

    print(f"\n⚡ CryptoLab — {strategy.name()}")
    print(f"   {symbol} | {timeframe} | {start} → {end}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")
    print()

    # Load data + detail data via centralized helper
    data, detail_info, symbol, timeframe = _load_data(args, timeframe)

    # Market config
    market = MarketConfig.detect(symbol)

    # Create engine with detail data
    no_intrabar = args.get('no_intrabar', False)
    engine_factory = _make_engine_factory(capital, detail_info, market, no_intrabar=no_intrabar)

    # Run backtest
    if no_intrabar:
        print("🚀 Running backtest... [bar-close mode — no intrabar signal timing]")
    else:
        print("🚀 Running backtest... [intrabar mode — incremental Ehlers, 1m signal timing]")
    t0 = time.time()

    engine = engine_factory()

    # ── Warmup fix: signals only after user's --start date ──
    if not args.get('sample', False):
        try:
            from datetime import datetime as _dt, timezone as _tz
            _start_str = args.get('start', '')
            if _start_str:
                _start_ts = int(_dt.fromisoformat(_start_str).replace(
                    tzinfo=_tz.utc).timestamp() * 1000)
                engine.set_signal_start(_start_ts)
        except (ValueError, OSError, AttributeError):
            pass

    result = engine.run(strategy, data, symbol, timeframe)
    elapsed = time.time() - t0
    bars = len(data['close'])
    print(f"   Done in {elapsed:.2f}s ({bars / elapsed:.0f} bars/sec)")
    print()

    # Print report
    print(format_result(result))


    # Monthly breakdown — equity-based (consistent with total_return)
    try:
        from optimize.grid_search import (
            compute_monthly_stats_from_equity, compute_monthly_stats
        )

        timestamps = data.get('timestamp', None)
        if timestamps is not None and len(result.equity_curve) > 1:
            ms = compute_monthly_stats_from_equity(
                result.equity_curve, timestamps,
                trades=result.trades, initial_capital=capital)
        else:
            ms = compute_monthly_stats(result.trades, initial_capital=capital)

        if ms['n_months'] >= 2:
            print(f"\\n  📅 MONTHLY BREAKDOWN")
            print(f"  {'─' * 60}")
            for m in ms['months']:
                icon = '✅' if m['pnl_pct'] >= 0 else '❌'
                bar = '█' * max(1, int(abs(m['pnl_pct']) / 3))
                sign = '+' if m['pnl_pct'] >= 0 else ''
                print(f"  {m['year']}-{m['month']:02d}  {sign}{m['pnl_pct']:>6.1f}%  "
                      f"{m['n_trades']:>3}t  WR={m['win_rate']:>4.0f}%  "
                      f"{icon} {'▓' if m['pnl_pct'] >= 0 else '░'}{bar}")
            print(f"  {'─' * 60}")

            # Compound verification
            compound = 1.0
            for m in ms['months']:
                compound *= (1 + m['pnl_pct'] / 100)
            compound_ret = (compound - 1) * 100

            print(f"  Avg: {ms['avg_monthly_return']:+.1f}%/mo | "
                  f"Positive: {ms['pct_positive']:.0f}% | "
                  f"mSR: {ms['monthly_sharpe']:.2f} | "
                  f"Best: {ms['best_month']:+.1f}% | "
                  f"Worst: {ms['worst_month']:+.1f}%")
            print(f"  Σ Compound: {compound_ret:+.1f}%  "
                  f"(Total Return: {result.total_return:+.1f}%)")
    except Exception as e:
        print(f"  ⚠️ Monthly error: {e}")

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


# ═══════════════════════════════════════════════════════════════
#  VALIDATE — Parallel (4-8x speedup with --n-jobs)
# ═══════════════════════════════════════════════════════════════

def cmd_validate(args):
    """Run anti-overfitting validation pipeline (Phase 3)."""
    from core.engine import BacktestEngine
    from data.bitget_client import DataCache, MarketConfig, generate_sample_data
    from optimize.anti_overfit import AntiOverfitPipeline

    strategy_name = args.get('strategy', 'cybercycle')
    symbol = args.get('symbol', 'BTCUSDT')
    timeframe = args.get('timeframe', '4h')
    capital = args.get('capital', 1000.0)       # FIX: era 10000.0, ahora igual que cmd_backtest
    leverage = args.get('leverage', 3.0)         # FIX: era None, ahora default 3.0 como cmd_backtest

    strategy = get_strategy(strategy_name)

    # FIX: Siempre aplicar leverage (igual que cmd_backtest)
    _load_params_file(args, strategy)        # ← JSON primero
    if 'leverage' in args:                   # ← CLI gana si fue pasado
        strategy.set_params({'leverage': args['leverage']})
    leverage = strategy.get_param('leverage', 3.0)

    print(f"\n🔬 CryptoLab — Anti-Overfitting Validation")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {timeframe}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")
    print()

    # Load data + detail data
    data, detail_info, symbol, timeframe = _load_data(args, timeframe)

    # FIX: Engine factory WITH detail data AND market config (antes no se pasaba market)
    market = MarketConfig.detect(symbol)
    no_intrabar = args.get('no_intrabar', False)
    engine_factory = _make_engine_factory(capital, detail_info, market,
                                          no_intrabar=no_intrabar)

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
    """
    Build a param grid for WFA validation centered on the strategy's
    CURRENT params (from --params-file / --trial).

    Two-phase approach:
      Phase A — STABILITY: include the current values as-is
               (tests if the trial works across time windows)
      Phase B — SENSITIVITY: include current ± small perturbation
               (tests if nearby params also work, proving robustness)

    For each param: [current - step, current, current + step]
    Clamped to [min_val, max_val].

    Only uses top 3 most impactful params to keep grid at 27 combos.
    The other ~30 params stay FIXED at the trial's values.

    WHY THIS IS CORRECT:
      - Bayesian already explored 150+ combos globally.
      - WFA's job is NOT to re-discover params (Bayesian did that).
      - WFA's job IS to verify that the CHOSEN params are temporally stable.
      - Small perturbations test that we're not on a "cliff" in param space
        (where moving confidence_min from 72 to 73 destroys the strategy).
    """
    grid = {}

    for pdef in strategy.parameter_defs():
        # Get current value from loaded params (or default)
        current = strategy.params.get(pdef.name, pdef.default)

        if pdef.ptype == 'categorical':
            # Include current + 1-2 alternatives
            options = list(pdef.options)
            if current in options:
                others = [o for o in options if o != current][:2]
                grid[pdef.name] = [current] + others
            else:
                grid[pdef.name] = options[:3]

        elif pdef.ptype == 'float' and pdef.min_val is not None:
            lo = pdef.min_val
            hi = pdef.max_val
            step = pdef.step if pdef.step else (hi - lo) / 10.0

            # Small perturbation: ±2 steps from current
            v_lo = round(max(lo, float(current) - step * 2), 4)
            v_mid = round(float(current), 4)
            v_hi = round(min(hi, float(current) + step * 2), 4)

            values = sorted(set([v_lo, v_mid, v_hi]))
            grid[pdef.name] = values[:3]

        elif pdef.ptype == 'int' and pdef.min_val is not None:
            lo = pdef.min_val
            hi = pdef.max_val
            step = pdef.step if pdef.step else max(1, (hi - lo) // 5)

            v_lo = max(lo, int(current) - step)
            v_mid = int(current)
            v_hi = min(hi, int(current) + step)

            values = sorted(set([v_lo, v_mid, v_hi]))
            grid[pdef.name] = values[:3]

        elif pdef.ptype == 'bool':
            grid[pdef.name] = [bool(current)]  # Don't vary booleans

    # ── Select top 3 most impactful params ──
    # These are the params where small changes have the biggest effect.
    # Order matters: first found = first included.
    priority_keys = [
        'confidence_min',      # Controls signal frequency directly
        'sl_atr_mult',         # Controls risk/reward ratio
        'tp1_rr',              # Controls profit taking
        'alpha_method',        # Changes entire signal generation
        'leverage',            # Scales everything (but often fixed by user)
        'trail_pullback_pct',  # Trailing stop sensitivity
        'be_pct',              # Break-even trigger
        'signal_mode',         # GaussBands
        'length',              # GaussBands / SMC
        'swing_length',        # SMC
    ]

    small_grid = {}
    for key in priority_keys:
        if key in grid and len(grid[key]) > 1:  # Only include if there's variation
            small_grid[key] = grid[key]
        if len(small_grid) >= 3:
            break

    # If we don't have 3 varying params, allow single-value params too
    # (this means WFA just tests stability without perturbation for that param)
    if len(small_grid) < 3:
        for key in priority_keys:
            if key in grid and key not in small_grid:
                small_grid[key] = grid[key]
            if len(small_grid) >= 3:
                break

    # Last resort: grab any remaining
    if len(small_grid) < 3:
        for key in grid:
            if key not in small_grid:
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
    exchange = args.get('exchange', 'bitget')

    # Allow user override of detail TF (e.g. --detail-tf 1m)
    detail_tf_override = args.get('detail_tf', None)

    if use_sample:
        print("📊 Generating sample data...")
        df = generate_sample_data(n_bars=5000, timeframe=tf)
    else:
        # Use DataManager: checks cache, downloads if needed
        dm = DataManager(exchange=exchange, verbose=True)

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
        dm = DataManager(exchange=exchange, verbose=True)

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


def _make_engine_factory(capital, detail_info=None, market_config=None,
                         no_intrabar: bool = False):
    """
    Create an engine_factory that produces BacktestEngines
    with detail data and market config pre-loaded.

    This ensures that ALL commands (validate, optimize, regime, etc.)
    use the same intra-bar simulation as backtest.

    Args:
        capital:      Initial capital in USDT
        detail_info:  {'data': dict|None, 'tf': str|None} from _load_data
        market_config: MarketConfig dict or None
        no_intrabar:  If True, force BacktestEngine (bar-close signals,
                      no IncrementalCyberCycle). Detail data is still
                      loaded for intra-bar exit simulation (SL/TP/trailing).
                      Use --no-intrabar to compare both signal modes.

    Returns:
        Callable that returns a configured BacktestEngine
    """
    _detail = detail_info or {'data': None, 'tf': None}
    _dd = _detail.get('data')
    _dtf = _detail.get('tf')

    def factory():
        if no_intrabar:
            # Bar-close signal mode: signals fire at main-TF bar close.
            # Still uses detail data for exit precision (SL/TP/trailing).
            from core.engine import BacktestEngine
            engine = BacktestEngine(
                initial_capital=capital,
                market_config=market_config,
            )
        else:
            try:
                from core.engine_intrabar import IntrabarBacktestEngine
                engine = IntrabarBacktestEngine(
                    initial_capital=capital,
                    market_config=market_config,
                )
            except ImportError:
                from core.engine import BacktestEngine
                engine = BacktestEngine(
                    initial_capital=capital,
                    market_config=market_config,
                )
        if _dd is not None and _dtf is not None:
            engine.set_detail_data(_dd, _dtf)
        return engine

    return factory


# ═══════════════════════════════════════════════════════════════
#  OPTIMIZE — Parallel (uses --n-jobs for bayesian/grid)
# ═══════════════════════════════════════════════════════════════

def cmd_optimize(args):
    """Optimization with multiple methods and objectives — with parallel support."""
    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 1000.0)
    leverage = args.get('leverage', None)
    objective = args.get('objective', 'sharpe')
    method = args.get('method', 'grid')

    strategy = get_strategy(strategy_name)
    if leverage is not None:
        strategy.set_params({'leverage': leverage})

    # Load params from JSON as warm start baseline
    _load_params_file(args, strategy)

    param_subset = None
    valid_names = {pd.name for pd in strategy.parameter_defs()}

    opt_params_str = args.get('optimize_params')
    excl_params_str = args.get('exclude_params')

    if opt_params_str:
        # Explicit include list (existing behavior)
        param_subset = [p.strip() for p in opt_params_str.split(',')]
        invalid = [p for p in param_subset if p not in valid_names]
        if invalid:
            print(f"   ⚠️  Unknown params: {', '.join(invalid)}")
            print(f"       Valid: {', '.join(sorted(valid_names))}")
            param_subset = [p for p in param_subset if p in valid_names]

    elif excl_params_str:
        # Exclude list → param_subset = all EXCEPT excluded
        excluded = {p.strip() for p in excl_params_str.split(',')}
        invalid = excluded - valid_names
        if invalid:
            print(f"   ⚠️  Unknown params to exclude: {', '.join(invalid)}")
            print(f"       Valid: {', '.join(sorted(valid_names))}")
        excluded &= valid_names  # Only keep valid names

        # Build subset = all params minus excluded
        param_subset = [pd.name for pd in strategy.parameter_defs()
                        if pd.name not in excluded]

        if excluded:
            print(f"   🚫 Excluded {len(excluded)} params: {', '.join(sorted(excluded))}")
            print(f"   ✅ Optimizing {len(param_subset)} of {len(valid_names)} params")

    data, detail_info, symbol, tf = _load_data(args)
    from data.bitget_client import MarketConfig
    market = MarketConfig.detect(symbol)  # FIX: agregar market_config
    no_intrabar = args.get('no_intrabar', False)
    engine_factory = _make_engine_factory(capital, detail_info, market,
                                          no_intrabar=no_intrabar)

    # ── Parallel setup ──
    n_jobs = _get_n_jobs(args)
    engine_config = _build_engine_config(capital, detail_info,market, no_intrabar=no_intrabar)

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
    _print_parallel_info(n_jobs)
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
                    n_jobs=n_jobs,
                    verbose=True,
                )
                result = optimizer.run(
                    strategy, data, engine_factory,
                    symbol=symbol, timeframe=tf,
                    param_subset=param_subset,
                    engine_config=engine_config,
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

            optimizer = GridSearchOptimizer(
                objective=objective,
                n_jobs=n_jobs,
                verbose=True,
            )
            result = optimizer.run(
                strategy, data, engine_factory,
                param_grid, symbol=symbol, timeframe=tf,
                engine_config=engine_config,
            )

    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n   ⚠️  Ctrl+C — compiling partial results...\n")
        # Recover partial result from optimizer's internal state
        if result is None and optimizer is not None:
            result = getattr(optimizer, 'last_result', None)

    # ── Print results (complete OR partial) ──────────────────────
    # ── Print results (complete OR partial) ──────────────────────
    if result is not None:
        trials = getattr(result, 'trials', None)
        if trials is None:
            trials = getattr(result, 'hall_of_fame', [])

        if trials:
            # ◀ FIX: importar AMBAS funciones de monthly stats
            from optimize.grid_search import compute_monthly_stats, compute_monthly_stats_from_equity

            n_shown = min(10, len(trials))
            label = "PARTIAL " if interrupted else ""
            print(f"\\n{'═' * 80}")
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
                print(f"\\n  ┌─ #{i+1} {'─' * 60}")
                print(f"  │ SR={sr:.2f}  Ret={ret:+.1f}%  WR={wr:.1f}%  "
                      f"DD={dd:.1f}%  PF={pf:.2f}  Trades={nt}")

                # ◀ FIX 3: key_params dinámicos según sltp_type + optimized params
                key_params = {}
                sltp = params.get('sltp_type', 'sltp_fixed')
                if sltp == 'sltp_fixed':
                    sltp_keys = ['sl_fixed_pct', 'tp1_fixed_pct', 'tp1_fixed_size',
                                 'tp2_fixed_pct']
                else:
                    sltp_keys = ['sl_atr_mult', 'tp1_rr', 'tp1_size', 'tp2_rr']

                display_keys = ['alpha_method', 'confidence_min', 'leverage'] + sltp_keys + [
                    'use_trend', 'use_htf', 'close_on_signal',
                    'be_pct', 'trail_activate_pct', 'trail_pullback_pct']

                # Agregar parámetros optimizados que no estén en la lista base
                optimized = getattr(trial, 'optimized_params', None)
                if optimized and isinstance(optimized, dict):
                    for ok in optimized:
                        if ok not in display_keys:
                            display_keys.append(ok)
                # Fallback: si param_subset está disponible, agregar también
                if param_subset:
                    for ps in param_subset:
                        if ps not in display_keys:
                            display_keys.append(ps)

                for k in display_keys:
                    if k in params:
                        v = params[k]
                        key_params[k] = f'{v:.1f}' if isinstance(v, float) else str(v)
                print(f"  │ {key_params}")

                # ◀ FIX 2: Monthly breakdown — equity-based (consistent with backtest)
                try:
                    trial_strat = copy.deepcopy(strategy)
                    trial_strat.set_params(params)
                    trial_engine = engine_factory()
                    trial_result = trial_engine.run(trial_strat, data, symbol, tf)

                    # ◀ FIX: usar compute_monthly_stats_from_equity
                    # (misma función que cmd_backtest) para que los %
                    # mensuales sean consistentes con total_return y
                    # max_drawdown del engine.
                    timestamps = data.get('timestamp', None)
                    if timestamps is not None and len(trial_result.equity_curve) > 1:
                        ms = compute_monthly_stats_from_equity(
                            trial_result.equity_curve, timestamps,
                            trades=trial_result.trades, initial_capital=capital)
                    else:
                        ms = compute_monthly_stats(trial_result.trades, initial_capital=capital)

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
            print("\\n   ❌ No completed trials to show.")
    elif interrupted:
        print("\\n   ❌ No trials completed before interruption.")

    return result

def cmd_repair(args):
    """
    Diagnosticar y reparar gaps en datos cacheados.

    Tres modos:
      --diagnose   Solo reportar gaps (no modifica nada)
      --fill       Forward-fill gaps con close anterior (default)
      --redownload Borrar cache y re-descargar completo

    Uso:
      python cli.py repair --symbol SOLUSDT --tf 1m
      python cli.py repair --symbol SOLUSDT --tf 1m --diagnose
      python cli.py repair --symbol SOLUSDT --tf 1h --redownload --start 2022-01-01 --end 2025-06-25
    """
    import shutil
    from pathlib import Path
    from data.bitget_client import DataCache, TIMEFRAME_SECONDS

    symbol = args.get('symbol', 'SOLUSDT')
    tf = args.get('timeframe', '1h')
    mode = 'fill'  # default
    if args.get('diagnose'):
        mode = 'diagnose'
    elif args.get('redownload'):
        mode = 'redownload'

    cache = DataCache()
    tf_ms = TIMEFRAME_SECONDS.get(tf, 3600) * 1000

    # ── MODO REDOWNLOAD: borrar cache y descargar de nuevo ──
    if mode == 'redownload':
        start = args.get('start')
        end = args.get('end')
        if not start or not end:
            print("❌ --redownload requiere --start y --end")
            return

        print(f"\n🔄 Redownload: {symbol} {tf}")
        print(f"   Range: {start} → {end}")

        # Borrar cache existente
        cache.delete(symbol, tf)
        print(f"   Cache borrado")

        # Re-descargar con reintentos agresivos
        from data.bitget_client import BitgetClient
        import asyncio

        client = BitgetClient(verbose=True)

        async def _download_robust():
            """Descarga con reintentos por segmento."""
            try:
                df = await client.download_ohlcv(symbol, tf, start, end)
                return df
            finally:
                await client.close()

        df = asyncio.run(_download_robust())

        if df is None or len(df) == 0:
            print("   ❌ No se pudo descargar data")
            return

        cache.save(df, symbol, tf)
        print(f"   ✅ {len(df):,} bars descargadas y guardadas")

        # Diagnosticar gaps en lo descargado
        _diagnose_gaps(df, tf_ms, symbol, tf)
        return

    # ── CARGAR DATA EXISTENTE ──
    if not cache.has_data(symbol, tf):
        print(f"❌ No hay data cacheada para {symbol} {tf}")
        print(f"   Usa: python cli.py download --symbol {symbol} --tf {tf}")
        return

    df = cache.load(symbol, tf)
    print(f"\n🔍 Diagnóstico: {symbol} {tf}")
    print(f"   Bars: {len(df):,}")

    if len(df) == 0:
        print("   ❌ Dataset vacío")
        return

    # ── DETECTAR GAPS ──
    gap_info = _diagnose_gaps(df, tf_ms, symbol, tf)

    if mode == 'diagnose':
        return

    # ── DETECTAR VELAS CON VOLUMEN 0 (fake fills previos) ──
    zero_vol = (df['volume'] == 0).sum()
    flat = ((df['open'] == df['high']) & (df['high'] == df['low']) &
            (df['low'] == df['close'])).sum()

    if zero_vol > 0:
        print(f"\n   ⚠️  Velas con volumen=0: {zero_vol:,}")
    if flat > 0:
        print(f"   ⚠️  Velas flat (O=H=L=C): {flat:,}")

    if gap_info['n_gaps'] == 0 and zero_vol == 0:
        print(f"\n   ✅ Data OK — sin gaps ni velas fantasma")
        return

    # ── MODO FILL: forward-fill gaps ──
    if gap_info['n_gaps'] > 0:
        print(f"\n🔧 Reparando {gap_info['n_gaps']} gaps...")

        import numpy as np
        ts_start = int(df['timestamp'].iloc[0])
        ts_end = int(df['timestamp'].iloc[-1])

        full_ts = np.arange(ts_start, ts_end + tf_ms, tf_ms, dtype=np.int64)
        full_df = pd.DataFrame({'timestamp': full_ts})

        merged = full_df.merge(df, on='timestamp', how='left')

        # Forward-fill: usar close anterior para OHLC de gaps
        merged['close'] = merged['close'].ffill()
        missing = merged['open'].isna()
        merged.loc[missing, 'open'] = merged.loc[missing, 'close']
        merged.loc[missing, 'high'] = merged.loc[missing, 'close']
        merged.loc[missing, 'low'] = merged.loc[missing, 'close']
        merged.loc[missing, 'volume'] = 0.0

        if 'datetime' in merged.columns:
            merged['datetime'] = pd.to_datetime(merged['timestamp'], unit='ms', utc=True)
        merged['volume'] = merged['volume'].fillna(0.0)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            merged[col] = merged[col].astype(np.float64)

        n_filled = missing.sum()
        print(f"   Bars rellenadas: {n_filled:,}")
        print(f"   Total después: {len(merged):,}")

        # Backup + guardar
        cache_dir = cache.cache_dir
        for ext in ['.parquet', '.csv']:
            src = cache_dir / f"{symbol}_{tf}{ext}"
            bak = cache_dir / f"{symbol}_{tf}_backup{ext}"
            if src.exists() and not bak.exists():
                shutil.copy2(src, bak)
                print(f"   Backup: {bak.name}")

        cache.save(merged, symbol, tf)
        print(f"   ✅ Reparado y guardado")
# ═══════════════════════════════════════════════════════════════
#  REGIME — Concurrent detection + backtest (~1.3x speedup)
# ═══════════════════════════════════════════════════════════════
def _diagnose_gaps(df, tf_ms, symbol, tf):
    """Diagnosticar gaps en un DataFrame de velas."""
    from datetime import datetime, timezone
    import numpy as np

    diffs = df['timestamp'].diff().dropna()
    gap_mask = diffs > tf_ms * 1.5
    n_gaps = int(gap_mask.sum())

    ts_min = df['timestamp'].iloc[0]
    ts_max = df['timestamp'].iloc[-1]
    expected = int((ts_max - ts_min) / tf_ms) + 1
    actual = len(df)
    missing = expected - actual

    dt_start = datetime.fromtimestamp(ts_min / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
    dt_end = datetime.fromtimestamp(ts_max / 1000, tz=timezone.utc).strftime('%Y-%m-%d')

    print(f"   Rango: {dt_start} → {dt_end}")
    print(f"   Esperadas: {expected:,} | Actual: {actual:,} | Faltantes: {missing:,}")
    print(f"   Gaps (saltos): {n_gaps}")

    if n_gaps > 0:
        gap_indices = gap_mask[gap_mask].index
        gap_sizes = []
        for idx in gap_indices:
            t1 = df['timestamp'].iloc[idx - 1]
            t2 = df['timestamp'].iloc[idx]
            gap_bars = int((t2 - t1) / tf_ms) - 1
            t1_str = datetime.fromtimestamp(t1 / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            t2_str = datetime.fromtimestamp(t2 / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            gap_sizes.append((gap_bars, t1_str, t2_str))

        gap_sizes.sort(reverse=True)
        print(f"\n   Top 10 gaps más grandes:")
        for bars, t1, t2 in gap_sizes[:10]:
            print(f"     {t1} → {t2}  ({bars} bars faltantes)")

    return {'n_gaps': n_gaps, 'missing': missing, 'expected': expected}
def cmd_regime(args):
    """Detect market regimes and analyze per-regime performance."""
    from data.bitget_client import MarketConfig

    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 1000.0)       # FIX: era 10000.0, ahora igual que cmd_backtest        # FIX: era None, ahora default 3.0 como cmd_backtest

    strategy = get_strategy(strategy_name)
    _load_params_file(args, strategy)        # ← JSON primero
    if 'leverage' in args:                   # ← CLI gana si fue pasado
        strategy.set_params({'leverage': args['leverage']})
    leverage = strategy.get_param('leverage', 3.0)
    itrend_alpha = strategy.get_param('itrend_alpha', 0.07)
    # FIX: Siempre aplicar leverage (igual que cmd_backtest)
    strategy.set_params({'leverage': leverage})

    # Load params from JSON file (puede sobreescribir leverage)
    _load_params_file(args, strategy)

    # Apply any extra params
    if 'params' in args:
        strategy.set_params(args['params'])

    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n🔄 CryptoLab — Regime Detection & Analysis")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {tf}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")
    print()

    # FIX: Pasar market_config (antes no se pasaba)
    market = MarketConfig.detect(symbol)
    no_intrabar = args.get('no_intrabar', False)
    engine_factory = _make_engine_factory(capital, detail_info, market,
                                          no_intrabar=no_intrabar)

    from ml.regime_detector import detect_regime, strategy_regime_performance

    # Detect regimes
    rr = detect_regime(data, method='vt', verbose=True,itrend_alpha=itrend_alpha)

    # Analyze per-regime performance
    perf = strategy_regime_performance(
        strategy, data, engine_factory, rr,
        symbol=symbol, timeframe=tf, verbose=True)

    return rr, perf

def cmd_ensemble(args):
    """Run ensemble of all 3 strategies."""
    from data.bitget_client import MarketConfig

    capital = args.get('capital', 1000.0)       # FIX: era 10000.0, ahora igual que cmd_backtest
    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n🎯 CryptoLab — Strategy Ensemble")
    print(f"   Methods: CyberCycle + GaussBands + SmartMoney")
    print(f"   {symbol} | {tf}\n")

    # FIX: Pasar market_config (antes no se pasaba)
    market = MarketConfig.detect(symbol)
    engine_factory = _make_engine_factory(capital, detail_info, market)

    from ml.ensemble import EnsembleBuilder

    builder = EnsembleBuilder()
    builder.add('CyberCycle', get_strategy('cybercycle'))
    builder.add('GaussBands', get_strategy('gaussbands'))
    builder.add('SmartMoney', get_strategy('smartmoney'))

    result = builder.evaluate(data, engine_factory, method='confidence_vote',
                              symbol=symbol, timeframe=tf, verbose=True)
    return result

# ═══════════════════════════════════════════════════════════════
#  TARGETS — Optimized with pre-computed buckets (~1.2x speedup)
# ═══════════════════════════════════════════════════════════════

def cmd_targets(args):
    """Evaluate temporal targets for a strategy."""
    from data.bitget_client import MarketConfig

    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 1000.0)       # FIX: era 10000.0, ahora igual que cmd_backtest
    leverage = args.get('leverage', 3.0)         # FIX: era None, ahora default 3.0 como cmd_backtest
    target_preset = args.get('targets', 'conservative')

    strategy = get_strategy(strategy_name)

    # FIX: Siempre aplicar leverage (igual que cmd_backtest)
    strategy.set_params({'leverage': leverage})

    # Load params from JSON file
    _load_params_file(args, strategy)

    data, detail_info, symbol, tf = _load_data(args)

    print(f"\n📅 CryptoLab — Temporal Target Analysis")
    print(f"   Strategy: {strategy.name()}")
    print(f"   {symbol} | {tf}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")
    print(f"   Targets: {target_preset}\n")

    # FIX: Create engine with detail data AND market config (antes no se pasaba market)
    market = MarketConfig.detect(symbol)
    no_intrabar = args.get('no_intrabar', False)
    engine_factory = _make_engine_factory(capital, detail_info, market,
                                            no_intrabar=no_intrabar)
    engine = engine_factory()
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
# ═══════════════════════════════════════════════════════════════
# cmd_combinatorial() — CORREGIDA
# Soporta --params-file, --leverage, MarketConfig
# Reemplazar la funcion completa en cli.py
# ═══════════════════════════════════════════════════════════════

def cmd_combinatorial(args):
    """
    Search for optimal strategy combinations (portfolio).

    Now supports:
    - --params-file to apply optimized params to the primary strategy
    - --leverage to set leverage for all strategies
    - MarketConfig detection for consistent fees
    - Dynamic config generation: primary strategy with optimized params
      + default variants of other strategies
    """
    from data.bitget_client import MarketConfig

    strategy_name = args.get('strategy', 'cybercycle')
    capital = args.get('capital', 1000.0)
    leverage = args.get('leverage', 3.0)

    data, detail_info, symbol, tf = _load_data(args)

    # Market config
    market = MarketConfig.detect(symbol)
    engine_factory = _make_engine_factory(capital, detail_info, market)

    print(f"\n🧬 CryptoLab — Combinatorial Strategy Search")
    print(f"   {symbol} | {tf}")
    print(f"   Leverage: {leverage}x | Capital: ${capital:,.0f}")

    from ml.combinatorial import CombinatorialSearch, StrategyConfig

    # ── Build configs dynamically ──
    configs = []

    # Primary strategy with optimized params (if --params-file provided)
    primary = get_strategy(strategy_name)
    primary.set_params({'leverage': leverage})

    params_file = args.get('params_file')
    has_optimized = False
    if params_file:
        has_optimized = _load_params_file(args, primary)

    # Always include the primary strategy with its (possibly optimized) params
    primary_params = copy.deepcopy(primary.params)
    configs.append(StrategyConfig(
        f'{strategy_name.upper()}-optimized' if has_optimized else f'{strategy_name.upper()}-default',
        get_strategy(strategy_name),
        primary_params,
    ))

    # If --params-file has top_trials, add top 3 variants of the primary strategy
    if params_file and has_optimized:
        try:
            with open(Path(params_file)) as f:
                loaded = json.load(f)
            top_trials = loaded.get('top_trials', loaded.get('top_5', []))

            # Add trial #2 and #3 as variants (trial #1 is already the primary)
            for t_idx, trial in enumerate(top_trials[1:3], start=2):
                t_params = trial.get('params', {})
                t_params['leverage'] = leverage  # Ensure same leverage
                t_metrics = trial.get('metrics', trial)
                label = (f"{strategy_name.upper()}-t{t_idx} "
                         f"(SR={t_metrics.get('sharpe',0):.1f})")
                configs.append(StrategyConfig(
                    label,
                    get_strategy(strategy_name),
                    t_params,
                ))
        except Exception:
            pass

    # Add other strategies with defaults
    all_strategies = ['cybercycle', 'gaussbands', 'smartmoney']
    for other_name in all_strategies:
        if other_name == strategy_name.lower():
            continue  # Skip if it's the primary (already added above)
        try:
            other = get_strategy(other_name)
            other_params = {'leverage': leverage}
            configs.append(StrategyConfig(
                other_name.upper(),
                other,
                other_params,
            ))
        except Exception:
            pass

    # If we have CyberCycle, add a variant with a different alpha_method
    if strategy_name.lower() != 'cybercycle':
        try:
            cc_alt = get_strategy('cybercycle')
            configs.append(StrategyConfig(
                'CC-mama',
                cc_alt,
                {'alpha_method': 'mama', 'leverage': leverage},
            ))
        except Exception:
            pass

    print(f"   Configs: {len(configs)}")
    for i, c in enumerate(configs):
        print(f"     [{i}] {c.name}")
    print()

    cs = CombinatorialSearch(max_portfolio_size=3, verbose=True)
    result = cs.run(configs, data, engine_factory, symbol, tf)

    print(f"\n{'─' * 60}")
    print(f"Top 5 Portfolios:")
    for i, p in enumerate(result.portfolios[:5]):
        names = [c.name for c in p.configs]
        weights = [f"{w:.0%}" for w in p.weights]
        print(f"  #{i + 1}: {' + '.join(names)}")
        print(f"       Weights: {weights}")
        print(f"       SR={p.combined_sharpe:.2f} Ret={p.combined_return:+.1f}% "
              f"DD={p.combined_max_dd:.1f}% Synergy={p.synergy_score:+.2f}")

    return result
# ═══════════════════════════════════════════════════════════════
#  DOWNLOAD — Parallel batch (3-5x speedup with asyncio.gather)
# ═══════════════════════════════════════════════════════════════

def cmd_download(args):
    """
    Download OHLCV candle data from Bitget API.

    Examples:
        python cli.py download --symbol BTCUSDT --tf 4h --start 2023-01-01 --end 2025-01-01
        python cli.py download --symbol ETHUSDT --tf 1h --start 2024-01-01 --end 2025-01-01
        python cli.py download --batch crypto --tf 4h --start 2023-01-01 --end 2025-01-01
    """
    from data.data_manager import DataManager, POPULAR_CRYPTO, POPULAR_STOCKS

    symbol = args.get('symbol', None)
    tf = args.get('timeframe', '4h')
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')
    batch = args.get('batch', None)  # 'crypto' or 'stocks' or 'all'
    exchange = args.get('exchange', 'bitget')

    dm = DataManager(exchange=exchange, verbose=True)

    if batch:
        if batch == 'crypto':
            symbols = POPULAR_CRYPTO
        elif batch == 'stocks':
            symbols = POPULAR_STOCKS
        elif batch == 'all':
            symbols = POPULAR_CRYPTO + POPULAR_STOCKS
        else:
            symbols = batch.split(',')

        # ── Try parallel batch download ──
        try:
            from data.download_parallel import download_batch_parallel
            download_batch_parallel(
                symbols, tf, start, end,
                cache_dir=str(dm.cache.cache_dir),
                max_concurrent=4,
                rate_limit=15,
                verbose=True,
            )
            return
        except ImportError:
            pass

        # ── Sequential fallback ──
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
def cmd_download_batch(args):
    """
    Descarga múltiples símbolos y timeframes en paralelo.

    Uso:
        python cli.py download-batch --symbols BTCUSDT,ETHUSDT,SOLUSDT \\
            --timeframes 4h,1h,15m,1m --start 2023-01-01 --end 2025-01-01 \\
            --exchange binance --workers 4
    """
    from data.data_manager import DataManager
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, as_completed

    symbols = args.get('symbols', '').split(',')
    timeframes = args.get('timeframes', '').split(',')
    exchange = args.get('exchange', 'bitget')
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')
    workers = int(args.get('workers', 4))

    if not symbols or not timeframes:
        print("❌ Debes especificar --symbols y --timeframes")
        return

    print(f"\n📡 Descarga masiva ({exchange})")
    print(f"   Símbolos: {len(symbols)} | Timeframes: {len(timeframes)}")
    print(f"   Rango: {start} → {end}")
    print(f"   Workers: {workers}\n")

    tasks = []
    total_combos = len(symbols) * len(timeframes)

    # Usamos ThreadPoolExecutor para lanzar descargas en paralelo (cada una es síncrona por dentro)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_combo = {}
        for sym in symbols:
            for tf in timeframes:
                dm = DataManager(exchange=exchange, verbose=False)
                future = executor.submit(
                    dm.get_data, sym, tf, start, end,
                    warmup=True, validate=False, force_download=False
                )
                future_to_combo[future] = (sym, tf)

        completed = 0
        for future in as_completed(future_to_combo):
            sym, tf = future_to_combo[future]
            try:
                df = future.result()
                bars = len(df)
                status = "✅" if bars > 0 else "⚠️ vacío"
                print(f"  {status} {sym} {tf}: {bars:,} barras")
            except Exception as e:
                print(f"  ❌ {sym} {tf}: {e}")
            completed += 1
            print(f"  Progreso: {completed}/{total_combos}")

    print("\n✅ Descarga masiva completada.")

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
    exchange = args.get('exchange', 'bitget')
    dm = DataManager(exchange=exchange, verbose=True)

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
║                    CryptoLab Engine v0.7                    ║
║  Backtesting, Validation & ML for Perp Futures (Parallel)   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Data Commands:                                              ║
║    download                Download candle data from Bitget  ║
║    data list               List cached datasets              ║
║    data info               Show info for a dataset           ║
║    data validate           Validate data integrity           ║
║    data delete             Delete cached data                ║
║                                           ║
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
║    --no-intrabar           Bar-close signals (no incremental ║
║                            Ehlers). Detail still used for    ║
║                            SL/TP exits. Compare both modes.  ║
║    --params-file PATH      Load params from JSON file        ║
║    --objective  STR        sharpe|return|calmar|composite    ║
║                           monthly|monthly_robust            ║
║                           weekly|weekly_robust              ║
║    --method     STR        grid|bayesian|genetic             ║
║    --targets    STR        conservative|aggressive|consistency║
║    --n-trials   INT        Trials for bayesian (default 100) ║
║    --n-jobs     INT        Parallel workers (-1=auto, 1=seq) ║
║    --optimize-params STR   Params to optimize (comma-sep)    ║
║    --trial   INT          Pick trial from top 10 (1-10)     ║
║                                                              ║
║  Examples:                                                   ║
║    python cli.py backtest --strategy cybercycle --sample     ║
║    python cli.py backtest --strategy cybercycle \\             ║
║      --symbol SOLUSDT --tf 1h --leverage 10                  ║
║    python cli.py optimize --strategy cybercycle \\             ║
║      --method bayesian --objective monthly --n-trials 200   ║
║    python cli.py optimize --strategy cybercycle \\             ║
║      --method bayesian --n-jobs -1 --n-trials 300           ║
║    python cli.py validate --strategy cybercycle \\             ║
║      --params-file output/params_*.json --n-jobs -1         ║
║    python cli.py download --batch crypto --tf 4h            ║
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
        elif arg == '--no-intrabar':
            args['no_intrabar'] = True
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
        elif arg == '--n-jobs':
            args['n_jobs'] = int(sys.argv[i + 1])
            i += 2
        elif arg == '--optimize-params':
            args['optimize_params'] = sys.argv[i + 1]
            i += 2
        elif arg == '--exclude-params':
            args['exclude_params'] = sys.argv[i + 1]
            i += 2
        elif arg == '--trial':
            args['trial'] = int(sys.argv[i + 1])
            i += 2
        elif arg == '--symbols':
            args['symbols'] = sys.argv[i + 1]
            i += 2
        elif arg == '--timeframes':
            args['timeframes'] = sys.argv[i + 1]
            i += 2
        elif arg == '--exchange':
            args['exchange'] = sys.argv[i + 1]
            i += 2
        elif arg == '--workers':
            args['workers'] = int(sys.argv[i + 1])
            i += 2
        elif not arg.startswith('--'):
            # Sub-command for 'data' command
            args['data_cmd'] = arg
            i += 1
        elif arg == '--diagnose':
             args['diagnose'] = True
             i += 1
        elif arg == '--redownload':
             args['redownload'] = True
             i += 1
        else:
            i += 1

    commands = {
        'download-batch': lambda: cmd_download_batch(args),
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
        'repai':lambda : cmd_repair(args),
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands.keys())}")


if __name__ == '__main__':
    main()