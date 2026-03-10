"""
CryptoLab — Parallel Trial Evaluation

Uses fork-based multiprocessing for zero-copy data sharing on Linux.
Parent process sets module-level globals, fork() children inherit via COW.
Workers only receive small param dicts — data arrays stay shared.

Usage:
    setup_workers(strategy, data, engine_config, objective_name)
    with Pool(n_jobs) as pool:
        results = pool.map(evaluate_trial, [(id, params), ...])
    cleanup_workers()
"""
import os
import copy
import time
import pickle
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Force fork on Linux for COW memory sharing
# Python 3.14+ defaults to 'spawn' which creates fresh processes without shared state
_MP_CONTEXT = mp.get_context('fork') if hasattr(mp, 'get_context') else None

# ═══════════════════════════════════════════════════════════════
#  MODULE-LEVEL SHARED STATE (inherited by fork COW)
# ═══════════════════════════════════════════════════════════════

_shared = {
    'strategy': None,       # Base strategy (deep-copied per trial)
    'data': None,           # OHLCV data dict (read-only, shared COW)
    'engine_config': None,  # {capital, market_config, detail_data, detail_tf}
    'objective_fn': None,   # Objective function
    'objective_name': None,
    'min_trades': 10,
    'symbol': '',
    'timeframe': '',
}


def setup_workers(strategy, data: dict, engine_config: dict,
                  objective_name: str, min_trades: int = 10,
                  symbol: str = '', timeframe: str = ''):
    """
    Set shared state BEFORE creating Pool.
    Fork will share these via COW — no serialization overhead.

    engine_config: {
        'capital': float,
        'market_config': dict or None,
        'detail_data': dict or None,
        'detail_tf': str or None,
    }
    """
    from optimize.grid_search import OBJECTIVES

    _shared['strategy'] = strategy
    _shared['data'] = data
    _shared['engine_config'] = engine_config
    _shared['objective_fn'] = OBJECTIVES.get(objective_name, OBJECTIVES['sharpe'])
    _shared['objective_name'] = objective_name
    _shared['min_trades'] = min_trades
    _shared['symbol'] = symbol
    _shared['timeframe'] = timeframe


def cleanup_workers():
    """Release shared references."""
    _shared['strategy'] = None
    _shared['data'] = None
    _shared['engine_config'] = None


def create_pool(n_jobs: int):
    """
    Create a multiprocessing Pool with fork context.
    Fork is required for COW memory sharing of numpy arrays.
    """
    if _MP_CONTEXT is not None:
        return _MP_CONTEXT.Pool(processes=n_jobs)
    else:
        return Pool(processes=n_jobs)


def evaluate_trial(args) -> dict:
    """
    Evaluate a single parameter set. Called in worker process.
    Receives only (trial_id, params_dict) — everything else is shared.

    Uses IntrabarBacktestEngine when detail data is available so that
    signal generation during optimization is IDENTICAL to backtest
    (intrabar processor, same min_bars scaling, same SL/TP resolution).

    Returns dict with all trial metrics.
    """
    trial_id, params_to_set = args

    t0 = time.time()

    # FIX: Don't use copy.deepcopy (breaks complex strategy objects like
    # CyberCycleStrategy with Ehlers filters, numpy state, etc.)
    # Create a fresh instance and apply base params + trial params.
    original_strategy = _shared['strategy']
    strategy_class = type(original_strategy)
    strategy = strategy_class()
    strategy.set_params(original_strategy.params)  # Base params (from JSON/defaults)
    strategy.set_params(params_to_set)              # Override with trial's optimized params

    cfg = _shared['engine_config']
    dd = cfg.get('detail_data')
    dtf = cfg.get('detail_tf')
    no_intrabar = cfg.get('no_intrabar', False)

    # Use IntrabarBacktestEngine when:
    #   1. Detail data is available AND
    #   2. --no-intrabar was NOT passed
    # --no-intrabar forces BacktestEngine (bar-close signals + detail exits)
    engine = None
    if dd is not None and dtf is not None and not no_intrabar:
        try:
            from core.engine_intrabar import IntrabarBacktestEngine
            engine = IntrabarBacktestEngine(
                initial_capital=cfg.get('capital', 1000.0),
                market_config=cfg.get('market_config'),
            )
            engine.set_detail_data(dd, dtf)
        except ImportError:
            engine = None

    if engine is None:
        from core.engine import BacktestEngine
        engine = BacktestEngine(
            initial_capital=cfg.get('capital', 1000.0),
            market_config=cfg.get('market_config'),
        )
        if dd is not None and dtf is not None:
            engine.set_detail_data(dd, dtf)

    # Run backtest
    result = engine.run(strategy, _shared['data'],
                        _shared['symbol'], _shared['timeframe'])

    # Compute objective
    obj_val = (_shared['objective_fn'](result)
               if result.n_trades >= _shared['min_trades'] else -999.0)

    # Full param snapshot (defaults + loaded + optimized)
    full_params = strategy.default_params()
    full_params.update(strategy.params)

    elapsed = time.time() - t0

    return {
        'trial_id': trial_id,
        'params': full_params,
        'optimized_params': params_to_set,
        'sharpe_ratio': result.sharpe_ratio,
        'total_return': result.total_return,
        'sortino_ratio': getattr(result, 'sortino_ratio', 0),
        'max_drawdown': result.max_drawdown,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'calmar_ratio': getattr(result, 'calmar_ratio', 0),
        'n_trades': result.n_trades,
        'objective_value': obj_val,
        'elapsed': elapsed,
    }

def get_n_jobs(requested: int = -1) -> int:
    """
    Resolve n_jobs:
        -1  → cpu_count - 1 (leave one for OS)
         0  → sequential (no pool)
         1  → sequential (no pool)
        >1  → that many workers
    """
    n_cpu = cpu_count() or 4

    if requested == -1:
        return max(1, n_cpu - 1)
    elif requested <= 0:
        return 1
    else:
        return min(requested, n_cpu)


def detect_hardware() -> dict:
    """Detect CPU/GPU info for display."""
    info = {
        'cpu_count': cpu_count() or 0,
        'cpu_name': 'Unknown',
        'gpu_name': None,
        'gpu_vram': None,
    }

    # CPU name
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    info['cpu_name'] = line.split(':')[1].strip()
                    break
    except Exception:
        pass

    # GPU (NVIDIA)
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader,nounits'],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        if out:
            parts = out.split(',')
            info['gpu_name'] = parts[0].strip()
            info['gpu_vram'] = f"{int(parts[1].strip())} MB"
    except Exception:
        pass

    return info