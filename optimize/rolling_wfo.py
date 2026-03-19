"""
CryptoLab — Rolling Walk-Forward Optimization (WFO) v3
========================================================
Dual-test rolling optimization for bot auto-reoptimization simulation.

Per window:
  1. Optimize on IS data → top N trials
  2. Evaluate top N on TEST-1 (backtest + validate + regime + targets)
  3. Pick best trial from TEST-1
  4. Backtest ONLY the winner on TEST-2 (truly unseen — simulates live)

TEST-2 is the "real" metric. The T1→T2 degradation ratio tells you
how reliable your selection process is for a live bot.

Window layout:
  |--- opt-days ---|--- test1-days (select) ---|--- test2-days (verify) ---|
                    ^ evaluate top N here        ^ only winner tested here

Usage:
    python wfo_pipeline.py \\
      --symbol ETHUSDT --strategy cybercycle --tf 1h \\
      --start 2023-01-01 --end 2025-06-01 \\
      --opt-days 90 --test1-days 60 --test2-days 30 --step-days 30 \\
      --method bayesian --objective monthly_robust \\
      --n-trials 150 --n-jobs -1 --top-n 10
"""

import numpy as np
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class WFOWindow:
    """Single window with dual test evaluation."""
    window_id: int
    opt_start: str
    opt_end: str
    test1_start: str
    test1_end: str
    test2_start: str
    test2_end: str
    # All evaluated trials (on TEST-1)
    trials: List[Dict[str, Any]] = field(default_factory=list)
    best_trial: Optional[Dict[str, Any]] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    # Best trial TEST-1 metrics (selection)
    t1_sharpe: float = 0.0
    t1_return: float = 0.0
    t1_max_drawdown: float = 0.0
    t1_win_rate: float = 0.0
    t1_profit_factor: float = 0.0
    t1_n_trades: int = 0
    # Validation of best on TEST-1
    validated: bool = False
    validate_layers: int = 0
    targets_passed: int = 0
    targets_total: int = 0
    # IS metrics of best trial
    is_sharpe: float = 0.0
    is_return: float = 0.0
    # TEST-2 metrics (verification — the "real" result)
    t2_sharpe: float = 0.0
    t2_return: float = 0.0
    t2_max_drawdown: float = 0.0
    t2_win_rate: float = 0.0
    t2_profit_factor: float = 0.0
    t2_n_trades: int = 0
    t2_calmar: float = 0.0
    t2_sortino: float = 0.0
    # Degradation T1 → T2
    degradation_sharpe: float = 0.0   # (t1 - t2) / t1
    degradation_return: float = 0.0
    # T2 equity and trades (for aggregation)
    t2_equity: Optional[np.ndarray] = None
    t2_trades: list = field(default_factory=list)
    # Timing
    elapsed_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class WFOResult:
    """Complete Rolling WFO results."""
    windows: List[WFOWindow]
    # Config
    opt_days: int = 0
    test1_days: int = 0
    test2_days: int = 0
    step_days: int = 0
    start_date: str = ""
    end_date: str = ""
    strategy: str = ""
    symbol: str = ""
    timeframe: str = ""
    method: str = ""
    objective: str = ""
    n_trials: int = 0
    top_n: int = 10
    # Aggregated TEST-2 — per-window metrics (primary, honest)
    avg_return_per_window: float = 0.0
    median_return_per_window: float = 0.0
    std_return_per_window: float = 0.0
    window_sharpe: float = 0.0
    window_sortino: float = 0.0
    # Compound metrics (secondary, for reference)
    compound_return: float = 0.0
    annual_return: float = 0.0
    total_test_days: int = 0
    total_years: float = 0.0
    # Risk
    t2_max_drawdown: float = 0.0
    t2_calmar: float = 0.0
    # Trades
    t2_win_rate: float = 0.0
    t2_profit_factor: float = 0.0
    t2_n_trades: int = 0
    # Aggregated TEST-1 metrics (for comparison)
    t1_avg_sharpe: float = 0.0
    t1_avg_return: float = 0.0
    # Degradation metrics
    avg_degradation_sharpe: float = 0.0
    avg_degradation_return: float = 0.0
    # Consistency
    n_profitable_windows_t2: int = 0
    n_total_windows: int = 0
    pct_profitable_t2: float = 0.0
    n_validated_windows: int = 0
    param_stability: float = 0.0
    # Concatenated T2 equity
    t2_equity_curve: Optional[np.ndarray] = None
    total_elapsed: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  WINDOW GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_wfo_windows(
    start_date: str, end_date: str,
    opt_days: int, test1_days: int, test2_days: int, step_days: int,
) -> List[Dict[str, str]]:
    """
    Generate rolling windows with dual test periods.

    |--- opt_days ---|--- test1_days ---|--- test2_days ---|
                      (select)           (verify/live)
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    windows = []
    wid = 0
    cur = start

    while True:
        opt_end = cur + timedelta(days=opt_days)
        t1_start = opt_end
        t1_end = t1_start + timedelta(days=test1_days)
        t2_start = t1_end
        t2_end = t2_start + timedelta(days=test2_days)

        if t2_end > end:
            break

        windows.append({
            'window_id': wid,
            'opt_start': cur.strftime("%Y-%m-%d"),
            'opt_end': opt_end.strftime("%Y-%m-%d"),
            'test1_start': t1_start.strftime("%Y-%m-%d"),
            'test1_end': t1_end.strftime("%Y-%m-%d"),
            'test2_start': t2_start.strftime("%Y-%m-%d"),
            'test2_end': t2_end.strftime("%Y-%m-%d"),
        })
        wid += 1
        cur += timedelta(days=step_days)

    return windows


# ═══════════════════════════════════════════════════════════════
#  SINGLE TRIAL EVALUATION (on TEST-1, mirrors pipeline.py)
# ═══════════════════════════════════════════════════════════════

def _evaluate_single_trial(
    trial_idx, trial_params, is_metrics,
    strategy_name, leverage, capital,
    data, detail_info, engine_factory,
    symbol, tf, mc_sims=500, target_preset='conservative',
    min_trades: int = 3, verbose=True,
    signal_start_date: str = None,  # ← AGREGAR
) -> dict:
    """
    Backtest + conditional validate + regime + targets on TEST-1.
    Gate: ret>0, dd<ret, trades>=min_trades (dynamic per window).
    """
    from cli import get_strategy, _build_validation_grid

    t0 = time.time()
    result = {
        'trial_rank': trial_idx,
        'params': trial_params,
        'in_sample': _sanitize(is_metrics),
        'backtest': None, 'monthly': None,
        'validate': None, 'regime': None, 'targets': None,
        'skipped_reason': None,
    }

    bt_result = None
    try:
        if verbose:
            print(f"    🚀 Backtest...", end=" ", flush=True)

        strategy = get_strategy(strategy_name)
        strategy.set_params({'leverage': leverage})
        strategy.set_params(trial_params)

        engine = engine_factory()
        if signal_start_date:
            from datetime import datetime as _dt, timezone as _tz
            _t1_ts = int(_dt.strptime(signal_start_date, '%Y-%m-%d').replace(
                tzinfo=_tz.utc).timestamp() * 1000)
            engine.set_signal_start(_t1_ts)
        bt_result = engine.run(strategy, data, symbol, tf)

        bt = {
            'total_return': round(bt_result.total_return, 2),
            'annual_return': round(bt_result.annual_return, 2),
            'sharpe_ratio': round(bt_result.sharpe_ratio, 2),
            'sortino_ratio': round(bt_result.sortino_ratio, 2),
            'max_drawdown': round(bt_result.max_drawdown, 2),
            'calmar_ratio': round(bt_result.calmar_ratio, 2),
            'win_rate': round(bt_result.win_rate, 1),
            'profit_factor': round(bt_result.profit_factor, 2),
            'n_trades': bt_result.n_trades,
            'n_longs': bt_result.n_longs,
            'n_shorts': bt_result.n_shorts,
            'avg_win': round(bt_result.avg_win, 2),
            'avg_loss': round(bt_result.avg_loss, 2),
        }
        n_liq = sum(1 for t in bt_result.trades
                    if t.exit_reason == 'liquidation')
        if n_liq > 0:
            bt['liquidations'] = n_liq
        result['backtest'] = bt

        if verbose:
            print(f"SR={bt_result.sharpe_ratio:.2f} "
                  f"Ret={bt_result.total_return:+.1f}% "
                  f"WR={bt_result.win_rate:.1f}% "
                  f"DD={bt_result.max_drawdown:.1f}% "
                  f"T={bt_result.n_trades}"
                  f"{f' ⚠️{n_liq}LIQ' if n_liq else ''}")

        # Monthly
        try:
            from optimize.grid_search import compute_monthly_stats
            ms = compute_monthly_stats(bt_result.trades)
            if ms['n_months'] >= 2:
                result['monthly'] = {
                    'avg_return': round(ms['avg_monthly_return'], 2),
                    'pct_positive': round(ms['pct_positive'], 1),
                    'monthly_sharpe': round(ms['monthly_sharpe'], 2),
                    'worst_month': round(ms['worst_month'], 2),
                    'best_month': round(ms['best_month'], 2),
                    'n_months': ms['n_months'],
                    'months': ms['months'],
                }
        except Exception:
            pass

        # Gate (dynamic min_trades based on test period length)
        ret = bt_result.total_return
        dd = bt_result.max_drawdown
        nt = bt_result.n_trades
        passes = (ret > 0 and dd < ret and nt >= min_trades)

        if not passes:
            result['skipped_reason'] = (
                f"gate: ret={ret:+.1f}% dd={dd:.1f}% t={nt} "
                f"(min={min_trades})")
        else:
            # Validate
            try:
                if verbose:
                    print(f"    🔬 Validate...", end=" ", flush=True)
                from optimize.anti_overfit import AntiOverfitPipeline
                sv = get_strategy(strategy_name)
                sv.set_params({'leverage': leverage})
                sv.set_params(trial_params)
                pg = _build_validation_grid(sv)
                vp = AntiOverfitPipeline(
                    wfa_windows=4, mc_simulations=mc_sims,
                    fail_fast=False, verbose=False)
                vr = vp.run(sv, data, engine_factory, pg, symbol, tf)
                val = {
                    'all_passed': getattr(vr, 'all_passed', False),
                    'layers_passed': getattr(vr, 'layers_passed', 0),
                    'wfa_passed': getattr(vr, 'wfa_passed', False),
                    'mc_passed': getattr(vr, 'mc_passed', False),
                    'sensitivity_passed': getattr(vr, 'sensitivity_passed', False),
                    'overfit_passed': getattr(vr, 'overfit_passed', False),
                }
                result['validate'] = val
                p = val['layers_passed']
                m = "✅" if val['all_passed'] else "⚠️"
                if verbose:
                    print(f"{m} {p}/4 layers")
            except Exception as e:
                if verbose:
                    print(f"❌ {e}")
                result['validate'] = {'status': 'FAILED', 'error': str(e)}

            # Regime
            try:
                if verbose:
                    print(f"    🔄 Regime...", end=" ", flush=True)
                from ml.regime_detector import (
                    detect_regime, strategy_regime_performance)
                sr = get_strategy(strategy_name)
                sr.set_params({'leverage': leverage})
                sr.set_params(trial_params)
                rr = detect_regime(data, method='vt', verbose=False)
                perf = strategy_regime_performance(
                    sr, data, engine_factory, rr,
                    symbol=symbol, timeframe=tf, verbose=False)
                reg = {}
                if isinstance(perf, dict):
                    reg = {str(k): v for k, v in perf.items()
                           if isinstance(v, dict)}
                result['regime'] = reg
                if verbose:
                    print("✅")
            except Exception as e:
                if verbose:
                    print(f"❌ {e}")
                result['regime'] = {'status': 'FAILED', 'error': str(e)}

            # Targets
            try:
                if verbose:
                    print(f"    📅 Targets...", end=" ", flush=True)
                from ml.temporal_targets import (
                    evaluate_targets,
                    CONSERVATIVE_TARGETS, AGGRESSIVE_TARGETS,
                    CONSISTENCY_TARGETS)
                tmap = {
                    'conservative': CONSERVATIVE_TARGETS,
                    'aggressive': AGGRESSIVE_TARGETS,
                    'consistency': CONSISTENCY_TARGETS,
                }
                tspecs = tmap.get(target_preset, CONSERVATIVE_TARGETS)
                tt = evaluate_targets(
                    tspecs, bt_result.trades,
                    data.get('timestamp', np.arange(len(data['close']))),
                    bt_result.equity_curve,
                    initial_capital=capital, verbose=False)
                tgt = {
                    'all_passed': getattr(tt, 'all_passed', False),
                    'n_passed': getattr(tt, 'n_passed', 0),
                    'n_targets': getattr(tt, 'n_targets', 0),
                }
                result['targets'] = tgt
                m = "✅" if tgt['all_passed'] else "⚠️"
                if verbose:
                    print(f"{m} {tgt['n_passed']}/{tgt['n_targets']} targets")
            except Exception as e:
                if verbose:
                    print(f"❌ {e}")
                result['targets'] = {'status': 'FAILED', 'error': str(e)}

    except Exception as e:
        if verbose:
            print(f"❌ {e}")
        result['backtest'] = {'status': 'FAILED', 'error': str(e)}

    result['_bt_result'] = bt_result
    result['elapsed_seconds'] = round(time.time() - t0, 1)
    return result


def _sanitize(m):
    safe = {}
    for k, v in m.items():
        if isinstance(v, (np.integer,)):
            safe[k] = int(v)
        elif isinstance(v, (np.floating,)):
            safe[k] = float(v)
        elif isinstance(v, (int, float, str, bool)):
            safe[k] = v
    return safe


def _find_best_trial(trials, min_validate_layers=0, min_win_rate=0.0):
    """
    Pick best trial.

    Priority:
      1. Filter by min_validate_layers AND min_win_rate
      2. Among qualified: prefer all_passed, then highest OOS Sharpe
      3. If none qualify: return None (caller decides what to do)

    Args:
        trials: list of trial result dicts
        min_validate_layers: minimum validation layers (0=any)
        min_win_rate: minimum win rate % (0=any, e.g. 40.0 for 40%)

    Returns:
        best trial dict, or None if no trial qualifies
    """
    valid = [t for t in trials if t is not None]
    if not valid:
        return None

    def _get_layers(t):
        val = t.get('validate', {})
        if isinstance(val, dict) and 'layers_passed' in val:
            return val['layers_passed']
        return 0

    def _get_wr(t):
        bt = t.get('backtest', {})
        if isinstance(bt, dict):
            return bt.get('win_rate', 0)
        return 0

    # Filter by validation + win rate
    if min_validate_layers > 0 or min_win_rate > 0:
        qualified = [
            t for t in valid
            if isinstance(t.get('backtest'), dict)
            and 'sharpe_ratio' in t['backtest']
            and _get_layers(t) >= min_validate_layers
            and _get_wr(t) >= min_win_rate
        ]
        if not qualified:
            return None  # triggers retry
        valid = qualified

    def score(t):
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            return (-999, -999)
        v = 1 if (isinstance(val, dict) and val.get('all_passed')) else 0
        return (v, bt['sharpe_ratio'])

    return max(valid, key=score)


# ═══════════════════════════════════════════════════════════════
#  OPTIMIZATION (incremental — keeps Optuna study alive)
# ═══════════════════════════════════════════════════════════════

class _OptimizerState:
    """Holds optimizer state between retry attempts.
    For bayesian: keeps the Optuna study so TPE model accumulates knowledge.
    For grid/genetic: stateless (fresh each retry).
    """
    def __init__(self):
        self.study = None        # Optuna study (bayesian only)
        self.total_trials = 0    # Total trials run so far
        self.all_bt_trials = []  # All BayesianTrial/GridTrial results


def _optimize_increment(
    state: _OptimizerState,
    strategy_name, data, detail_info, capital, leverage,
    method, objective, n_trials_this_round, n_jobs,
    symbol, timeframe, no_intrabar,
    param_subset=None, top_n=10, verbose=True,
) -> list:
    """
    Run optimization increment and return top N trials.

    For bayesian: continues the SAME Optuna study from previous round.
      Round 1: study.optimize(100) → TPE has 100 trials
      Round 2: study.optimize(+100) → TPE has 200 trials (incremental!)
      The TPE posterior is NEVER reset between retries.

    For grid/genetic: fresh optimization each round (no incremental support).

    Returns list of top_n trial dicts (same format as before).
    """
    from cli import (get_strategy, _make_engine_factory,
                     _build_validation_grid)
    from data.bitget_client import MarketConfig

    strategy = get_strategy(strategy_name)
    strategy.set_params({'leverage': leverage})

    market = MarketConfig.detect(symbol)
    ef = _make_engine_factory(capital, detail_info, market,
                              no_intrabar=no_intrabar)
    ec = {
        'capital': capital, 'market_config': market,
        'detail_data': detail_info.get('data') if detail_info else None,
        'detail_tf': detail_info.get('tf') if detail_info else None,
        'no_intrabar': no_intrabar,
    }

    if param_subset:
        vn = {pd.name for pd in strategy.parameter_defs()}
        param_subset = [p for p in param_subset if p in vn]

    if method == 'bayesian':
        return _bayesian_increment(
            state, strategy, data, ef, ec, n_trials_this_round,
            n_jobs, objective, symbol, timeframe, param_subset,
            top_n, verbose)
    elif method == 'genetic':
        from optimize.genetic import GeneticOptimizer
        opt = GeneticOptimizer(
            n_generations=max(20, n_trials_this_round // 20),
            pop_size=min(40, n_trials_this_round), objective=objective,
            n_jobs=n_jobs, verbose=verbose)
        r = opt.run(strategy=strategy, data=data, engine_factory=ef,
                    symbol=symbol, timeframe=timeframe,
                    param_subset=param_subset,
                    engine_config=ec if n_jobs > 1 else None)
        trials = getattr(r, 'hall_of_fame', [])
    else:
        from optimize.grid_search import GridSearchOptimizer
        sg = get_strategy(strategy_name)
        sg.set_params({'leverage': leverage})
        pg = _build_validation_grid(sg)
        opt = GridSearchOptimizer(objective=objective, n_jobs=n_jobs,
                                  verbose=verbose)
        r = opt.run(strategy=sg, data=data, engine_factory=ef,
                    param_grid=pg, symbol=symbol, timeframe=timeframe,
                    engine_config=ec if n_jobs > 1 else None)
        trials = getattr(r, 'trials', [])

    return [
        {
            'rank': i + 1,
            'params': getattr(t, 'params', {}),
            'metrics': {
                'sharpe': getattr(t, 'sharpe_ratio', 0),
                'return': getattr(t, 'total_return', 0),
                'win_rate': getattr(t, 'win_rate', 0),
                'max_drawdown': getattr(t, 'max_drawdown', 0),
                'profit_factor': getattr(t, 'profit_factor', 0),
                'n_trades': getattr(t, 'n_trades', 0),
            }
        }
        for i, t in enumerate(trials[:top_n])
    ]


def _bayesian_increment(
    state, strategy, data, engine_factory, engine_config,
    n_trials_this_round, n_jobs, objective,
    symbol, timeframe, param_subset, top_n, verbose,
):
    """
    Incremental Bayesian optimization.

    FIRST call: creates study + runs n_trials
    SUBSEQUENT calls: reuses SAME study + runs n_trials MORE
    The TPE model has full history → each round is smarter.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optimize.bayesian import BayesianOptimizer, BayesianTrial
    from optimize.grid_search import OBJECTIVES

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create study on first call, reuse on retries
    if state.study is None:
        sampler = TPESampler(seed=42, n_startup_trials=20)
        state.study = optuna.create_study(
            direction="maximize", sampler=sampler)

    study = state.study
    prev_total = state.total_trials

    # Build objective function
    opt = BayesianOptimizer(
        n_trials=n_trials_this_round, objective=objective,
        n_jobs=n_jobs, verbose=False)

    param_defs = strategy.parameter_defs()

    if n_jobs > 1 and engine_config:
        # Parallel: use ask-tell API
        from optimize.parallel import (
            setup_workers, evaluate_trial, create_pool)

        setup_workers(
            strategy=strategy, data=data,
            engine_config=engine_config,
            objective_name=objective,
            min_trades=opt.min_trades,
            symbol=symbol, timeframe=timeframe,
        )

        batch_size = n_jobs
        completed = 0
        try:
            best_obj = study.best_value
        except ValueError:
            best_obj = -999.0
        t0 = time.time()

        if verbose:
            print(f"\n  ⚡ Parallel: {n_jobs} workers, batch_size={batch_size}")

        try:
            with create_pool(n_jobs) as pool:
                while completed < n_trials_this_round:
                    remaining = n_trials_this_round - completed
                    current_batch = min(batch_size, remaining)

                    batch_items = []
                    for _ in range(current_batch):
                        trial = study.ask()
                        params = opt._build_search_space(
                            trial, param_defs, param_subset)
                        batch_items.append((trial, params))

                    work_items = [
                        (prev_total + completed + i, params)
                        for i, (_, params) in enumerate(batch_items)
                    ]
                    results = pool.map(evaluate_trial, work_items)

                    for idx, ((optuna_trial, _params), res) in enumerate(
                            zip(batch_items, results)):
                        study.tell(optuna_trial, res['objective_value'])

                        bt = BayesianTrial(
                            trial_id=res['trial_id'],
                            params=res['params'],
                            sharpe_ratio=res['sharpe_ratio'],
                            total_return=res['total_return'],
                            max_drawdown=res['max_drawdown'],
                            win_rate=res['win_rate'],
                            profit_factor=res['profit_factor'],
                            n_trades=res['n_trades'],
                            objective_value=res['objective_value'],
                        )
                        state.all_bt_trials.append(bt)
                        completed += 1

                        # Per-trial progress (same format as BayesianOptimizer)
                        if verbose:
                            tid = prev_total + completed
                            total = prev_total + n_trials_this_round
                            is_new_best = (res['objective_value'] > best_obj
                                           and res['objective_value'] > -900)
                            if is_new_best:
                                best_obj = res['objective_value']
                            star = "★" if is_new_best else " "
                            elapsed = time.time() - t0
                            avg_s = elapsed / max(1, completed)
                            # Show key params (first 5)
                            p = res['params']
                            p_keys = list(p.keys())[:5]
                            p_str = ", ".join(
                                f"{k}: {p[k]}" for k in p_keys)
                            print(
                                f"  {star} {tid:>4}/{total} "
                                f"obj={res['objective_value']:>7.3f} "
                                f"SR={res['sharpe_ratio']:>5.2f} "
                                f"Ret={res['total_return']:>+7.1f}% "
                                f"WR={res['win_rate']:>4.1f}% "
                                f"DD={res['max_drawdown']:>5.1f}% "
                                f"PF={res['profit_factor']:.2f} "
                                f"T={res['n_trades']:>4} "
                                f"[{avg_s:.0f}s] "
                                f"{{{p_str}}}")

        except KeyboardInterrupt:
            if verbose:
                print(f"  ⚠️ Interrupted at trial {prev_total + completed}")

        if verbose and completed > 0:
            elapsed = time.time() - t0
            print(f"\n  ────────────────────────────────────────")
            print(f"  ✅ {completed} trials in {elapsed:.1f}s "
                  f"(best obj: {best_obj:.4f})")

    else:
        # Sequential
        import copy

        def _objective(trial):
            params = opt._build_search_space(trial, param_defs, param_subset)
            strat = copy.deepcopy(strategy)
            strat.set_params(params)
            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)
            if result.n_trades < opt.min_trades:
                return -999.0
            obj_val = opt._compute_objective(result)

            bt = BayesianTrial(
                trial_id=prev_total + len(state.all_bt_trials),
                params=params,
                sharpe_ratio=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                n_trades=result.n_trades,
                objective_value=obj_val,
            )
            state.all_bt_trials.append(bt)
            return obj_val

        study.optimize(_objective, n_trials=n_trials_this_round,
                       show_progress_bar=False)

    state.total_trials = prev_total + n_trials_this_round

    # Return top_n from ALL trials in the study (not just this round)
    sorted_trials = sorted(state.all_bt_trials,
                           key=lambda t: t.objective_value, reverse=True)

    if verbose and sorted_trials:
        best_t = sorted_trials[0]
        print(f"\n  Total trials in study: {state.total_trials} "
              f"({n_trials_this_round} new this round)")
        print(f"  Best trial #{best_t.trial_id}:")
        print(f"    Objective:  {best_t.objective_value:.4f}")
        print(f"    Sharpe:     {best_t.sharpe_ratio:.2f}")
        print(f"    Return:     {best_t.total_return:+.1f}%")
        print(f"    Win Rate:   {best_t.win_rate:.1f}%")
        print(f"    Max DD:     {best_t.max_drawdown:.1f}%")
        print(f"    PF:         {best_t.profit_factor:.2f}")
        print(f"    Trades:     {best_t.n_trades}")

        # Show top-N summary
        print(f"\n  Top {min(top_n, len(sorted_trials))} trials:")
        for i, t in enumerate(sorted_trials[:top_n]):
            print(f"    #{i+1:>2} obj={t.objective_value:>7.3f} "
                  f"SR={t.sharpe_ratio:>5.2f} "
                  f"Ret={t.total_return:>+7.1f}% "
                  f"WR={t.win_rate:>4.1f}% "
                  f"DD={t.max_drawdown:>5.1f}% "
                  f"PF={t.profit_factor:.2f} "
                  f"T={t.n_trades:>4}")

    return [
        {
            'rank': i + 1,
            'params': t.params,
            'metrics': {
                'sharpe': t.sharpe_ratio,
                'return': t.total_return,
                'win_rate': t.win_rate,
                'max_drawdown': t.max_drawdown,
                'profit_factor': t.profit_factor,
                'n_trades': t.n_trades,
            }
        }
        for i, t in enumerate(sorted_trials[:top_n])
    ]


# ═══════════════════════════════════════════════════════════════
#  SINGLE WINDOW (full pipeline with dual test)
# ═══════════════════════════════════════════════════════════════

def _run_single_window(
    win, strategy_name, symbol, timeframe,
    capital, leverage, method, objective, n_trials, n_jobs,
    top_n=10, mc_sims=500, target_preset='conservative',
    no_intrabar=False, param_subset=None, exclude_params=None,
    min_trades_per_week: float = 1.5,
    min_validate_layers: int = 0,
    min_win_rate: float = 0.0,
    max_retries: int = 0,
    retry_trials_delta: int = 100,
    verbose=True,
) -> WFOWindow:
    """
    Full pipeline per window:
      Step 1: Optimize on IS
      Step 2: Evaluate top-N on TEST-1 (select)
      Step 3: Pick best trial (must have >= min_validate_layers + min_win_rate)
              If none qualifies and retries remain, add more trials to same study
      Step 4: Backtest winner on TEST-2 (verify)

    Retry logic (incremental, Bayesian only):
      - The Optuna study is KEPT ALIVE across retries
      - Each retry adds retry_trials_delta NEW trials to the same study
      - TPE model has FULL history → each round is smarter than the last
      - Round 1: 100 trials → no qualified → retry
      - Round 2: +100 trials (study now has 200) → finds qualified trial
      - After max_retries, pick best available (ignoring gate) or skip
    """
    from cli import get_strategy, _load_data, _make_engine_factory
    from data.bitget_client import MarketConfig

    wid = win['window_id']
    t0 = time.time()

    result = WFOWindow(
        window_id=wid,
        opt_start=win['opt_start'], opt_end=win['opt_end'],
        test1_start=win['test1_start'], test1_end=win['test1_end'],
        test2_start=win['test2_start'], test2_end=win['test2_end'],
    )

    try:
        if verbose:
            print(f"\n  {'═' * 70}")
            print(f"  WINDOW #{wid + 1}: "
                  f"OPT [{win['opt_start']}→{win['opt_end']}] "
                  f"T1 [{win['test1_start']}→{win['test1_end']}] "
                  f"T2 [{win['test2_start']}→{win['test2_end']}]")
            print(f"  {'═' * 70}")

        # Build param subset
        actual_subset = None
        if param_subset:
            actual_subset = param_subset
        elif exclude_params:
            st = get_strategy(strategy_name)
            vn = {pd.name for pd in st.parameter_defs()}
            excl = {p for p in exclude_params if p in vn}
            actual_subset = [pd.name for pd in st.parameter_defs()
                             if pd.name not in excl]

        # ═══ LOAD IS DATA (once, reused across retries) ═══
        if verbose:
            print(f"\n  📊 Loading IS data "
                  f"({win['opt_start']}→{win['opt_end']})")

        is_args = {'symbol': symbol, 'timeframe': timeframe,
                   'start': win['opt_start'], 'end': win['opt_end']}
        is_data, is_detail, _, _ = _load_data(is_args, timeframe)

        if verbose:
            print(f"     {len(is_data['close'])} bars loaded")

        # ═══ LOAD T1 DATA (once, reused across retries) ═══
        if verbose:
            print(f"\n  📡 Loading TEST-1 data "
                  f"({win['test1_start']}→{win['test1_end']})")

        t1_args = {'symbol': symbol, 'timeframe': timeframe,
                   'start': win['test1_start'], 'end': win['test1_end']}
        t1_data, t1_detail, _, _ = _load_data(t1_args, timeframe)
        market = MarketConfig.detect(symbol)
        t1_ef = _make_engine_factory(capital, t1_detail, market,
                                     no_intrabar=no_intrabar)

        if verbose:
            print(f"     {len(t1_data['close'])} bars loaded")

        # Dynamic min trades
        t1_start_dt = datetime.strptime(win['test1_start'], "%Y-%m-%d")
        t1_end_dt = datetime.strptime(win['test1_end'], "%Y-%m-%d")
        t1_days = (t1_end_dt - t1_start_dt).days
        t1_weeks = max(1.0, t1_days / 7.0)
        dyn_min_trades = max(2, int(t1_weeks * min_trades_per_week))

        if verbose:
            print(f"     Min trades: {dyn_min_trades} "
                  f"({min_trades_per_week}/week × {t1_weeks:.1f} weeks)")

        # ═══ RETRY LOOP: OPTIMIZE → EVALUATE → PICK ═══
        # Bayesian: SAME Optuna study across retries (truly incremental)
        # Each retry adds more trials to the existing TPE model.
        best = None
        all_trial_results = []
        opt_state = _OptimizerState()  # Keeps study alive across retries
        current_n_trials = n_trials

        has_quality_gate = (min_validate_layers > 0 or min_win_rate > 0)
        gate_desc = []
        if min_validate_layers > 0:
            gate_desc.append(f"≥{min_validate_layers}/4 val")
        if min_win_rate > 0:
            gate_desc.append(f"WR≥{min_win_rate:.0f}%")
        gate_str = " + ".join(gate_desc) if gate_desc else ""

        for attempt in range(1 + max_retries):
            if attempt > 0:
                # Add more trials to the SAME study
                extra = abs(retry_trials_delta)
                if verbose:
                    prev = opt_state.total_trials
                    print(f"\n  🔄 RETRY {attempt}/{max_retries}: "
                          f"+{extra} trials (study has {prev}, "
                          f"will have {prev + extra})")
                current_n_trials = extra  # Only the NEW trials this round
            else:
                if verbose:
                    print(f"\n  🔧 Optimizing ({method}, {current_n_trials} trials, "
                          f"obj={objective})...")

            top_trials = _optimize_increment(
                opt_state,
                strategy_name, is_data, is_detail, capital, leverage,
                method, objective, current_n_trials, n_jobs,
                symbol, timeframe, no_intrabar,
                actual_subset, top_n, verbose=verbose)

            if not top_trials:
                if verbose:
                    print(f"  ⚠️ No trials from optimizer")
                continue

            if verbose:
                print(f"  ✅ {len(top_trials)} top trials → evaluating on T1...")

            # Step 2: Evaluate on T1
            trial_results = []
            for ti in top_trials:
                tidx = ti.get('rank', 1)
                tparams = ti.get('params', {})
                ism = ti.get('metrics', {})

                if verbose:
                    print(f"\n  {'━' * 60}")
                    print(f"  Trial #{tidx}/{len(top_trials)}  "
                          f"(IS: SR={ism.get('sharpe',0):.2f} "
                          f"Ret={ism.get('return',0):+.1f}%)")
                    print(f"  {'━' * 60}")

                tr = _evaluate_single_trial(
                    tidx, tparams, ism, strategy_name,
                    leverage, capital, t1_data, t1_detail,
                    t1_ef, symbol, timeframe,
                    mc_sims, target_preset, dyn_min_trades, verbose,signal_start_date=win['test1_start'])
                trial_results.append(tr)

            all_trial_results.extend(trial_results)

            # Step 3: Pick best with quality gate
            best = _find_best_trial(
                trial_results, min_validate_layers, min_win_rate)

            if verbose:
                _print_leaderboard(trial_results, best)

            if best is not None:
                if verbose and attempt > 0:
                    layers = best.get('validate', {}).get('layers_passed', 0)
                    wr = best.get('backtest', {}).get('win_rate', 0)
                    print(f"\n  ✅ Found qualified trial on attempt "
                          f"{attempt + 1} ({layers}/4 layers, WR={wr:.1f}%) "
                          f"[study total: {opt_state.total_trials} trials]")
                break

            if verbose and has_quality_gate:
                print(f"\n  ⚠️ No trial passed quality gate ({gate_str})")

        # Fallback: pick best ignoring quality gate
        is_fallback = False
        if best is None and all_trial_results:
            best = _find_best_trial(all_trial_results, 0, 0.0)
            is_fallback = True
            if verbose and best:
                wr = best.get('backtest', {}).get('win_rate', 0)
                print(f"\n  ⚠️ Fallback: best unqualified trial "
                      f"(#{best.get('trial_rank', '?')}, WR={wr:.1f}%)")

        # Strip _bt_result from all trials
        for tr in all_trial_results:
            bt_obj = tr.pop('_bt_result', None)
            if (best is not None
                    and tr.get('trial_rank') == best.get('trial_rank')
                    and bt_obj is not None):
                result.oos_equity = bt_obj.equity_curve  # T1 equity for reference
                # (T2 equity replaces this below)

        result.trials = all_trial_results
        result.best_trial = best

        if best and isinstance(best.get('backtest'), dict):
            bt = best['backtest']
            result.best_params = best.get('params', {})
            result.t1_sharpe = bt.get('sharpe_ratio', 0)
            result.t1_return = bt.get('total_return', 0)
            result.t1_max_drawdown = bt.get('max_drawdown', 0)
            result.t1_win_rate = bt.get('win_rate', 0)
            result.t1_profit_factor = bt.get('profit_factor', 0)
            result.t1_n_trades = bt.get('n_trades', 0)

            ism = best.get('in_sample', {})
            result.is_sharpe = ism.get('sharpe', 0)
            result.is_return = ism.get('return', 0)

            val = best.get('validate', {})
            if isinstance(val, dict):
                result.validated = val.get('all_passed', False)
                result.validate_layers = val.get('layers_passed', 0)
            tgt = best.get('targets', {})
            if isinstance(tgt, dict):
                result.targets_passed = tgt.get('n_passed', 0)
                result.targets_total = tgt.get('n_targets', 0)
        else:
            result.error = "No valid trial on TEST-1"
            result.elapsed_seconds = time.time() - t0
            return result

        # ═══ STEP 4: BACKTEST WINNER ON TEST-2 ═══
        if is_fallback and has_quality_gate:
            if verbose:
                print(f"\n  ⏭️  Skip T2: no trial passed quality gate — "
                      f"flat period (0% return)")
            result.t2_sharpe = 0.0
            result.t2_return = 0.0
            result.t2_max_drawdown = 0.0
            result.t2_win_rate = 0.0
            result.t2_profit_factor = 0.0
            result.t2_n_trades = 0
            result.t2_calmar = 0.0
            result.t2_sortino = 0.0
            result.t2_equity = np.array([])
            result.t2_trades = []
            result.degradation_sharpe = 0.0
            result.degradation_return = 0.0
            result.error = "skipped_t2_fallback"
            result.elapsed_seconds = time.time() - t0
            return result

        if verbose:
            print(f"\n  🎯 Step 4: Verify winner (Trial #{best.get('trial_rank', '?')}) "
                  f"on TEST-2 ({win['test2_start']}→{win['test2_end']})")

        t2_args = {'symbol': symbol, 'timeframe': timeframe,
                   'start': win['test2_start'], 'end': win['test2_end']}
        t2_data, t2_detail, _, _ = _load_data(t2_args, timeframe)
        t2_ef = _make_engine_factory(capital, t2_detail, market,
                                     no_intrabar=no_intrabar)

        if verbose:
            print(f"     {len(t2_data['close'])} bars loaded")
            print(f"    🚀 Backtest...", end=" ", flush=True)

        s2 = get_strategy(strategy_name)
        s2.set_params({'leverage': leverage})
        s2.set_params(result.best_params)

        engine2 = t2_ef()
        from datetime import datetime as _dt, timezone as _tz
        _t2_ts = int(_dt.strptime(win['test2_start'], '%Y-%m-%d').replace(
            tzinfo=_tz.utc).timestamp() * 1000)
        engine2.set_signal_start(_t2_ts)
        bt2 = engine2.run(s2, t2_data, symbol, timeframe)

        result.t2_sharpe = bt2.sharpe_ratio
        result.t2_return = bt2.total_return
        result.t2_max_drawdown = bt2.max_drawdown
        result.t2_win_rate = bt2.win_rate
        result.t2_profit_factor = bt2.profit_factor
        result.t2_n_trades = bt2.n_trades
        result.t2_calmar = bt2.calmar_ratio
        result.t2_sortino = bt2.sortino_ratio
        result.t2_equity = bt2.equity_curve
        result.t2_trades = bt2.trades

        if verbose:
            print(f"SR={bt2.sharpe_ratio:.2f} "
                  f"Ret={bt2.total_return:+.1f}% "
                  f"WR={bt2.win_rate:.1f}% "
                  f"DD={bt2.max_drawdown:.1f}% "
                  f"T={bt2.n_trades}")

        # Degradation
        if abs(result.t1_sharpe) > 0.01:
            result.degradation_sharpe = round(
                (result.t1_sharpe - result.t2_sharpe) / abs(result.t1_sharpe) * 100, 1)
        if abs(result.t1_return) > 0.01:
            result.degradation_return = round(
                (result.t1_return - result.t2_return) / abs(result.t1_return) * 100, 1)

        if verbose:
            icon = "🟢" if result.t2_return > 0 else "🔴"
            deg = result.degradation_sharpe
            deg_icon = "✅" if abs(deg) < 30 else "⚠️" if abs(deg) < 60 else "❌"
            print(f"    {icon} T2 Result | "
                  f"{deg_icon} Degradation: {deg:+.1f}% Sharpe")

    except Exception as e:
        result.error = str(e)
        if verbose:
            print(f"\n  ❌ Window error: {e}")
            import traceback
            traceback.print_exc()

    result.elapsed_seconds = time.time() - t0
    return result


def _print_leaderboard(trials, best):
    if not trials:
        return
    print(f"\n  {'─' * 70}")
    print(f"  {'#':>3} {'SR':>6} {'Ret%':>8} {'WR%':>6} {'DD%':>6} "
          f"{'PF':>5} {'T':>4} {'Val':>5} {'Tgt':>5}")
    print(f"  {'─' * 70}")
    for t in trials:
        bt = t.get('backtest', {})
        val = t.get('validate', {})
        tgt = t.get('targets', {})
        if not isinstance(bt, dict) or 'sharpe_ratio' not in bt:
            print(f"  #{t['trial_rank']:>2}  — failed —")
            continue
        vs = (f"{val.get('layers_passed','?')}/4"
              if isinstance(val, dict) and 'layers_passed' in val else '—')
        ts = (f"{tgt.get('n_passed','?')}/{tgt.get('n_targets','?')}"
              if isinstance(tgt, dict) and 'n_passed' in tgt else '—')
        mk = " ★" if (best and t.get('trial_rank') == best.get('trial_rank')) else ""
        print(f"  #{t['trial_rank']:>2} "
              f"{bt.get('sharpe_ratio',0):>5.2f} "
              f"{bt.get('total_return',0):>+7.1f}% "
              f"{bt.get('win_rate',0):>5.1f}% "
              f"{bt.get('max_drawdown',0):>5.1f}% "
              f"{bt.get('profit_factor',0):>4.2f} "
              f"{bt.get('n_trades',0):>4} "
              f"{vs:>5} {ts:>5}{mk}")


# ═══════════════════════════════════════════════════════════════
#  AGGREGATION (uses TEST-2 as "real" results)
# ═══════════════════════════════════════════════════════════════

def _aggregate(windows, capital):
    valid = [w for w in windows if w.error is None and w.t2_n_trades > 0]
    if not valid:
        return {}

    # ── Per-window returns (the honest primary metrics) ──
    w_rets = [w.t2_return for w in valid]
    avg_ret = float(np.mean(w_rets))
    median_ret = float(np.median(w_rets))
    std_ret = float(np.std(w_rets)) if len(w_rets) > 1 else 0

    # Window-level Sharpe (most stable metric)
    w_sharpe = avg_ret / std_ret if std_ret > 0 else 0

    # ── Compound equity (for max DD and annualized return) ──
    all_eq = []
    for w in valid:
        if w.t2_equity is not None and len(w.t2_equity) > 0:
            if all_eq:
                s = all_eq[-1][-1] / w.t2_equity[0] if w.t2_equity[0] != 0 else 1.0
                all_eq.append(w.t2_equity * s)
            else:
                all_eq.append(w.t2_equity)

    ceq = np.concatenate(all_eq) if all_eq else np.array([capital])

    # Compound return
    compound_ret = (ceq[-1] / ceq[0] - 1) * 100 if len(ceq) > 1 else 0

    # Annualized return (from compound, using actual test days)
    total_test_days = sum(
        (datetime.strptime(w.test2_end, "%Y-%m-%d") -
         datetime.strptime(w.test2_start, "%Y-%m-%d")).days
        for w in valid
    )
    years = total_test_days / 365.0 if total_test_days > 0 else 1.0
    annual_ret = ((ceq[-1] / ceq[0]) ** (1.0 / years) - 1) * 100 if years > 0 else 0

    # Max drawdown from compound equity
    peak = np.maximum.accumulate(ceq)
    dd = (peak - ceq) / peak * 100
    mdd = float(np.max(dd)) if len(dd) > 0 else 0

    # Calmar = annualized return / max DD (correct)
    calmar = annual_ret / mdd if mdd > 0 else 0

    # Sortino from per-window returns
    neg_rets = [r for r in w_rets if r < 0]
    if neg_rets and np.std(neg_rets) > 0:
        w_sortino = avg_ret / np.std(neg_rets)
    else:
        w_sortino = w_sharpe  # no negative windows → use sharpe

    # Trades
    all_t = []
    for w in valid:
        all_t.extend(w.t2_trades)
    nt = len(all_t)
    if nt == 0:
        return {}

    wins = [t for t in all_t if t.net_pnl > 0]
    losses = [t for t in all_t if t.net_pnl <= 0]
    wr = len(wins) / nt * 100
    gp = sum(t.net_pnl for t in wins) if wins else 0
    gl = abs(sum(t.net_pnl for t in losses)) if losses else 1
    pf = gp / gl if gl > 0 else 0

    # Degradation
    degs_sr = [w.degradation_sharpe for w in valid
               if abs(w.degradation_sharpe) > 0.001]
    degs_rt = [w.degradation_return for w in valid
               if abs(w.degradation_return) > 0.001]

    stab = _param_stability(valid)

    return {
        # Per-window metrics (primary — honest)
        'avg_return_per_window': round(avg_ret, 2),
        'median_return_per_window': round(median_ret, 2),
        'std_return_per_window': round(std_ret, 2),
        'window_sharpe': round(w_sharpe, 2),
        'window_sortino': round(w_sortino, 2),
        # Compound metrics (secondary — for reference)
        'compound_return': round(compound_ret, 2),
        'annual_return': round(annual_ret, 2),
        'total_test_days': total_test_days,
        'total_years': round(years, 2),
        # Risk
        't2_max_drawdown': round(mdd, 2),
        'calmar': round(calmar, 2),
        # Trades
        't2_win_rate': round(wr, 1),
        't2_profit_factor': round(pf, 2),
        't2_n_trades': nt,
        # Consistency
        'n_profitable_t2': sum(1 for w in valid if w.t2_return > 0),
        'n_validated': sum(1 for w in valid if w.validated),
        'n_total': len(valid),
        'pct_profitable_t2': round(
            sum(1 for w in valid if w.t2_return > 0) / len(valid) * 100, 1),
        # Degradation
        'avg_deg_sharpe': round(np.mean(degs_sr), 1) if degs_sr else 0,
        'avg_deg_return': round(np.mean(degs_rt), 1) if degs_rt else 0,
        # Comparison
        't1_avg_sharpe': round(np.mean([w.t1_sharpe for w in valid]), 2),
        't1_avg_return': round(np.mean([w.t1_return for w in valid]), 2),
        # Meta
        'param_stability': round(stab, 3),
        'equity': ceq,
    }


def _param_stability(windows):
    if len(windows) < 2:
        return 1.0
    ap = [w.best_params for w in windows if w.best_params]
    if len(ap) < 2:
        return 1.0
    nk = [k for k in ap[0]
          if isinstance(ap[0][k], (int, float))
          and all(k in p and isinstance(p[k], (int, float)) for p in ap)]
    if not nk:
        return 1.0
    vecs = []
    for p in ap:
        v = np.array([float(p[k]) for k in nk])
        n = np.linalg.norm(v)
        vecs.append(v / n if n > 0 else v)
    sims = [float(np.dot(vecs[i], vecs[i+1]))
            for i in range(len(vecs) - 1)]
    return float(np.mean(sims))


# ═══════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ═══════════════════════════════════════════════════════════════

class RollingWFO:
    """
    Rolling WFO with dual test (bot auto-reoptimization simulation).

    TEST-1: Select best trial (full pipeline evaluation)
    TEST-2: Verify winner on unseen data (simulates live)
    """

    def __init__(self, opt_days=90, test1_days=60, test2_days=30,
                 step_days=30, method='bayesian',
                 objective='monthly_robust', n_trials=150,
                 n_jobs=-1, top_n=10, mc_sims=500,
                 target_preset='conservative', no_intrabar=False,
                 param_subset=None, exclude_params=None,
                 min_trades_per_week=1.5,
                 min_validate_layers=0,
                 min_win_rate=0.0,
                 max_retries=0,
                 retry_trials_delta=100,
                 verbose=True):
        self.opt_days = opt_days
        self.test1_days = test1_days
        self.test2_days = test2_days
        self.step_days = step_days
        self.method = method
        self.objective = objective
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.top_n = top_n
        self.mc_sims = mc_sims
        self.target_preset = target_preset
        self.no_intrabar = no_intrabar
        self.param_subset = param_subset
        self.exclude_params = exclude_params
        self.min_trades_per_week = min_trades_per_week
        self.min_validate_layers = min_validate_layers
        self.min_win_rate = min_win_rate
        self.max_retries = max_retries
        self.retry_trials_delta = retry_trials_delta
        self.verbose = verbose

    def run(self, strategy_name, symbol, timeframe,
            start_date, end_date,
            capital=1000.0, leverage=3.0) -> WFOResult:

        t0 = time.time()
        winfo = generate_wfo_windows(
            start_date, end_date,
            self.opt_days, self.test1_days, self.test2_days,
            self.step_days)

        if not winfo:
            raise ValueError(
                f"No windows. {start_date}→{end_date}, need ≥"
                f"{self.opt_days + self.test1_days + self.test2_days} days.")

        if self.verbose:
            self._banner(strategy_name, symbol, timeframe, capital,
                         leverage, start_date, end_date, len(winfo))

        wresults = []
        for i, wi in enumerate(winfo):
            if self.verbose:
                print(f"\n  ▶▶▶ Window {i+1}/{len(winfo)}")
            wr = _run_single_window(
                wi, strategy_name, symbol, timeframe,
                capital, leverage, self.method, self.objective,
                self.n_trials, self.n_jobs, self.top_n,
                self.mc_sims, self.target_preset,
                self.no_intrabar, self.param_subset,
                self.exclude_params, self.min_trades_per_week,
                self.min_validate_layers, self.min_win_rate,
                self.max_retries, self.retry_trials_delta,
                self.verbose)
            wresults.append(wr)

        agg = _aggregate(wresults, capital)

        r = WFOResult(
            windows=wresults,
            opt_days=self.opt_days, test1_days=self.test1_days,
            test2_days=self.test2_days, step_days=self.step_days,
            start_date=start_date, end_date=end_date,
            strategy=strategy_name, symbol=symbol, timeframe=timeframe,
            method=self.method, objective=self.objective,
            n_trials=self.n_trials, top_n=self.top_n,
            avg_return_per_window=agg.get('avg_return_per_window', 0),
            median_return_per_window=agg.get('median_return_per_window', 0),
            std_return_per_window=agg.get('std_return_per_window', 0),
            window_sharpe=agg.get('window_sharpe', 0),
            window_sortino=agg.get('window_sortino', 0),
            compound_return=agg.get('compound_return', 0),
            annual_return=agg.get('annual_return', 0),
            total_test_days=agg.get('total_test_days', 0),
            total_years=agg.get('total_years', 0),
            t2_max_drawdown=agg.get('t2_max_drawdown', 0),
            t2_calmar=agg.get('calmar', 0),
            t2_win_rate=agg.get('t2_win_rate', 0),
            t2_profit_factor=agg.get('t2_profit_factor', 0),
            t2_n_trades=agg.get('t2_n_trades', 0),
            t1_avg_sharpe=agg.get('t1_avg_sharpe', 0),
            t1_avg_return=agg.get('t1_avg_return', 0),
            avg_degradation_sharpe=agg.get('avg_deg_sharpe', 0),
            avg_degradation_return=agg.get('avg_deg_return', 0),
            n_profitable_windows_t2=agg.get('n_profitable_t2', 0),
            n_total_windows=agg.get('n_total', 0),
            pct_profitable_t2=agg.get('pct_profitable_t2', 0),
            n_validated_windows=agg.get('n_validated', 0),
            param_stability=agg.get('param_stability', 0),
            t2_equity_curve=agg.get('equity'),
            total_elapsed=time.time() - t0,
        )

        if self.verbose:
            self._summary(r)

        return r

    def _banner(self, strat, sym, tf, cap, lev, s, e, nw):
        print("\n╔" + "═" * 68 + "╗")
        print("║  ROLLING WFO — Dual Test (Bot Simulation)                       ║")
        print("╠" + "═" * 68 + "╣")
        print(f"║  Strategy:  {strat}")
        print(f"║  Symbol:    {sym} | {tf}")
        print(f"║  Capital:   ${cap:,.0f} | Leverage: {lev}x")
        print(f"║  Period:    {s} → {e}")
        print(f"║  Windows:   {nw} (opt={self.opt_days}d, "
              f"T1={self.test1_days}d, T2={self.test2_days}d, "
              f"step={self.step_days}d)")
        print(f"║  Method:    {self.method} | Obj: {self.objective}")
        print(f"║  Trials:    {self.n_trials} | Top-N: {self.top_n} "
              f"| Workers: {self.n_jobs}")
        print(f"║  Validate:  MC={self.mc_sims} | Targets: {self.target_preset}")
        print(f"║  Min trades: {self.min_trades_per_week}/week (dynamic per window)")
        if self.min_validate_layers > 0 or self.min_win_rate > 0:
            extra = abs(self.retry_trials_delta)
            gates = []
            if self.min_validate_layers > 0:
                gates.append(f"val≥{self.min_validate_layers}/4")
            if self.min_win_rate > 0:
                gates.append(f"WR≥{self.min_win_rate:.0f}%")
            print(f"║  Quality:   {' + '.join(gates)} | "
                  f"retries: {self.max_retries} "
                  f"(+{extra} trials/retry, same study)")
        print("╚" + "═" * 68 + "╝")

    def _summary(self, r):
        print("\n" + "═" * 90)
        print("  ROLLING WFO — FINAL SUMMARY (dual test)")
        print("═" * 90)

        print(f"\n  {'W':>2} {'OPT':^14} {'T1 (select)':^14} "
              f"{'T2 (verify)':^14} {'Best':>4} "
              f"{'T1 SR':>6} {'T2 SR':>6} {'T2 Ret':>7} "
              f"{'T2 DD':>6} {'Deg%':>6} {'Val':>4}")
        print(f"  {'─' * 100}")

        for w in r.windows:
            if w.error:
                print(f"  {w.window_id+1:>2} "
                      f"{w.opt_start}→{w.opt_end[:5]}  ... "
                      f"❌ {w.error[:30]}")
                continue
            brank = (w.best_trial.get('trial_rank', '?')
                     if w.best_trial else '?')
            deg = w.degradation_sharpe
            di = "✅" if abs(deg) < 30 else "⚠️" if abs(deg) < 60 else "❌"
            ri = "🟢" if w.t2_return > 0 else "🔴"
            vi = "✓" if w.validated else "—"

            print(f"  {w.window_id+1:>2} "
                  f"{w.opt_start[:5]}→{w.opt_end[:5]}  "
                  f"{w.test1_start[:5]}→{w.test1_end[:5]}  "
                  f"{w.test2_start[:5]}→{w.test2_end[:5]}  "
                  f"  #{brank:>2} "
                  f"{w.t1_sharpe:>5.2f} "
                  f"{w.t2_sharpe:>5.2f} "
                  f"{w.t2_return:>+6.1f}% "
                  f"{w.t2_max_drawdown:>5.1f}% "
                  f"{deg:>+5.0f}% "
                  f"  {vi} {ri}")

        # Aggregated
        print(f"\n  {'─' * 70}")
        print(f"  PER-WINDOW METRICS (primary — honest)")
        print(f"  {'─' * 70}")
        print(f"    Avg Return/Window:   {r.avg_return_per_window:+.2f}%")
        print(f"    Median Return/Win:   {r.median_return_per_window:+.2f}%")
        print(f"    Std Return/Window:   {r.std_return_per_window:.2f}%")
        print(f"    Window Sharpe:       {r.window_sharpe:.2f}  "
              f"(avg_ret / std_ret per window)")
        print(f"    Window Sortino:      {r.window_sortino:.2f}")
        print(f"  {'─' * 70}")
        print(f"  COMPOUND METRICS (full reinvestment — reference)")
        print(f"  {'─' * 70}")
        print(f"    Compound Return:     {r.compound_return:+,.1f}%  "
              f"({r.total_test_days}d = {r.total_years:.1f}y)")
        print(f"    Annualized Return:   {r.annual_return:+,.1f}%")
        print(f"    Max Drawdown:        {r.t2_max_drawdown:.2f}%")
        print(f"    Calmar (ann/DD):     {r.t2_calmar:.2f}")
        print(f"  {'─' * 70}")
        print(f"  TRADES & CONSISTENCY")
        print(f"  {'─' * 70}")
        print(f"    Win Rate:            {r.t2_win_rate:.1f}%")
        print(f"    Profit Factor:       {r.t2_profit_factor:.2f}")
        print(f"    Total Trades:        {r.t2_n_trades}")
        print(f"    Profitable Windows:  {r.n_profitable_windows_t2}/"
              f"{r.n_total_windows} "
              f"({r.pct_profitable_t2:.0f}%)")
        print(f"    Validated Windows:   {r.n_validated_windows}/"
              f"{r.n_total_windows}")
        print(f"  {'─' * 70}")
        print(f"  COMPARISON T1 → T2")
        print(f"  {'─' * 70}")
        print(f"    T1 Avg Sharpe:       {r.t1_avg_sharpe:.2f}")
        print(f"    T1 Avg Return:       {r.t1_avg_return:+.2f}%")
        print(f"    Avg Degradation SR:  {r.avg_degradation_sharpe:+.1f}%")
        print(f"    Avg Degradation Ret: {r.avg_degradation_return:+.1f}%")
        print(f"    Param Stability:     {r.param_stability:.3f}")
        print(f"    Total Time:          {r.total_elapsed:.1f}s")
        print("═" * 90)


# ═══════════════════════════════════════════════════════════════
#  JSON EXPORT
# ═══════════════════════════════════════════════════════════════

def _jd(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def export_wfo_results(result: WFOResult, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = f"wfo_{result.strategy}_{result.symbol}_{result.timeframe}_{ts}"

    report = {
        'config': {
            'strategy': result.strategy, 'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start': result.start_date, 'end': result.end_date,
            'opt_days': result.opt_days,
            'test1_days': result.test1_days,
            'test2_days': result.test2_days,
            'step_days': result.step_days,
            'method': result.method, 'objective': result.objective,
            'n_trials': result.n_trials, 'top_n': result.top_n,
        },
        'per_window_metrics': {
            'avg_return': result.avg_return_per_window,
            'median_return': result.median_return_per_window,
            'std_return': result.std_return_per_window,
            'window_sharpe': result.window_sharpe,
            'window_sortino': result.window_sortino,
        },
        'compound_metrics': {
            'compound_return': result.compound_return,
            'annual_return': result.annual_return,
            'total_test_days': result.total_test_days,
            'total_years': result.total_years,
            'max_drawdown': result.t2_max_drawdown,
            'calmar': result.t2_calmar,
        },
        'trades': {
            'win_rate': result.t2_win_rate,
            'profit_factor': result.t2_profit_factor,
            'n_trades': result.t2_n_trades,
        },
        'comparison': {
            't1_avg_sharpe': result.t1_avg_sharpe,
            't1_avg_return': result.t1_avg_return,
            'avg_degradation_sharpe_pct': result.avg_degradation_sharpe,
            'avg_degradation_return_pct': result.avg_degradation_return,
            'n_profitable_t2': result.n_profitable_windows_t2,
            'n_validated': result.n_validated_windows,
            'n_total': result.n_total_windows,
            'pct_profitable_t2': result.pct_profitable_t2,
            'param_stability': result.param_stability,
            'total_elapsed_s': round(result.total_elapsed, 1),
        },
        'windows': [],
    }

    for w in result.windows:
        wd = {
            'window_id': w.window_id,
            'opt': f"{w.opt_start}→{w.opt_end}",
            'test1': f"{w.test1_start}→{w.test1_end}",
            'test2': f"{w.test2_start}→{w.test2_end}",
            'best_trial_rank': (w.best_trial.get('trial_rank')
                                if w.best_trial else None),
            'best_params': w.best_params,
            'test1_metrics': {
                'sharpe': w.t1_sharpe, 'return': w.t1_return,
                'max_dd': w.t1_max_drawdown, 'win_rate': w.t1_win_rate,
                'pf': w.t1_profit_factor, 'trades': w.t1_n_trades,
            },
            'test2_metrics': {
                'sharpe': w.t2_sharpe, 'return': w.t2_return,
                'max_dd': w.t2_max_drawdown, 'win_rate': w.t2_win_rate,
                'pf': w.t2_profit_factor, 'trades': w.t2_n_trades,
                'calmar': w.t2_calmar, 'sortino': w.t2_sortino,
            },
            'degradation': {
                'sharpe_pct': w.degradation_sharpe,
                'return_pct': w.degradation_return,
            },
            'validation': {
                'validated': w.validated,
                'layers': w.validate_layers,
                'targets_passed': w.targets_passed,
                'targets_total': w.targets_total,
            },
            'all_trials': w.trials,
            'elapsed_s': round(w.elapsed_seconds, 1),
            'error': w.error,
        }
        report['windows'].append(wd)

    jp = out / f"{fn}.json"
    with open(jp, 'w') as f:
        json.dump(report, f, indent=2, default=_jd)

    if result.t2_equity_curve is not None and len(result.t2_equity_curve) > 0:
        import csv
        cp = out / f"{fn}_equity_t2.csv"
        with open(cp, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['bar', 'equity'])
            for i, eq in enumerate(result.t2_equity_curve):
                wr.writerow([i, round(float(eq), 4)])

    return jp