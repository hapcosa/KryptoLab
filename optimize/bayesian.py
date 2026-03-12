"""
CryptoLab — Bayesian Optimizer (Optuna TPE)
Intelligent parameter optimization using Tree-structured Parzen Estimators.

Features:
- Automatic search space from strategy ParamDef
- TPE sampler (better than random for structured spaces)
- Median pruning (early stop losing trials)
- Multi-objective support (Sharpe + Drawdown)
- Conditional spaces: alpha_method + sltp_type (v7.1)
- Integration with anti-overfit pipeline (returns n_trials for DSR)
- Warm start from grid search results

Reference: Bergstra et al. "Algorithms for Hyper-Parameter Optimization" (2011)
"""
import numpy as np
import copy
import time
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ── Single source of truth for parameter dependencies ──
from optimize.param_dependencies import (
    ALPHA_METHOD_PARAMS, ALL_METHOD_PARAMS,
    SLTP_MODE_PARAMS, ALL_SLTP_PARAMS,
    count_effective_params,
)


@dataclass
class BayesianTrial:
    """Result of a single Optuna trial."""
    trial_id: int
    params: Dict[str, Any]
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    objective_value: float = 0.0
    state: str = "COMPLETE"


@dataclass
class BayesianResult:
    """Complete Bayesian optimization results."""
    trials: List[BayesianTrial]
    best_trial: Optional[BayesianTrial] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    n_trials_total: int = 0
    n_trials_complete: int = 0
    n_trials_pruned: int = 0
    elapsed_seconds: float = 0.0
    objective_name: str = ""
    param_importances: Dict[str, float] = field(default_factory=dict)


class BayesianOptimizer:
    """
    Bayesian optimization via Optuna TPE.

    More efficient than grid search for high-dimensional parameter spaces.
    TPE models P(params|good) and P(params|bad) separately, then samples
    from the ratio — focusing search on promising regions.

    Conditional spaces (v7.1):
      - alpha_method → only relevant alpha sub-params explored
      - sltp_type    → only relevant SL/TP params explored

    Typical usage: 100-500 trials for 10-30 parameter strategies.
    """

    def __init__(self,
                 n_trials: int = 200,
                 objective: str = 'sharpe',
                 min_trades: int = 10,
                 timeout_seconds: Optional[int] = None,
                 pruning: bool = True,
                 seed: int = 42,
                 n_jobs: int = 1,
                 verbose: bool = True):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required: pip install optuna")

        self.n_trials = n_trials
        self.objective_name = objective
        self.min_trades = min_trades
        self.timeout = timeout_seconds
        self.pruning = pruning
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.last_result = None

    # ── Conditional parameter groups (imported from param_dependencies) ──
    ALPHA_METHOD_PARAMS = ALPHA_METHOD_PARAMS
    ALL_METHOD_PARAMS = ALL_METHOD_PARAMS
    SLTP_MODE_PARAMS = SLTP_MODE_PARAMS
    ALL_SLTP_PARAMS = ALL_SLTP_PARAMS

    def _build_search_space(self, trial: 'optuna.Trial',
                            param_defs: list,
                            param_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Construct Optuna search space from ParamDef list.

        Uses TWO conditional parameter spaces:
          1. alpha_method → only relevant alpha sub-params suggested
          2. sltp_type    → only relevant SL/TP params suggested

        This reduces effective dimensionality significantly,
        making TPE much more efficient.
        """
        params = {}

        # ═══════════════════════════════════════════════════════════
        #  Phase 1: Determine alpha_method (suggest or use default)
        # ═══════════════════════════════════════════════════════════
        alpha_method = None
        alpha_pdef = None
        for pdef in param_defs:
            if pdef.name == 'alpha_method':
                alpha_pdef = pdef
                break

        if alpha_pdef:
            if param_subset is None or 'alpha_method' in param_subset:
                alpha_method = trial.suggest_categorical(
                    'alpha_method', alpha_pdef.options)
            else:
                alpha_method = alpha_pdef.default
            params['alpha_method'] = alpha_method

        # Determine which alpha params to skip
        active_alpha = self.ALPHA_METHOD_PARAMS.get(alpha_method, set()) if alpha_method else set()
        skip_alpha = self.ALL_METHOD_PARAMS - active_alpha

        # ═══════════════════════════════════════════════════════════
        #  Phase 2: Determine sltp_type (suggest or use default)
        # ═══════════════════════════════════════════════════════════
        sltp_type = None
        sltp_pdef = None
        for pdef in param_defs:
            if pdef.name == 'sltp_type':
                sltp_pdef = pdef
                break

        if sltp_pdef:
            if param_subset is None or 'sltp_type' in param_subset:
                sltp_type = trial.suggest_categorical(
                    'sltp_type', sltp_pdef.options)
            else:
                sltp_type = sltp_pdef.default
            params['sltp_type'] = sltp_type

        # Determine which sltp params to skip
        active_sltp = self.SLTP_MODE_PARAMS.get(sltp_type, set()) if sltp_type else set()
        skip_sltp = self.ALL_SLTP_PARAMS - active_sltp

        # Combined skip set
        skip_params = skip_alpha | skip_sltp

        # ═══════════════════════════════════════════════════════════
        #  Phase 3: Suggest remaining params
        # ═══════════════════════════════════════════════════════════
        for pdef in param_defs:
            if pdef.name in ('alpha_method', 'sltp_type'):
                continue  # Already handled in Phase 1/2
            if param_subset and pdef.name not in param_subset:
                continue  # User excluded this param
            if pdef.name in skip_params:
                continue  # Irrelevant for current mode

            if pdef.ptype == 'float':
                val = trial.suggest_float(
                    pdef.name, pdef.min_val, pdef.max_val)
                if pdef.step and pdef.step > 0:
                    val = round(val / pdef.step) * pdef.step
                    val = max(pdef.min_val, min(pdef.max_val, val))
                params[pdef.name] = val
            elif pdef.ptype == 'int':
                step = pdef.step if pdef.step else 1
                params[pdef.name] = trial.suggest_int(
                    pdef.name, pdef.min_val, pdef.max_val, step=step)
            elif pdef.ptype == 'bool':
                params[pdef.name] = trial.suggest_categorical(
                    pdef.name, [True, False])
            elif pdef.ptype == 'categorical':
                params[pdef.name] = trial.suggest_categorical(
                    pdef.name, pdef.options)

        return params

    def _compute_objective(self, result) -> float:
        """Compute objective value from BacktestResult using shared objective functions."""
        from optimize.grid_search import (
            objective_sharpe, objective_return, objective_composite,
            objective_calmar, objective_monthly, objective_monthly_robust,
            objective_weekly, objective_weekly_robust,
        )
        obj_map = {
            'sharpe': objective_sharpe,
            'return': objective_return,
            'composite': objective_composite,
            'calmar': objective_calmar,
            'monthly': objective_monthly,
            'monthly_robust': objective_monthly_robust,
            'weekly': objective_weekly,
            'weekly_robust': objective_weekly_robust,
        }
        fn = obj_map.get(self.objective_name, objective_sharpe)
        return fn(result)

    def _print_header(self, param_defs, param_subset):
        """Print optimization header with effective param count."""
        n_params = len(param_subset) if param_subset else len(param_defs)
        # Worst case effective count (largest alpha + largest sltp)
        max_alpha = max(len(v) for v in self.ALPHA_METHOD_PARAMS.values())
        max_sltp = max(len(v) for v in self.SLTP_MODE_PARAMS.values())
        n_conditional = len(self.ALL_METHOD_PARAMS) + len(self.ALL_SLTP_PARAMS)
        n_effective = n_params - n_conditional + max_alpha + max_sltp

        print(f"\n  Bayesian Optimizer (TPE): {self.n_trials} trials, "
              f"{n_params} params ({n_effective} effective), objective={self.objective_name}")
        if param_subset:
            print(f"  Optimizing: {', '.join(param_subset)}")
        print(f"  Conditional spaces: alpha_method + sltp_type → only relevant sub-params explored")
        print(f"  {'─' * 80}")

    # ─── Sequential optimization ─────────────────────────────────

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            symbol: str = "",
            timeframe: str = "",
            param_subset: Optional[List[str]] = None,
            warm_start: Optional[List[Dict[str, Any]]] = None,
            engine_config: dict = None
            ) -> 'BayesianResult':
        """
        Execute Bayesian optimization.

        Args:
            strategy: IStrategy instance
            data: Full OHLCV data dict
            engine_factory: Callable returning BacktestEngine (used if n_jobs=1)
            symbol: For reporting
            timeframe: For reporting
            param_subset: Only optimize these params (None = all)
            warm_start: List of param dicts to seed the optimizer
            engine_config: For parallel execution (n_jobs > 1)

        Returns:
            BayesianResult with ranked trials and importances
        """
        if self.n_jobs > 1 and engine_config is not None:
            return self._run_parallel(
                strategy, data, engine_factory,
                symbol, timeframe,
                param_subset, warm_start, engine_config)

        return self._run_sequential(
            strategy, data, engine_factory,
            symbol, timeframe,
            param_subset, warm_start)

    def _run_sequential(self, strategy, data, engine_factory,
                         symbol="", timeframe="",
                         param_subset=None,
                         warm_start=None) -> 'BayesianResult':
        """Sequential Bayesian optimization."""
        t0 = time.time()
        param_defs = strategy.parameter_defs()
        all_trials = []

        if self.verbose:
            self._print_header(param_defs, param_subset)

        # Configure Optuna
        sampler = TPESampler(seed=self.seed, n_startup_trials=20)
        pruner = MedianPruner(n_startup_trials=10) if self.pruning else optuna.pruners.NopPruner()

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Warm start
        if warm_start:
            for ws_params in warm_start[:10]:
                study.enqueue_trial(ws_params)

        def _objective(trial):
            optimized_params = self._build_search_space(trial, param_defs, param_subset)

            strat = copy.deepcopy(strategy)
            strat.set_params(optimized_params)

            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)

            if result.n_trades < self.min_trades:
                return -999.0

            obj_val = self._compute_objective(result)

            # Debug: show rejection reason for -999 trials (only when verbose)
            if obj_val <= -999.0 and self.verbose:
                try:
                    from optimize.grid_search import diagnose_rejection
                    reason = diagnose_rejection(result)
                    if not reason.startswith('PASS'):
                        print(f"         ⚠️  Rejected: {reason}")
                except Exception:
                    pass

            # Store trial data
            bt = BayesianTrial(
                trial_id=trial.number,
                params=optimized_params,
                sharpe_ratio=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                n_trades=result.n_trades,
                objective_value=obj_val,
            )
            all_trials.append(bt)

            if self.verbose:
                best_so_far = max((t.objective_value for t in all_trials), default=-999)
                marker = '★' if obj_val >= best_so_far else ' '

                if param_subset:
                    p_disp = {k: (f'{v:.2f}' if isinstance(v, float) else str(v))
                              for k, v in optimized_params.items() if k in param_subset}
                else:
                    # Show mode selectors + top varying params
                    p_disp = {}
                    for k in ('alpha_method', 'sltp_type'):
                        if k in optimized_params:
                            p_disp[k] = str(optimized_params[k])
                    remaining = {k: v for k, v in optimized_params.items()
                                 if k not in ('alpha_method', 'sltp_type')}
                    for k, v in list(remaining.items())[:3]:
                        p_disp[k] = f'{v:.2f}' if isinstance(v, float) else str(v)

                params_str = str(p_disp).replace("'", "")
                print(f"  {marker}{trial.number+1:>4}/{self.n_trials} "
                      f"obj={obj_val:>7.3f} "
                      f"SR={result.sharpe_ratio:>5.2f} "
                      f"Ret={result.total_return:>+6.1f}% "
                      f"WR={result.win_rate:>4.1f}% "
                      f"DD={result.max_drawdown:>5.1f}% "
                      f"PF={result.profit_factor:>4.2f} "
                      f"T={result.n_trades:>3} "
                      f"{params_str}")

            return obj_val

        # Run optimization
        interrupted = False
        try:
            study.optimize(
                _objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            interrupted = True
            if self.verbose:
                print(f"\n  ⚠️  Interrupted after {len(all_trials)} trials — compiling partial results...")

        # Compile results
        all_trials.sort(key=lambda t: t.objective_value, reverse=True)

        best = all_trials[0] if all_trials else None
        best_params = best.params if best else {}

        importances = {}
        try:
            imp = optuna.importance.get_param_importances(study)
            importances = dict(imp)
        except Exception:
            pass

        try:
            n_complete = sum(1 for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE)
            n_pruned = sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED)
        except Exception:
            n_complete = len(all_trials)
            n_pruned = 0

        elapsed = time.time() - t0

        result = BayesianResult(
            trials=all_trials,
            best_trial=best,
            best_params=best_params,
            n_trials_total=len(all_trials),
            n_trials_complete=n_complete,
            n_trials_pruned=n_pruned,
            elapsed_seconds=elapsed,
            objective_name=self.objective_name,
            param_importances=importances,
        )

        self.last_result = result

        if self.verbose:
            self._print_summary(result, interrupted)

        return result

    def _run_parallel(self, strategy, data, engine_factory,
                       symbol="", timeframe="",
                       param_subset=None,
                       warm_start=None, engine_config=None) -> 'BayesianResult':
        """
        Parallel Bayesian optimization using ask-and-tell API.

        Flow per batch:
          1. Main process: study.ask() × batch_size → trials
          2. Main process: build_search_space() for each → param dicts (fast)
          3. Worker pool: evaluate_trial() in parallel (slow - this is the win)
          4. Main process: study.tell() with results
        """
        from optimize.parallel import setup_workers, evaluate_trial, cleanup_workers, create_pool

        t0 = time.time()
        param_defs = strategy.parameter_defs()
        all_trials = []
        batch_size = self.n_jobs

        if self.verbose:
            self._print_header(param_defs, param_subset)
            print(f"  ⚡ Parallel: {self.n_jobs} workers, batch_size={batch_size}")

        # Configure Optuna
        sampler = TPESampler(seed=self.seed, n_startup_trials=20)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        if warm_start:
            for ws_params in warm_start[:10]:
                study.enqueue_trial(ws_params)

        # Setup shared state for workers
        setup_workers(
            strategy=strategy,
            data=data,
            engine_config=engine_config,
            objective_name=self.objective_name,
            min_trades=self.min_trades,
            symbol=symbol,
            timeframe=timeframe,
        )

        interrupted = False
        completed = 0
        best_obj = -999.0

        try:
            with create_pool(self.n_jobs) as pool:
                while completed < self.n_trials:
                    if self.timeout and (time.time() - t0) > self.timeout:
                        break

                    remaining = self.n_trials - completed
                    current_batch = min(batch_size, remaining)

                    # Phase 1: Ask + suggest params (main process, fast)
                    batch_items = []
                    for _ in range(current_batch):
                        trial = study.ask()
                        params = self._build_search_space(trial, param_defs, param_subset)
                        batch_items.append((trial, params))

                    # Phase 2: Evaluate in parallel (workers, slow)
                    work_items = [
                        (completed + i, params)
                        for i, (_, params) in enumerate(batch_items)
                    ]
                    results = pool.map(evaluate_trial, work_items)

                    # Phase 3: Tell study + collect results (main process, fast)
                    for (optuna_trial, optimized_params), res in zip(batch_items, results):
                        obj_val = res['objective_value']

                        study.tell(optuna_trial, obj_val)

                        bt = BayesianTrial(
                            trial_id=res['trial_id'],
                            params=res['params'],
                            sharpe_ratio=res['sharpe_ratio'],
                            total_return=res['total_return'],
                            max_drawdown=res['max_drawdown'],
                            win_rate=res['win_rate'],
                            profit_factor=res['profit_factor'],
                            n_trades=res['n_trades'],
                            objective_value=obj_val,
                        )
                        all_trials.append(bt)
                        completed += 1

                        is_best = obj_val > best_obj
                        if is_best:
                            best_obj = obj_val

                        if self.verbose:
                            marker = '★' if is_best else ' '

                            if param_subset:
                                p_disp = {k: (f'{v:.2f}' if isinstance(v, float) else str(v))
                                          for k, v in optimized_params.items() if k in param_subset}
                            else:
                                p_disp = {}
                                for k in ('alpha_method', 'sltp_type'):
                                    if k in optimized_params:
                                        p_disp[k] = str(optimized_params[k])
                                remaining_p = {k: v for k, v in optimized_params.items()
                                               if k not in ('alpha_method', 'sltp_type')}
                                for k, v in list(remaining_p.items())[:3]:
                                    p_disp[k] = f'{v:.2f}' if isinstance(v, float) else str(v)

                            params_str = str(p_disp).replace("'", "")
                            print(f"  {marker}{completed:>4}/{self.n_trials} "
                                  f"obj={obj_val:>7.3f} "
                                  f"SR={res['sharpe_ratio']:>5.2f} "
                                  f"Ret={res['total_return']:>+6.1f}% "
                                  f"WR={res['win_rate']:>4.1f}% "
                                  f"DD={res['max_drawdown']:>5.1f}% "
                                  f"PF={res['profit_factor']:>4.2f} "
                                  f"T={res['n_trades']:>3} "
                                  f"[{res['elapsed']:.0f}s] "
                                  f"{params_str}")

        except KeyboardInterrupt:
            interrupted = True
            if self.verbose:
                print(f"\n  ⚠️  Interrupted after {len(all_trials)} trials — compiling partial results...")
        finally:
            cleanup_workers()

        # Compile results
        all_trials.sort(key=lambda t: t.objective_value, reverse=True)

        best = all_trials[0] if all_trials else None
        best_params = best.params if best else {}

        importances = {}
        try:
            imp = optuna.importance.get_param_importances(study)
            importances = dict(imp)
        except Exception:
            pass

        try:
            n_complete = sum(1 for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE)
            n_pruned = sum(1 for t in study.trials
                           if t.state == optuna.trial.TrialState.PRUNED)
        except Exception:
            n_complete = len(all_trials)
            n_pruned = 0

        elapsed = time.time() - t0

        result = BayesianResult(
            trials=all_trials,
            best_trial=best,
            best_params=best_params,
            n_trials_total=len(all_trials),
            n_trials_complete=n_complete,
            n_trials_pruned=n_pruned,
            elapsed_seconds=elapsed,
            objective_name=self.objective_name,
            param_importances=importances,
        )

        self.last_result = result

        if self.verbose:
            self._print_summary(result, interrupted)

        return result

    def _print_summary(self, result: 'BayesianResult', interrupted: bool = False):
        """Print optimization summary."""
        print(f"\n  {'─' * 80}")
        if interrupted:
            print(f"  ⚠️  Partial results ({result.n_trials_total} trials)")
        else:
            print(f"  ✅ Complete ({result.n_trials_total} trials, "
                  f"{result.n_trials_pruned} pruned)")

        if result.best_trial:
            bt = result.best_trial
            print(f"\n  Best trial #{bt.trial_id}:")
            print(f"    Objective:  {bt.objective_value:.4f}")
            print(f"    Sharpe:     {bt.sharpe_ratio:.2f}")
            print(f"    Return:     {bt.total_return:+.1f}%")
            print(f"    Win Rate:   {bt.win_rate:.1f}%")
            print(f"    Max DD:     {bt.max_drawdown:.1f}%")
            print(f"    PF:         {bt.profit_factor:.2f}")
            print(f"    Trades:     {bt.n_trades}")

            # Show effective param count for best trial
            try:
                from strategies.base import ParamDef
                # We can't easily get param_defs here, so show modes
                am = bt.params.get('alpha_method', '?')
                st = bt.params.get('sltp_type', '?')
                print(f"    Modes:      alpha={am}, sltp={st}")
            except Exception:
                pass

        if result.param_importances:
            print(f"\n  Parameter importances:")
            for name, imp in sorted(result.param_importances.items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
                bar = '█' * int(imp * 40)
                print(f"    {name:>25s}: {imp:.3f} {bar}")

        print(f"\n  Elapsed: {result.elapsed_seconds:.1f}s")
        print(f"  {'─' * 80}\n")