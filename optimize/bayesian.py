"""
CryptoLab — Bayesian Optimizer (Optuna TPE)
Intelligent parameter optimization using Tree-structured Parzen Estimators.

Features:
- Automatic search space from strategy ParamDef
- TPE sampler (better than random for structured spaces)
- Median pruning (early stop losing trials)
- Multi-objective support (Sharpe + Drawdown)
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

    Typical usage: 100-500 trials for 10-30 parameter strategies.
    """

    def __init__(self,
                 n_trials: int = 200,
                 objective: str = 'sharpe',
                 min_trades: int = 10,
                 timeout_seconds: Optional[int] = None,
                 pruning: bool = True,
                 seed: int = 42,
                 verbose: bool = True):
        """
        Args:
            n_trials: Maximum number of trials
            objective: 'sharpe', 'return', 'composite'
            min_trades: Minimum trades for valid trial
            timeout_seconds: Max wall time (None = unlimited)
            pruning: Enable median pruner
            seed: Random seed for reproducibility
            verbose: Print progress
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna required: pip install optuna")

        self.n_trials = n_trials
        self.objective_name = objective
        self.min_trades = min_trades
        self.timeout = timeout_seconds
        self.pruning = pruning
        self.seed = seed
        self.verbose = verbose
        self.last_result = None  # Stores partial result for Ctrl+C recovery

    # ── Conditional parameter groups ──
    # Params that only matter for specific alpha_method values.
    # When alpha_method='mama', kalman/homodyne/autocorrelation params are ignored
    # by the strategy, so we skip them in the search space to avoid wasting trials.
    ALPHA_METHOD_PARAMS = {
        'manual':          {'manual_alpha'},
        'homodyne':        {'hd_min_period', 'hd_max_period', 'alpha_floor'},
        'mama':            {'mama_fast', 'mama_slow', 'alpha_floor'},
        'autocorrelation': {'ac_min_period', 'ac_max_period', 'ac_avg_length', 'alpha_floor'},
        'kalman':          {'kal_process_noise', 'kal_meas_noise',
                            'kal_alpha_fast', 'kal_alpha_slow', 'kal_sensitivity', 'alpha_floor'},
    }

    # All method-specific params (union of all groups)
    ALL_METHOD_PARAMS = set()
    for v in ALPHA_METHOD_PARAMS.values():
        ALL_METHOD_PARAMS |= v

    def _build_search_space(self, trial: 'optuna.Trial',
                            param_defs: list,
                            param_subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Construct Optuna search space from ParamDef list.

        Uses conditional parameter spaces: alpha_method is suggested first,
        then only the relevant method-specific params are suggested.
        Other method params use strategy defaults (not optimized).

        This reduces effective dimensionality from ~35 to ~22 params,
        making TPE much more efficient.
        """
        params = {}

        # Phase 1: Determine alpha_method (suggest or use default)
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

        # Phase 2: Determine which method-specific params to include
        active_params = self.ALPHA_METHOD_PARAMS.get(alpha_method, set()) if alpha_method else set()
        skip_params = self.ALL_METHOD_PARAMS - active_params  # Skip irrelevant method params

        # Phase 3: Suggest remaining params
        for pdef in param_defs:
            if pdef.name == 'alpha_method':
                continue  # Already handled
            if param_subset and pdef.name not in param_subset:
                continue  # User excluded this param
            if pdef.name in skip_params:
                continue  # Irrelevant for current alpha_method

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
        if result.n_trades < self.min_trades:
            return -999.0

        # Use the centralized objective functions from grid_search
        try:
            from optimize.grid_search import OBJECTIVES
            fn = OBJECTIVES.get(self.objective_name)
            if fn:
                return fn(result)
        except ImportError:
            pass

        # Fallback: inline objectives
        if self.objective_name == 'sharpe':
            return result.sharpe_ratio
        elif self.objective_name == 'return':
            return result.total_return
        elif self.objective_name == 'calmar':
            return result.calmar_ratio
        elif self.objective_name == 'composite':
            sr = max(0, result.sharpe_ratio)
            pf = min(5, result.profit_factor) / 5.0 if result.profit_factor > 0 else 0
            cal = min(5, result.calmar_ratio) / 5.0 if result.calmar_ratio > 0 else 0
            wr = result.win_rate / 100.0
            return sr * 0.4 + pf * 0.2 + cal * 0.2 + wr * 0.2
        else:
            return result.sharpe_ratio

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            symbol: str = "",
            timeframe: str = "",
            param_subset: Optional[List[str]] = None,
            warm_start: Optional[List[Dict[str, Any]]] = None
            ) -> BayesianResult:
        """
        Execute Bayesian optimization.

        Args:
            strategy: IStrategy instance
            data: Full OHLCV data dict
            engine_factory: Callable returning BacktestEngine
            symbol: For reporting
            timeframe: For reporting
            param_subset: Only optimize these params (None = all)
            warm_start: List of param dicts to seed the optimizer

        Returns:
            BayesianResult with ranked trials and importances
        """
        t0 = time.time()
        param_defs = strategy.parameter_defs()
        all_trials = []

        if self.verbose:
            n_params = len(param_subset) if param_subset else len(param_defs)
            n_method = len(self.ALL_METHOD_PARAMS)
            n_effective = n_params - n_method + max(len(v) for v in self.ALPHA_METHOD_PARAMS.values())
            print(f"\n  Bayesian Optimizer (TPE): {self.n_trials} trials, "
                  f"{n_params} params ({n_effective} effective), objective={self.objective_name}")
            if param_subset:
                print(f"  Optimizing: {', '.join(param_subset)}")
            print(f"  Conditional spaces: alpha_method → only relevant sub-params explored")
            print(f"  {'─' * 80}")

        # Configure Optuna
        sampler = TPESampler(seed=self.seed, n_startup_trials=20)
        pruner = MedianPruner(n_startup_trials=10) if self.pruning else optuna.pruners.NopPruner()

        verbosity = optuna.logging.WARNING if not self.verbose else optuna.logging.WARNING
        optuna.logging.set_verbosity(verbosity)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # Warm start: enqueue known good configs
        if warm_start:
            for ws_params in warm_start[:10]:
                study.enqueue_trial(ws_params)

        def _objective(trial):
            optimized_params = self._build_search_space(trial, param_defs, param_subset)

            strat = copy.deepcopy(strategy)
            strat.set_params(optimized_params)

            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)

            obj_val = self._compute_objective(result)

            # Full snapshot: defaults + phase1 (from loaded strategy) + phase2 (optimized)
            # This ensures ALL params persist across optimization phases
            full_params = strat.default_params()
            full_params.update(strat.params)

            # Store trial info with FULL params
            bt = BayesianTrial(
                trial_id=trial.number,
                params=full_params,
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
                try:
                    best_val = study.best_value
                except (ValueError, AttributeError):
                    best_val = obj_val
                is_best = obj_val >= best_val
                marker = '★' if is_best else ' '
                n = trial.number + 1

                # Compact param display — only show optimized params
                if param_subset:
                    p_disp = {k: (f'{v:.2f}' if isinstance(v, float) else str(v))
                              for k, v in optimized_params.items() if k in param_subset}
                else:
                    # Show top 4 most changed params
                    p_disp = {k: (f'{v:.2f}' if isinstance(v, float) else str(v))
                              for k, v in list(optimized_params.items())[:4]}

                params_str = str(p_disp).replace("'", "")
                print(f"  {marker}{n:>4}/{self.n_trials} "
                      f"obj={obj_val:>7.3f} "
                      f"SR={result.sharpe_ratio:>5.2f} "
                      f"Ret={result.total_return:>+6.1f}% "
                      f"WR={result.win_rate:>4.1f}% "
                      f"DD={result.max_drawdown:>5.1f}% "
                      f"PF={result.profit_factor:>4.2f} "
                      f"T={result.n_trades:>3} "
                      f"{params_str}")

            return obj_val

        # Run optimization (Ctrl+C → graceful stop, show partial results)
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

        # Extract results (works with partial data too)
        all_trials.sort(key=lambda t: t.objective_value, reverse=True)

        best = all_trials[0] if all_trials else None
        best_params = best.params if best else {}

        # Parameter importances
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

        result = BayesianResult(
            trials=all_trials[:50],  # Keep top 50
            best_trial=best,
            best_params=best_params,
            n_trials_total=n_complete + n_pruned,
            n_trials_complete=n_complete,
            n_trials_pruned=n_pruned,
            elapsed_seconds=time.time() - t0,
            objective_name=self.objective_name,
            param_importances=importances,
        )
        self.last_result = result  # Store for Ctrl+C recovery from CLI

        if self.verbose and best:
            status = "INTERRUPTED" if interrupted else "Complete"
            print(f"\n  {'═' * 60}")
            print(f"  Bayesian Optimization {status} ({result.elapsed_seconds:.1f}s)")
            print(f"  Trials: {n_complete} complete, {n_pruned} pruned")
            print(f"  Best: obj={best.objective_value:.3f} "
                  f"SR={best.sharpe_ratio:.2f} Ret={best.total_return:+.1f}% "
                  f"WR={best.win_rate:.1f}% DD={best.max_drawdown:.1f}%")
            if importances:
                top_imp = sorted(importances.items(),
                                key=lambda x: x[1], reverse=True)[:5]
                print(f"  Importances: {', '.join(f'{k}={v:.2f}' for k,v in top_imp)}")

        return result

    def run_multi_objective(self,
                            strategy,
                            data: dict,
                            engine_factory: Callable,
                            symbol: str = "",
                            timeframe: str = "",
                            param_subset: Optional[List[str]] = None
                            ) -> BayesianResult:
        """
        Multi-objective optimization: maximize Sharpe, minimize MaxDrawdown.
        Returns Pareto-optimal set.
        """
        t0 = time.time()
        param_defs = strategy.parameter_defs()
        all_trials = []

        if self.verbose:
            print(f"\n  Multi-Objective Bayesian: Sharpe↑ + MaxDD↓")

        sampler = TPESampler(seed=self.seed, n_startup_trials=15)

        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=sampler,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def _mo_objective(trial):
            optimized_params = self._build_search_space(trial, param_defs, param_subset)
            strat = copy.deepcopy(strategy)
            strat.set_params(optimized_params)

            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)

            full_params = strat.default_params()
            full_params.update(strat.params)

            bt = BayesianTrial(
                trial_id=trial.number,
                params=full_params,
                sharpe_ratio=result.sharpe_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                n_trades=result.n_trades,
                objective_value=result.sharpe_ratio,
            )
            all_trials.append(bt)

            return result.sharpe_ratio, result.max_drawdown

        study.optimize(_mo_objective, n_trials=self.n_trials,
                       timeout=self.timeout, show_progress_bar=False)

        # Pareto front
        pareto = study.best_trials
        all_trials.sort(key=lambda t: t.objective_value, reverse=True)

        best = all_trials[0] if all_trials else None

        result = BayesianResult(
            trials=all_trials[:50],
            best_trial=best,
            best_params=best.params if best else {},
            n_trials_total=len(study.trials),
            n_trials_complete=len(all_trials),
            elapsed_seconds=time.time() - t0,
            objective_name='multi(sharpe,maxdd)',
        )

        if self.verbose:
            print(f"  Pareto front: {len(pareto)} solutions")
            for i, pt in enumerate(pareto[:5]):
                print(f"    #{i}: SR={pt.values[0]:.2f} MaxDD={pt.values[1]:.1f}%")

        return result