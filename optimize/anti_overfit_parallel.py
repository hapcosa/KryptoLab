"""
CryptoLab — Parallel Anti-Overfitting Pipeline
Extends anti_overfit.py with multiprocessing for WFA, K-Fold, and Monte Carlo.

Drop-in replacement: same API, automatic parallelism when n_jobs > 1.

Usage in cli.py:
    from optimize.anti_overfit_parallel import ParallelAntiOverfitPipeline

    pipeline = ParallelAntiOverfitPipeline(
        n_jobs=n_jobs,           # from --n-jobs flag
        wfa_windows=4,
        mc_simulations=1000,
        fail_fast=False,
        verbose=True,
    )
    result = pipeline.run(strategy, data, engine_factory,
                          param_grid, symbol, timeframe,
                          engine_config=engine_config)
"""
import numpy as np
import copy
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

from optimize.anti_overfit import (
    AntiOverfitPipeline, AntiOverfitResult, _annualization_factor
)
from optimize.walk_forward import WalkForwardResult
from optimize.purged_kfold import PurgedKFoldResult
from optimize.monte_carlo import MonteCarloResult


# ═══════════════════════════════════════════════════════════════
#  WORKER STATE (module-level for fork COW — same pattern as
#  optimize/parallel.py)
# ═══════════════════════════════════════════════════════════════

_VALIDATE_STATE = {}


def _setup_validate_workers(strategy, data, engine_config,
                            symbol="", timeframe=""):
    """Initialize module-level state BEFORE forking the pool."""
    global _VALIDATE_STATE
    _VALIDATE_STATE = {
        'strategy': strategy,
        'data': data,
        'engine_config': engine_config,
        'symbol': symbol,
        'timeframe': timeframe,
    }


def _make_engine_from_config(cfg):
    """Reconstruct a BacktestEngine from serializable config dict."""
    from core.engine import BacktestEngine
    engine = BacktestEngine(
        initial_capital=cfg.get('capital', 10000),
        market_config=cfg.get('market_config'),
    )
    dd = cfg.get('detail_data')
    dtf = cfg.get('detail_tf')
    if dd is not None and dtf is not None:
        engine.set_detail_data(dd, dtf)
    return engine


# ═══════════════════════════════════════════════════════════════
#  TOP-LEVEL WORKER FUNCTIONS (must be pickleable)
# ═══════════════════════════════════════════════════════════════

def _worker_wfa_window(args):
    """
    Evaluate one Walk-Forward window.
    args: (window_id, is_indices, oos_indices)

    Returns dict with IS and OOS backtest metrics.
    """
    window_id, is_idx, oos_idx = args
    state = _VALIDATE_STATE
    strategy = copy.deepcopy(state['strategy'])
    data = state['data']
    cfg = state['engine_config']
    symbol = state['symbol']
    timeframe = state['timeframe']

    def _slice_data(d, indices):
        sliced = {}
        for key, val in d.items():
            if isinstance(val, np.ndarray) and len(val) > 0:
                sliced[key] = val[indices].copy()
            else:
                sliced[key] = val
        return sliced

    try:
        # IS backtest
        is_data = _slice_data(data, is_idx)
        engine_is = _make_engine_from_config(cfg)
        strat_is = copy.deepcopy(strategy)
        res_is = engine_is.run(strat_is, is_data, symbol, timeframe)

        # OOS backtest
        oos_data = _slice_data(data, oos_idx)
        engine_oos = _make_engine_from_config(cfg)
        strat_oos = copy.deepcopy(strategy)
        res_oos = engine_oos.run(strat_oos, oos_data, symbol, timeframe)

        return {
            'window_id': window_id,
            'is_sharpe': res_is.sharpe_ratio,
            'is_return': res_is.total_return,
            'is_trades': res_is.n_trades,
            'oos_sharpe': res_oos.sharpe_ratio,
            'oos_return': res_oos.total_return,
            'oos_trades': res_oos.n_trades,
            'error': None,
        }
    except Exception as e:
        return {
            'window_id': window_id,
            'is_sharpe': 0, 'is_return': 0, 'is_trades': 0,
            'oos_sharpe': 0, 'oos_return': 0, 'oos_trades': 0,
            'error': str(e),
        }


def _worker_kfold_path(args):
    """
    Evaluate one Purged K-Fold path.
    args: (fold_id, train_indices, test_indices)
    """
    fold_id, train_idx, test_idx = args
    state = _VALIDATE_STATE
    strategy = copy.deepcopy(state['strategy'])
    data = state['data']
    cfg = state['engine_config']
    symbol = state['symbol']
    timeframe = state['timeframe']

    def _slice_data(d, indices):
        sliced = {}
        for key, val in d.items():
            if isinstance(val, np.ndarray) and len(val) > 0:
                sliced[key] = val[indices].copy()
            else:
                sliced[key] = val
        return sliced

    try:
        # Train backtest
        train_data = _slice_data(data, train_idx)
        engine_train = _make_engine_from_config(cfg)
        strat_train = copy.deepcopy(strategy)
        res_train = engine_train.run(strat_train, train_data, symbol, timeframe)

        # Test backtest
        test_data = _slice_data(data, test_idx)
        engine_test = _make_engine_from_config(cfg)
        strat_test = copy.deepcopy(strategy)
        res_test = engine_test.run(strat_test, test_data, symbol, timeframe)

        return {
            'fold_id': fold_id,
            'train_sharpe': res_train.sharpe_ratio,
            'train_return': res_train.total_return,
            'train_trades': res_train.n_trades,
            'test_sharpe': res_test.sharpe_ratio,
            'test_return': res_test.total_return,
            'test_trades': res_test.n_trades,
            'error': None,
        }
    except Exception as e:
        return {
            'fold_id': fold_id,
            'train_sharpe': 0, 'train_return': 0, 'train_trades': 0,
            'test_sharpe': 0, 'test_return': 0, 'test_trades': 0,
            'error': str(e),
        }


def _worker_mc_permutation(args):
    """
    Run one Monte Carlo permutation (trade shuffle).
    args: (perm_id, seed)
    """
    perm_id, seed = args
    state = _VALIDATE_STATE
    strategy = copy.deepcopy(state['strategy'])
    data = state['data']
    cfg = state['engine_config']
    symbol = state['symbol']
    timeframe = state['timeframe']

    try:
        # Run base backtest
        engine = _make_engine_from_config(cfg)
        result = engine.run(strategy, data, symbol, timeframe)

        if not result.trades:
            return {'perm_id': perm_id, 'sharpe': 0.0, 'error': None}

        # Shuffle trade PnLs with deterministic seed
        rng = np.random.RandomState(seed)
        pnls = np.array([t.net_pnl for t in result.trades])
        rng.shuffle(pnls)

        # Compute Sharpe of shuffled PnLs
        if len(pnls) > 1 and np.std(pnls) > 0:
            shuffled_sr = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
        else:
            shuffled_sr = 0.0

        return {
            'perm_id': perm_id,
            'sharpe': shuffled_sr,
            'error': None,
        }
    except Exception as e:
        return {'perm_id': perm_id, 'sharpe': 0.0, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════
#  OPTIMIZED MC: Run base backtest ONCE, then shuffle in workers
#  (much faster — avoids N redundant backtests)
# ═══════════════════════════════════════════════════════════════

_MC_PNLS = None  # Set before forking
_MC_BASE_SR = None


def _setup_mc_state(pnls, base_sr):
    """Set MC-specific state before creating the pool."""
    global _MC_PNLS, _MC_BASE_SR
    _MC_PNLS = pnls
    _MC_BASE_SR = base_sr


def _worker_mc_shuffle_only(args):
    """
    Pure shuffle worker — no backtest, just permute pre-computed PnLs.
    MUCH faster: ~0.01ms per permutation vs ~50ms per backtest.
    args: (perm_id, seed)
    """
    perm_id, seed = args
    rng = np.random.RandomState(seed)
    pnls = _MC_PNLS.copy()
    rng.shuffle(pnls)

    n = len(pnls)
    if n > 1 and np.std(pnls) > 0:
        sr = np.mean(pnls) / np.std(pnls) * np.sqrt(n)
    else:
        sr = 0.0

    return {'perm_id': perm_id, 'sharpe': sr}


# ═══════════════════════════════════════════════════════════════
#  PARALLEL PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════

class ParallelAntiOverfitPipeline(AntiOverfitPipeline):
    """
    Drop-in replacement for AntiOverfitPipeline with parallel execution.

    When n_jobs > 1:
      - WFA windows run in parallel
      - K-Fold paths run in parallel
      - Monte Carlo permutations run in parallel (optimized: 1 backtest + N shuffles)
      - DSR stays sequential (already instant)

    When n_jobs == 1, falls back to the original sequential behavior.
    """

    def __init__(self, n_jobs: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_jobs = n_jobs

    def run(self, strategy, data, engine_factory, param_grid,
            symbol="", timeframe="", engine_config=None):
        """
        Execute the full pipeline with parallel layers.

        Args:
            engine_config: Serializable dict for worker processes.
                           Required when n_jobs > 1.
                           Format: {'capital': float, 'market_config': dict|None,
                                    'detail_data': dict|None, 'detail_tf': str|None}
        """
        if self.n_jobs <= 1 or engine_config is None:
            # Fallback to sequential
            return super().run(strategy, data, engine_factory,
                               param_grid, symbol, timeframe)

        return self._run_parallel(
            strategy, data, engine_factory, engine_config,
            param_grid, symbol, timeframe)

    def _run_parallel(self, strategy, data, engine_factory, engine_config,
                      param_grid, symbol, timeframe):
        """Parallel execution of all 4 validation layers."""
        from optimize.parallel import create_pool, get_n_jobs

        t0 = time.time()
        n_backtests = 0

        result = AntiOverfitResult()

        if self.verbose:
            print(f"  ⚡ Parallel Validation: {self.n_jobs} workers")
            print(f"  {'═' * 60}")

        # ── Layer 1: Walk-Forward Analysis (parallel windows) ──
        if self.verbose:
            print(f"\n  🔄 Layer 1: Walk-Forward Analysis ({self.wfa.n_windows} windows)...")

        _setup_validate_workers(strategy, data, engine_config, symbol, timeframe)

        try:
            wfa_result = self._parallel_wfa(data, symbol, timeframe)
            result.wfa_result = wfa_result
            result.wfa_passed = (wfa_result.wfe_sharpe >= self.wfa.wfe_threshold)
            n_backtests += self.wfa.n_windows * 2

            if self.verbose:
                icon = "✅" if result.wfa_passed else "❌"
                print(f"  {icon} WFE={wfa_result.wfe_sharpe:.3f} "
                      f"(threshold: {self.wfa.wfe_threshold})")
        except Exception as e:
            if self.verbose:
                print(f"  ❌ WFA error: {e}")

        if self.fail_fast and not result.wfa_passed:
            result.rejection_layer = "WFA"
            result.rejection_reason = "Walk-Forward efficiency below threshold"
            self._finalize_result(result, t0, n_backtests)
            return result

        # ── Layer 2: Purged K-Fold CV (parallel paths) ──
        if self.verbose:
            print(f"\n  🔄 Layer 2: Purged K-Fold CV ({self.pkfold.n_splits} splits)...")

        try:
            pkfold_result = self._parallel_kfold(data, symbol, timeframe)
            result.pkfold_result = pkfold_result
            result.pkfold_passed = (pkfold_result.sharpe_degradation
                                    <= self.pkfold.max_degradation)
            n_backtests += len(pkfold_result.folds) * 2

            if self.verbose:
                icon = "✅" if result.pkfold_passed else "❌"
                print(f"  {icon} Degradation={pkfold_result.sharpe_degradation:.1f}% "
                      f"(threshold: {self.pkfold.max_degradation}%)")
        except Exception as e:
            if self.verbose:
                print(f"  ❌ K-Fold error: {e}")

        if self.fail_fast and not result.pkfold_passed:
            result.rejection_layer = "PurgedKFold"
            result.rejection_reason = "OOS degradation exceeds threshold"
            self._finalize_result(result, t0, n_backtests)
            return result

        # ── Layer 3: DSR (sequential — instant) ──
        if self.verbose:
            print(f"\n  🔄 Layer 3: Deflated Sharpe Ratio...")

        try:
            engine = engine_factory()
            strat = copy.deepcopy(strategy)
            base_result = engine.run(strat, data, symbol, timeframe)
            n_backtests += 1

            from optimize.deflated_sharpe import compute_dsr_from_backtest
            n_trials_est = max(1, len(param_grid.get(
                list(param_grid.keys())[0], [1])) if param_grid else 1)
            for v in param_grid.values():
                n_trials_est *= len(v) if isinstance(v, list) else 1

            ann = _annualization_factor(timeframe)
            dsr = compute_dsr_from_backtest(
                base_result.equity_curve, n_trials_est,
                self.dsr_threshold, ann)

            result.dsr_result = dsr
            result.dsr_passed = dsr.passed

            if self.verbose:
                icon = "✅" if result.dsr_passed else "❌"
                print(f"  {icon} DSR={dsr.deflated_sharpe:.3f} "
                      f"(threshold: {self.dsr_threshold})")
        except Exception as e:
            if self.verbose:
                print(f"  ❌ DSR error: {e}")

        if self.fail_fast and not result.dsr_passed:
            result.rejection_layer = "DSR"
            result.rejection_reason = "Deflated Sharpe below threshold"
            self._finalize_result(result, t0, n_backtests)
            return result

        # ── Layer 4: Monte Carlo (parallel shuffles — optimized) ──
        if self.verbose:
            print(f"\n  🔄 Layer 4: Monte Carlo ({self.mc_simulations} permutations)...")

        try:
            mc_result = self._parallel_monte_carlo(
                base_result, symbol, timeframe)
            result.mc_results = {'trade_shuffle': mc_result}
            result.mc_passed = (mc_result.p_value < self.mc_significance)
            # MC permutations are pure shuffles, not full backtests
            # but count them for reporting
            n_backtests += 1  # only the base backtest counts

            if self.verbose:
                icon = "✅" if result.mc_passed else "❌"
                print(f"  {icon} p-value={mc_result.p_value:.4f} "
                      f"(threshold: {self.mc_significance})")
        except Exception as e:
            if self.verbose:
                print(f"  ❌ MC error: {e}")

        # ── Finalize ──
        self._finalize_result(result, t0, n_backtests)

        if self.verbose:
            print(result.summary())

        return result

    def _parallel_wfa(self, data, symbol, timeframe):
        """Run Walk-Forward windows in parallel."""
        from optimize.parallel import create_pool
        from optimize.walk_forward import WalkForwardResult, WindowResult

        n = len(data['close'])
        n_windows = self.wfa.n_windows
        oos_ratio = self.wfa.oos_ratio

        # Build window indices (same logic as WalkForwardAnalyzer)
        window_size = n // n_windows
        oos_size = max(50, int(window_size * oos_ratio))
        is_size = window_size - oos_size

        work_items = []
        for w in range(n_windows):
            start = w * window_size
            is_end = start + is_size
            oos_end = min(start + window_size, n)

            is_idx = np.arange(start, is_end)
            oos_idx = np.arange(is_end, oos_end)

            if len(is_idx) < 100 or len(oos_idx) < 30:
                continue

            work_items.append((w, is_idx, oos_idx))

        with create_pool(min(self.n_jobs, len(work_items))) as pool:
            results = list(pool.imap_unordered(
                _worker_wfa_window, work_items))

        # Assemble WalkForwardResult
        windows = []
        is_sharpes = []
        oos_sharpes = []

        for res in sorted(results, key=lambda r: r['window_id']):
            if res['error']:
                continue
            is_sharpes.append(res['is_sharpe'])
            oos_sharpes.append(res['oos_sharpe'])

        avg_is = np.mean(is_sharpes) if is_sharpes else 0
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0
        wfe = avg_oos / avg_is if avg_is > 0 else 0

        return WalkForwardResult(
            windows=windows,
            wfe_sharpe=wfe,
            wfe_return=0,
            avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos,
            consistency=sum(1 for s in oos_sharpes if s > 0) / max(1, len(oos_sharpes)),
            n_windows=len(results),
            passed=wfe >= self.wfa.wfe_threshold,
        )

    def _parallel_kfold(self, data, symbol, timeframe):
        """Run Purged K-Fold paths in parallel."""
        import itertools
        from optimize.parallel import create_pool
        from optimize.purged_kfold import PurgedKFoldResult, KFoldInfo

        n = len(data['close'])
        k = self.pkfold.n_splits
        purge = self.pkfold.purge_gap

        # Create groups
        group_size = n // k
        groups = [(i * group_size,
                   (i + 1) * group_size if i < k - 1 else n)
                  for i in range(k)]

        # Generate test combinations
        all_combos = list(itertools.combinations(range(k), 1))
        if len(all_combos) > 30:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(all_combos), 30, replace=False)
            all_combos = [all_combos[i] for i in sorted(idx)]

        # Build index arrays with purging
        work_items = []
        for fold_id, test_combo in enumerate(all_combos):
            test_groups = set(test_combo)
            test_idx = []
            train_idx = []

            for gid, (gs, ge) in enumerate(groups):
                indices = np.arange(gs, ge)
                if gid in test_groups:
                    test_idx.extend(indices)
                else:
                    train_idx.extend(indices)

            # Apply purge at boundaries
            test_set = set(test_idx)
            train_idx_purged = [
                i for i in train_idx
                if not any(abs(i - t) <= purge for t in
                           [test_idx[0], test_idx[-1]] if test_idx)
            ]

            if len(train_idx_purged) < 100 or len(test_idx) < 30:
                continue

            work_items.append((
                fold_id,
                np.array(train_idx_purged),
                np.array(test_idx),
            ))

        with create_pool(min(self.n_jobs, len(work_items))) as pool:
            results = list(pool.imap_unordered(
                _worker_kfold_path, work_items))

        # Assemble result
        train_sharpes = []
        test_sharpes = []
        folds = []

        for res in sorted(results, key=lambda r: r['fold_id']):
            if res['error']:
                continue
            train_sharpes.append(res['train_sharpe'])
            test_sharpes.append(res['test_sharpe'])
            folds.append(KFoldInfo(
                fold_id=res['fold_id'],
                train_sharpe=res['train_sharpe'],
                test_sharpe=res['test_sharpe'],
                train_return=res['train_return'],
                test_return=res['test_return'],
                train_size=0,
                test_size=0,
            ))

        avg_train = np.mean(train_sharpes) if train_sharpes else 0
        avg_test = np.mean(test_sharpes) if test_sharpes else 0
        degradation = ((avg_train - avg_test) / abs(avg_train) * 100
                       if avg_train != 0 else 0)

        return PurgedKFoldResult(
            folds=folds,
            avg_train_sharpe=avg_train,
            avg_test_sharpe=avg_test,
            std_test_sharpe=np.std(test_sharpes) if test_sharpes else 0,
            sharpe_degradation=max(0, degradation),
            passed=degradation <= self.pkfold.max_degradation,
        )

    def _parallel_monte_carlo(self, base_result, symbol, timeframe):
        """
        Optimized parallel Monte Carlo.

        Runs base backtest ONCE, then shuffles PnLs in parallel.
        This is ~1000x faster than running N full backtests.
        """
        from optimize.parallel import create_pool

        pnls = np.array([t.net_pnl for t in base_result.trades])
        n = len(pnls)

        if n < 5:
            return MonteCarloResult(
                p_value=1.0,
                observed_sharpe=0,
                simulated_sharpes=np.array([0]),
                ci_lower=0, ci_upper=0,
                n_simulations=0,
                passed=False,
            )

        # Base Sharpe
        base_sr = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0

        # Setup MC state (fork COW)
        _setup_mc_state(pnls, base_sr)

        # Build work items with deterministic seeds
        work_items = [(i, 42 + i) for i in range(self.mc_simulations)]

        with create_pool(self.n_jobs) as pool:
            results = list(pool.map(
                _worker_mc_shuffle_only, work_items,
                chunksize=max(1, self.mc_simulations // (self.n_jobs * 4))
            ))

        # Compute p-value
        sim_sharpes = np.array([r['sharpe'] for r in results])
        p_value = np.mean(sim_sharpes >= base_sr)

        # Bootstrap CI (95%)
        sorted_sr = np.sort(sim_sharpes)
        ci_lower = sorted_sr[int(0.025 * len(sorted_sr))]
        ci_upper = sorted_sr[int(0.975 * len(sorted_sr))]

        return MonteCarloResult(
            p_value=p_value,
            observed_sharpe=base_sr,
            simulated_sharpes=sim_sharpes,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_simulations=self.mc_simulations,
            passed=p_value < self.mc_significance,
        )

    def _finalize_result(self, result, t0, n_backtests):
        """Compute final scores and timing."""
        result.layers_passed = sum([
            result.wfa_passed,
            result.pkfold_passed,
            result.dsr_passed,
            result.mc_passed,
        ])
        result.all_passed = (result.layers_passed == result.total_layers)

        # Robustness score
        score = 0.0
        if result.wfa_result:
            wfe = max(0, min(1, result.wfa_result.wfe_sharpe))
            score += wfe * 30.0
        if result.pkfold_result:
            deg = result.pkfold_result.sharpe_degradation
            deg_score = max(0, 1.0 - deg / 100.0) if deg > 0 else 1.0
            score += deg_score * 25.0
        if result.dsr_result:
            score += min(1.0, result.dsr_result.deflated_sharpe) * 25.0
        if result.mc_results and 'trade_shuffle' in result.mc_results:
            p = result.mc_results['trade_shuffle'].p_value
            mc_score = max(0, 1.0 - p / self.mc_significance)
            score += mc_score * 20.0

        result.robustness_score = min(100, score)
        result.elapsed_seconds = time.time() - t0
        result.n_backtests_run = n_backtests

        if not result.all_passed:
            failed = []
            if not result.wfa_passed: failed.append("WFA")
            if not result.pkfold_passed: failed.append("PurgedKFold")
            if not result.dsr_passed: failed.append("DSR")
            if not result.mc_passed: failed.append("MonteCarlo")
            result.rejection_layer = ", ".join(failed)
            result.rejection_reason = f"Failed {len(failed)} layer(s)"