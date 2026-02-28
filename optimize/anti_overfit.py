"""
CryptoLab — Anti-Overfitting Pipeline
Integrates all 4 anti-overfitting layers into a sequential gate system.

Pipeline:
  Layer 1: Walk-Forward Analysis     → WFE < 0.3      → REJECT
  Layer 2: Purged K-Fold CV          → degradation > 40% → REJECT
  Layer 3: Deflated Sharpe Ratio     → DSR < 0.5      → REJECT
  Layer 4: Monte Carlo Permutation   → p-value > 0.05  → REJECT

All 4 layers must pass for the strategy to be considered robust.
Each layer addresses a different type of overfitting risk:
  - WFA: Are optimized params stable across time?
  - Purged K-Fold: Does performance hold on unseen segments?
  - DSR: Is the Sharpe statistically significant given N trials?
  - Monte Carlo: Could this result occur by chance?
"""
import numpy as np
import copy
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field

from data.bitget_client import TIMEFRAME_SECONDS
from optimize.walk_forward import WalkForwardAnalyzer, WalkForwardResult
from optimize.purged_kfold import PurgedKFoldCV, PurgedKFoldResult
from optimize.deflated_sharpe import (
    deflated_sharpe_ratio, compute_dsr_from_backtest, DSRResult
)
from optimize.monte_carlo import (
    run_full_monte_carlo, MonteCarloResult
)


def _annualization_factor(timeframe: str) -> float:
    """Compute √(bars_per_year) for correct Sharpe annualization."""
    tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 14400)
    bars_per_year = (365.25 * 86400) / tf_seconds
    return np.sqrt(bars_per_year)


@dataclass
class AntiOverfitResult:
    """Complete anti-overfitting analysis result."""
    # Layer results
    wfa_result: Optional[WalkForwardResult] = None
    pkfold_result: Optional[PurgedKFoldResult] = None
    dsr_result: Optional[DSRResult] = None
    mc_results: Optional[Dict[str, MonteCarloResult]] = None

    # Layer verdicts
    wfa_passed: bool = False
    pkfold_passed: bool = False
    dsr_passed: bool = False
    mc_passed: bool = False

    # Overall
    all_passed: bool = False
    layers_passed: int = 0
    total_layers: int = 4
    rejection_layer: str = ""
    rejection_reason: str = ""
    robustness_score: float = 0.0  # 0-100, weighted composite

    # Execution
    elapsed_seconds: float = 0.0
    n_backtests_run: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("═" * 60)
        lines.append("  ANTI-OVERFITTING ANALYSIS")
        lines.append("═" * 60)

        def _icon(passed):
            return "✅" if passed else "❌"

        # Layer 1: WFA
        lines.append(f"\n  {_icon(self.wfa_passed)} Layer 1: Walk-Forward Analysis")
        if self.wfa_result:
            lines.append(f"     WFE (Sharpe): {self.wfa_result.wfe_sharpe:.3f}"
                         f"  (threshold: 0.3)")
            lines.append(f"     Avg IS→OOS Sharpe: {self.wfa_result.avg_is_sharpe:.2f}"
                         f" → {self.wfa_result.avg_oos_sharpe:.2f}")

        # Layer 2: Purged K-Fold
        lines.append(f"\n  {_icon(self.pkfold_passed)} Layer 2: Purged K-Fold CV")
        if self.pkfold_result:
            lines.append(f"     Degradation: {self.pkfold_result.sharpe_degradation:.1f}%"
                         f"  (threshold: 40%)")
            lines.append(f"     Train→Test Sharpe: "
                         f"{self.pkfold_result.avg_train_sharpe:.2f}"
                         f" → {self.pkfold_result.avg_test_sharpe:.2f}"
                         f" ± {self.pkfold_result.std_test_sharpe:.2f}")

        # Layer 3: DSR
        lines.append(f"\n  {_icon(self.dsr_passed)} Layer 3: Deflated Sharpe Ratio")
        if self.dsr_result:
            lines.append(f"     DSR: {self.dsr_result.deflated_sharpe:.3f}"
                         f"  (threshold: 0.5)")
            lines.append(f"     Observed SR: {self.dsr_result.observed_sharpe:.2f}"
                         f"  E[max SR|{self.dsr_result.n_trials} trials]: "
                         f"{self.dsr_result.expected_max_sharpe:.2f}")
            lines.append(f"     Skew: {self.dsr_result.skewness:.2f}"
                         f"  Kurt: {self.dsr_result.kurtosis:.2f}")

        # Layer 4: Monte Carlo
        lines.append(f"\n  {_icon(self.mc_passed)} Layer 4: Monte Carlo Permutation")
        if self.mc_results and 'trade_shuffle' in self.mc_results:
            ts = self.mc_results['trade_shuffle']
            lines.append(f"     p-value (trade shuffle): {ts.p_value:.4f}"
                         f"  (threshold: 0.05)")
            lines.append(f"     90% CI: [{ts.ci_lower:.2f}, {ts.ci_upper:.2f}]")

        # Overall verdict
        lines.append("\n" + "─" * 60)
        verdict = "✅ ROBUST" if self.all_passed else "❌ REJECTED"
        lines.append(f"  VERDICT: {verdict}")
        lines.append(f"  Layers passed: {self.layers_passed}/{self.total_layers}")
        lines.append(f"  Robustness score: {self.robustness_score:.0f}/100")
        if self.rejection_layer:
            lines.append(f"  Failed at: {self.rejection_layer}")
            lines.append(f"  Reason: {self.rejection_reason}")
        lines.append(f"  Analysis time: {self.elapsed_seconds:.1f}s "
                     f"({self.n_backtests_run} backtests)")
        lines.append("═" * 60)

        return "\n".join(lines)


class AntiOverfitPipeline:
    """
    Sequential Anti-Overfitting Pipeline.

    Runs 4 layers of validation. By default, stops at the first failure
    (fail-fast mode). Can also run all layers regardless.
    """

    def __init__(self,
                 # WFA params
                 wfa_windows: int = 5,
                 wfa_oos_ratio: float = 0.25,
                 wfa_threshold: float = 0.3,
                 # Purged K-Fold params
                 pkfold_splits: int = 5,
                 pkfold_purge: int = 10,
                 pkfold_max_degradation: float = 40.0,
                 # DSR params
                 dsr_threshold: float = 0.5,
                 # Monte Carlo params
                 mc_simulations: int = 2000,
                 mc_significance: float = 0.05,
                 # Pipeline params
                 fail_fast: bool = False,
                 verbose: bool = True):
        """
        Args:
            fail_fast: If True, stop at first failed layer
            verbose: Print progress and results
        """
        self.wfa = WalkForwardAnalyzer(
            n_windows=wfa_windows,
            oos_ratio=wfa_oos_ratio,
            wfe_threshold=wfa_threshold,
        )
        self.pkfold = PurgedKFoldCV(
            n_splits=pkfold_splits,
            purge_gap=pkfold_purge,
            max_degradation=pkfold_max_degradation,
        )
        self.dsr_threshold = dsr_threshold
        self.mc_simulations = mc_simulations
        self.mc_significance = mc_significance
        self.fail_fast = fail_fast
        self.verbose = verbose

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            param_grid: Dict[str, List[Any]],
            symbol: str = "",
            timeframe: str = "") -> AntiOverfitResult:
        """
        Execute the full anti-overfitting pipeline.

        Args:
            strategy: IStrategy instance with default/initial params
            data: Full dataset dict
            engine_factory: Callable returning a BacktestEngine
            param_grid: Parameter grid for WFA optimization
            symbol: For reporting
            timeframe: For reporting

        Returns:
            AntiOverfitResult with all layer results
        """
        t0 = time.time()
        result = AntiOverfitResult()
        n_backtests = 0

        if self.verbose:
            print("\n" + "═" * 60)
            print("  ANTI-OVERFITTING PIPELINE")
            print("═" * 60)

        # Count total param combos for DSR
        n_trials = 1
        for v in param_grid.values():
            n_trials *= len(v)

        # ═══════════════════════════════════════════════════
        # LAYER 1: Walk-Forward Analysis
        # ═══════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'─'*60}")
            print("  Layer 1: Walk-Forward Analysis")
            print(f"{'─'*60}")

        wfa_result = self.wfa.run(
            strategy=strategy,
            data=data,
            engine_factory=engine_factory,
            param_grid=param_grid,
            symbol=symbol,
            timeframe=timeframe,
            verbose=self.verbose,
        )
        result.wfa_result = wfa_result
        result.wfa_passed = wfa_result.passed
        n_backtests += len(wfa_result.windows) * n_trials

        if self.fail_fast and not result.wfa_passed:
            result.rejection_layer = "Walk-Forward Analysis"
            result.rejection_reason = wfa_result.rejection_reason
            result.elapsed_seconds = time.time() - t0
            result.n_backtests_run = n_backtests
            result.layers_passed = 0
            result.robustness_score = 0.0
            return result

        # ═══════════════════════════════════════════════════
        # LAYER 2: Purged K-Fold CV
        # ═══════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'─'*60}")
            print("  Layer 2: Purged K-Fold CV")
            print(f"{'─'*60}")

        # Use best params from last WFA window (or defaults)
        best_params = {}
        if wfa_result.windows:
            best_params = wfa_result.windows[-1].best_params
        if best_params:
            strat_pkfold = copy.deepcopy(strategy)
            strat_pkfold.set_params(best_params)
        else:
            strat_pkfold = copy.deepcopy(strategy)

        pkfold_result = self.pkfold.run(
            strategy=strat_pkfold,
            data=data,
            engine_factory=engine_factory,
            symbol=symbol,
            timeframe=timeframe,
            verbose=self.verbose,
        )
        result.pkfold_result = pkfold_result
        result.pkfold_passed = pkfold_result.passed
        n_backtests += len(pkfold_result.folds) * 2

        if self.fail_fast and not result.pkfold_passed:
            result.rejection_layer = "Purged K-Fold CV"
            result.rejection_reason = pkfold_result.rejection_reason
            result.elapsed_seconds = time.time() - t0
            result.n_backtests_run = n_backtests
            result.layers_passed = 1 if result.wfa_passed else 0
            result.robustness_score = result.layers_passed * 25.0
            return result

        # ═══════════════════════════════════════════════════
        # LAYER 3: Deflated Sharpe Ratio
        # ═══════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'─'*60}")
            print("  Layer 3: Deflated Sharpe Ratio")
            print(f"{'─'*60}")

        # Run a full backtest with best params to get equity curve
        strat_full = copy.deepcopy(strategy)
        if best_params:
            strat_full.set_params(best_params)
        engine_full = engine_factory()
        full_result = engine_full.run(strat_full, data, symbol, timeframe)
        n_backtests += 1

        # Compute correct annualization for this timeframe
        ann = _annualization_factor(timeframe)

        dsr_result = compute_dsr_from_backtest(
            equity_curve=full_result.equity_curve,
            n_trials=n_trials,
            threshold=self.dsr_threshold,
            annualization=ann,
        )
        result.dsr_result = dsr_result
        result.dsr_passed = dsr_result.passed

        if self.verbose:
            status = "✅ PASS" if dsr_result.passed else "❌ REJECT"
            print(f"  DSR Result: {status}")
            print(f"    DSR = {dsr_result.deflated_sharpe:.3f} "
                  f"(threshold: {self.dsr_threshold})")
            print(f"    Observed SR: {dsr_result.observed_sharpe:.2f}  "
                  f"E[max SR|{n_trials} trials]: "
                  f"{dsr_result.expected_max_sharpe:.2f}")

        if self.fail_fast and not result.dsr_passed:
            result.rejection_layer = "Deflated Sharpe Ratio"
            result.rejection_reason = dsr_result.rejection_reason
            result.elapsed_seconds = time.time() - t0
            result.n_backtests_run = n_backtests
            result.layers_passed = sum([result.wfa_passed, result.pkfold_passed])
            result.robustness_score = result.layers_passed * 25.0
            return result

        # ═══════════════════════════════════════════════════
        # LAYER 4: Monte Carlo Permutation Test
        # ═══════════════════════════════════════════════════
        if self.verbose:
            print(f"\n{'─'*60}")
            print("  Layer 4: Monte Carlo Permutation Test")
            print(f"{'─'*60}")

        trade_pnls = np.array([t.net_pnl for t in full_result.trades])

        mc_results = run_full_monte_carlo(
            trade_pnls=trade_pnls,
            equity_curve=full_result.equity_curve,
            observed_sharpe=full_result.sharpe_ratio,
            initial_capital=engine_full.initial_capital,
            n_simulations=self.mc_simulations,
            significance=self.mc_significance,
            annualization=ann,
            verbose=self.verbose,
        )
        result.mc_results = mc_results
        result.mc_passed = mc_results.get('passed', False)

        # ═══════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════
        result.layers_passed = sum([
            result.wfa_passed,
            result.pkfold_passed,
            result.dsr_passed,
            result.mc_passed,
        ])
        result.all_passed = (result.layers_passed == result.total_layers)

        # Robustness score (weighted composite)
        # WFA=30%, K-Fold=25%, DSR=25%, MC=20%
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

        if not result.all_passed:
            failed = []
            if not result.wfa_passed:
                failed.append("WFA")
            if not result.pkfold_passed:
                failed.append("PurgedKFold")
            if not result.dsr_passed:
                failed.append("DSR")
            if not result.mc_passed:
                failed.append("MonteCarlo")
            result.rejection_layer = ", ".join(failed)
            result.rejection_reason = f"Failed {len(failed)} layer(s): {', '.join(failed)}"

        result.elapsed_seconds = time.time() - t0
        result.n_backtests_run = n_backtests

        if self.verbose:
            print(result.summary())

        return result


def quick_validate(strategy,
                   data: dict,
                   engine_factory: Callable,
                   param_grid: Dict[str, List[Any]] = None,
                   symbol: str = "",
                   timeframe: str = "",
                   verbose: bool = True) -> AntiOverfitResult:
    """
    Quick validation with default settings.
    Convenience function for rapid strategy assessment.

    Args:
        strategy: Configured IStrategy
        data: OHLCV data dict
        engine_factory: Callable returning BacktestEngine
        param_grid: Optional param grid (uses small defaults if None)
        symbol: Symbol name
        timeframe: Timeframe string
    """
    if param_grid is None:
        # Use a minimal grid from strategy defaults
        param_grid = {}

    pipeline = AntiOverfitPipeline(
        wfa_windows=4,
        mc_simulations=1000,
        fail_fast=False,
        verbose=verbose,
    )

    return pipeline.run(
        strategy=strategy,
        data=data,
        engine_factory=engine_factory,
        param_grid=param_grid,
        symbol=symbol,
        timeframe=timeframe,
    )