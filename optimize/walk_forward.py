"""
CryptoLab — Walk-Forward Analysis
Based on Robert Pardo "The Evaluation and Optimization of Trading Strategies" (2008)

Walk-Forward Analysis (WFA) splits data into sequential In-Sample (IS) / Out-of-Sample (OOS)
windows. The strategy is optimized on IS, then validated on OOS. The Walk-Forward Efficiency
(WFE) measures how well IS performance translates to OOS.

Pipeline rule: WFE < 0.3 → REJECT (strategy is likely overfit)
"""
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field
import itertools
import copy


@dataclass
class WFWindow:
    """A single Walk-Forward window."""
    window_id: int
    is_start: int        # In-sample start index
    is_end: int          # In-sample end index
    oos_start: int       # Out-of-sample start index
    oos_end: int         # Out-of-sample end index
    best_params: Dict[str, Any] = field(default_factory=dict)
    is_sharpe: float = 0.0
    is_return: float = 0.0
    oos_sharpe: float = 0.0
    oos_return: float = 0.0


@dataclass
class WalkForwardResult:
    """Complete WFA results."""
    windows: List[WFWindow]
    wfe_sharpe: float = 0.0       # WFE based on Sharpe ratio
    wfe_return: float = 0.0       # WFE based on total return
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    avg_is_return: float = 0.0
    avg_oos_return: float = 0.0
    oos_equity_curve: Optional[np.ndarray] = None
    passed: bool = False
    rejection_reason: str = ""


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis engine.

    Splits data into n_windows sequential IS/OOS segments.
    For each window:
      1. Optimize parameters on IS data
      2. Validate with best params on OOS data
      3. Record IS/OOS performance

    WFE = avg(OOS_metric) / avg(IS_metric)
    A WFE > 0.3 indicates acceptable robustness.
    """

    def __init__(self,
                 n_windows: int = 5,
                 oos_ratio: float = 0.25,
                 anchored: bool = False,
                 wfe_threshold: float = 0.3,
                 metric: str = 'sharpe'):
        """
        Args:
            n_windows: Number of WF windows
            oos_ratio: Fraction of each window used for OOS (0.2-0.4 typical)
            anchored: If True, IS always starts from bar 0 (expanding window)
            wfe_threshold: Minimum WFE to pass
            metric: 'sharpe' or 'return' — primary metric for optimization
        """
        self.n_windows = n_windows
        self.oos_ratio = oos_ratio
        self.anchored = anchored
        self.wfe_threshold = wfe_threshold
        self.metric = metric

    def create_windows(self, n_bars: int) -> List[WFWindow]:
        """
        Create WF windows from total bar count.

        Non-anchored (rolling):
          Each window has equal total size, IS+OOS consecutive.
          Windows advance by OOS size (no gap, no overlap in OOS).

        Anchored (expanding):
          IS always starts at 0, grows each window.
          OOS is the next segment after IS.
        """
        total_window_size = n_bars / self.n_windows
        oos_size = int(total_window_size * self.oos_ratio)
        is_size = int(total_window_size * (1 - self.oos_ratio))

        windows = []
        for w in range(self.n_windows):
            if self.anchored:
                is_start = 0
                is_end = is_size + w * oos_size
                oos_start = is_end
                oos_end = min(oos_start + oos_size, n_bars)
            else:
                is_start = w * (is_size + oos_size)
                is_end = is_start + is_size
                oos_start = is_end
                oos_end = min(oos_start + oos_size, n_bars)

            if oos_start >= n_bars or is_end >= n_bars:
                break

            windows.append(WFWindow(
                window_id=w,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            ))

        return windows

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            param_grid: Dict[str, List[Any]],
            symbol: str = "",
            timeframe: str = "",
            verbose: bool = True) -> WalkForwardResult:
        """
        Execute Walk-Forward Analysis.

        Args:
            strategy: IStrategy instance (will be deep-copied for each trial)
            data: Full dataset dict with numpy arrays
            engine_factory: Callable that returns a BacktestEngine instance
            param_grid: Dict of parameter name → list of values to search
            symbol: For reporting
            timeframe: For reporting
            verbose: Print progress

        Returns:
            WalkForwardResult with all windows and WFE scores
        """
        n_bars = len(data['close'])
        windows = self.create_windows(n_bars)

        if verbose:
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)
            print(f"  WFA: {len(windows)} windows × {n_combos} param combos")

        all_oos_equity = []

        for win in windows:
            # Slice IS data
            is_data = self._slice_data(data, win.is_start, win.is_end)
            oos_data = self._slice_data(data, win.oos_start, win.oos_end)

            if len(is_data['close']) < 50 or len(oos_data['close']) < 20:
                continue

            # Grid search on IS
            best_metric = -np.inf
            best_params = {}

            for combo in self._param_combinations(param_grid):
                strat_copy = copy.deepcopy(strategy)
                strat_copy.set_params(combo)

                engine = engine_factory()
                result = engine.run(strat_copy, is_data, symbol, timeframe)

                val = result.sharpe_ratio if self.metric == 'sharpe' else result.total_return
                if val > best_metric:
                    best_metric = val
                    best_params = dict(combo)
                    win.is_sharpe = result.sharpe_ratio
                    win.is_return = result.total_return

            win.best_params = best_params

            # Validate on OOS with best params
            strat_oos = copy.deepcopy(strategy)
            strat_oos.set_params(best_params)

            engine = engine_factory()
            oos_result = engine.run(strat_oos, oos_data, symbol, timeframe)

            win.oos_sharpe = oos_result.sharpe_ratio
            win.oos_return = oos_result.total_return

            if len(oos_result.equity_curve) > 0:
                all_oos_equity.append(oos_result.equity_curve)

            if verbose:
                print(f"    Window {win.window_id}: IS Sharpe={win.is_sharpe:.2f} "
                      f"OOS Sharpe={win.oos_sharpe:.2f} "
                      f"IS Ret={win.is_return:+.1f}% OOS Ret={win.oos_return:+.1f}%")

        # Compute WFE
        valid_windows = [w for w in windows if w.is_sharpe != 0]

        if not valid_windows:
            return WalkForwardResult(
                windows=windows,
                passed=False,
                rejection_reason="No valid windows (all IS Sharpe = 0)",
            )

        avg_is_sharpe = np.mean([w.is_sharpe for w in valid_windows])
        avg_oos_sharpe = np.mean([w.oos_sharpe for w in valid_windows])
        avg_is_return = np.mean([w.is_return for w in valid_windows])
        avg_oos_return = np.mean([w.oos_return for w in valid_windows])

        wfe_sharpe = avg_oos_sharpe / avg_is_sharpe if abs(avg_is_sharpe) > 1e-6 else 0.0
        wfe_return = avg_oos_return / avg_is_return if abs(avg_is_return) > 1e-6 else 0.0

        # Concatenate OOS equity curves
        oos_equity = np.concatenate(all_oos_equity) if all_oos_equity else None

        primary_wfe = wfe_sharpe if self.metric == 'sharpe' else wfe_return
        passed = primary_wfe >= self.wfe_threshold

        reason = ""
        if not passed:
            reason = (f"WFE ({self.metric}) = {primary_wfe:.3f} < {self.wfe_threshold} threshold. "
                      f"IS performance does not translate to OOS.")

        result = WalkForwardResult(
            windows=valid_windows,
            wfe_sharpe=wfe_sharpe,
            wfe_return=wfe_return,
            avg_is_sharpe=avg_is_sharpe,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_is_return=avg_is_return,
            avg_oos_return=avg_oos_return,
            oos_equity_curve=oos_equity,
            passed=passed,
            rejection_reason=reason,
        )

        if verbose:
            status = "✅ PASS" if passed else "❌ REJECT"
            print(f"\n  WFA Result: {status}")
            print(f"    WFE (Sharpe): {wfe_sharpe:.3f}  |  WFE (Return): {wfe_return:.3f}")
            print(f"    Avg IS Sharpe: {avg_is_sharpe:.2f}  →  Avg OOS Sharpe: {avg_oos_sharpe:.2f}")

        return result

    def _slice_data(self, data: dict, start: int, end: int) -> dict:
        """Slice all arrays in data dict."""
        sliced = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                sliced[key] = val[start:end].copy()
            else:
                sliced[key] = val
        return sliced

    def _param_combinations(self, grid: Dict[str, List[Any]]):
        """Generate all parameter combinations from grid."""
        if not grid:
            yield {}
            return
        keys = list(grid.keys())
        values = list(grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))
