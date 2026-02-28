"""
CryptoLab — Combinatorial Purged K-Fold Cross-Validation
Based on Marcos Lopez de Prado "Advances in Financial Machine Learning" (2018), Chapter 12.

Standard K-Fold CV is invalid for time series because:
  1. Temporal autocorrelation causes information leakage
  2. Future data can leak into training through overlapping features

Purged K-Fold solves this by:
  1. Purge gap: removes bars between train/test to prevent autocorrelation leakage
  2. Embargo: removes bars after test set from training to prevent look-ahead bias
  3. Combinatorial: tests C(k,t) combinations for more robust estimates

Pipeline rule: OOS degradation > 40% → REJECT
"""
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field
import itertools
import copy


@dataclass
class FoldResult:
    """Result from a single fold evaluation."""
    fold_id: int
    train_indices: Tuple[int, ...]   # Which groups form training set
    test_indices: Tuple[int, ...]    # Which groups form test set
    train_sharpe: float = 0.0
    train_return: float = 0.0
    test_sharpe: float = 0.0
    test_return: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    n_train_trades: int = 0
    n_test_trades: int = 0


@dataclass
class PurgedKFoldResult:
    """Complete purged K-Fold results."""
    folds: List[FoldResult]
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    sharpe_degradation: float = 0.0   # (train - test) / train * 100
    return_degradation: float = 0.0
    std_test_sharpe: float = 0.0
    std_test_return: float = 0.0
    passed: bool = False
    rejection_reason: str = ""


class PurgedKFoldCV:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Splits data into k groups. For each C(k,t) combination of test groups:
      1. Assign t groups as test set
      2. Assign remaining k-t groups as training set
      3. Purge: remove `purge_gap` bars at train/test boundaries
      4. Embargo: remove `embargo_pct` bars after test set from training
      5. Run backtest on both train and test sets
      6. Record performance degradation

    Typical: k=5, t=1 gives 5 folds (standard). k=6, t=2 gives C(6,2)=15 paths.
    """

    def __init__(self,
                 n_splits: int = 5,
                 n_test_groups: int = 1,
                 purge_gap: int = 10,
                 embargo_pct: float = 0.01,
                 max_degradation: float = 40.0):
        """
        Args:
            n_splits: Number of groups (k)
            n_test_groups: Number of groups used for testing (t)
            purge_gap: Number of bars to remove at train/test boundaries
            embargo_pct: Fraction of test size to embargo after test
            max_degradation: Max acceptable OOS degradation (%)
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.max_degradation = max_degradation

    def _create_groups(self, n_bars: int) -> List[Tuple[int, int]]:
        """Create k equal-sized groups of bar indices."""
        group_size = n_bars // self.n_splits
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_bars
            groups.append((start, end))
        return groups

    def _build_indices(self, groups: List[Tuple[int, int]],
                       test_group_ids: Tuple[int, ...],
                       n_bars: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build train and test index arrays with purging and embargo.

        Purge: remove purge_gap bars before each test segment start
               and after each test segment end.
        Embargo: remove embargo bars after the last test segment from training.
        """
        test_ranges = []
        for gid in test_group_ids:
            test_ranges.append(groups[gid])

        # Sort test ranges
        test_ranges.sort(key=lambda x: x[0])

        # Build test indices
        test_indices = set()
        for start, end in test_ranges:
            test_indices.update(range(start, end))

        # Build purge zones
        purge_indices = set()
        embargo_size = int(len(test_indices) * self.embargo_pct)

        for start, end in test_ranges:
            # Purge before test
            purge_start = max(0, start - self.purge_gap)
            purge_indices.update(range(purge_start, start))
            # Purge after test
            purge_end = min(n_bars, end + self.purge_gap)
            purge_indices.update(range(end, purge_end))

        # Embargo after last test segment
        last_test_end = max(end for _, end in test_ranges)
        embargo_end = min(n_bars, last_test_end + embargo_size)
        embargo_indices = set(range(last_test_end, embargo_end))

        # Train = everything except test + purge + embargo
        excluded = test_indices | purge_indices | embargo_indices
        train_indices = np.array(sorted(set(range(n_bars)) - excluded))
        test_indices = np.array(sorted(test_indices))

        return train_indices, test_indices

    def _slice_by_indices(self, data: dict, indices: np.ndarray) -> dict:
        """Slice data dict by arbitrary indices (may be non-contiguous)."""
        sliced = {}
        for key, val in data.items():
            if isinstance(val, np.ndarray) and len(val) > 0:
                sliced[key] = val[indices].copy()
            else:
                sliced[key] = val
        return sliced

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            symbol: str = "",
            timeframe: str = "",
            verbose: bool = True) -> PurgedKFoldResult:
        """
        Execute Combinatorial Purged K-Fold CV.

        Args:
            strategy: IStrategy instance (already configured with params)
            data: Full dataset dict
            engine_factory: Callable returning BacktestEngine
            symbol: For reporting
            timeframe: For reporting

        Returns:
            PurgedKFoldResult with degradation analysis
        """
        n_bars = len(data['close'])
        groups = self._create_groups(n_bars)

        # Generate all C(k,t) test combinations
        all_test_combos = list(itertools.combinations(
            range(self.n_splits), self.n_test_groups))

        # Limit for computational sanity (C(10,3) = 120 is fine, but cap at 30)
        if len(all_test_combos) > 30:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(all_test_combos), 30, replace=False)
            all_test_combos = [all_test_combos[i] for i in sorted(idx)]

        if verbose:
            print(f"  Purged K-Fold: k={self.n_splits}, t={self.n_test_groups}, "
                  f"paths={len(all_test_combos)}, purge={self.purge_gap}, "
                  f"embargo={self.embargo_pct:.1%}")

        folds = []
        for fold_id, test_combo in enumerate(all_test_combos):
            train_idx, test_idx = self._build_indices(groups, test_combo, n_bars)

            if len(train_idx) < 100 or len(test_idx) < 30:
                continue

            train_data = self._slice_by_indices(data, train_idx)
            test_data = self._slice_by_indices(data, test_idx)

            # Run on training set
            strat_train = copy.deepcopy(strategy)
            engine_train = engine_factory()
            train_result = engine_train.run(strat_train, train_data, symbol, timeframe)

            # Run on test set (same params)
            strat_test = copy.deepcopy(strategy)
            engine_test = engine_factory()
            test_result = engine_test.run(strat_test, test_data, symbol, timeframe)

            fold = FoldResult(
                fold_id=fold_id,
                train_indices=tuple(i for i in range(self.n_splits)
                                    if i not in test_combo),
                test_indices=test_combo,
                train_sharpe=train_result.sharpe_ratio,
                train_return=train_result.total_return,
                test_sharpe=test_result.sharpe_ratio,
                test_return=test_result.total_return,
                train_win_rate=train_result.win_rate,
                test_win_rate=test_result.win_rate,
                n_train_trades=train_result.n_trades,
                n_test_trades=test_result.n_trades,
            )
            folds.append(fold)

            if verbose and fold_id < 8:
                print(f"    Fold {fold_id}: Train Sharpe={fold.train_sharpe:.2f} "
                      f"Test Sharpe={fold.test_sharpe:.2f} "
                      f"({fold.n_train_trades}/{fold.n_test_trades} trades)")

        if not folds:
            return PurgedKFoldResult(
                folds=[],
                passed=False,
                rejection_reason="No valid folds could be created",
            )

        # Aggregate
        avg_train_sharpe = np.mean([f.train_sharpe for f in folds])
        avg_test_sharpe = np.mean([f.test_sharpe for f in folds])
        avg_train_return = np.mean([f.train_return for f in folds])
        avg_test_return = np.mean([f.test_return for f in folds])
        std_test_sharpe = np.std([f.test_sharpe for f in folds])
        std_test_return = np.std([f.test_return for f in folds])

        # Degradation
        if abs(avg_train_sharpe) > 1e-6:
            sharpe_deg = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe) * 100
        else:
            sharpe_deg = 0.0 if abs(avg_test_sharpe) < 1e-6 else 100.0

        if abs(avg_train_return) > 1e-6:
            return_deg = (avg_train_return - avg_test_return) / abs(avg_train_return) * 100
        else:
            return_deg = 0.0 if abs(avg_test_return) < 1e-6 else 100.0

        # Determine pass/fail
        passed = sharpe_deg <= self.max_degradation
        reason = ""
        if not passed:
            reason = (f"OOS degradation {sharpe_deg:.1f}% > {self.max_degradation}% threshold. "
                      f"Avg train Sharpe {avg_train_sharpe:.2f} → test {avg_test_sharpe:.2f}.")

        result = PurgedKFoldResult(
            folds=folds,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            avg_train_return=avg_train_return,
            avg_test_return=avg_test_return,
            sharpe_degradation=sharpe_deg,
            return_degradation=return_deg,
            std_test_sharpe=std_test_sharpe,
            std_test_return=std_test_return,
            passed=passed,
            rejection_reason=reason,
        )

        if verbose:
            status = "✅ PASS" if passed else "❌ REJECT"
            print(f"\n  Purged K-Fold Result: {status}")
            print(f"    Sharpe degradation: {sharpe_deg:.1f}% "
                  f"(train {avg_train_sharpe:.2f} → test {avg_test_sharpe:.2f} "
                  f"± {std_test_sharpe:.2f})")
            if len(all_test_combos) > 8:
                print(f"    ({len(folds)} paths evaluated)")

        return result
