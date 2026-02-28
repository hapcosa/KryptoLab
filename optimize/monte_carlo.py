"""
CryptoLab — Monte Carlo Permutation Test
Based on White "A Reality Check for Data Snooping" (2000)
and standard permutation testing methodology.

Tests whether observed strategy performance is statistically significant
or could arise from chance alone.

Methods:
  1. Trade Shuffle: randomly permute the sequence of trade PnLs
  2. Return Shuffle: permute period returns to destroy signal structure
  3. Bootstrap: resample trades with replacement for confidence intervals

Pipeline rule: p-value > 0.05 → REJECT (strategy lacks statistical significance)
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class MonteCarloResult:
    """Monte Carlo test results."""
    observed_metric: float           # Original strategy metric
    simulated_distribution: np.ndarray  # Distribution under null
    p_value: float                   # Probability of observing metric by chance
    ci_lower: float                  # 5th percentile of simulations
    ci_upper: float                  # 95th percentile of simulations
    ci_99_lower: float               # 1st percentile
    ci_99_upper: float               # 99th percentile
    n_simulations: int
    method: str                      # 'trade_shuffle', 'return_shuffle', 'bootstrap'
    passed: bool = False
    rejection_reason: str = ""


class MonteCarloTest:
    """
    Monte Carlo Permutation and Bootstrap Testing.

    Generates a null distribution by repeatedly randomizing the strategy's
    results, then computes the probability that the observed performance
    could arise from chance.
    """

    def __init__(self,
                 n_simulations: int = 2000,
                 significance: float = 0.05,
                 random_seed: int = 42):
        """
        Args:
            n_simulations: Number of random permutations (≥1000 recommended)
            significance: p-value threshold for rejection
            random_seed: For reproducibility
        """
        self.n_simulations = n_simulations
        self.significance = significance
        self.rng = np.random.RandomState(random_seed)

    def trade_shuffle_test(self,
                           trade_pnls: np.ndarray,
                           observed_sharpe: float,
                           initial_capital: float = 10000.0,
                           metric: str = 'sharpe',
                           annualization: float = None,
                           ) -> MonteCarloResult:
        """
        Trade Shuffle Test.

        Null hypothesis: the ORDER of trades doesn't matter — any permutation
        of the same trade PnLs could have occurred.

        If the observed Sharpe is not significantly better than the shuffled
        distribution, the strategy's risk-adjusted returns are indistinguishable
        from random ordering.

        IMPORTANT: Both the observed metric and simulated metrics are computed
        identically from trade PnLs → equity → returns → Sharpe, ensuring
        an apples-to-apples comparison. The observed_sharpe parameter from
        the engine (bar-by-bar) is NOT used for comparison — we recompute it
        from the trade PnLs to match the simulation methodology.

        Args:
            trade_pnls: Array of trade net PnL values
            observed_sharpe: The strategy's observed Sharpe ratio (for reporting only)
            initial_capital: Starting capital for equity curve construction
            metric: 'sharpe' or 'total_return'
            annualization: √(bars_per_year) — if None, uses √365

        Returns:
            MonteCarloResult with p-value and distribution
        """
        ann = annualization if annualization is not None else np.sqrt(365)

        if len(trade_pnls) < 5:
            return MonteCarloResult(
                observed_metric=observed_sharpe,
                simulated_distribution=np.array([]),
                p_value=1.0,
                ci_lower=0.0, ci_upper=0.0,
                ci_99_lower=0.0, ci_99_upper=0.0,
                n_simulations=0,
                method='trade_shuffle',
                passed=False,
                rejection_reason="Insufficient trades (need ≥ 5)",
            )

        def _equity_metric(pnls, cap):
            """Compute metric from trade PnLs — same logic for observed & simulated."""
            equity = np.zeros(len(pnls) + 1)
            equity[0] = cap
            for i, pnl in enumerate(pnls):
                equity[i + 1] = equity[i] + pnl

            if metric == 'sharpe':
                returns = np.diff(equity) / equity[:-1]
                returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
                if len(returns) > 1 and np.std(returns) > 1e-12:
                    return np.mean(returns) / np.std(returns) * ann
                return 0.0
            else:
                return (equity[-1] - cap) / cap * 100

        # Compute observed metric using the SAME methodology as simulations
        observed_trade_metric = _equity_metric(trade_pnls, initial_capital)

        simulated = np.zeros(self.n_simulations)
        for sim in range(self.n_simulations):
            shuffled = self.rng.permutation(trade_pnls)
            simulated[sim] = _equity_metric(shuffled, initial_capital)

        return self._build_result(
            observed=observed_trade_metric,
            simulated=simulated,
            method='trade_shuffle',
        )

    def return_shuffle_test(self,
                            returns: np.ndarray,
                            observed_sharpe: float,
                            block_size: int = 5,
                            metric: str = 'sharpe',
                            annualization: float = None,
                            ) -> MonteCarloResult:
        """
        Block Return Shuffle Test (Stationary Bootstrap variant).

        Instead of shuffling individual returns (which destroys autocorrelation
        structure), shuffle blocks of consecutive returns. This preserves local
        dependencies while breaking the signal-to-return mapping.

        Null hypothesis: the strategy's entry/exit timing provides no edge
        beyond what random block-shuffled returns would produce.

        Args:
            returns: Array of period returns
            observed_sharpe: Strategy's observed Sharpe
            block_size: Size of blocks for shuffling (preserves local structure)
            metric: 'sharpe' or 'total_return'
            annualization: √(bars_per_year) — if None, uses √365
        """
        ann = annualization if annualization is not None else np.sqrt(365)
        n = len(returns)
        if n < 20:
            return MonteCarloResult(
                observed_metric=observed_sharpe,
                simulated_distribution=np.array([]),
                p_value=1.0,
                ci_lower=0.0, ci_upper=0.0,
                ci_99_lower=0.0, ci_99_upper=0.0,
                n_simulations=0,
                method='return_shuffle',
                passed=False,
                rejection_reason="Insufficient returns (need ≥ 20)",
            )

        # Create blocks
        n_blocks = n // block_size
        if n_blocks < 3:
            block_size = max(1, n // 3)
            n_blocks = n // block_size

        blocks = [returns[i * block_size:(i + 1) * block_size]
                  for i in range(n_blocks)]

        # Handle remainder
        remainder = returns[n_blocks * block_size:]

        simulated = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            # Shuffle block order
            shuffled_blocks = [blocks[i] for i in
                               self.rng.permutation(n_blocks)]
            shuffled_returns = np.concatenate(shuffled_blocks)
            if len(remainder) > 0:
                shuffled_returns = np.concatenate([shuffled_returns, remainder])

            if metric == 'sharpe':
                std = np.std(shuffled_returns)
                if std > 1e-12:
                    simulated[sim] = (np.mean(shuffled_returns) / std *
                                      ann)
                else:
                    simulated[sim] = 0.0
            else:
                simulated[sim] = np.sum(shuffled_returns) * 100

        return self._build_result(
            observed=observed_sharpe,
            simulated=simulated,
            method='return_shuffle',
        )

    def bootstrap_confidence(self,
                             trade_pnls: np.ndarray,
                             observed_sharpe: float,
                             initial_capital: float = 10000.0,
                             metric: str = 'sharpe',
                             annualization: float = None,
                             ) -> MonteCarloResult:
        """
        Bootstrap Confidence Interval.

        Resamples trades WITH REPLACEMENT to estimate the distribution
        of the metric. This provides confidence intervals rather than
        a formal hypothesis test.

        Useful for estimating the uncertainty of the observed Sharpe.

        Args:
            trade_pnls: Array of trade net PnLs
            observed_sharpe: Strategy's observed metric (for reporting only)
            initial_capital: Starting capital
            metric: 'sharpe' or 'total_return'
            annualization: √(bars_per_year) — if None, uses √365
        """
        ann = annualization if annualization is not None else np.sqrt(365)
        n = len(trade_pnls)
        if n < 5:
            return MonteCarloResult(
                observed_metric=observed_sharpe,
                simulated_distribution=np.array([]),
                p_value=0.5,
                ci_lower=0.0, ci_upper=0.0,
                ci_99_lower=0.0, ci_99_upper=0.0,
                n_simulations=0,
                method='bootstrap',
                passed=False,
                rejection_reason="Insufficient trades for bootstrap",
            )

        def _equity_metric(pnls, cap):
            equity = np.zeros(len(pnls) + 1)
            equity[0] = cap
            for i, pnl in enumerate(pnls):
                equity[i + 1] = equity[i] + pnl
            if metric == 'sharpe':
                returns = np.diff(equity) / equity[:-1]
                returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
                if len(returns) > 1 and np.std(returns) > 1e-12:
                    return np.mean(returns) / np.std(returns) * ann
                return 0.0
            else:
                return (equity[-1] - cap) / cap * 100

        # Observed metric computed same as simulations
        observed_trade_metric = _equity_metric(trade_pnls, initial_capital)

        simulated = np.zeros(self.n_simulations)
        for sim in range(self.n_simulations):
            resampled = self.rng.choice(trade_pnls, size=n, replace=True)
            simulated[sim] = _equity_metric(resampled, initial_capital)

        return self._build_result(
            observed=observed_trade_metric,
            simulated=simulated,
            method='bootstrap',
        )

    def _build_result(self, observed: float, simulated: np.ndarray,
                      method: str) -> MonteCarloResult:
        """Build MonteCarloResult from simulation output."""
        n_sims = len(simulated)

        if n_sims == 0:
            return MonteCarloResult(
                observed_metric=observed,
                simulated_distribution=simulated,
                p_value=1.0,
                ci_lower=0.0, ci_upper=0.0,
                ci_99_lower=0.0, ci_99_upper=0.0,
                n_simulations=0,
                method=method,
                passed=False,
                rejection_reason="No simulations completed",
            )

        # p-value: fraction of simulations >= observed
        p_value = np.mean(simulated >= observed)

        # Confidence intervals
        ci_lower = np.percentile(simulated, 5)
        ci_upper = np.percentile(simulated, 95)
        ci_99_lower = np.percentile(simulated, 1)
        ci_99_upper = np.percentile(simulated, 99)

        passed = p_value <= self.significance
        reason = ""
        if not passed:
            reason = (f"p-value = {p_value:.4f} > {self.significance} significance level. "
                      f"Observed metric {observed:.3f} is within the null distribution "
                      f"[{ci_lower:.3f}, {ci_upper:.3f}] (90% CI).")

        return MonteCarloResult(
            observed_metric=observed,
            simulated_distribution=simulated,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper,
            n_simulations=n_sims,
            method=method,
            passed=passed,
            rejection_reason=reason,
        )


def run_full_monte_carlo(trade_pnls: np.ndarray,
                         equity_curve: np.ndarray,
                         observed_sharpe: float,
                         initial_capital: float = 10000.0,
                         n_simulations: int = 2000,
                         significance: float = 0.05,
                         annualization: float = None,
                         verbose: bool = True
                         ) -> dict:
    """
    Run all three Monte Carlo tests and return combined results.

    Args:
        annualization: √(bars_per_year) for the timeframe used.
                       If None, defaults to √365 (daily).

    Returns dict with keys: 'trade_shuffle', 'return_shuffle', 'bootstrap'
    """
    mc = MonteCarloTest(n_simulations=n_simulations,
                        significance=significance)

    # Compute returns from equity curve
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    results = {}

    if verbose:
        print(f"  Monte Carlo: {n_simulations} simulations, "
              f"α={significance}")

    # 1. Trade shuffle
    r1 = mc.trade_shuffle_test(trade_pnls, observed_sharpe, initial_capital,
                                annualization=annualization)
    results['trade_shuffle'] = r1
    if verbose:
        status = "✅" if r1.passed else "❌"
        print(f"    {status} Trade Shuffle: p={r1.p_value:.4f} "
              f"CI=[{r1.ci_lower:.2f}, {r1.ci_upper:.2f}]")

    # 2. Return shuffle (block)
    r2 = mc.return_shuffle_test(returns, observed_sharpe,
                                 annualization=annualization)
    results['return_shuffle'] = r2
    if verbose:
        status = "✅" if r2.passed else "❌"
        print(f"    {status} Return Shuffle: p={r2.p_value:.4f} "
              f"CI=[{r2.ci_lower:.2f}, {r2.ci_upper:.2f}]")

    # 3. Bootstrap CI
    r3 = mc.bootstrap_confidence(trade_pnls, observed_sharpe, initial_capital,
                                  annualization=annualization)
    results['bootstrap'] = r3
    if verbose:
        print(f"    📊 Bootstrap CI: [{r3.ci_lower:.2f}, {r3.ci_upper:.2f}] (90%)")
        print(f"                     [{r3.ci_99_lower:.2f}, {r3.ci_99_upper:.2f}] (98%)")

    # Overall: pass if at least trade_shuffle passes
    overall_pass = r1.passed
    results['passed'] = overall_pass
    results['primary_p_value'] = r1.p_value

    if verbose:
        status = "✅ PASS" if overall_pass else "❌ REJECT"
        print(f"\n  Monte Carlo Result: {status} (primary p={r1.p_value:.4f})")

    return results