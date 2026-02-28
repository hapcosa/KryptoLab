"""
CryptoLab — Combinatorial Strategy Search
Finds optimal strategy combinations and weight allocations.

Searches across:
- Single strategy with different param configs
- Dual/triple strategy portfolios with capital allocation
- Multi-timeframe variants of the same strategy
- Synergy scoring via return correlation analysis

The goal is to find strategy combinations where the portfolio
performs better risk-adjusted than any individual component.
"""
import numpy as np
import copy
import itertools
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """A strategy with specific parameters."""
    name: str
    strategy: Any               # IStrategy instance
    params: Dict[str, Any]
    weight: float = 1.0         # Capital allocation weight


@dataclass
class PortfolioResult:
    """Result of a portfolio backtest."""
    configs: List[StrategyConfig]
    weights: List[float]
    combined_sharpe: float = 0.0
    combined_return: float = 0.0
    combined_max_dd: float = 0.0
    combined_sortino: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    individual_sharpes: List[float] = field(default_factory=list)
    individual_returns: List[float] = field(default_factory=list)
    synergy_score: float = 0.0     # How much better portfolio is vs best individual
    diversification_ratio: float = 0.0
    n_total_trades: int = 0


@dataclass
class CombinatorialResult:
    """Complete combinatorial search results."""
    portfolios: List[PortfolioResult]
    best_portfolio: Optional[PortfolioResult] = None
    n_combinations_tested: int = 0
    elapsed_seconds: float = 0.0


class CombinatorialSearch:
    """
    Search for optimal strategy combinations.

    Tests all C(N,k) combinations of strategy configs for k ∈ {1,2,3}
    and optimizes weight allocation for each combo.

    Weight optimization uses mean-variance optimization (simplified):
    maximize Sharpe of weighted equity curves.
    """

    def __init__(self,
                 max_portfolio_size: int = 3,
                 weight_steps: int = 5,
                 min_trades: int = 10,
                 verbose: bool = True):
        """
        Args:
            max_portfolio_size: Max strategies in a portfolio (2 or 3)
            weight_steps: Granularity of weight search (5 → 0.2 increments)
            min_trades: Min trades per strategy
            verbose: Print progress
        """
        self.max_size = max_portfolio_size
        self.weight_steps = weight_steps
        self.min_trades = min_trades
        self.verbose = verbose

    def run(self,
            configs: List[StrategyConfig],
            data: dict,
            engine_factory: Callable,
            symbol: str = "",
            timeframe: str = "") -> CombinatorialResult:
        """
        Search strategy combinations.

        Args:
            configs: List of StrategyConfig (each is a strategy+params pair)
            data: Full OHLCV data dict
            engine_factory: Callable returning BacktestEngine
            symbol: For reporting
            timeframe: For reporting

        Returns:
            CombinatorialResult ranked by combined Sharpe
        """
        import time
        t0 = time.time()

        if self.verbose:
            print(f"\n  Combinatorial Search: {len(configs)} configs, "
                  f"max portfolio size={self.max_size}")

        # Step 1: Run individual backtests
        individual_results = {}
        individual_equities = {}

        for i, cfg in enumerate(configs):
            strat = copy.deepcopy(cfg.strategy)
            strat.set_params(cfg.params)

            engine = engine_factory()
            result = engine.run(strat, data, symbol, timeframe)

            individual_results[i] = result
            individual_equities[i] = result.equity_curve

            if self.verbose:
                print(f"    [{i}] {cfg.name}: SR={result.sharpe_ratio:.2f} "
                      f"Ret={result.total_return:+.1f}% "
                      f"Trades={result.n_trades}")

        # Step 2: Test all combinations
        portfolios = []
        n_tested = 0

        for k in range(1, self.max_size + 1):
            for combo in itertools.combinations(range(len(configs)), k):
                if k == 1:
                    # Single strategy — just report individual
                    idx = combo[0]
                    r = individual_results[idx]
                    port = PortfolioResult(
                        configs=[configs[idx]],
                        weights=[1.0],
                        combined_sharpe=r.sharpe_ratio,
                        combined_return=r.total_return,
                        combined_max_dd=r.max_drawdown,
                        combined_sortino=r.sortino_ratio,
                        individual_sharpes=[r.sharpe_ratio],
                        individual_returns=[r.total_return],
                        synergy_score=0.0,
                        n_total_trades=r.n_trades,
                    )
                    portfolios.append(port)
                else:
                    # Multi-strategy — optimize weights
                    equities = [individual_equities[i] for i in combo]
                    min_len = min(len(e) for e in equities)
                    equities = [e[:min_len] for e in equities]

                    best_port = self._optimize_weights(
                        combo, equities, configs, individual_results)

                    if best_port:
                        portfolios.append(best_port)

                n_tested += 1

        # Sort by combined Sharpe
        portfolios.sort(key=lambda p: p.combined_sharpe, reverse=True)

        result = CombinatorialResult(
            portfolios=portfolios[:20],  # Top 20
            best_portfolio=portfolios[0] if portfolios else None,
            n_combinations_tested=n_tested,
            elapsed_seconds=time.time() - t0,
        )

        if self.verbose and result.best_portfolio:
            bp = result.best_portfolio
            names = [c.name for c in bp.configs]
            print(f"\n  Best Portfolio: {' + '.join(names)}")
            print(f"    Weights: {[f'{w:.0%}' for w in bp.weights]}")
            print(f"    Combined: SR={bp.combined_sharpe:.2f} "
                  f"Ret={bp.combined_return:+.1f}% "
                  f"DD={bp.combined_max_dd:.1f}%")
            print(f"    Synergy: {bp.synergy_score:+.2f} "
                  f"Diversification: {bp.diversification_ratio:.2f}")

        return result

    def _optimize_weights(self, combo, equities, configs, results):
        """Find optimal weights for a portfolio combination."""
        n = len(combo)

        # Compute returns for each component
        returns_list = []
        for eq in equities:
            if len(eq) < 2:
                return None
            rets = np.diff(eq) / eq[:-1]
            rets = np.nan_to_num(rets, nan=0, posinf=0, neginf=0)
            returns_list.append(rets)

        min_len = min(len(r) for r in returns_list)
        returns_matrix = np.column_stack([r[:min_len] for r in returns_list])

        # Correlation matrix
        if returns_matrix.shape[1] > 1:
            corr = np.corrcoef(returns_matrix.T)
        else:
            corr = np.array([[1.0]])

        # Generate weight combinations that sum to 1
        best_sharpe = -np.inf
        best_weights = [1.0 / n] * n
        best_equity = None

        weight_options = np.linspace(0.1, 0.9, self.weight_steps)

        if n == 2:
            for w1 in weight_options:
                w2 = 1.0 - w1
                if w2 < 0.05:
                    continue
                weights = [w1, w2]
                port_returns = sum(w * r[:min_len] for w, r
                                   in zip(weights, returns_list))
                sr = self._sharpe(port_returns)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_weights = weights
                    best_equity = self._equity_from_returns(port_returns)
        elif n == 3:
            for w1 in weight_options:
                for w2 in weight_options:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05 or w1 + w2 > 0.95:
                        continue
                    weights = [w1, w2, w3]
                    port_returns = sum(w * r[:min_len] for w, r
                                       in zip(weights, returns_list))
                    sr = self._sharpe(port_returns)
                    if sr > best_sharpe:
                        best_sharpe = sr
                        best_weights = weights
                        best_equity = self._equity_from_returns(port_returns)

        if best_equity is None:
            return None

        # Compute portfolio metrics
        port_returns = sum(w * r[:min_len] for w, r
                           in zip(best_weights, returns_list))

        peak = np.maximum.accumulate(best_equity)
        dd = (peak - best_equity) / np.maximum(peak, 1e-12) * 100
        max_dd = np.max(dd)

        total_return = (best_equity[-1] - best_equity[0]) / best_equity[0] * 100

        # Sortino
        neg_rets = port_returns[port_returns < 0]
        sortino = (np.mean(port_returns) / np.std(neg_rets) * np.sqrt(365)
                   if len(neg_rets) > 1 and np.std(neg_rets) > 0 else 0)

        # Synergy: how much better than the best individual?
        ind_sharpes = [results[i].sharpe_ratio for i in combo]
        best_individual = max(ind_sharpes)
        synergy = best_sharpe - best_individual

        # Diversification ratio
        weighted_vol = sum(w * np.std(r[:min_len]) for w, r
                          in zip(best_weights, returns_list))
        port_vol = np.std(port_returns)
        div_ratio = weighted_vol / port_vol if port_vol > 1e-12 else 1.0

        port = PortfolioResult(
            configs=[configs[i] for i in combo],
            weights=best_weights,
            combined_sharpe=best_sharpe,
            combined_return=total_return,
            combined_max_dd=max_dd,
            combined_sortino=sortino,
            correlation_matrix=corr,
            individual_sharpes=ind_sharpes,
            individual_returns=[results[i].total_return for i in combo],
            synergy_score=synergy,
            diversification_ratio=div_ratio,
            n_total_trades=sum(results[i].n_trades for i in combo),
        )

        return port

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) < 1e-12:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(365)

    @staticmethod
    def _equity_from_returns(returns: np.ndarray,
                              initial: float = 10000.0) -> np.ndarray:
        """Build equity curve from returns."""
        equity = np.zeros(len(returns) + 1)
        equity[0] = initial
        for i, r in enumerate(returns):
            equity[i + 1] = equity[i] * (1 + r)
        return equity
