"""
CryptoLab — Deflated Sharpe Ratio (DSR)
Based on Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)
and Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier" (2012)

The standard Sharpe Ratio is inflated by:
  1. Multiple testing: trying N parameter combinations inflates the best SR
  2. Non-normality: skewed/fat-tailed returns bias SR estimates
  3. Short samples: small T overestimates SR

The Deflated Sharpe Ratio adjusts for all three by computing:
  PSR(SR*) = P(SR > SR_benchmark)
  where SR_benchmark = E[max(SR)] under multiple testing

Pipeline rule: DSR < 0.5 → REJECT (likely a statistical fluke)
"""
import numpy as np
from scipy import stats
from typing import Optional
from dataclasses import dataclass


@dataclass
class DSRResult:
    """Deflated Sharpe Ratio result."""
    observed_sharpe: float       # Raw Sharpe from backtest
    expected_max_sharpe: float   # E[max SR] under N trials
    deflated_sharpe: float       # DSR = PSR(observed | benchmark = E[max SR])
    psr: float                   # Probabilistic SR (before deflation)
    n_trials: int                # Number of trials/combos tested
    n_observations: int          # Number of return observations
    skewness: float              # Sample skewness of returns
    kurtosis: float              # Sample excess kurtosis
    sr_std_error: float          # Standard error of SR estimate
    passed: bool = False
    rejection_reason: str = ""


def _sr_std_error(sr: float, n: int, skew: float, kurt: float) -> float:
    """
    Standard error of the Sharpe Ratio estimate.

    From Lo (2002) and Bailey & López de Prado (2012):
    SE(SR) = sqrt((1 - γ₃·SR + (γ₄-1)/4 · SR²) / (n-1))

    where γ₃ = skewness, γ₄ = excess kurtosis

    Kurtosis is Winsorized at 30 to prevent extreme leverage-driven
    fat tails from making the SE blow up (>30 is effectively noise
    in the estimation of SR uncertainty).
    """
    if n <= 1:
        return 1.0

    # Winsorize: kurtosis > 30 adds noise, not information
    kurt_w = min(kurt, 30.0)

    term1 = 1.0
    term2 = -skew * sr
    term3 = ((kurt_w - 1.0) / 4.0) * sr ** 2

    variance = (term1 + term2 + term3) / (n - 1)
    # Protect against negative variance from extreme inputs
    variance = max(variance, 1e-12)

    return np.sqrt(variance)


def _expected_max_sharpe(n_trials: int, n_obs: int,
                         skew: float = 0.0, kurt: float = 3.0) -> float:
    """
    Expected maximum Sharpe Ratio under N independent trials.

    From Bailey & López de Prado (2014), Eq. 17:
    E[max(SR)] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))

    where γ ≈ 0.5772 (Euler-Mascheroni constant)

    Simplified using Extreme Value Theory approximation:
    E[max(SR)] ≈ √(2·ln(N)) - (ln(π) + ln(ln(N))) / (2·√(2·ln(N)))
    """
    if n_trials <= 1:
        return 0.0

    # Approximation via order statistics of standard normal
    # More accurate: use the exact formula from the paper
    euler_gamma = 0.5772156649

    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    e_max = (1.0 - euler_gamma) * z1 + euler_gamma * z2

    # Adjust for non-normality (first-order correction)
    # Higher kurtosis → higher expected max
    # Winsorize kurtosis same as in SE calculation
    kurt_w = min(kurt, 30.0)
    kurt_correction = 1.0 + (kurt_w - 3.0) / (4.0 * max(n_obs, 10))
    e_max *= max(0.5, kurt_correction)

    return max(0.0, e_max)


def probabilistic_sharpe_ratio(observed_sr: float,
                                benchmark_sr: float,
                                n_obs: int,
                                skew: float,
                                kurt: float) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).

    PSR(SR*) = Φ((SR_obs - SR_benchmark) / SE(SR_obs))

    where Φ is the standard normal CDF.
    This gives the probability that the true SR exceeds the benchmark.
    """
    se = _sr_std_error(observed_sr, n_obs, skew, kurt)

    if se < 1e-12:
        return 1.0 if observed_sr > benchmark_sr else 0.0

    z = (observed_sr - benchmark_sr) / se
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: np.ndarray,
                          n_trials: int,
                          annualization: float = np.sqrt(365),
                          threshold: float = 0.5,
                          benchmark_sr: Optional[float] = None
                          ) -> DSRResult:
    """
    Compute the Deflated Sharpe Ratio.

    Args:
        returns: Array of period returns (not annualized)
        n_trials: Number of parameter combinations tested
        annualization: Factor to annualize SR (√365 for daily crypto, √252 for stocks)
        threshold: Minimum DSR to pass
        benchmark_sr: Override benchmark SR (default: use expected max)

    Returns:
        DSRResult with all components
    """
    # Clean returns
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    n = len(returns)

    if n < 10:
        return DSRResult(
            observed_sharpe=0.0,
            expected_max_sharpe=0.0,
            deflated_sharpe=0.0,
            psr=0.0,
            n_trials=n_trials,
            n_observations=n,
            skewness=0.0,
            kurtosis=3.0,
            sr_std_error=1.0,
            passed=False,
            rejection_reason="Insufficient observations (need ≥ 10)",
        )

    # Sample statistics
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    skew = float(stats.skew(returns)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(returns, fisher=False)) if n > 3 else 3.0  # Excess → raw

    # Observed Sharpe (annualized)
    if sigma > 1e-12:
        sr_obs = (mu / sigma) * annualization
    else:
        sr_obs = 0.0

    # Expected maximum SR under N trials
    if benchmark_sr is not None:
        e_max_sr = benchmark_sr
    else:
        e_max_sr = _expected_max_sharpe(n_trials, n, skew, kurt)

    # Standard error of SR
    # Use per-period SR (not annualized) for the SE calculation
    sr_per_period = mu / sigma if sigma > 1e-12 else 0.0
    se = _sr_std_error(sr_per_period, n, skew, kurt) * annualization

    # Deflated SR = PSR(observed | benchmark = E[max SR])
    dsr = probabilistic_sharpe_ratio(sr_obs, e_max_sr, n, skew, kurt)

    # Also compute standard PSR against 0 benchmark
    psr = probabilistic_sharpe_ratio(sr_obs, 0.0, n, skew, kurt)

    passed = dsr >= threshold
    reason = ""
    if not passed:
        reason = (f"DSR = {dsr:.3f} < {threshold} threshold. "
                  f"Observed Sharpe {sr_obs:.2f} is likely a statistical artifact "
                  f"from testing {n_trials} combinations "
                  f"(E[max SR] = {e_max_sr:.2f}).")

    return DSRResult(
        observed_sharpe=sr_obs,
        expected_max_sharpe=e_max_sr,
        deflated_sharpe=dsr,
        psr=psr,
        n_trials=n_trials,
        n_observations=n,
        skewness=skew,
        kurtosis=kurt,
        sr_std_error=se,
        passed=passed,
        rejection_reason=reason,
    )


def compute_dsr_from_backtest(equity_curve: np.ndarray,
                              n_trials: int,
                              threshold: float = 0.5,
                              annualization: float = np.sqrt(365)
                              ) -> DSRResult:
    """
    Convenience function: compute DSR directly from an equity curve.

    Args:
        equity_curve: Array of equity values over time
        n_trials: Number of param combos tested during optimization
        threshold: Minimum DSR to pass
        annualization: SR annualization factor

    Returns:
        DSRResult
    """
    if len(equity_curve) < 2:
        return DSRResult(
            observed_sharpe=0.0, expected_max_sharpe=0.0,
            deflated_sharpe=0.0, psr=0.0,
            n_trials=n_trials, n_observations=0,
            skewness=0.0, kurtosis=3.0, sr_std_error=1.0,
            passed=False,
            rejection_reason="Equity curve too short",
        )

    # Compute returns
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    return deflated_sharpe_ratio(returns, n_trials, annualization, threshold)