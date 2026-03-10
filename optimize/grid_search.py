"""
CryptoLab — Grid Search Optimizer
Exhaustive parameter search with parallel execution.

Features:
- Full cartesian product of parameter grid
- Multi-metric objective (Sharpe, return, profit factor, composite)
- Early stopping via callback (prune losing combos fast)
- Result ranking and top-N extraction
- Integration with anti-overfit pipeline
"""
import numpy as np
import copy
import time
import itertools
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed


@dataclass
class GridSearchTrial:
    """Result of a single parameter combination trial."""
    trial_id: int
    params: Dict[str, Any]
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    n_trades: int = 0
    objective_value: float = 0.0
    elapsed: float = 0.0


@dataclass
class GridSearchResult:
    """Complete grid search results."""
    trials: List[GridSearchTrial]
    best_trial: Optional[GridSearchTrial] = None
    best_params: Dict[str, Any] = field(default_factory=dict)
    total_combinations: int = 0
    evaluated: int = 0
    elapsed_seconds: float = 0.0
    objective_name: str = ""


# ─── Objective functions ───



def _total_calendar_weeks(trades, active_weeks_fallback: int) -> float:
    """
    Calendar weeks between first and last trade — NOT just active weeks.
    Bug was: divide by ws['n_weeks'] (= weeks WITH trades).
    Result: 26 trades in 156 calendar weeks → 26/26 = 1.0/wk (passes gate).
    Fix:    26 trades in 156 calendar weeks → 26/156 = 0.17/wk (fails gate).
    """
    if not trades:
        return float(max(1, active_weeks_fallback))
    def _ts(t):
        return getattr(t, 'exit_time', 0) or getattr(t, 'entry_time', 0)
    all_ts = [_ts(t) for t in trades if _ts(t) > 0]
    if len(all_ts) < 2:
        return float(max(1, active_weeks_fallback))
    span_days = (max(all_ts) - min(all_ts)) / (1000.0 * 86400.0)
    return max(float(active_weeks_fallback), span_days / 7.0)


def _total_calendar_months(trades, active_months_fallback: int) -> float:
    """Calendar months between first and last trade."""
    if not trades:
        return float(max(1, active_months_fallback))
    from datetime import datetime
    def _ts(t):
        return getattr(t, 'exit_time', 0) or getattr(t, 'entry_time', 0)
    all_ts = [_ts(t) for t in trades if _ts(t) > 0]
    if len(all_ts) < 2:
        return float(max(1, active_months_fallback))
    def _dt(ms):
        return datetime.utcfromtimestamp(ms / 1000 if ms > 1e12 else ms)
    first, last = _dt(min(all_ts)), _dt(max(all_ts))
    months = (last.year - first.year) * 12 + (last.month - first.month) + 1
    return float(max(active_months_fallback, months))

def objective_sharpe(result) -> float:
    return result.sharpe_ratio

def objective_return(result) -> float:
    return result.total_return

def objective_calmar(result) -> float:
    return result.calmar_ratio

def objective_composite(result) -> float:
    """Balanced composite: Sharpe(40%) + PF(20%) + Calmar(20%) + WR(20%)."""
    sr = max(0, result.sharpe_ratio)
    pf = min(5, result.profit_factor) / 5.0 if result.profit_factor > 0 else 0
    cal = min(5, result.calmar_ratio) / 5.0 if result.calmar_ratio > 0 else 0
    wr = result.win_rate / 100.0
    return sr * 0.4 + pf * 0.2 + cal * 0.2 + wr * 0.2


def compute_monthly_stats(trades, initial_capital: float = 1000.0) -> dict:
    """
    Compute monthly breakdown from trade list.

    FIX: pnl_pct is now calculated relative to the running capital
    at the start of each month, not relative to avg position size.
    This matches the engine's total_return calculation which uses
    the equity curve.

    Returns dict with monthly_returns, pct_positive, worst_month,
    monthly_sharpe, and a list of (year, month, pnl_pct, n_trades, wr).
    """
    from collections import defaultdict
    from datetime import datetime

    if not trades or len(trades) < 2:
        return {
            'months': [],
            'monthly_returns': [],
            'pct_positive': 0.0,
            'worst_month': 0.0,
            'best_month': 0.0,
            'monthly_sharpe': 0.0,
            'avg_monthly_return': 0.0,
            'n_months': 0,
        }

    # Group trades by (year, month) using exit_time
    monthly = defaultdict(list)
    for t in trades:
        ts = getattr(t, 'exit_time', 0)
        if ts > 1e9:  # unix timestamp
            dt = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
        else:
            dt = datetime(2024, 1, 1)  # fallback for bar indices
        key = (dt.year, dt.month)
        monthly[key].append(t)

    # Sort trades by exit_time to reconstruct capital progression
    all_trades_sorted = sorted(trades, key=lambda t: getattr(t, 'exit_time', 0))

    # Build a map: for each month, what was the capital at its start?
    # We walk through all trades in order, accumulating net_pnl
    running_capital = initial_capital
    month_start_capital = {}
    current_month_key = None

    for t in all_trades_sorted:
        ts = getattr(t, 'exit_time', 0)
        if ts > 1e9:
            dt = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
        else:
            dt = datetime(2024, 1, 1)
        key = (dt.year, dt.month)

        # First time we see this month → record starting capital
        if key not in month_start_capital:
            month_start_capital[key] = running_capital

        # Accumulate PnL (net_pnl includes commissions and funding)
        running_capital += getattr(t, 'net_pnl', 0.0)

    # Compute per-month stats
    months_data = []
    monthly_rets = []
    for (y, m), month_trades in sorted(monthly.items()):
        total_pnl = sum(t.net_pnl for t in month_trades)
        n = len(month_trades)
        wins = sum(1 for t in month_trades if t.net_pnl > 0)
        wr = (wins / n * 100) if n > 0 else 0

        # PnL as % of capital at START of this month
        start_cap = month_start_capital.get((y, m), initial_capital)
        pnl_pct = (total_pnl / start_cap * 100) if start_cap > 0 else 0

        months_data.append({
            'year': y, 'month': m,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'n_trades': n,
            'win_rate': wr,
        })
        monthly_rets.append(pnl_pct)

    monthly_rets = np.array(monthly_rets)
    n_months = len(monthly_rets)
    pos_months = np.sum(monthly_rets > 0)
    pct_positive = (pos_months / n_months * 100) if n_months > 0 else 0

    avg_ret = np.mean(monthly_rets) if n_months > 0 else 0
    std_ret = np.std(monthly_rets) if n_months > 1 else 1
    monthly_sharpe = (avg_ret / std_ret) if std_ret > 0 else 0

    return {
        'months': months_data,
        'monthly_returns': monthly_rets.tolist(),
        'pct_positive': pct_positive,
        'worst_month': float(np.min(monthly_rets)) if n_months > 0 else 0,
        'best_month': float(np.max(monthly_rets)) if n_months > 0 else 0,
        'monthly_sharpe': monthly_sharpe,
        'avg_monthly_return': avg_ret,
        'n_months': n_months,
    }
def compute_weekly_stats(trades) -> dict:
    """
    Compute per-week PnL and consistency metrics.
    Uses ISO week numbers for grouping.
    """
    from collections import defaultdict
    from datetime import datetime

    if not trades:
        return {'weeks': [], 'weekly_returns': [], 'pct_positive': 0,
                'worst_week': 0, 'best_week': 0, 'weekly_sharpe': 0,
                'avg_weekly_return': 0, 'n_weeks': 0}

    weekly = defaultdict(list)
    for t in trades:
        ts = getattr(t, 'exit_time', 0)
        if ts > 1e9:
            dt = datetime.utcfromtimestamp(ts / 1000 if ts > 1e12 else ts)
        else:
            dt = datetime(2024, 1, 1)
        iso_year, iso_week, _ = dt.isocalendar()
        key = (iso_year, iso_week)
        weekly[key].append(t)

    weeks_data = []
    weekly_rets = []
    for (y, w), week_trades in sorted(weekly.items()):
        total_pnl = sum(t.net_pnl for t in week_trades)
        n = len(week_trades)
        wins = sum(1 for t in week_trades if t.net_pnl > 0)
        wr = (wins / n * 100) if n > 0 else 0

        # Use margin (real capital committed) as denominator, consistent
        # with how monthly_stats uses running_capital. Using t.size (notional
        # with leverage) would inflate pnl_pct by the leverage factor.
        avg_margin = np.mean([abs(getattr(t, 'margin', t.size)) for t in week_trades]) if week_trades else 1
        pnl_pct = (total_pnl / avg_margin * 100) if avg_margin > 0 else 0

        weeks_data.append({
            'year': y, 'week': w,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'n_trades': n,
            'win_rate': wr,
        })
        weekly_rets.append(pnl_pct)

    weekly_rets = np.array(weekly_rets)
    n_weeks = len(weekly_rets)
    pos_weeks = np.sum(weekly_rets > 0)
    pct_positive = (pos_weeks / n_weeks * 100) if n_weeks > 0 else 0

    avg_ret = np.mean(weekly_rets) if n_weeks > 0 else 0
    std_ret = np.std(weekly_rets) if n_weeks > 1 else 1
    weekly_sharpe = (avg_ret / std_ret) if std_ret > 0 else 0

    return {
        'weeks': weeks_data,
        'weekly_returns': weekly_rets.tolist(),
        'pct_positive': pct_positive,
        'worst_week': float(np.min(weekly_rets)) if n_weeks > 0 else 0,
        'best_week': float(np.max(weekly_rets)) if n_weeks > 0 else 0,
        'weekly_sharpe': weekly_sharpe,
        'avg_weekly_return': avg_ret,
        'n_weeks': n_weeks,
    }


def objective_monthly(result) -> float:
    """
    Monthly consistency objective.
    Rewards strategies that are profitable EVERY month, not just overall.

    Score = monthly_sharpe × pct_positive_months × (1 + min(0, worst_month)/10)

    - monthly_sharpe: mean/std of monthly returns (higher = more consistent)
    - pct_positive_months: fraction of months with profit > 0 (0-1)
    - worst_month penalty: if worst month is -20%, multiply by 0.8

    A strategy with SR=2 overall but 3 losing months out of 12 will score
    much lower than one with SR=1.5 but 0 losing months.
    """
    ms = compute_monthly_stats(result.trades, initial_capital=getattr(result, "initial_capital", 1000.0))

    if ms['n_months'] < 2:
        return -999.0

    monthly_sr = ms['monthly_sharpe']
    pct_pos = ms['pct_positive'] / 100.0  # normalize to 0-1

    # Worst month penalty: -20% worst → factor 0.80
    worst_penalty = 1.0 + min(0, ms['worst_month']) / 100.0
    worst_penalty = max(0.1, worst_penalty)  # floor at 0.1

    # Win rate bonus
    wr = result.win_rate / 100.0

    # Must have enough trades per month
    cal_months = _total_calendar_months(result.trades, ms['n_months'])
    if result.n_trades / cal_months < 1.0:  # min ~1 trade/month
        return -999.0
    coverage = min(1.0, ms['n_months'] / cal_months)
    coverage_factor = coverage ** 0.5

    score = monthly_sr * pct_pos * worst_penalty * (0.5 + 0.5 * wr) * coverage_factor
    return score


def objective_monthly_robust(result) -> float:
    """
    Monthly consistency + robustness constraints.

    Like 'monthly' but adds hard penalties for:
    - Win rate < 40%  → -999 (lottery strategies)
    - Leverage > 20x  → progressive penalty
    - PF < 1.0        → -999 (net loser)
    - Max DD > 30%    → progressive penalty

    This prevents the optimizer from finding "exploits" like:
    WR=10%, leverage=30x → a few lucky trades per month.

    Score = monthly_sharpe × pct_pos × wr_factor × leverage_factor × dd_factor
    """
    # ── Hard gates ──
    if result.win_rate < 40.0:
        return -999.0

    if result.profit_factor < 1.0:
        return -999.0

    ms = compute_monthly_stats(result.trades, initial_capital=getattr(result, "initial_capital", 1000.0))

    if ms['n_months'] < 2:
        return -999.0

    cal_months = _total_calendar_months(result.trades, ms['n_months'])
    if result.n_trades / cal_months < 3.5:  # min ~1.5 trades/month
        return -999.0
    coverage = min(1.0, ms['n_months'] / cal_months)
    coverage_factor = coverage ** 0.5

    # ── Monthly consistency score ──
    monthly_sr = ms['monthly_sharpe']
    pct_pos = ms['pct_positive'] / 100.0

    # Worst month penalty
    worst_penalty = 1.0 + min(0, ms['worst_month']) / 100.0
    worst_penalty = max(0.1, worst_penalty)

    # ── Robustness factors ──

    # Win rate: linear 0.4→1.0 mapped to 0.5→1.0
    wr = result.win_rate / 100.0
    wr_factor = 0.5 + 0.5 * min(1.0, (wr - 0.4) / 0.6)

    # Leverage penalty: no penalty ≤15x, linear decay 15x→30x
    lev = result.params.get('leverage', 1.0) if hasattr(result, 'params') else 1.0
    if lev <= 25:
        lev_factor = 1.0
    else:
        lev_factor = max(0.3, 1.0 - (lev - 15) / 30.0)  # 30x → 0.5, 45x → 0.0

    # Drawdown penalty: no penalty ≤15%, linear decay 15%→50%
    dd = abs(result.max_drawdown)
    if dd <= 15:
        dd_factor = 1.0
    else:
        dd_factor = max(0.2, 1.0 - (dd - 15) / 35.0)

    # Profit factor bonus: PF=1.5 → 1.0, PF=3.0 → 1.3
    pf_bonus = min(1.5, result.profit_factor / 2.0) if result.profit_factor > 0 else 0.5

    score = (monthly_sr * pct_pos * worst_penalty
             * wr_factor * lev_factor * dd_factor * pf_bonus
             * coverage_factor)
    return score


def objective_weekly(result) -> float:
    """
    Weekly consistency objective.
    Like monthly but at weekly granularity — stricter test.
    Requires min 4 weeks, min 1 trade/week.
    """
    ws = compute_weekly_stats(result.trades)

    if ws['n_weeks'] < 4:
        return -999.0

    weekly_sr = ws['weekly_sharpe']
    pct_pos = ws['pct_positive'] / 100.0

    worst_penalty = 1.0 + min(0, ws['worst_week']) / 50.0
    worst_penalty = max(0.1, worst_penalty)

    wr = result.win_rate / 100.0

    cal_weeks = _total_calendar_weeks(result.trades, ws['n_weeks'])
    if result.n_trades / cal_weeks < 0.2:  # min 1 trade per 5 calendar weeks
        return -999.0
    coverage = min(1.0, ws['n_weeks'] / cal_weeks)
    coverage_factor = coverage ** 0.5

    score = weekly_sr * pct_pos * worst_penalty * (0.5 + 0.5 * wr) * coverage_factor
    return score


def objective_weekly_robust(result) -> float:
    """
    Weekly consistency + robustness constraints.
    Hard gates: WR >= 40%, PF >= 1.0, leverage penalty, DD penalty.
    """
    if result.win_rate < 40.0:
        return -999.0
    if result.profit_factor < 1.0:
        return -999.0

    ws = compute_weekly_stats(result.trades)
    if ws['n_weeks'] < 4:
        return -999.0

    # KEY FIX: calendar weeks, not active weeks.
    # Old: 26 trades / 26 active_weeks  = 1.0/wk → passes (BUG)
    # New: 26 trades / 156 cal_weeks    = 0.17/wk → -999  (CORRECT)
    cal_weeks = _total_calendar_weeks(result.trades, ws['n_weeks'])
    if result.n_trades / cal_weeks < 0.8:  # 1 trade per 5 calendar weeks
        return -999.0

    # Smooth coverage penalty (can't escape by perfecting few active weeks)
    # 26/156=17% → ×0.41 | 74/156=47% → ×0.69 | 156/156=100% → ×1.0
    coverage = min(1.0, ws['n_weeks'] / cal_weeks)
    coverage_factor = coverage ** 0.5

    weekly_sr = ws['weekly_sharpe']
    pct_pos = ws['pct_positive'] / 100.0
    worst_penalty = 1.0 + min(0, ws['worst_week']) / 50.0
    worst_penalty = max(0.1, worst_penalty)

    wr = result.win_rate / 100.0
    wr_factor = 0.5 + 0.5 * min(1.0, (wr - 0.4) / 0.6)

    lev = result.params.get('leverage', 1.0) if hasattr(result, 'params') else 1.0
    lev_factor = 1.0 if lev <= 15 else max(0.3, 1.0 - (lev - 15) / 30.0)

    dd = abs(result.max_drawdown)
    dd_factor = 1.0 if dd <= 15 else max(0.2, 1.0 - (dd - 15) / 35.0)

    pf_bonus = min(1.5, result.profit_factor / 2.0) if result.profit_factor > 0 else 0.5

    score = (weekly_sr * pct_pos * worst_penalty
             * wr_factor * lev_factor * dd_factor * pf_bonus
             * coverage_factor)
    return score


def check_liquidation_safety(result, max_leverage_sl_pct: float = 85.0) -> bool:
    """
    Verify no trade has leverage × SL_distance exceeding liquidation threshold.

    In perpetual futures, liquidation occurs at approximately:
        price_move% ≈ 100% / leverage  (simplified, ignoring maintenance margin)

    If leverage × sl_distance_pct >= 85%, the position is dangerously close
    to liquidation and the SL may not protect you (slippage, gaps, etc.)

    Args:
        result: BacktestResult with trades and params
        max_leverage_sl_pct: Maximum allowed leverage × sl_distance_pct

    Returns:
        True if all trades are safe, False if any trade exceeds threshold
    """
    if not hasattr(result, 'trades') or not result.trades:
        return True

    for trade in result.trades:
        # Any actual liquidation = immediate fail
        if trade.exit_reason == 'liquidation':
            return False

        # Check SL trades for proximity to liquidation
        if trade.exit_reason == 'SL':
            if trade.entry_price <= 0:
                continue

            # Calculate actual price movement at SL
            if trade.direction == 1:  # LONG
                sl_distance_pct = abs(trade.entry_price - trade.exit_price) / trade.entry_price * 100
            else:  # SHORT
                sl_distance_pct = abs(trade.exit_price - trade.entry_price) / trade.entry_price * 100

            effective_risk = trade.leverage * sl_distance_pct
            if effective_risk > max_leverage_sl_pct:
                return False

    return True


def apply_liquidation_gate(objective_fn):
    """
    Decorator: wraps any objective function with liquidation safety check.
    Returns -999.0 (rejection) if any trade exceeds liquidation threshold.
    """

    def wrapped(result) -> float:
        if not check_liquidation_safety(result, max_leverage_sl_pct=85.0):
            return -999.0
        return objective_fn(result)

    wrapped.__name__ = objective_fn.__name__
    wrapped.__doc__ = objective_fn.__doc__
    return wrapped


OBJECTIVES = {
    'sharpe': apply_liquidation_gate(objective_sharpe),
    'return': apply_liquidation_gate(objective_return),
    'calmar': apply_liquidation_gate(objective_calmar),
    'composite': apply_liquidation_gate(objective_composite),
    'monthly': apply_liquidation_gate(objective_monthly),
    'monthly_robust': apply_liquidation_gate(objective_monthly_robust),
    'weekly': apply_liquidation_gate(objective_weekly),
    'weekly_robust': apply_liquidation_gate(objective_weekly_robust),
}


class GridSearchOptimizer:
    """
    Exhaustive grid search over parameter space.

    Evaluates every combination in the cartesian product of the param grid.
    Suitable for small-to-medium grids (< 10,000 combinations).
    For larger spaces, use BayesianOptimizer or GeneticOptimizer.
    """

    def __init__(self,
                 objective: str = 'sharpe',
                 min_trades: int = 10,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """
        Args:
            objective: 'sharpe', 'return', 'calmar', 'composite'
            min_trades: Minimum trades required (skip combos with fewer)
            n_jobs: Parallel workers (1 = sequential)
            verbose: Print progress
        """
        self.objective_fn = OBJECTIVES.get(objective, objective_sharpe)
        self.objective_name = objective
        self.min_trades = min_trades
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self,
            strategy,
            data: dict,
            engine_factory: Callable,
            param_grid: Dict[str, List[Any]],
            symbol: str = "",
            timeframe: str = "",
            top_n: int = 10,
            engine_config: dict = None) -> GridSearchResult:
        """
        Execute grid search.

        Args:
            strategy: IStrategy instance (will be deep-copied per trial)
            data: Full OHLCV data dict
            engine_factory: Callable returning BacktestEngine (used if n_jobs=1)
            param_grid: Dict[param_name → list_of_values]
            symbol: For reporting
            timeframe: For reporting
            top_n: Return top N trials sorted by objective
            engine_config: For parallel execution (n_jobs > 1)

        Returns:
            GridSearchResult with ranked trials
        """
        t0 = time.time()

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combos = list(itertools.product(*values))
        total = len(all_combos)

        if self.verbose:
            jobs_str = f", {self.n_jobs} workers" if self.n_jobs > 1 else ""
            print(f"\n  Grid Search: {total} combinations, "
                  f"objective={self.objective_name}{jobs_str}")
            print(f"  Parameters: {', '.join(keys)}")
            print(f"  {'─' * 72}")

        trials = []
        interrupted = False

        if self.n_jobs > 1 and engine_config is not None and total > 1:
            # ── PARALLEL EXECUTION ──
            trials, interrupted = self._run_parallel(
                strategy, data, keys, all_combos, total,
                engine_config, symbol, timeframe)
        else:
            # ── SEQUENTIAL EXECUTION ──
            if self.verbose and self.n_jobs <= 1:
                print(f"  {'#':>4} {'Obj':>7} {'SR':>6} {'Ret%':>7} {'WR%':>5} "
                      f"{'DD%':>6} {'PF':>5} {'Trd':>4}  Params")
                print(f"  {'─' * 72}")

            try:
                for trial_id, combo in enumerate(all_combos):
                    optimized_params = dict(zip(keys, combo))

                    t1 = time.time()
                    strat = copy.deepcopy(strategy)
                    strat.set_params(optimized_params)

                    engine = engine_factory()
                    result = engine.run(strat, data, symbol, timeframe)

                    obj_val = self.objective_fn(result) if result.n_trades >= self.min_trades else -999.0

                    full_params = strat.default_params()
                    full_params.update(strat.params)

                    trial = GridSearchTrial(
                        trial_id=trial_id,
                        params=full_params,
                        sharpe_ratio=result.sharpe_ratio,
                        total_return=result.total_return,
                        sortino_ratio=result.sortino_ratio,
                        max_drawdown=result.max_drawdown,
                        win_rate=result.win_rate,
                        profit_factor=result.profit_factor,
                        calmar_ratio=result.calmar_ratio,
                        n_trades=result.n_trades,
                        objective_value=obj_val,
                        elapsed=time.time() - t1,
                    )
                    trials.append(trial)

                    if self.verbose:
                        marker = '★' if not trials or obj_val >= max(t.objective_value for t in trials) else ' '
                        short_params = {k: (f'{v:.1f}' if isinstance(v, float) else str(v)) for k, v in optimized_params.items()}
                        params_str = str(short_params).replace("'", "")
                        print(f"  {marker}{trial_id+1:>3} {obj_val:>7.3f} "
                              f"{result.sharpe_ratio:>6.2f} {result.total_return:>+6.1f}% "
                              f"{result.win_rate:>5.1f} {result.max_drawdown:>5.1f}% "
                              f"{result.profit_factor:>5.2f} {result.n_trades:>4}  {params_str}")
            except KeyboardInterrupt:
                interrupted = True
                if self.verbose:
                    print(f"\n  ⚠️  Interrupted after {len(trials)}/{total} — compiling partial results...")

        # Sort by objective (works with partial results)
        trials.sort(key=lambda t: t.objective_value, reverse=True)
        trials = trials[:top_n] if len(trials) > top_n else trials

        best = trials[0] if trials else None
        best_params = best.params if best else {}

        result = GridSearchResult(
            trials=trials,
            best_trial=best,
            best_params=best_params,
            total_combinations=total,
            evaluated=len(trials) if interrupted else len(all_combos),
            elapsed_seconds=time.time() - t0,
            objective_name=self.objective_name,
        )

        if self.verbose and best:
            status = "INTERRUPTED" if interrupted else "Complete"
            print(f"\n  Grid Search {status} ({result.elapsed_seconds:.1f}s)")
            print(f"  Evaluated: {result.evaluated}/{total}")
            print(f"  Best: obj={best.objective_value:.3f} "
                  f"SR={best.sharpe_ratio:.2f} Ret={best.total_return:+.1f}% "
                  f"WR={best.win_rate:.1f}% DD={best.max_drawdown:.1f}% "
                  f"Trades={best.n_trades}")
            print(f"  Params: {best.params}")

        return result

    def _run_parallel(self, strategy, data, keys, all_combos, total,
                      engine_config, symbol, timeframe):
        """Run grid search trials in parallel using multiprocessing."""
        from optimize.parallel import setup_workers, evaluate_trial, cleanup_workers, create_pool

        setup_workers(
            strategy=strategy,
            data=data,
            engine_config=engine_config,
            objective_name=self.objective_name,
            min_trades=self.min_trades,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Build work items: (trial_id, params_dict)
        work_items = [
            (tid, dict(zip(keys, combo)))
            for tid, combo in enumerate(all_combos)
        ]

        trials = []
        interrupted = False

        try:
            with create_pool(self.n_jobs) as pool:
                # Use imap_unordered for progress reporting
                completed = 0
                best_obj = -999.0

                for res in pool.imap_unordered(evaluate_trial, work_items,
                                                chunksize=max(1, total // (self.n_jobs * 4))):
                    trial = GridSearchTrial(
                        trial_id=res['trial_id'],
                        params=res['params'],
                        sharpe_ratio=res['sharpe_ratio'],
                        total_return=res['total_return'],
                        sortino_ratio=res['sortino_ratio'],
                        max_drawdown=res['max_drawdown'],
                        win_rate=res['win_rate'],
                        profit_factor=res['profit_factor'],
                        calmar_ratio=res['calmar_ratio'],
                        n_trades=res['n_trades'],
                        objective_value=res['objective_value'],
                        elapsed=res['elapsed'],
                    )
                    trials.append(trial)
                    completed += 1

                    is_best = trial.objective_value > best_obj
                    if is_best:
                        best_obj = trial.objective_value

                    if self.verbose:
                        marker = '★' if is_best else ' '
                        pct = completed / total * 100
                        p = res.get('optimized_params', {})
                        p_str = str({k: (f'{v:.1f}' if isinstance(v, float) else str(v))
                                     for k, v in p.items()}).replace("'", "")
                        print(f"  {marker}{completed:>4}/{total} ({pct:4.0f}%) "
                              f"obj={trial.objective_value:>7.3f} "
                              f"SR={trial.sharpe_ratio:>5.2f} "
                              f"Ret={trial.total_return:>+6.1f}% "
                              f"WR={trial.win_rate:>4.1f}% "
                              f"T={trial.n_trades:>4} "
                              f"[{res['elapsed']:.1f}s] {p_str}")

        except KeyboardInterrupt:
            interrupted = True
            if self.verbose:
                print(f"\n  ⚠️  Interrupted after {len(trials)}/{total} — compiling partial results...")
        finally:
            cleanup_workers()

        return trials, interrupted

    def run_with_validation(self,
                            strategy,
                            data: dict,
                            engine_factory: Callable,
                            param_grid: Dict[str, List[Any]],
                            val_ratio: float = 0.3,
                            symbol: str = "",
                            timeframe: str = "",
                            engine_config: dict = None) -> GridSearchResult:
        """
        Grid search with train/validation split.
        Optimizes on training data, reports validation performance.
        """
        n = len(data['close'])
        split = int(n * (1 - val_ratio))

        train_data = {k: v[:split].copy() if isinstance(v, np.ndarray) else v
                      for k, v in data.items()}
        val_data = {k: v[split:].copy() if isinstance(v, np.ndarray) else v
                    for k, v in data.items()}

        if self.verbose:
            print(f"  Train: {split} bars | Validation: {n - split} bars")

        # Optimize on training
        result = self.run(strategy, train_data, engine_factory,
                         param_grid, symbol, timeframe,
                         engine_config=engine_config)

        # Validate top 5
        if self.verbose and result.trials:
            print(f"\n  Validating top 5 on held-out data:")

        try:
            for trial in result.trials[:5]:
                strat = copy.deepcopy(strategy)
                strat.set_params(trial.params)
                engine = engine_factory()
                val_result = engine.run(strat, val_data, symbol, timeframe)

                if self.verbose:
                    print(f"    Train SR={trial.sharpe_ratio:.2f} → "
                          f"Val SR={val_result.sharpe_ratio:.2f} "
                          f"({val_result.n_trades} trades)")
        except KeyboardInterrupt:
            if self.verbose:
                print(f"\n  ⚠️  Validation interrupted — returning training results")

        return result