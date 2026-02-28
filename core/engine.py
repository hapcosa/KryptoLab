"""
CryptoLab — Core Backtesting Engine
High-performance backtester for perpetual futures.

Features:
- Bar-by-bar simulation with intra-bar detail (Phase 2)
- Multi-level TP with partial closes
- Break-even activation
- Progressive trailing stop
- Funding rate simulation
- Liquidation detection
- Full trade logging
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from strategies.base import IStrategy, Signal
from data.bitget_client import MarketConfig, TIMEFRAME_SECONDS


# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Active position in the portfolio."""
    id: int
    direction: int          # 1=long, -1=short
    entry_price: float
    size: float             # USDT notional
    leverage: float
    margin: float           # Actual margin used
    
    sl_price: float
    original_sl: float
    tp_levels: List[float]  = field(default_factory=list)
    tp_sizes: List[float]   = field(default_factory=list)
    tp_hit: List[bool]      = field(default_factory=list)
    
    be_trigger: float       = 0.0
    be_activated: bool      = False
    trailing: bool          = False
    trailing_distance: float = 0.0
    trailing_activated: bool = False
    trailing_high: float    = 0.0   # Best price since entry (for trailing)
    
    entry_bar: int          = 0
    entry_timestamp: float  = 0.0
    
    unrealized_pnl: float   = 0.0
    realized_pnl: float     = 0.0
    funding_paid: float     = 0.0
    commission_paid: float  = 0.0
    
    remaining_size: float   = 1.0   # Fraction remaining (1.0 = full)
    confidence: float       = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Completed trade record."""
    id: int
    direction: int
    entry_price: float
    exit_price: float
    entry_bar: int
    exit_bar: int
    entry_time: float
    exit_time: float
    size: float
    leverage: float
    pnl: float
    pnl_pct: float
    commission: float
    funding: float
    net_pnl: float
    exit_reason: str        # 'SL', 'TP1', 'TP2', ..., 'trailing', 'signal', 'liquidation'
    duration_bars: int
    confidence: float
    max_favorable: float    # Max favorable excursion
    max_adverse: float      # Max adverse excursion
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[Trade]
    equity_curve: np.ndarray
    drawdown_curve: np.ndarray
    
    # Summary metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    n_trades: int = 0
    n_longs: int = 0
    n_shorts: int = 0
    avg_duration: float = 0.0
    
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    start_date: str = ""
    end_date: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    annualization_factor: float = 0.0   # √(bars_per_year) — depends on timeframe


# ═══════════════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ═══════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Core backtesting engine for perpetual futures.

    Usage:
        engine = BacktestEngine(initial_capital=10000)
        result = engine.run(strategy, data)
    """

    def __init__(self,
                 initial_capital: float = 500.0,
                 fee_maker: float = 0.0002,
                 fee_taker: float = 0.0006,
                 funding_rate: float = 0.0001,  # per interval
                 funding_interval_hours: int = 8,
                 max_positions: int = 1,
                 market_config: dict = None):

        self.initial_capital = initial_capital
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.funding_rate = funding_rate
        self.funding_interval_hours = funding_interval_hours
        self.max_positions = max_positions

        if market_config:
            self.fee_maker = market_config.get('fee_maker', fee_maker)
            self.fee_taker = market_config.get('fee_taker', fee_taker)
            self.funding_interval_hours = market_config.get(
                'funding_interval_hours', funding_interval_hours)

        # State
        self._capital = initial_capital
        self._positions: List[Position] = []
        self._trades: List[Trade] = []
        self._equity: List[float] = []
        self._trade_counter = 0
        self._detail_data: Optional[dict] = None

    def set_detail_data(self, detail_data: dict, detail_tf: str):
        """
        Set intra-bar detail data for high-TF backtesting.
        When backtesting on 4H, this provides 5m data for SL/TP simulation.
        """
        self._detail_data = detail_data
        self._detail_tf = detail_tf

    def run(self, strategy: IStrategy, data: dict,
            symbol: str = "", timeframe: str = "",
            callback=None) -> BacktestResult:
        """
        Execute complete backtest.

        Args:
            strategy: IStrategy implementation
            data: dict with numpy arrays (open, high, low, close, volume, timestamp)
            symbol: symbol name for reporting
            timeframe: timeframe string for reporting
            callback: optional function called each bar (for optimizer pruning)

        Returns:
            BacktestResult with all metrics and trade log
        """
        # Reset state
        self._capital = self.initial_capital
        self._positions = []
        self._trades = []
        self._equity = []
        self._trade_counter = 0

        n = len(data['close'])

        # Build TimeframeDetail if detail data is available
        self._tf_detail = None
        if self._detail_data is not None and 'timestamp' in data:
            self._tf_detail = TimeframeDetail(
                data['timestamp'], self._detail_data, timeframe
            )

        # Pre-calculate all indicators
        indicators = strategy.calculate_indicators(data)

        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 14400)
        funding_interval_bars = max(1, (self.funding_interval_hours * 3600) // tf_seconds)

        # Main loop
        for i in range(n):
            bar = {
                'open': data['open'][i],
                'high': data['high'][i],
                'low': data['low'][i],
                'close': data['close'][i],
                'volume': data['volume'][i],
                'timestamp': data['timestamp'][i] if 'timestamp' in data else i,
            }

            # 1. Check exits (SL/TP/trailing/liquidation)
            #    Uses intra-bar detail data if available
            self._check_exits(bar, i, data)

            # 2. Apply funding rate
            if i > 0 and i % funding_interval_bars == 0:
                self._apply_funding(bar)

            # 3. Generate new signal (+ close-on-opposite-signal)
            signal = strategy.generate_signal(indicators, i, data)

            if signal is not None:
                if len(self._positions) < self.max_positions:
                    # No position — open normally
                    self._open_position(signal, bar, i)
                elif signal.metadata.get('close_on_signal', False):
                    # Position exists — check if signal is opposite
                    for pos in list(self._positions):
                        if pos.direction != signal.direction:
                            ts = bar.get('timestamp', i)
                            self._close_position(pos, bar['close'], i, ts, 'signal')
                    # Now open new if we freed a slot
                    if len(self._positions) < self.max_positions:
                        self._open_position(signal, bar, i)

            # 4. Update position state (trailing, etc.)
            self._update_positions(bar, i, strategy, data)

            # 5. Record equity
            equity = self._calculate_equity(bar)
            self._equity.append(equity)

            # 6. Callback for optimizer
            if callback and i % 100 == 0:
                callback({
                    'bar': i,
                    'equity': equity,
                    'n_trades': len(self._trades),
                    'drawdown': self._current_drawdown(),
                })

        # Close remaining positions at last bar close
        if self._positions:
            last_bar = {
                'open': data['open'][-1],
                'high': data['high'][-1],
                'low': data['low'][-1],
                'close': data['close'][-1],
                'volume': data['volume'][-1],
                'timestamp': data['timestamp'][-1] if 'timestamp' in data else n-1,
            }
            for pos in list(self._positions):
                self._close_position(pos, last_bar['close'], n-1,
                                    last_bar['timestamp'], 'end_of_data')

        # Compile results
        return self._compile_results(strategy, data, symbol, timeframe)

    # ─── POSITION MANAGEMENT ───

    def _open_position(self, signal: Signal, bar: dict, bar_idx: int):
        """Open a new position from a signal."""
        # Calculate position size
        risk_capital = self._capital * 0.02  # 2% risk per trade

        # Commission on entry
        size = risk_capital * signal.leverage
        commission = size * self.fee_taker

        if commission >= self._capital:
            return  # Can't afford

        self._trade_counter += 1

        pos = Position(
            id=self._trade_counter,
            direction=signal.direction,
            entry_price=signal.entry_price,
            size=size,
            leverage=signal.leverage,
            margin=size / signal.leverage,
            sl_price=signal.sl_price,
            original_sl=signal.sl_price,
            tp_levels=list(signal.tp_levels),
            tp_sizes=list(signal.tp_sizes),
            tp_hit=[False] * len(signal.tp_levels),
            be_trigger=signal.be_trigger,
            trailing=signal.trailing,
            trailing_distance=signal.trailing_distance,
            trailing_high=signal.entry_price,
            entry_bar=bar_idx,
            entry_timestamp=bar.get('timestamp', bar_idx),
            commission_paid=commission,
            confidence=signal.confidence,
            metadata=signal.metadata,
        )

        self._capital -= commission
        self._capital -= pos.margin
        self._positions.append(pos)

    def _check_exits(self, bar: dict, bar_idx: int, data: dict):
        """
        Check all positions for exits.

        When TimeframeDetail is available, iterates through intra-bar
        candles (e.g. 5m candles within a 4h bar) for accurate SL/TP
        sequencing. This solves the problem of not knowing whether
        SL or TP was hit first within a single bar.

        Without detail data, uses bar high/low with the conservative
        assumption that adverse move happens first (SL checked before TP).
        """
        if self._tf_detail is not None:
            intrabar = self._tf_detail.get_intrabar_candles(bar_idx)
            if intrabar is not None and len(intrabar['high']) > 0:
                self._check_exits_detailed(bar, bar_idx, intrabar)
                return

        # Fallback: standard bar-level exit check
        self._check_exits_bar(bar, bar_idx)

    def _check_exits_detailed(self, main_bar: dict, bar_idx: int,
                               intrabar: dict):
        """
        Check exits using intra-bar candles for precise SL/TP ordering.

        Iterates each detail candle (e.g. 5m within 4h) chronologically.
        First candle that triggers SL, TP, or liquidation determines the exit.
        """
        n_detail = len(intrabar['high'])

        for pos in list(self._positions):
            exited = False

            for j in range(n_detail):
                if exited:
                    break

                d_high = intrabar['high'][j]
                d_low = intrabar['low'][j]
                d_close = intrabar['close'][j]
                ts = main_bar.get('timestamp', bar_idx)

                if pos.direction == 1:  # LONG
                    # Liquidation
                    liq_price = pos.entry_price * (1 - 1.0 / pos.leverage * 0.9)
                    if d_low <= liq_price:
                        self._close_position(pos, liq_price, bar_idx, ts, 'liquidation')
                        exited = True
                        continue

                    # Stop Loss (or trailing stop if SL was moved by trailing)
                    if d_low <= pos.sl_price:
                        reason = 'trailing' if pos.trailing_activated else 'SL'
                        self._close_position(pos, pos.sl_price, bar_idx, ts, reason)
                        exited = True
                        continue

                    # Break-even: check against actual intrabar price (not just TP)
                    if (pos.be_trigger > 0 and not pos.be_activated
                            and d_high >= pos.be_trigger):
                        pos.be_activated = True
                        pos.sl_price = pos.entry_price

                    # Take Profits
                    for tp_idx, (tp_price, tp_size, tp_hit) in enumerate(
                            zip(pos.tp_levels, pos.tp_sizes, pos.tp_hit)):
                        if not tp_hit and d_high >= tp_price:
                            pos.tp_hit[tp_idx] = True
                            # tp_size = fraction of ORIGINAL position (not remaining)
                            partial_size = pos.size * tp_size
                            pnl = partial_size * (tp_price - pos.entry_price) / pos.entry_price
                            commission = partial_size * self.fee_taker
                            pos.realized_pnl += pnl - commission
                            pos.commission_paid += commission
                            pos.remaining_size = max(0.0, pos.remaining_size - tp_size)
                            self._capital += pnl - commission + (partial_size / pos.leverage)

                            self._record_partial(pos, tp_price, bar_idx, ts,
                                               f'TP{tp_idx+1}', pnl - commission)

                    if pos.remaining_size < 0.01:
                        self._positions.remove(pos)
                        exited = True
                        continue

                    # Update trailing high (tracks best price for long)
                    if d_high > pos.trailing_high:
                        pos.trailing_high = d_high

                else:  # SHORT
                    liq_price = pos.entry_price * (1 + 1.0 / pos.leverage * 0.9)
                    if d_high >= liq_price:
                        self._close_position(pos, liq_price, bar_idx, ts, 'liquidation')
                        exited = True
                        continue

                    if d_high >= pos.sl_price:
                        reason = 'trailing' if pos.trailing_activated else 'SL'
                        self._close_position(pos, pos.sl_price, bar_idx, ts, reason)
                        exited = True
                        continue

                    # Break-even: check against actual intrabar price (not just TP)
                    if (pos.be_trigger > 0 and not pos.be_activated
                            and d_low <= pos.be_trigger):
                        pos.be_activated = True
                        pos.sl_price = pos.entry_price

                    for tp_idx, (tp_price, tp_size, tp_hit) in enumerate(
                            zip(pos.tp_levels, pos.tp_sizes, pos.tp_hit)):
                        if not tp_hit and d_low <= tp_price:
                            pos.tp_hit[tp_idx] = True
                            # tp_size = fraction of ORIGINAL position (not remaining)
                            partial_size = pos.size * tp_size
                            pnl = partial_size * (pos.entry_price - tp_price) / pos.entry_price
                            commission = partial_size * self.fee_taker
                            pos.realized_pnl += pnl - commission
                            pos.commission_paid += commission
                            pos.remaining_size = max(0.0, pos.remaining_size - tp_size)
                            self._capital += pnl - commission + (partial_size / pos.leverage)

                            self._record_partial(pos, tp_price, bar_idx, ts,
                                               f'TP{tp_idx+1}', pnl - commission)

                    if pos.remaining_size < 0.01:
                        self._positions.remove(pos)
                        exited = True
                        continue

                    # Shorts track the lowest price reached in trailing_high
                    if d_low < pos.trailing_high:
                        pos.trailing_high = d_low

    def _check_exits_bar(self, bar: dict, bar_idx: int):
        """Standard bar-level exit check (no intra-bar detail)."""
        for pos in list(self._positions):
            price_high = bar['high']
            price_low = bar['low']

            # Determine intra-bar price path
            if pos.direction == 1:
                # Long: check SL first (assume low hit first if both hit)
                # Then check TP

                # Liquidation check
                liq_price = pos.entry_price * (1 - 1.0 / pos.leverage * 0.9)
                if price_low <= liq_price:
                    self._close_position(pos, liq_price, bar_idx,
                                        bar.get('timestamp', bar_idx), 'liquidation')
                    continue

                # Stop Loss (or trailing stop)
                if price_low <= pos.sl_price:
                    reason = 'trailing' if pos.trailing_activated else 'SL'
                    self._close_position(pos, pos.sl_price, bar_idx,
                                        bar.get('timestamp', bar_idx), reason)
                    continue

                # Break-even: check against bar high directly
                if (pos.be_trigger > 0 and not pos.be_activated
                        and price_high >= pos.be_trigger):
                    pos.be_activated = True
                    pos.sl_price = pos.entry_price

                # Take Profits (check each level)
                for tp_idx, (tp_price, tp_size, tp_hit) in enumerate(
                        zip(pos.tp_levels, pos.tp_sizes, pos.tp_hit)):
                    if not tp_hit and price_high >= tp_price:
                        pos.tp_hit[tp_idx] = True
                        # tp_size = fraction of ORIGINAL position (not remaining)
                        partial_size = pos.size * tp_size
                        pnl = partial_size * (tp_price - pos.entry_price) / pos.entry_price
                        commission = partial_size * self.fee_taker
                        pos.realized_pnl += pnl - commission
                        pos.commission_paid += commission
                        pos.remaining_size = max(0.0, pos.remaining_size - tp_size)

                        self._capital += pnl - commission + (partial_size / pos.leverage)

                        # Record partial close trade
                        self._record_partial(pos, tp_price, bar_idx,
                                           bar.get('timestamp', bar_idx),
                                           f'TP{tp_idx+1}', pnl - commission)

                # Check if fully closed
                if pos.remaining_size < 0.01:
                    self._positions.remove(pos)
                    continue

                # Update trailing high
                if price_high > pos.trailing_high:
                    pos.trailing_high = price_high

            else:
                # Short: mirror logic
                liq_price = pos.entry_price * (1 + 1.0 / pos.leverage * 0.9)
                if price_high >= liq_price:
                    self._close_position(pos, liq_price, bar_idx,
                                        bar.get('timestamp', bar_idx), 'liquidation')
                    continue

                if price_high >= pos.sl_price:
                    reason = 'trailing' if pos.trailing_activated else 'SL'
                    self._close_position(pos, pos.sl_price, bar_idx,
                                        bar.get('timestamp', bar_idx), reason)
                    continue

                # Break-even: check against bar low directly
                if (pos.be_trigger > 0 and not pos.be_activated
                        and price_low <= pos.be_trigger):
                    pos.be_activated = True
                    pos.sl_price = pos.entry_price

                for tp_idx, (tp_price, tp_size, tp_hit) in enumerate(
                        zip(pos.tp_levels, pos.tp_sizes, pos.tp_hit)):
                    if not tp_hit and price_low <= tp_price:
                        pos.tp_hit[tp_idx] = True
                        # tp_size = fraction of ORIGINAL position (not remaining)
                        partial_size = pos.size * tp_size
                        pnl = partial_size * (pos.entry_price - tp_price) / pos.entry_price
                        commission = partial_size * self.fee_taker
                        pos.realized_pnl += pnl - commission
                        pos.commission_paid += commission
                        pos.remaining_size = max(0.0, pos.remaining_size - tp_size)
                        self._capital += pnl - commission + (partial_size / pos.leverage)

                        self._record_partial(pos, tp_price, bar_idx,
                                           bar.get('timestamp', bar_idx),
                                           f'TP{tp_idx+1}', pnl - commission)

                if pos.remaining_size < 0.01:
                    self._positions.remove(pos)
                    continue

                if price_low < pos.trailing_high:
                    pos.trailing_high = price_low

    def _update_positions(self, bar: dict, bar_idx: int,
                          strategy: IStrategy, data: dict):
        """Update trailing stops, break-even, etc."""
        for pos in self._positions:
            if not pos.trailing:
                continue

            if pos.direction == 1:
                # Long: trail SL up
                if pos.be_activated and pos.trailing_high > pos.entry_price:
                    new_sl = pos.trailing_high - pos.trailing_distance
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_activated = True
            else:
                # Short: trail SL down
                if pos.be_activated and pos.trailing_high < pos.entry_price:
                    new_sl = pos.trailing_high + pos.trailing_distance
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl
                        pos.trailing_activated = True

    def _close_position(self, pos: Position, exit_price: float,
                        bar_idx: int, timestamp: float, reason: str):
        """Fully close a position."""
        remaining_notional = pos.size * pos.remaining_size

        if pos.direction == 1:
            pnl = remaining_notional * (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = remaining_notional * (pos.entry_price - exit_price) / pos.entry_price

        commission = remaining_notional * self.fee_taker
        net_pnl = pnl - commission + pos.realized_pnl

        # Return margin + PnL
        self._capital += (remaining_notional / pos.leverage) + pnl - commission

        pnl_pct = net_pnl / pos.margin * 100 if pos.margin > 0 else 0.0

        trade = Trade(
            id=pos.id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_bar=pos.entry_bar,
            exit_bar=bar_idx,
            entry_time=pos.entry_timestamp,
            exit_time=timestamp,
            size=pos.size,
            leverage=pos.leverage,
            pnl=pnl + pos.realized_pnl,
            pnl_pct=pnl_pct,
            commission=pos.commission_paid + commission,
            funding=pos.funding_paid,
            net_pnl=net_pnl - pos.funding_paid,
            exit_reason=reason,
            duration_bars=bar_idx - pos.entry_bar,
            confidence=pos.confidence,
            max_favorable=0.0,
            max_adverse=0.0,
            metadata=pos.metadata,
        )

        self._trades.append(trade)

        if pos in self._positions:
            self._positions.remove(pos)

    def _record_partial(self, pos: Position, price: float,
                        bar_idx: int, timestamp: float,
                        reason: str, pnl: float):
        """Record a partial close as a sub-trade."""
        # Partial closes are tracked in the position's realized_pnl
        # The final close will include the full accounting
        pass

    def _apply_funding(self, bar: dict):
        """Apply funding rate to all open positions."""
        for pos in self._positions:
            notional = pos.size * pos.remaining_size
            funding = notional * self.funding_rate
            # Long pays funding, short receives (typical)
            if pos.direction == 1:
                pos.funding_paid += funding
                self._capital -= funding
            else:
                pos.funding_paid -= funding
                self._capital += funding

    def _calculate_equity(self, bar: dict) -> float:
        """Calculate total equity (capital + unrealized PnL)."""
        equity = self._capital
        for pos in self._positions:
            notional = pos.size * pos.remaining_size
            if pos.direction == 1:
                upnl = notional * (bar['close'] - pos.entry_price) / pos.entry_price
            else:
                upnl = notional * (pos.entry_price - bar['close']) / pos.entry_price
            equity += upnl + pos.margin * pos.remaining_size
        return equity

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity."""
        if not self._equity:
            return 0.0
        peak = max(self._equity)
        current = self._equity[-1]
        return (peak - current) / peak * 100 if peak > 0 else 0.0

    # ─── RESULTS COMPILATION ───

    def _compile_results(self, strategy: IStrategy, data: dict,
                         symbol: str, timeframe: str) -> BacktestResult:
        """Compile all results into BacktestResult."""
        equity = np.array(self._equity)
        n = len(equity)

        if n == 0:
            return BacktestResult(
                trades=self._trades,
                equity_curve=equity,
                drawdown_curve=np.array([]),
                strategy_name=strategy.name(),
                symbol=symbol,
                timeframe=timeframe,
            )

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        drawdown[peak == 0] = 0

        # Returns
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        # Daily returns for Sharpe/Sortino
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        # Sharpe (annualized — factor depends on timeframe)
        # Each return is per-bar, so bars_per_year = seconds_in_year / seconds_per_bar
        tf_seconds_bar = TIMEFRAME_SECONDS.get(timeframe, 14400)
        bars_per_year = (365.25 * 86400) / tf_seconds_bar
        ann_factor = np.sqrt(bars_per_year)

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * ann_factor
        else:
            sharpe = 0.0

        # Sortino
        neg_returns = returns[returns < 0]
        if len(neg_returns) > 1 and np.std(neg_returns) > 0:
            sortino = np.mean(returns) / np.std(neg_returns) * ann_factor
        else:
            sortino = 0.0

        # Trade stats
        trades = self._trades
        n_trades = len(trades)

        if n_trades > 0:
            wins = [t for t in trades if t.net_pnl > 0]
            losses = [t for t in trades if t.net_pnl <= 0]

            win_rate = len(wins) / n_trades * 100
            avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0.0
            avg_loss = np.mean([abs(t.net_pnl) for t in losses]) if losses else 0.0

            gross_profit = sum(t.net_pnl for t in wins)
            gross_loss = abs(sum(t.net_pnl for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

            avg_duration = np.mean([t.duration_bars for t in trades])
            n_longs = sum(1 for t in trades if t.direction == 1)
            n_shorts = sum(1 for t in trades if t.direction == -1)
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_duration = 0.0
            n_longs = n_shorts = 0

        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Annualized return estimate
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 14400)
        total_seconds = n * tf_seconds
        years = total_seconds / (365.25 * 86400)
        annual_return = total_return / years if years > 0 else 0.0

        calmar = annual_return / max_dd if max_dd > 0 else 0.0

        # Dates
        start_date = ""
        end_date = ""
        if 'timestamp' in data and len(data['timestamp']) > 0:
            start_date = datetime.fromtimestamp(
                data['timestamp'][0] / 1000).strftime('%Y-%m-%d')
            end_date = datetime.fromtimestamp(
                data['timestamp'][-1] / 1000).strftime('%Y-%m-%d')

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            drawdown_curve=drawdown,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            n_trades=n_trades,
            n_longs=n_longs,
            n_shorts=n_shorts,
            avg_duration=avg_duration,
            strategy_name=strategy.name(),
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            params=strategy.params,
            annualization_factor=ann_factor,
        )


# ═══════════════════════════════════════════════════════════════
#  TIMEFRAME DETAIL (Phase 2)
# ═══════════════════════════════════════════════════════════════

class TimeframeDetail:
    """
    Provides intra-bar price paths for accurate SL/TP simulation
    when backtesting on higher timeframes.

    Example: Backtesting on 4H with 5m detail data.
    The signal fires on the 4H close, but SL/TP are checked
    against every 5m candle within the next 4H bar.
    """

    DETAIL_MAP = {
        '15m': '1m',
        '4h': '5m',
        '1h': '1m',
        '1d': '15m',
        '1w': '1h',
    }

    def __init__(self, main_timestamps: np.ndarray,
                 detail_data: dict,
                 main_tf: str):
        self.main_timestamps = main_timestamps
        self.detail_timestamps = detail_data['timestamp']
        self.detail_high = detail_data['high']
        self.detail_low = detail_data['low']
        self.detail_close = detail_data['close']
        self.detail_open = detail_data['open']
        self.main_tf = main_tf

        main_ms = TIMEFRAME_SECONDS.get(main_tf, 14400) * 1000
        self._main_ms = main_ms

    def get_intrabar_candles(self, main_bar_idx: int) -> dict:
        """
        Get all detail candles within a main timeframe bar.
        Returns dict with open, high, low, close arrays.
        """
        if main_bar_idx >= len(self.main_timestamps) - 1:
            return None

        start_ts = self.main_timestamps[main_bar_idx]
        end_ts = self.main_timestamps[main_bar_idx + 1]

        mask = ((self.detail_timestamps >= start_ts) &
                (self.detail_timestamps < end_ts))

        if not np.any(mask):
            return None

        return {
            'open': self.detail_open[mask],
            'high': self.detail_high[mask],
            'low': self.detail_low[mask],
            'close': self.detail_close[mask],
        }


# ═══════════════════════════════════════════════════════════════
#  RESULT FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_result(result: BacktestResult) -> str:
    """Format backtest result as a readable report string."""
    lines = []
    lines.append("═" * 60)
    lines.append(f"  CryptoLab Report — {result.strategy_name}")
    lines.append(f"  {result.symbol} | {result.timeframe} | {result.start_date} → {result.end_date}")
    lines.append("═" * 60)
    lines.append("")
    lines.append("  PERFORMANCE SUMMARY")
    lines.append("─" * 60)
    lines.append(f"  Total Return:     {result.total_return:+.1f}%          Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    lines.append(f"  Annual Return:    {result.annual_return:+.1f}%          Sortino Ratio: {result.sortino_ratio:.2f}")
    lines.append(f"  Max Drawdown:     {result.max_drawdown:.1f}%           Calmar Ratio:  {result.calmar_ratio:.2f}")
    lines.append(f"  Win Rate:         {result.win_rate:.1f}%           Profit Factor: {result.profit_factor:.2f}")
    lines.append(f"  Avg Win:          {result.avg_win:+.2f}          Avg Loss:      {result.avg_loss:.2f}")
    lines.append(f"  Total Trades:     {result.n_trades}              Longs: {result.n_longs}  Shorts: {result.n_shorts}")
    lines.append(f"  Avg Duration:     {result.avg_duration:.0f} bars")
    lines.append("")

    if result.trades:
        lines.append("  LAST 10 TRADES")
        lines.append("─" * 60)
        lines.append(f"  {'Dir':>5} {'Entry':>10} {'Exit':>10} {'PnL%':>8} {'Reason':>10}")
        for t in result.trades[-10:]:
            d = "LONG" if t.direction == 1 else "SHORT"
            lines.append(f"  {d:>5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.pnl_pct:>+7.1f}% {t.exit_reason:>10}")

    lines.append("═" * 60)
    return "\n".join(lines)


def result_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert trades to DataFrame for analysis."""
    if not result.trades:
        return pd.DataFrame()

    records = []
    for t in result.trades:
        records.append({
            'id': t.id,
            'direction': 'LONG' if t.direction == 1 else 'SHORT',
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'entry_bar': t.entry_bar,
            'exit_bar': t.exit_bar,
            'size': t.size,
            'leverage': t.leverage,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'commission': t.commission,
            'funding': t.funding,
            'net_pnl': t.net_pnl,
            'exit_reason': t.exit_reason,
            'duration_bars': t.duration_bars,
            'confidence': t.confidence,
        })

    return pd.DataFrame(records)