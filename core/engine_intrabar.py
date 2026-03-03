"""
CryptoLab — Intrabar Backtesting Engine
========================================
Subclass of BacktestEngine that adds intrabar signal execution.

Strategies that return an incremental processor via
create_incremental_processor() are iterated over detail bars
(e.g. 5m candles within 4H), evaluating signals on every sub-bar.

Strategies without a processor (GaussBands, SmartMoney) fall through
to the standard bar-close engine unchanged.

Usage:
    # In cli.py, replace BacktestEngine with IntrabarBacktestEngine
    # in _make_engine_factory():

    from core.engine_intrabar import IntrabarBacktestEngine

    engine = IntrabarBacktestEngine(initial_capital=capital,
                                     market_config=market_config)
    engine.set_detail_data(detail_data, detail_tf)
    result = engine.run(strategy, data, symbol, timeframe)

    # GaussBands → standard bar-close path (no change)
    # CyberCycle → intrabar path (signals on 5m bars)
"""

import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

from core.engine import (
    BacktestEngine, BacktestResult, Position, Trade, TimeframeDetail
)
from strategies.base import IStrategy, Signal
from data.bitget_client import TIMEFRAME_SECONDS


class IntrabarBacktestEngine(BacktestEngine):
    """
    Extended backtest engine with intrabar signal execution.

    Routing logic:
    1. If strategy provides create_incremental_processor() AND detail data exists:
       → _run_intrabar(): iterate detail bars, processor generates signals
    2. Otherwise:
       → super().run(): standard bar-close execution (identical to current)
    """

    def run(self, strategy: IStrategy, data: dict,
            symbol: str = "", timeframe: str = "",
            callback=None) -> BacktestResult:
        """
        Route to intrabar or bar-close depending on strategy.
        """
        processor = None
        if hasattr(strategy, 'create_incremental_processor'):
            processor = strategy.create_incremental_processor()

        if processor is not None and self._detail_data is not None:
            return self._run_intrabar(
                processor, strategy, data, symbol, timeframe, callback
            )

        # Standard bar-close path — unchanged
        return super().run(strategy, data, symbol, timeframe, callback)

    # ═════════════════════════════════════════════════════════
    #  INTRABAR EXECUTION
    # ═════════════════════════════════════════════════════════

    def _run_intrabar(self, processor, strategy: IStrategy,
                      main_data: dict, symbol: str, timeframe: str,
                      callback=None) -> BacktestResult:
        """
        Run backtest with signals evaluated on detail bars.

        The processor (IncrementalCyberCycle) is called on every
        detail bar. When it returns a Signal, the engine opens a
        position at that detail bar's close — exactly as TradingView
        would execute on a live chart.

        SL/TP/trailing are checked on the same detail bars,
        providing natural intrabar resolution without extra cost.
        """
        # ── Reset engine state ──
        self._capital = self.initial_capital
        self._positions = []
        self._trades = []
        self._equity = []
        self._trade_counter = 0
        self._daily_signal_count = {}

        detail = self._detail_data
        n_detail = len(detail['close'])

        if n_detail == 0:
            return self._empty_result(strategy, symbol, timeframe)

        # We do NOT build TimeframeDetail here — we're already on
        # detail resolution. Exits use _check_exits_bar directly.
        self._tf_detail = None

        # Determine detail timeframe for annualization
        detail_tf = getattr(self, '_detail_tf', '5m')
        tf_seconds = TIMEFRAME_SECONDS.get(detail_tf, 300)
        funding_interval_bars = max(
            1, (self.funding_interval_hours * 3600) // tf_seconds
        )

        # Equity curve sampling: record every N detail bars to keep
        # the array manageable (~2000-4000 points regardless of
        # detail resolution).
        equity_sample = max(1, n_detail // 3000)

        # ── Main loop over detail bars ──
        for i in range(n_detail):
            bar = {
                'open': detail['open'][i],
                'high': detail['high'][i],
                'low': detail['low'][i],
                'close': detail['close'][i],
                'volume': detail['volume'][i],
                'timestamp': detail['timestamp'][i],
            }

            # 1. Check exits (SL/TP/trailing/liquidation)
            #    Uses bar-level check since we're already on detail bars.
            self._check_exits_bar(bar, i)

            # 2. Apply funding rate
            if i > 0 and i % funding_interval_bars == 0:
                self._apply_funding(bar)

            # 3. Generate signal via incremental processor
            signal = processor.update(
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar['volume'],
                timestamp=int(bar['timestamp']),
            )

            # 4. Handle signal (same logic as standard engine)
            if signal is not None:
                # Daily signal limit (engine-level, redundant with
                # processor but kept for consistency)
                max_daily = signal.metadata.get('max_signals_per_day', 0)
                if max_daily > 0:
                    ts = bar['timestamp']
                    if ts > 1e9:
                        dt = datetime.utcfromtimestamp(
                            ts / 1000 if ts > 1e12 else ts
                        )
                        day_key = (dt.year, dt.month, dt.day)
                    else:
                        day_key = i // (86400 // tf_seconds)
                    count = self._daily_signal_count.get(day_key, 0)
                    if count >= max_daily:
                        signal = None
                    else:
                        self._daily_signal_count[day_key] = count + 1

            if signal is not None:
                if len(self._positions) < self.max_positions:
                    self._open_position(signal, bar, i)
                elif signal.metadata.get('close_on_signal', False):
                    for pos in list(self._positions):
                        if pos.direction != signal.direction:
                            ts = bar.get('timestamp', i)
                            self._close_position(
                                pos, bar['close'], i, ts, 'signal'
                            )
                    if len(self._positions) < self.max_positions:
                        self._open_position(signal, bar, i)

            # 5. Update position state (trailing, etc.)
            self._update_positions(bar, i, strategy, detail)

            # 6. Record equity (sampled)
            if i % equity_sample == 0:
                equity = self._calculate_equity(bar)
                self._equity.append(equity)

            # 7. Callback for optimizer (less frequent to reduce overhead)
            if callback and i % 500 == 0:
                callback({
                    'bar': i,
                    'equity': self._calculate_equity(bar),
                    'n_trades': len(self._trades),
                    'drawdown': self._current_drawdown(),
                })

        # ── Close remaining positions at last bar ──
        if self._positions and n_detail > 0:
            last_bar = {
                'open': detail['open'][-1],
                'high': detail['high'][-1],
                'low': detail['low'][-1],
                'close': detail['close'][-1],
                'volume': detail['volume'][-1],
                'timestamp': detail['timestamp'][-1],
            }
            for pos in list(self._positions):
                self._close_position(
                    pos, last_bar['close'], n_detail - 1,
                    last_bar['timestamp'], 'end_of_data'
                )

        # ── Compile results (using detail_tf for proper annualization) ──
        return self._compile_results(
            strategy, detail, symbol, detail_tf
        )

    def _empty_result(self, strategy, symbol, timeframe) -> BacktestResult:
        """Return empty result when no data is available."""
        return BacktestResult(
            trades=[],
            equity_curve=np.array([self.initial_capital]),
            drawdown_curve=np.array([0.0]),
            total_return=0.0,
            strategy_name=strategy.name(),
            symbol=symbol,
            timeframe=timeframe,
            params=strategy.params,
        )