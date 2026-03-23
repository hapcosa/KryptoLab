"""
CryptoLab — Intrabar Backtesting Engine v2
==========================================
Tick-accurate signal generation by rebuilding partial higher-timeframe
bars from 1m data — mirrors TradingView's real-time indicator evaluation.

Architecture
------------
For each main TF bar (e.g. 1h), we iterate its constituent 1m bars and
maintain a running partial OHLCV:

    1m[0]: partial = {open=1h_open, high=m0.high, low=m0.low, close=m0.close}
    1m[1]: partial = {open=1h_open, high=max(m0,m1), low=min(m0,m1), close=m1.close}
    ...
    1m[59]: partial = closed 1h bar

At each 1m bar:
  - Exits (SL/TP/trailing/liquidation) are checked on the raw 1m bar.
  - IncrementalCyberCycle.update_partial() is called with the PARTIAL 1h
    OHLCV, not the raw 1m values.  This is the key difference from any
    prior implementation — the CyberCycle sees a 1h bar in progress, not
    an isolated 1m bar.
  - If a crossover fires, Signal.entry_price = current 1m close.

Routing
-------
  - Strategy has create_incremental_processor() AND detail data loaded
    → partial bar mode (this file's _run_partial_bar)
  - Otherwise → standard bar-close engine (super().run())

This means --no-detail still works exactly as before.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from core.engine import BacktestEngine, BacktestResult
from strategies.base import IStrategy
from data.bitget_client import TIMEFRAME_SECONDS


class IntrabarBacktestEngine(BacktestEngine):
    """
    Backtest engine with partial-bar signal timing.

    Inherits all position management, exit logic, SL/TP/trailing/
    break-even/liquidation and result compilation from BacktestEngine.
    Only signal generation is changed: signals fire at the 1m close
    where the CyberCycle first crosses the trigger, not at the 1h close.
    """

    def run(self, strategy: IStrategy, data: dict,
            symbol: str = "", timeframe: str = "",
            callback=None) -> BacktestResult:

        processor = None
        if hasattr(strategy, 'create_incremental_processor'):
            main_tf_sec   = TIMEFRAME_SECONDS.get(timeframe, 3600)
            detail_tf_sec = TIMEFRAME_SECONDS.get(
                getattr(self, '_detail_tf', '1m'), 60)
            ratio = max(1, main_tf_sec // detail_tf_sec)
            processor = strategy.create_incremental_processor(
                detail_tf_ratio=ratio)

        if processor is not None and self._detail_data is not None:
            return self._run_partial_bar(
                processor, strategy, data, symbol, timeframe, callback)

        return super().run(strategy, data, symbol, timeframe, callback)

    # ══════════════════════════════════════════════════════════════
    #  PARTIAL BAR EXECUTION
    # ══════════════════════════════════════════════════════════════

    def _run_partial_bar(self, processor, strategy: IStrategy,
                         main_data: dict, symbol: str, timeframe: str,
                         callback=None) -> BacktestResult:
        """
        Backtest loop that rebuilds partial 1h bars from 1m data.

        Signal timing: fires on the exact 1m bar where the CyberCycle
        first crosses the trigger threshold within a 1h bar.
        Entry price: the 1m close at the moment of crossing.
        Exit timing: SL/TP checked on every subsequent 1m bar.
        """
        # ── Reset engine state ──
        self._capital   = self.initial_capital
        self._positions = []
        self._trades    = []
        self._equity    = []
        self._trade_counter      = 0
        self._daily_signal_count = {}

        detail      = self._detail_data
        main_ts     = main_data['timestamp']
        n_main      = len(main_data['close'])
        n_detail    = len(detail['close'])

        if n_detail == 0 or n_main == 0:
            return self._empty_result(strategy, symbol, timeframe)

        detail_tf_str = getattr(self, '_detail_tf', '1m')
        tf_sec        = TIMEFRAME_SECONDS.get(detail_tf_str, 60)
        main_tf_sec   = TIMEFRAME_SECONDS.get(timeframe, 3600)

        funding_interval_bars = max(1, (self.funding_interval_hours * 3600) // tf_sec)
        equity_sample         = max(1, n_detail // 3000)

        # ── Map each detail bar → its main bar index ──
        # main bar i spans [main_ts[i], main_ts[i+1])
        main_end     = np.empty(n_main, dtype=np.int64)
        main_end[:-1] = main_ts[1:]
        main_end[-1]  = main_ts[-1] + main_tf_sec * 1000

        detail_ts       = detail['timestamp']
        main_bar_idx    = np.clip(
            np.searchsorted(main_ts, detail_ts, side='right') - 1,
            0, n_main - 1
        )

        # Number of detail bars per main bar
        bar_counts = np.bincount(main_bar_idx, minlength=n_main)

        # ── Main loop ──
        current_main = -1
        d_in_bar     = 0     # position within current main bar
        p_high = p_low = p_vol = 0.0   # rolling partial 1h OHLCV accumulators
        n_in_bar     = 0     # total 1m bars in the current 1h bar

        for d_idx in range(n_detail):
            m_idx = int(main_bar_idx[d_idx])

            # ── Entering a new 1h bar ──
            if m_idx != current_main:
                current_main = m_idx
                d_in_bar     = 0
                p_high       = float(detail['high'][d_idx])
                p_low        = float(detail['low'][d_idx])
                p_vol        = float(detail['volume'][d_idx])
                n_in_bar     = int(bar_counts[m_idx])
            else:
                d_in_bar += 1
                p_high    = max(p_high, float(detail['high'][d_idx]))
                p_low     = min(p_low,  float(detail['low'][d_idx]))
                p_vol    += float(detail['volume'][d_idx])

            p_close   = float(detail['close'][d_idx])
            d_ts      = int(detail['timestamp'][d_idx])
            is_close  = (d_in_bar == n_in_bar - 1)

            # Current raw 1m bar (used for exit checks and position updates)
            bar_1m = {
                'open':      float(detail['open'][d_idx]),
                'high':      float(detail['high'][d_idx]),
                'low':       float(detail['low'][d_idx]),
                'close':     p_close,
                'volume':    float(detail['volume'][d_idx]),
                'timestamp': d_ts,
            }

            # 1. Check exits on raw 1m bar
            self._check_exits_bar(bar_1m, d_idx)

            # 2. Funding rate
            if d_idx > 0 and d_idx % funding_interval_bars == 0:
                self._apply_funding(bar_1m)

            # 3. Generate signal from PARTIAL 1h OHLCV
            #    ┌─────────────────────────────────────────────────┐
            #    │  Key: CyberCycle sees (p_high, p_low, p_close)  │
            #    │  = the 1h bar as it looks right now, not the    │
            #    │  isolated 1m bar values.                        │
            #    └─────────────────────────────────────────────────┘
            signal = processor.update_partial(
                high=p_high,
                low=p_low,
                close=p_close,  # ← "latest tick" of the 1h bar
                volume=p_vol,
                timestamp=d_ts,
                is_bar_close=is_close,
            )
            if signal is not None and self._signal_start_ts > 0 and d_ts < self._signal_start_ts:
                signal = None

            # 4. Apply daily signal cap (engine-level, redundant safety)
            if signal is not None:
                max_daily = signal.metadata.get('max_signals_per_day', 0)
                if max_daily > 0:
                    if d_ts > 1e9:
                        epoch_s = d_ts/1000 if d_ts > 1e12 else d_ts
                        dt      = datetime.utcfromtimestamp(epoch_s)
                        day_key = (dt.year, dt.month, dt.day)
                    else:
                        day_key = d_idx // (86400 // tf_sec)
                    cnt = self._daily_signal_count.get(day_key, 0)
                    if cnt >= max_daily:
                        signal = None
                    else:
                        self._daily_signal_count[day_key] = cnt + 1

            # 5. Handle signal
            if signal is not None:
                if len(self._positions) < self.max_positions:
                    self._open_position(signal, bar_1m, d_idx)
                elif signal.metadata.get('close_on_signal', False):
                    for pos in list(self._positions):
                        if pos.direction != signal.direction:
                            self._close_position(
                                pos, p_close, d_idx, d_ts, 'signal')
                    if len(self._positions) < self.max_positions:
                        self._open_position(signal, bar_1m, d_idx)

            # 6. Trailing / break-even updates
            self._update_positions(bar_1m, d_idx, strategy, detail)

            # 7. Record equity (sampled)
            if d_idx % equity_sample == 0:
                self._equity.append(self._calculate_equity(bar_1m))

            # 8. Progress callback for optimizer
            if callback and d_idx % 500 == 0:
                callback({
                    'bar':       d_idx,
                    'equity':    self._calculate_equity(bar_1m),
                    'n_trades':  len(self._trades),
                    'drawdown':  self._current_drawdown(),
                })

        # ── Close any open positions at end of data ──
        if self._positions and n_detail > 0:
            last = {
                'open':      float(detail['open'][-1]),
                'high':      float(detail['high'][-1]),
                'low':       float(detail['low'][-1]),
                'close':     float(detail['close'][-1]),
                'volume':    float(detail['volume'][-1]),
                'timestamp': int(detail['timestamp'][-1]),
            }
            for pos in list(self._positions):
                self._close_position(
                    pos, last['close'], n_detail-1, last['timestamp'],
                    'end_of_data')

        # ── Compile using main TF bars for annualization stats ──
        return self._compile_results(strategy, main_data, symbol, timeframe)

    def _empty_result(self, strategy, symbol, timeframe) -> BacktestResult:
        return BacktestResult(
            trades=[], equity_curve=np.array([self.initial_capital]),
            drawdown_curve=np.array([0.0]), total_return=0.0,
            strategy_name=strategy.name(), symbol=symbol,
            timeframe=timeframe, params=strategy.params,
        )