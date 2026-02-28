"""
CryptoLab — Data Manager
Central data management: download, cache, validate, and serve OHLCV data.

Features:
- Download with user-specified date ranges (--start / --end)
- Intelligent caching: only downloads missing ranges (gap-fill)
- Progress reporting with ETA
- Multi-symbol batch download
- Data integrity validation (gaps, duplicates, OHLC sanity)
- Cache inspection and management
- Automatic warmup bars for indicators

Usage:
    from data.data_manager import DataManager
    
    dm = DataManager()
    
    # Download with date range
    df = dm.get_data('BTCUSDT', '4h', start='2023-01-01', end='2025-01-01')
    
    # Check what's cached
    dm.list_cached()
    
    # Get data info
    info = dm.data_info('BTCUSDT', '4h')
"""
import asyncio
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.bitget_client import (
    BitgetClient, DataCache, MarketConfig,
    TIMEFRAME_SECONDS, TIMEFRAME_MAP, generate_sample_data
)


# ═══════════════════════════════════════════════════════════════
#  DATA TYPES
# ═══════════════════════════════════════════════════════════════

@dataclass
class DateRange:
    """A time range with start/end as ISO strings and millisecond timestamps."""
    start: str          # ISO date: '2023-01-01'
    end: str            # ISO date: '2025-01-01'
    start_ms: int = 0
    end_ms: int = 0
    
    def __post_init__(self):
        if self.start_ms == 0:
            self.start_ms = int(datetime.fromisoformat(self.start).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
        if self.end_ms == 0:
            self.end_ms = int(datetime.fromisoformat(self.end).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
    
    @property
    def duration_days(self) -> float:
        return (self.end_ms - self.start_ms) / (86400 * 1000)
    
    def __repr__(self):
        return f"DateRange({self.start} → {self.end}, {self.duration_days:.0f} days)"


@dataclass
class DataInfo:
    """Information about cached data for a symbol/timeframe."""
    symbol: str
    timeframe: str
    bars: int = 0
    first_date: str = ""
    last_date: str = ""
    first_ts: int = 0
    last_ts: int = 0
    file_size_mb: float = 0.0
    gaps: int = 0
    duplicates: int = 0
    exists: bool = False
    
    def summary(self) -> str:
        if not self.exists:
            return f"  {self.symbol} {self.timeframe}: No data cached"
        
        lines = [
            f"  {self.symbol} {self.timeframe}:",
            f"    Bars: {self.bars:,}",
            f"    Range: {self.first_date} → {self.last_date}",
            f"    File: {self.file_size_mb:.2f} MB",
        ]
        if self.gaps > 0:
            lines.append(f"    ⚠ Gaps: {self.gaps}")
        if self.duplicates > 0:
            lines.append(f"    ⚠ Duplicates: {self.duplicates}")
        return "\n".join(lines)


@dataclass
class DownloadProgress:
    """Progress tracker for downloads."""
    symbol: str
    timeframe: str
    total_bars_expected: int = 0
    bars_downloaded: int = 0
    pages_downloaded: int = 0
    pages_total: int = 0
    start_time: float = 0.0
    errors: int = 0
    
    @property
    def pct(self) -> float:
        if self.total_bars_expected == 0:
            return 0.0
        return min(100.0, self.bars_downloaded / self.total_bars_expected * 100)
    
    @property
    def eta_seconds(self) -> float:
        if self.bars_downloaded == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        rate = self.bars_downloaded / elapsed
        remaining = self.total_bars_expected - self.bars_downloaded
        return remaining / rate if rate > 0 else 0.0
    
    def status_line(self) -> str:
        pct = self.pct
        bars_filled = int(pct / 5)
        bar = "█" * bars_filled + "░" * (20 - bars_filled)
        eta = self.eta_seconds
        eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
        return (f"  [{bar}] {pct:5.1f}% | "
                f"{self.bars_downloaded:,}/{self.total_bars_expected:,} bars | "
                f"ETA: {eta_str}")


@dataclass
class ValidationResult:
    """Result of data integrity validation."""
    is_valid: bool = True
    bars: int = 0
    gaps: List[Tuple[str, str]] = field(default_factory=list)
    duplicates: int = 0
    ohlc_violations: int = 0
    null_count: int = 0
    messages: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        status = "✅ VALID" if self.is_valid else "❌ ISSUES FOUND"
        lines = [f"  Data Validation: {status}", f"    Bars: {self.bars:,}"]
        
        if self.gaps:
            lines.append(f"    Gaps: {len(self.gaps)}")
            for s, e in self.gaps[:5]:
                lines.append(f"      {s} → {e}")
            if len(self.gaps) > 5:
                lines.append(f"      ... and {len(self.gaps)-5} more")
        
        if self.duplicates > 0:
            lines.append(f"    Duplicates: {self.duplicates}")
        if self.ohlc_violations > 0:
            lines.append(f"    OHLC violations: {self.ohlc_violations}")
        if self.null_count > 0:
            lines.append(f"    Null values: {self.null_count}")
        
        for msg in self.messages:
            lines.append(f"    ℹ {msg}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  DATA MANAGER
# ═══════════════════════════════════════════════════════════════

class DataManager:
    """
    Central data management for CryptoLab.
    
    Handles downloading, caching, validation, and serving of OHLCV data
    with intelligent gap-filling and user-specified date ranges.
    """
    
    # Detail TF mapping for intra-bar simulation
    DETAIL_TF_MAP = {
        '15m': '1m',   # 15 detail bars per main bar
        '4h': '5m',   # 48 detail bars per main bar
        '2h': '1m',   # 120 detail bars per main bar
        '1h': '5m',   # 60 detail bars per main bar
        '6h': '5m',   # 72 detail bars per main bar
        '12h': '15m', # 48 detail bars per main bar
        '1d': '15m',  # 96 detail bars per main bar
        '1w': '1h',   # 168 detail bars per main bar
    }

    def __init__(self, cache_dir: str = "./data/cache",
                 api_key: str = "", api_secret: str = "",
                 passphrase: str = "", verbose: bool = True):
        self.cache = DataCache(cache_dir)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.verbose = verbose
        self._warmup_bars = 300  # Extra bars before start for indicator warmup

    # ─────────────────────────────────────────────────
    #  PUBLIC API: Get Data (download if needed)
    # ─────────────────────────────────────────────────

    def get_data(self, symbol: str, timeframe: str,
                 start: str = '2023-01-01', end: str = '2025-01-01',
                 warmup: bool = True, validate: bool = True,
                 force_download: bool = False) -> pd.DataFrame:
        """
        Get OHLCV data for the given range. Downloads if not cached.

        Args:
            symbol: Trading pair (e.g. 'BTCUSDT', 'TSLAUSDT')
            timeframe: Candle period (e.g. '1h', '4h', '1d')
            start: Start date ISO format
            end: End date ISO format
            warmup: If True, downloads extra bars before start for indicators
            validate: If True, validates data integrity after load
            force_download: If True, re-downloads even if cached

        Returns:
            DataFrame with columns: timestamp, datetime, open, high, low, close, volume
        """
        date_range = DateRange(start, end)

        # Calculate warmup range
        if warmup:
            tf_sec = TIMEFRAME_SECONDS.get(timeframe, 14400)
            warmup_ms = self._warmup_bars * tf_sec * 1000
            actual_start_ms = date_range.start_ms - warmup_ms
            actual_start = datetime.fromtimestamp(
                actual_start_ms / 1000, tz=timezone.utc
            ).strftime('%Y-%m-%d')
        else:
            actual_start = start

        actual_range = DateRange(actual_start, end)

        # Check cache coverage
        if not force_download:
            missing = self._find_missing_ranges(symbol, timeframe, actual_range)
        else:
            missing = [actual_range]

        # Download missing ranges
        if missing:
            total_bars = sum(
                self._estimate_bars(r, timeframe) for r in missing
            )

            if self.verbose:
                print(f"\n📡 Downloading {symbol} {timeframe}")
                print(f"   Range: {actual_start} → {end}")
                if warmup:
                    print(f"   (includes {self._warmup_bars} warmup bars before {start})")
                print(f"   Expected: ~{total_bars:,} bars to download")
                print()

            for mr in missing:
                self._download_range(symbol, timeframe, mr)

        # Load from cache
        df = self.cache.load(symbol, timeframe, actual_start, end)

        if len(df) == 0 and self.verbose:
            print("⚠ No data available after download attempt.")
            return df

        # Validate
        if validate and len(df) > 0:
            vr = self.validate_data(df, timeframe)
            if not vr.is_valid and self.verbose:
                print(vr.summary())

        if self.verbose:
            print(f"   ✅ {len(df):,} bars loaded ({actual_start} → {end})")

        return df

    def get_data_numpy(self, symbol: str, timeframe: str,
                       start: str = '2023-01-01', end: str = '2025-01-01',
                       **kwargs) -> dict:
        """
        Get data as numpy dict (ready for engine consumption).
        Shortcut for get_data() + to_numpy().
        """
        df = self.get_data(symbol, timeframe, start, end, **kwargs)
        if len(df) == 0:
            return {}
        return self.cache.to_numpy(df)

    def get_detail_data(self, symbol: str, main_tf: str,
                        start: str, end: str,
                        **kwargs) -> Optional[dict]:
        """
        Download and return detail timeframe data for intra-bar simulation.

        For example, if main_tf='4h', downloads '5m' data for the same range.
        This enables accurate SL/TP ordering within each 4h bar.

        Args:
            symbol: Trading pair
            main_tf: Main timeframe (e.g. '4h')
            start: Start date
            end: End date

        Returns:
            Numpy dict of detail data, or None if no detail TF defined
        """
        detail_tf = self.DETAIL_TF_MAP.get(main_tf)
        if detail_tf is None:
            if self.verbose:
                print(f"   ℹ No detail TF for {main_tf} — using bar-level simulation")
            return None

        if self.verbose:
            print(f"   📡 Loading detail data: {detail_tf} (for intra-bar simulation)")

        df = self.get_data(symbol, detail_tf, start, end,
                           warmup=False, validate=False, **kwargs)

        if len(df) == 0:
            return None

        return self.cache.to_numpy(df)

    @staticmethod
    def detail_tf_for(main_tf: str) -> Optional[str]:
        """Return the detail timeframe for a given main timeframe."""
        return DataManager.DETAIL_TF_MAP.get(main_tf)

    # ─────────────────────────────────────────────────
    #  DOWNLOAD: Core download with pagination + progress
    # ─────────────────────────────────────────────────

    def _download_range(self, symbol: str, timeframe: str,
                        date_range: DateRange):
        """Download a specific date range and save to cache."""
        client = BitgetClient(self.api_key, self.api_secret, self.passphrase)

        progress = DownloadProgress(
            symbol=symbol,
            timeframe=timeframe,
            total_bars_expected=self._estimate_bars(date_range, timeframe),
            start_time=time.time(),
        )

        async def _do_download():
            try:
                df = await client.download_ohlcv(
                    symbol, timeframe,
                    date_range.start, date_range.end
                )
                return df
            finally:
                await client.close()

        # Run async download
        df = asyncio.run(_do_download())

        if df is None or len(df) == 0:
            if self.verbose:
                print(f"   ⚠ No candles returned for {date_range}")
            return

        # Save to cache (merges with existing data)
        self.cache.save(df, symbol, timeframe)

        elapsed = time.time() - progress.start_time
        if self.verbose:
            print(f"   ✅ Downloaded {len(df):,} bars in {elapsed:.1f}s")

    # ─────────────────────────────────────────────────
    #  DOWNLOAD: Batch multi-symbol
    # ─────────────────────────────────────────────────

    def download_batch(self, symbols: List[str], timeframe: str,
                       start: str, end: str):
        """
        Download data for multiple symbols.

        Usage:
            dm.download_batch(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], '4h',
                              '2023-01-01', '2025-01-01')
        """
        date_range = DateRange(start, end)

        if self.verbose:
            print(f"\n📡 Batch Download: {len(symbols)} symbols")
            print(f"   Timeframe: {timeframe}")
            print(f"   Range: {start} → {end} ({date_range.duration_days:.0f} days)")
            print()

        for i, symbol in enumerate(symbols, 1):
            if self.verbose:
                print(f"  [{i}/{len(symbols)}] {symbol}")

            try:
                self.get_data(symbol, timeframe, start, end,
                              warmup=True, validate=False)
            except Exception as e:
                print(f"   ❌ Error: {e}")

            if self.verbose:
                print()

    # ─────────────────────────────────────────────────
    #  CACHE: Inspection and management
    # ─────────────────────────────────────────────────

    def data_info(self, symbol: str, timeframe: str) -> DataInfo:
        """Get detailed info about cached data."""
        info = DataInfo(symbol=symbol, timeframe=timeframe)

        path = self.cache._cache_path(symbol, timeframe)
        # Also check alternate format
        pq_path = self.cache.cache_dir / f"{symbol}_{timeframe}.parquet"
        csv_path = self.cache.cache_dir / f"{symbol}_{timeframe}.csv"
        actual_path = pq_path if pq_path.exists() else csv_path if csv_path.exists() else path

        if not actual_path.exists():
            return info

        info.exists = True
        info.file_size_mb = actual_path.stat().st_size / (1024 * 1024)

        df = self.cache.load(symbol, timeframe)
        if len(df) == 0:
            return info

        info.bars = len(df)
        info.first_ts = int(df['timestamp'].iloc[0])
        info.last_ts = int(df['timestamp'].iloc[-1])
        info.first_date = datetime.fromtimestamp(
            info.first_ts / 1000, tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M')
        info.last_date = datetime.fromtimestamp(
            info.last_ts / 1000, tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M')

        # Check for gaps and duplicates
        info.duplicates = int(df['timestamp'].duplicated().sum())

        tf_ms = TIMEFRAME_SECONDS.get(timeframe, 14400) * 1000
        diffs = df['timestamp'].diff().dropna()
        gap_mask = diffs > tf_ms * 1.5  # Allow 50% tolerance
        info.gaps = int(gap_mask.sum())

        return info

    def list_cached(self) -> List[DataInfo]:
        """List all cached datasets."""
        cache_dir = self.cache.cache_dir
        infos = []

        for f in sorted(cache_dir.glob("*.parquet")) + sorted(cache_dir.glob("*.csv")):
            if f.name.endswith("_funding.parquet") or f.name.endswith("_funding.csv"):
                continue

            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue

            symbol, tf = parts
            info = self.data_info(symbol, tf)
            infos.append(info)

        return infos

    def delete_cached(self, symbol: str, timeframe: str) -> bool:
        """Delete cached data for a symbol/timeframe."""
        deleted = False
        for ext in ['.parquet', '.csv']:
            path = self.cache.cache_dir / f"{symbol}_{timeframe}{ext}"
            if path.exists():
                path.unlink()
                deleted = True
        return deleted

    def clear_cache(self):
        """Delete all cached data."""
        cache_dir = self.cache.cache_dir
        for ext in ['*.parquet', '*.csv']:
            for f in cache_dir.glob(ext):
                f.unlink()

    # ─────────────────────────────────────────────────
    #  VALIDATION: Data integrity checks
    # ─────────────────────────────────────────────────

    def validate_data(self, df: pd.DataFrame, timeframe: str) -> ValidationResult:
        """
        Validate data integrity.

        Checks:
        1. No null values in OHLCV
        2. OHLC consistency: high >= max(open, close), low <= min(open, close)
        3. No duplicate timestamps
        4. No unexpected gaps (considering market hours)
        5. Correct chronological order
        """
        vr = ValidationResult(bars=len(df))

        if len(df) == 0:
            vr.is_valid = False
            vr.messages.append("Empty dataset")
            return vr

        # 1. Null check
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                nulls = int(df[col].isnull().sum())
                vr.null_count += nulls

        if vr.null_count > 0:
            vr.is_valid = False
            vr.messages.append(f"{vr.null_count} null values found")

        # 2. OHLC sanity
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            high_ok = df['high'] >= df[['open', 'close']].max(axis=1) - 1e-10
            low_ok = df['low'] <= df[['open', 'close']].min(axis=1) + 1e-10
            violations = int((~high_ok).sum() + (~low_ok).sum())
            vr.ohlc_violations = violations
            if violations > 0:
                vr.messages.append(
                    f"{violations} OHLC violations (high<open/close or low>open/close)")

        # 3. Duplicates
        vr.duplicates = int(df['timestamp'].duplicated().sum())
        if vr.duplicates > 0:
            vr.messages.append(f"{vr.duplicates} duplicate timestamps")

        # 4. Gaps
        tf_ms = TIMEFRAME_SECONDS.get(timeframe, 14400) * 1000
        if len(df) > 1:
            diffs = df['timestamp'].diff().dropna()
            gap_mask = diffs > tf_ms * 2.0  # 2x threshold for gaps
            gap_indices = gap_mask[gap_mask].index

            for idx in gap_indices:
                t1 = datetime.fromtimestamp(
                    df['timestamp'].iloc[idx-1] / 1000, tz=timezone.utc
                ).strftime('%Y-%m-%d %H:%M')
                t2 = datetime.fromtimestamp(
                    df['timestamp'].iloc[idx] / 1000, tz=timezone.utc
                ).strftime('%Y-%m-%d %H:%M')
                vr.gaps.append((t1, t2))

        # 5. Chronological order
        if len(df) > 1:
            if not df['timestamp'].is_monotonic_increasing:
                vr.is_valid = False
                vr.messages.append("Data not in chronological order")

        # Determine overall validity (gaps are warnings, not failures for stocks)
        if vr.duplicates > 0 or vr.null_count > 0:
            vr.is_valid = False

        return vr

    # ─────────────────────────────────────────────────
    #  INTERNAL: Gap detection and range calculation
    # ─────────────────────────────────────────────────

    def _find_missing_ranges(self, symbol: str, timeframe: str,
                             requested: DateRange) -> List[DateRange]:
        """
        Find which sub-ranges need downloading by comparing
        requested range against cached data.
        """
        if not self.cache.has_data(symbol, timeframe):
            return [requested]

        df = self.cache.load(symbol, timeframe)
        if len(df) == 0:
            return [requested]

        cached_start = int(df['timestamp'].iloc[0])
        cached_end = int(df['timestamp'].iloc[-1])
        tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

        missing = []

        # Need data before cached start?
        if requested.start_ms < cached_start - tf_ms:
            missing.append(DateRange(
                start=requested.start,
                end=datetime.fromtimestamp(
                    cached_start / 1000, tz=timezone.utc
                ).strftime('%Y-%m-%d'),
                start_ms=requested.start_ms,
                end_ms=cached_start,
            ))

        # Need data after cached end?
        if requested.end_ms > cached_end + tf_ms:
            missing.append(DateRange(
                start=datetime.fromtimestamp(
                    cached_end / 1000, tz=timezone.utc
                ).strftime('%Y-%m-%d'),
                end=requested.end,
                start_ms=cached_end + tf_ms,
                end_ms=requested.end_ms,
            ))

        return missing

    def _estimate_bars(self, date_range: DateRange, timeframe: str) -> int:
        """Estimate number of bars in a date range."""
        tf_sec = TIMEFRAME_SECONDS.get(timeframe, 14400)
        duration_sec = (date_range.end_ms - date_range.start_ms) / 1000
        return max(1, int(duration_sec / tf_sec))


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def quick_download(symbol: str, timeframe: str,
                   start: str, end: str, **kwargs) -> pd.DataFrame:
    """
    One-liner to download/load data.

    Usage:
        df = quick_download('BTCUSDT', '4h', '2023-01-01', '2025-01-01')
    """
    dm = DataManager(**kwargs)
    return dm.get_data(symbol, timeframe, start, end)


def get_sample_data(n_bars: int = 5000, timeframe: str = '4h',
                    seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    """
    Get sample data as both DataFrame and numpy dict.

    Returns:
        (df, data_dict) ready for backtesting
    """
    df = generate_sample_data(n_bars=n_bars, timeframe=timeframe, seed=seed)
    cache = DataCache()
    data = cache.to_numpy(df)
    data['open'] = df['open'].values.astype(np.float64)
    return df, data


# ═══════════════════════════════════════════════════════════════
#  POPULAR SYMBOLS
# ═══════════════════════════════════════════════════════════════

POPULAR_CRYPTO = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
    'MATICUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
]

POPULAR_STOCKS = [
    'TSLAUSDT', 'NVDAUSDT', 'AAPLUSDT', 'AMZNUSDT', 'GOOGLUSDT',
    'METAUSDT', 'MSFTUSDT',
]

ALL_TIMEFRAMES = list(TIMEFRAME_MAP.keys())