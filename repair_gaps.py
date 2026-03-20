"""
Repair gaps in 1m candle data by forward-filling missing bars.

Usage:
    python repair_gaps.py --symbol DOGEUSDT --tf 1m

What it does:
    1. Loads the cached parquet file
    2. Finds all gaps (missing timestamps)
    3. For each gap: inserts bars with OHLC = previous close, volume = 0
    4. Saves the repaired file back to cache

This is standard practice for crypto data — exchanges have brief
outages or zero-volume periods that create gaps. Forward-filling
with the last close price ensures continuous data without introducing
artificial price movements.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TIMEFRAME_MS = {
    '1m': 60_000,
    '5m': 300_000,
    '15m': 900_000,
    '1h': 3_600_000,
    '4h': 14_400_000,
    '1d': 86_400_000,
}

def repair_gaps(symbol: str, timeframe: str, cache_dir: str = "./data/cache/bitget"):
    """Repair gaps in cached data by forward-filling."""

    cache_path = Path(cache_dir)
    pq_path = cache_path / f"{symbol}_{timeframe}.parquet"
    csv_path = cache_path / f"{symbol}_{timeframe}.csv"

    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        save_parquet = True
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        save_parquet = False
    else:
        print(f"❌ No data found for {symbol} {timeframe}")
        return

    n_before = len(df)
    tf_ms = TIMEFRAME_MS.get(timeframe)
    if tf_ms is None:
        print(f"❌ Unknown timeframe: {timeframe}")
        return

    # Remove duplicates first
    dupes = df['timestamp'].duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        print(f"  Removed {dupes} duplicate timestamps")

    # Find gaps
    diffs = df['timestamp'].diff().dropna()
    gap_mask = diffs > tf_ms * 1.5  # 50% tolerance
    n_gaps = gap_mask.sum()

    if n_gaps == 0:
        print(f"✅ No gaps found in {symbol} {timeframe} ({n_before:,} bars)")
        return

    print(f"🔧 Repairing {symbol} {timeframe}")
    print(f"   Bars before: {n_before:,}")
    print(f"   Gaps found: {n_gaps:,}")

    # Build complete timestamp range
    ts_start = int(df['timestamp'].iloc[0])
    ts_end = int(df['timestamp'].iloc[-1])

    full_ts = np.arange(ts_start, ts_end + tf_ms, tf_ms, dtype=np.int64)

    # Create full DataFrame with all timestamps
    full_df = pd.DataFrame({'timestamp': full_ts})

    # Merge with existing data
    merged = full_df.merge(df, on='timestamp', how='left')

    # Forward-fill OHLC (use previous close for O/H/L/C of missing bars)
    # First, forward-fill close
    merged['close'] = merged['close'].ffill()

    # For missing bars: O=H=L=C = previous close, volume=0
    missing_mask = merged['open'].isna()
    merged.loc[missing_mask, 'open'] = merged.loc[missing_mask, 'close']
    merged.loc[missing_mask, 'high'] = merged.loc[missing_mask, 'close']
    merged.loc[missing_mask, 'low'] = merged.loc[missing_mask, 'close']
    merged.loc[missing_mask, 'volume'] = 0.0

    # Fill datetime column if it exists
    if 'datetime' in merged.columns:
        merged['datetime'] = pd.to_datetime(merged['timestamp'], unit='ms', utc=True)

    # Fill any remaining NaN in volume
    merged['volume'] = merged['volume'].fillna(0.0)

    # Ensure correct dtypes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        merged[col] = merged[col].astype(np.float64)

    n_filled = missing_mask.sum()
    n_after = len(merged)

    print(f"   Bars filled: {n_filled:,}")
    print(f"   Bars after: {n_after:,}")

    # Verify no gaps remain
    diffs_after = merged['timestamp'].diff().dropna()
    gaps_after = (diffs_after > tf_ms * 1.5).sum()
    print(f"   Gaps remaining: {gaps_after}")

    # Save
    if save_parquet:
        # Backup original
        backup_path = cache_path / f"{symbol}_{timeframe}_backup.parquet"
        if not backup_path.exists():
            import shutil
            shutil.copy2(pq_path, backup_path)
            print(f"   Backup saved: {backup_path.name}")

        merged.to_parquet(pq_path, index=False, compression='snappy')
        print(f"   ✅ Saved: {pq_path.name} ({n_after:,} bars)")
    else:
        backup_path = cache_path / f"{symbol}_{timeframe}_backup.csv"
        if not backup_path.exists():
            import shutil
            shutil.copy2(csv_path, backup_path)
        merged.to_csv(csv_path, index=False)
        print(f"   ✅ Saved: {csv_path.name}")

    # Show top 5 largest gaps that were filled
    if n_filled > 0:
        gap_indices = gap_mask[gap_mask].index
        gap_sizes = []
        for idx in gap_indices:
            t1 = df['timestamp'].iloc[idx-1]
            t2 = df['timestamp'].iloc[idx]
            gap_bars = int((t2 - t1) / tf_ms) - 1
            t1_str = datetime.fromtimestamp(t1/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            t2_str = datetime.fromtimestamp(t2/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
            gap_sizes.append((gap_bars, t1_str, t2_str))

        gap_sizes.sort(reverse=True)
        print(f"\n   Top 5 largest gaps filled:")
        for bars, t1, t2 in gap_sizes[:5]:
            print(f"     {t1} → {t2} ({bars} bars)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Repair gaps in candle data")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--tf", default="1m")
    parser.add_argument("--cache-dir", default="./data/cache")
    args = parser.parse_args()

    repair_gaps(args.symbol, args.tf, args.cache_dir)