"""
CryptoLab — Parallel Batch Download
Replaces sequential download_batch() with asyncio.gather() + semaphore.

Usage:
    Replace in data/data_manager.py:

    # OLD (sequential):
    def download_batch(self, symbols, timeframe, start, end):
        for symbol in symbols:
            self.get_data(symbol, timeframe, start, end)

    # NEW (parallel):
    from data.download_parallel import download_batch_parallel

    def download_batch(self, symbols, timeframe, start, end):
        download_batch_parallel(
            symbols, timeframe, start, end,
            cache_dir=self.cache.cache_dir,
            max_concurrent=4,     # Symbols downloading simultaneously
            rate_limit=15,        # Max API requests per second (global)
            verbose=self.verbose,
        )

    Or call directly from cli.py in cmd_download().
"""
import asyncio
import time
from typing import List, Optional
from pathlib import Path


async def _download_symbol(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: 'AsyncRateLimiter',
    cache_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Download one symbol with rate limiting.
    Returns dict with status info.
    """
    from data.bitget_client import BitgetClient, DataCache, TIMEFRAME_SECONDS

    t0 = time.time()
    result = {
        'symbol': symbol,
        'bars': 0,
        'elapsed': 0,
        'error': None,
    }

    async with semaphore:
        try:
            cache = DataCache(cache_dir)

            # Check if already cached
            cached_df = cache.load(symbol, timeframe, start, end)
            if len(cached_df) > 0:
                from datetime import datetime, timezone
                tf_sec = TIMEFRAME_SECONDS.get(timeframe, 14400)
                start_ms = int(datetime.fromisoformat(start).replace(
                    tzinfo=timezone.utc).timestamp() * 1000)
                end_ms = int(datetime.fromisoformat(end).replace(
                    tzinfo=timezone.utc).timestamp() * 1000)

                if (cached_df['timestamp'].iloc[0] <= start_ms and
                        cached_df['timestamp'].iloc[-1] >= end_ms - tf_sec * 1000):
                    result['bars'] = len(cached_df)
                    result['elapsed'] = time.time() - t0
                    if verbose:
                        print(f"  ✅ {symbol}: {len(cached_df):,} bars (cached)")
                    return result

            # Download from API
            client = BitgetClient()
            try:
                # Rate limit: wait before each API page request
                # The client internally paginates, so we wrap the whole download
                await rate_limiter.acquire()
                df = await client.download_ohlcv(symbol, timeframe, start, end)
            finally:
                await client.close()

            if df is not None and len(df) > 0:
                cache.save(df, symbol, timeframe)
                result['bars'] = len(df)
            else:
                result['error'] = "No data returned"

        except Exception as e:
            result['error'] = str(e)

    result['elapsed'] = time.time() - t0

    if verbose:
        if result['error']:
            print(f"  ❌ {symbol}: {result['error']}")
        else:
            print(f"  ✅ {symbol}: {result['bars']:,} bars in {result['elapsed']:.1f}s")

    return result


class AsyncRateLimiter:
    """
    Token bucket rate limiter for async code.
    Ensures we don't exceed Bitget's rate limit (~20 req/s).
    """

    def __init__(self, max_per_second: float = 15.0):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()


def download_batch_parallel(
    symbols: List[str],
    timeframe: str,
    start: str,
    end: str,
    cache_dir: str = "./data/cache",
    max_concurrent: int = 4,
    rate_limit: float = 15.0,
    verbose: bool = True,
) -> List[dict]:
    """
    Download multiple symbols concurrently.

    Args:
        symbols: List of symbol strings (e.g. ['BTCUSDT', 'ETHUSDT'])
        timeframe: Candle timeframe
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        cache_dir: Path to cache directory
        max_concurrent: Max symbols downloading simultaneously
        rate_limit: Max API requests per second (global across all symbols)
        verbose: Print progress

    Returns:
        List of result dicts with status per symbol.
    """

    async def _run():
        semaphore = asyncio.Semaphore(max_concurrent)
        limiter = AsyncRateLimiter(rate_limit)

        tasks = [
            _download_symbol(
                sym, timeframe, start, end,
                semaphore, limiter, cache_dir, verbose)
            for sym in symbols
        ]

        return await asyncio.gather(*tasks, return_exceptions=True)

    t0 = time.time()

    if verbose:
        print(f"\n📡 Parallel Batch Download: {len(symbols)} symbols")
        print(f"   Timeframe: {timeframe}")
        print(f"   Range: {start} → {end}")
        print(f"   Concurrency: {max_concurrent} symbols | Rate limit: {rate_limit} req/s")
        print()

    # Run the event loop
    results = asyncio.run(_run())

    # Handle any exceptions from gather
    clean_results = []
    for r in results:
        if isinstance(r, Exception):
            clean_results.append({'symbol': '?', 'error': str(r), 'bars': 0, 'elapsed': 0})
        else:
            clean_results.append(r)

    elapsed = time.time() - t0

    if verbose:
        total_bars = sum(r['bars'] for r in clean_results)
        errors = sum(1 for r in clean_results if r.get('error'))
        print(f"\n{'─' * 50}")
        print(f"  Total: {total_bars:,} bars | {len(symbols) - errors}/{len(symbols)} OK "
              f"| {elapsed:.1f}s")
        if errors:
            print(f"  ⚠ {errors} symbol(s) failed")

    return clean_results


# ─── Integration helper for cmd_download in cli.py ───

def patch_cmd_download_batch(args, dm):
    """
    Drop-in replacement for the batch section in cmd_download().

    Usage in cli.py:
        if batch:
            from data.download_parallel import patch_cmd_download_batch
            patch_cmd_download_batch(args, dm)
            return
    """
    from data.data_manager import POPULAR_CRYPTO, POPULAR_STOCKS

    batch = args.get('batch', 'crypto')
    tf = args.get('timeframe', '4h')
    start = args.get('start', '2023-01-01')
    end = args.get('end', '2025-01-01')

    if batch == 'crypto':
        symbols = POPULAR_CRYPTO
    elif batch == 'stocks':
        symbols = POPULAR_STOCKS
    elif batch == 'all':
        symbols = POPULAR_CRYPTO + POPULAR_STOCKS
    else:
        symbols = batch.split(',')

    download_batch_parallel(
        symbols, tf, start, end,
        cache_dir=str(dm.cache.cache_dir),
        max_concurrent=4,
        rate_limit=15,
        verbose=True,
    )