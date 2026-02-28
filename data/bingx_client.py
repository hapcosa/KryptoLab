"""
CryptoLab — BingX Data Provider
Downloads OHLCV for perpetual futures (Standard & Perpetual contracts).

BingX API v3 endpoints (public, no auth needed for market data):
  - GET /openApi/swap/v3/quote/klines  → OHLCV candles, max 1440/page
  - GET /openApi/swap/v2/quote/fundingRate → current funding rate
  - GET /openApi/swap/v2/quote/fundingRateHistory → historical funding rates

Key API behaviors:
  - symbol format: 'BTC-USDT' (hyphen, not 'BTCUSDT')
  - interval: '1m','3m','5m','15m','30m','1h','2h','4h','6h','12h','1d','1w'
  - Pagination: forward or backward via startTime / endTime (ms timestamps)
  - Max 1440 candles per request
  - Response: list of {time, open, high, low, close, volume, quoteAssetVolume}
  - Rate limit: ~10 req/s on public endpoints

Usage:
    from data.bingx_client import download_data_bingx, BingXClient

    # Simple sync download
    df = download_data_bingx('BTC-USDT', '1h', '2024-01-01', '2025-01-01')

    # Async client
    async with BingXClient() as client:
        df = await client.download_ohlcv('BTC-USDT', '4h', '2024-01-01', '2025-01-01')
"""
import asyncio
import time
import json
import hmac
import hashlib
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None

# SSL setup (shared approach with bitget_client)
_SSL_CTX  = None
_SSL_MODE = 'disabled'
try:
    import certifi
    import ssl as _ssl
    _SSL_CTX  = _ssl.create_default_context(cafile=certifi.where())
    _SSL_MODE = 'certifi'
except ImportError:
    _SSL_MODE = 'no-verify'


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

BASE_URL = "https://open-api.bingx.com"

# Internal TF → BingX interval string
# BingX uses lowercase intervals (unlike Bitget's uppercase hours)
TIMEFRAME_MAP: Dict[str, str] = {
    '1m':  '1m',   '3m':  '3m',   '5m':  '5m',
    '15m': '15m',  '30m': '30m',
    '1h':  '1h',   '2h':  '2h',   '4h':  '4h',
    '6h':  '6h',   '12h': '12h',  '1d':  '1d',   '1w': '1w',
}

TIMEFRAME_SECONDS: Dict[str, int] = {
    '1m': 60,        '3m': 180,      '5m': 300,
    '15m': 900,      '30m': 1800,
    '1h': 3600,      '2h': 7200,     '4h': 14400,
    '6h': 21600,     '12h': 43200,   '1d': 86400,   '1w': 604800,
}

CANDLES_PER_PAGE = 1440     # BingX maximum per request
RATE_LIMIT_SLEEP = 0.12     # ~8 req/s, safe under 10 req/s limit


# ═══════════════════════════════════════════════════════════════
#  SYMBOL HELPERS
# ═══════════════════════════════════════════════════════════════

def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol to BingX format ('BTC-USDT').
    Accepts: 'BTC-USDT', 'BTCUSDT', 'btcusdt', 'BTC/USDT'
    """
    s = symbol.upper().replace('/', '-').replace('_', '-')
    # If no hyphen, try to insert before 'USDT' or 'BUSD'
    if '-' not in s:
        for quote in ('USDT', 'BUSD', 'BTC', 'ETH', 'BNB'):
            if s.endswith(quote) and len(s) > len(quote):
                base = s[:-len(quote)]
                return f"{base}-{quote}"
    return s


# ═══════════════════════════════════════════════════════════════
#  BINGX CLIENT
# ═══════════════════════════════════════════════════════════════

class BingXClient:
    """
    Async client for BingX REST API v3.
    No API key needed for public market data.

    Can be used as an async context manager:
        async with BingXClient() as client:
            df = await client.download_ohlcv(...)
    """

    def __init__(
        self,
        api_key:    str = "",
        api_secret: str = "",
        verbose:    bool = True,
    ):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.verbose    = verbose
        self._session: Optional['aiohttp.ClientSession'] = None

    # ── context manager support ──

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _get_session(self) -> 'aiohttp.ClientSession':
        if self._session is None or self._session.closed:
            connector = (
                aiohttp.TCPConnector(ssl=_SSL_CTX)
                if _SSL_CTX is not None
                else aiohttp.TCPConnector(ssl=False)
            )
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                connector=connector, timeout=timeout
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── private signing (for authenticated endpoints) ──

    def _sign(self, params: dict) -> dict:
        """
        Add HMAC-SHA256 signature to params (for private endpoints).
        BingX signs the query string: 'key=val&key=val&timestamp=...'
        """
        ts = str(int(time.time() * 1000))
        params = {**params, 'timestamp': ts}
        query  = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
        sig    = hmac.new(
            self.api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {**params, 'signature': sig}

    # ── low-level HTTP ──

    async def _raw_get(self, path: str, params: dict, signed: bool = False) -> dict:
        """GET with retry logic and rate-limit handling."""
        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret required for signed endpoints")
            params = self._sign(params)

        query = '&'.join(f"{k}={v}" for k, v in params.items())
        url   = f"{BASE_URL}{path}?{query}"

        session = await self._get_session()
        headers = {
            'X-BX-APIKEY': self.api_key,
            'Content-Type': 'application/json',
        }

        for attempt in range(3):
            try:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()

                    # Rate limit
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        if self.verbose:
                            print(f"      ⏳ Rate limited (BingX), waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    if resp.status != 200:
                        if self.verbose:
                            print(f"      ⚠ HTTP {resp.status}: {text[:300]}")
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return {}

                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"      ⚠ Invalid JSON: {text[:300]}")
                        return {}

            except Exception as e:
                ename = type(e).__name__
                if self.verbose:
                    print(f"      ❌ {ename} (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    return {}
                if 'SSL' in ename or 'Certificate' in str(e):
                    if self._session:
                        await self._session.close()
                    connector = aiohttp.TCPConnector(ssl=False)
                    self._session = aiohttp.ClientSession(connector=connector)
                    session = self._session
                await asyncio.sleep(1 + attempt)

        return {}

    # ── candle fetching ──

    def _parse_candle_response(self, data: dict) -> List[List]:
        """
        Parse BingX klines response into list of
        [timestamp_ms, open, high, low, close, volume].

        BingX v3 klines response format:
          {
            "code": 0,
            "msg": "",
            "data": [
              {"time": 1234567890000, "open": "...", "high": "...",
               "low": "...", "close": "...", "volume": "...", "quoteAssetVolume": "..."},
              ...
            ]
          }
        """
        if not data:
            return []

        code = data.get('code', -1)
        if code != 0:
            if self.verbose:
                print(f"      BingX API error [{code}]: {data.get('msg', '')}")
            return []

        raw = data.get('data') or []
        candles = []
        for item in raw:
            try:
                candles.append([
                    int(item['time']),
                    float(item['open']),
                    float(item['high']),
                    float(item['low']),
                    float(item['close']),
                    float(item['volume']),
                    float(item.get('quoteAssetVolume', 0)),
                ])
            except (KeyError, ValueError, TypeError):
                continue
        return candles

    async def fetch_candles_page(
        self,
        symbol:    str,
        timeframe: str,
        start_ms:  int,
        end_ms:    int,
    ) -> List[List]:
        """
        Fetch one page of candles from BingX.
        BingX paginates FORWARD: provide startTime, optionally endTime.
        Returns list of [ts_ms, open, high, low, close, volume, quoteVol].
        """
        interval = TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Valid: {list(TIMEFRAME_MAP.keys())}"
            )

        params = {
            'symbol':    symbol,
            'interval':  interval,
            'limit':     str(CANDLES_PER_PAGE),
            'startTime': str(int(start_ms)),
            'endTime':   str(int(end_ms)),
        }

        data = await self._raw_get('/openApi/swap/v3/quote/klines', params)
        return self._parse_candle_response(data)

    async def download_ohlcv(
        self,
        symbol:    str,
        timeframe: str,
        start:     str,
        end:       str,
    ) -> pd.DataFrame:
        """
        Download full OHLCV range with FORWARD pagination.

        BingX paginates differently from Bitget:
          - We step forward from start_ms to end_ms
          - Each page returns up to 1440 candles
          - We advance start_ms to (last_ts + 1 tf_ms) after each page

        Returns chronologically sorted DataFrame with columns:
          timestamp, datetime, open, high, low, close, volume
        """
        symbol = normalize_symbol(symbol)

        start_ms = int(datetime.fromisoformat(start).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

        all_candles      = []
        current_start    = start_ms
        max_pages        = 5000
        consecutive_empty = 0
        t0               = time.time()

        if self.verbose:
            total_est = max(1, (end_ms - start_ms) // tf_ms)
            interval  = TIMEFRAME_MAP.get(timeframe, timeframe)
            print(f"      BingX | symbol={symbol}, interval={interval}, "
                  f"ssl={_SSL_MODE}, ~{total_est:,} bars expected")

        for page in range(1, max_pages + 1):
            if current_start >= end_ms:
                break

            # BingX endTime is exclusive-ish; always clip to our desired end
            page_end = min(current_start + tf_ms * CANDLES_PER_PAGE, end_ms)

            candles = await self.fetch_candles_page(
                symbol, timeframe, current_start, page_end
            )

            if not candles:
                consecutive_empty += 1
                if consecutive_empty > 5:
                    if self.verbose:
                        print(f"\n      ❌ {consecutive_empty} consecutive empty pages, stopping")
                    break
                await asyncio.sleep(0.5)
                continue

            consecutive_empty = 0
            all_candles.extend(candles)

            # Advance cursor to after the last received candle
            timestamps    = [c[0] for c in candles]
            latest_ts     = max(timestamps)
            new_start     = latest_ts + tf_ms

            if new_start <= current_start:
                # No progress — safety break
                if self.verbose:
                    print(f"\n      ⚠ Pagination stuck at {current_start}, stopping")
                break
            current_start = new_start

            # Progress reporting
            if self.verbose:
                n       = len(all_candles)
                pct     = min(100.0, (latest_ts - start_ms) / max(1, end_ms - start_ms) * 100)
                elapsed = time.time() - t0
                eta     = (elapsed / max(pct, 0.1) * (100 - pct)) if pct > 1 else 0
                dt_str  = datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                sys.stdout.write(
                    f"\r      {n:>7,} bars | {pct:5.1f}% | latest={dt_str} | ETA {eta:.0f}s   "
                )
                sys.stdout.flush()

            # Small delay to stay under rate limit
            await asyncio.sleep(RATE_LIMIT_SLEEP)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"\n      Download complete: {len(all_candles):,} raw bars in {elapsed:.1f}s")

        if not all_candles:
            return pd.DataFrame(columns=[
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'
            ])

        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        df['timestamp'] = df['timestamp'].astype('int64')
        df['datetime']  = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        df = df[(df['timestamp'] >= start_ms) & (df['timestamp'] <= end_ms)]
        df = df.reset_index(drop=True)

        return df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]

    # ── market info ──

    async def get_symbols(self) -> List[dict]:
        """Return all available perpetual symbols."""
        data = await self._raw_get('/openApi/swap/v2/quote/contracts', {})
        if data and data.get('code') == 0:
            return data.get('data', {}).get('contracts', [])
        return []

    async def get_ticker(self, symbol: str) -> dict:
        """Get 24h ticker for a symbol."""
        symbol = normalize_symbol(symbol)
        data   = await self._raw_get(
            '/openApi/swap/v2/quote/ticker',
            {'symbol': symbol}
        )
        if data and data.get('code') == 0:
            return data.get('data', {})
        return {}

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for a symbol."""
        symbol = normalize_symbol(symbol)
        data   = await self._raw_get(
            '/openApi/swap/v2/quote/fundingRate',
            {'symbol': symbol}
        )
        if data and data.get('code') == 0:
            rate = data.get('data', {}).get('fundingRate')
            return float(rate) if rate is not None else None
        return None

    async def get_funding_rate_history(
        self,
        symbol: str,
        limit:  int = 100,
    ) -> List[dict]:
        """Get historical funding rates."""
        symbol = normalize_symbol(symbol)
        data   = await self._raw_get(
            '/openApi/swap/v2/quote/fundingRateHistory',
            {'symbol': symbol, 'limit': str(limit)}
        )
        if data and data.get('code') == 0:
            return data.get('data', {}).get('list', [])
        return []


# ═══════════════════════════════════════════════════════════════
#  SYNC DOWNLOAD WRAPPER
# ═══════════════════════════════════════════════════════════════

def download_data_bingx(
    symbol:    str,
    timeframe: str,
    start:     str,
    end:       str,
    cache_dir: str = "./data/cache",
    api_key:   str = "",
    api_secret: str = "",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Synchronous wrapper: download OHLCV from BingX (with optional cache).

    Args:
        symbol:     BingX symbol, e.g. 'BTC-USDT' or 'BTCUSDT'
        timeframe:  '1m','5m','15m','30m','1h','4h','1d', etc.
        start:      ISO date string, e.g. '2024-01-01'
        end:        ISO date string, e.g. '2025-01-01'
        cache_dir:  local Parquet/CSV cache directory
        api_key:    BingX API key (only for private endpoints)
        api_secret: BingX API secret
        use_cache:  skip cache when False (always re-download)

    Returns:
        pd.DataFrame with columns: timestamp, datetime, open, high, low, close, volume

    Example:
        df = download_data_bingx('BTC-USDT', '4h', '2023-01-01', '2025-01-01')
        print(df.head())
    """
    from data.bitget_client import DataCache, TIMEFRAME_SECONDS

    symbol_normalized = normalize_symbol(symbol)

    if use_cache:
        cache = DataCache(cache_dir)
        if cache.has_data(symbol_normalized, timeframe):
            cached = cache.load(symbol_normalized, timeframe, start, end)
            if len(cached) > 0:
                from datetime import datetime, timezone
                start_ms = int(datetime.fromisoformat(start).replace(
                    tzinfo=timezone.utc).timestamp() * 1000)
                end_ms = int(datetime.fromisoformat(end).replace(
                    tzinfo=timezone.utc).timestamp() * 1000)
                if (cached['timestamp'].iloc[0] <= start_ms and
                        cached['timestamp'].iloc[-1] >= end_ms - TIMEFRAME_SECONDS[timeframe] * 1000):
                    return cached
    else:
        cache = None

    client = BingXClient(api_key=api_key, api_secret=api_secret)

    async def _download():
        try:
            return await client.download_ohlcv(symbol_normalized, timeframe, start, end)
        finally:
            await client.close()

    df = asyncio.run(_download())

    if use_cache and len(df) > 0:
        if cache is None:
            cache = DataCache(cache_dir)
        cache.save(df, symbol_normalized, timeframe)

    return df


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTIC  (python -m data.bingx_client)
# ═══════════════════════════════════════════════════════════════

async def _run_diagnostics():
    print("=" * 60)
    print("  BingX API Diagnostics")
    print("=" * 60)

    async with BingXClient(verbose=True) as client:
        # 1. Ticker
        print("\n[1] Testing ticker endpoint (BTC-USDT)...")
        ticker = await client.get_ticker('BTC-USDT')
        if ticker:
            print(f"    ✅ Price: {ticker.get('lastPrice')} | "
                  f"24h vol: {ticker.get('volume')}")
        else:
            print("    ❌ No ticker data")

        # 2. Funding rate
        print("\n[2] Testing funding rate...")
        rate = await client.get_funding_rate('BTC-USDT')
        if rate is not None:
            print(f"    ✅ Current funding rate: {rate:.6f} ({rate*100:.4f}%)")
        else:
            print("    ❌ No funding rate data")

        # 3. Candles
        print("\n[3] Testing candle download (BTC-USDT, 1h, last 5 candles)...")
        candles = await client.fetch_candles_page(
            'BTC-USDT', '1h',
            start_ms=int(time.time() * 1000) - 5 * 3600 * 1000,
            end_ms=int(time.time() * 1000),
        )
        print(f"    Candles returned: {len(candles)}")
        if candles:
            print(f"    Sample: ts={candles[-1][0]}, "
                  f"O={candles[-1][1]}, H={candles[-1][2]}, "
                  f"L={candles[-1][3]}, C={candles[-1][4]}")
            print("    ✅ Candle endpoint OK")
        else:
            print("    ❌ No candles returned")

        # 4. Granularity check
        print("\n[4] Testing timeframe aliases...")
        for tf in ['1h', '4h', '1d']:
            candles_tf = await client.fetch_candles_page(
                'BTC-USDT', tf,
                start_ms=int(time.time() * 1000) - 3 * TIMEFRAME_SECONDS[tf] * 1000,
                end_ms=int(time.time() * 1000),
            )
            status = "✅" if candles_tf else "❌"
            print(f"    {status} interval={TIMEFRAME_MAP[tf]:>4s} → {len(candles_tf)} candles")

    print("\n" + "=" * 60)
    print("  BingX diagnostics complete.")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(_run_diagnostics())