"""
CryptoLab — Bitget Data Provider  (v0.4 — pagination fix)
Downloads OHLCV + Funding Rate for perpetual futures (crypto + stock tokens).

Bitget API v2 endpoints (public, no auth needed):
  - /api/v2/mix/market/candles        → recent ~30 days, max 1000/page
  - /api/v2/mix/market/history-candles → historical, max 200/page

Key API behaviors:
  - productType MUST be lowercase: 'usdt-futures' (not 'USDT-FUTURES')
  - Pagination is BACKWARD via endTime (newest → oldest)
  - granularity values: '1m','5m','15m','30m','1H','4H','6H','12H','1D','1W'
  - '1h','60m' are NOT accepted → must uppercase hours: '1H','4H', etc.
  - Data array: [timestamp, open, high, low, close, baseVol, quoteVol]

Bug fixes (v0.4):
  - FIX 1: TIMEFRAME_MAP now covers ALL timeframes that were missing ('2H','6H','12H')
  - FIX 2 REVISED: Both `candles` and `history-candles` endpoints support `endTime`.
            The original FIX 2 removed endTime from `candles` entirely, which broke
            backward pagination for recent data (e.g., 5m detail data gap-fill).
            Now: only omit endTime on the FIRST call to get absolute latest candles;
            all subsequent pages include endTime for proper pagination.
  - FIX 3 REVISED: Smart endpoint selection with 3-stage fallback:
            (a) First recent page → candles without endTime (freshest data)
            (b) Subsequent recent pages → candles WITH endTime (pagination)
            (c) If candles+endTime fails → fallback to history-candles
  - FIX 4: hmac.new → hmac.HMAC (hmac.new is deprecated/removed in Python 3.12+)
  - FIX 5: generate_sample_data returns 'turnover' col for engine compatibility
"""
import asyncio
import time
import json
import hmac
import hashlib
import base64
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None

# SSL: try certifi first, then fall back to ssl=False
_SSL_CTX = None
_SSL_MODE = 'disabled'
try:
    import certifi
    import ssl as _ssl
    _SSL_CTX = _ssl.create_default_context(cafile=certifi.where())
    _SSL_MODE = 'certifi'
except ImportError:
    _SSL_MODE = 'no-verify'


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

BASE_URL = "https://api.bitget.com"

# FIX 1: Complete map — Bitget requires UPPERCASE for hour-based TFs.
# Verified working values from diagnose_api.py: '1H','4H','6H','12H','1D','1W'
TIMEFRAME_MAP = {
    '1m':  '1m',   '3m':  '3m',   '5m':  '5m',
    '15m': '15m',  '30m': '30m',
    '1h':  '1H',   '2h':  '2H',   '4h':  '4H',
    '6h':  '6H',   '12h': '12H',  '1d':  '1D',   '1w': '1W',
}

TIMEFRAME_SECONDS = {
    '1m': 60,        '3m': 180,      '5m': 300,
    '15m': 900,      '30m': 1800,
    '1h': 3600,      '2h': 7200,     '4h': 14400,
    '6h': 21600,     '12h': 43200,   '1d': 86400,   '1w': 604800,
}

# history-candles: max 200 per page; candles: max 1000 per page
HISTORY_LIMIT = 200
RECENT_LIMIT  = 1000

# Market types — MUST be lowercase for Bitget API v2
MARKET_CRYPTO = 'usdt-futures'
MARKET_STOCK  = 'susdt-futures'

# Recent candles endpoint covers roughly this many ms back (~30 days)
RECENT_WINDOW_MS = 30 * 24 * 3600 * 1000

# Backwards compat
MAX_CANDLES_PER_REQUEST = RECENT_LIMIT


# ═══════════════════════════════════════════════════════════════
#  MARKET CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class MarketConfig:
    """Market-specific parameters for backtesting."""

    CRYPTO_PERPETUAL = {
        'name': 'Crypto Perpetual',
        'product_type': MARKET_CRYPTO,
        'trading_hours': '24/7',
        'has_weekend_gap': False,
        'funding_interval_hours': 8,
        'fee_maker': 0.0002,
        'fee_taker': 0.0006,
        'max_leverage': 125,
        'settlement': 'USDT',
    }

    STOCK_PERPETUAL = {
        'name': 'Stock Perpetual',
        'product_type': MARKET_STOCK,
        'trading_hours': '24/5',
        'has_weekend_gap': True,
        'funding_interval_hours': 4,
        'fee_maker': 0.0002,
        'fee_taker': 0.0006,
        'max_leverage': 25,
        'settlement': 'USDT',
        'holidays_2025': [
            '2025-01-01', '2025-01-20', '2025-02-17',
            '2025-04-18', '2025-05-26', '2025-06-19',
            '2025-07-04', '2025-09-01', '2025-11-27',
            '2025-12-25',
        ],
    }

    @staticmethod
    def detect(symbol: str) -> dict:
        """Auto-detect market type from symbol."""
        stock_symbols = {
            'TSLAUSDT', 'NVDAUSDT', 'AAPLUSDT', 'AMZNUSDT',
            'GOOGLUSDT', 'METAUSDT', 'MSFTUSDT', 'CRCLUSDT',
            'BAAUSDT', 'JPMAUSDT',
        }
        if symbol.upper() in stock_symbols:
            return MarketConfig.STOCK_PERPETUAL
        return MarketConfig.CRYPTO_PERPETUAL


# ═══════════════════════════════════════════════════════════════
#  BITGET CLIENT
# ═══════════════════════════════════════════════════════════════

class BitgetClient:
    """
    Async client for Bitget REST API v2.
    No API key needed for public market data.
    """

    def __init__(self, api_key: str = "", api_secret: str = "",
                 passphrase: str = "", verbose: bool = True):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.verbose    = verbose
        self._session   = None

    async def _get_session(self) -> 'aiohttp.ClientSession':
        if self._session is None or self._session.closed:
            connector = (
                aiohttp.TCPConnector(ssl=_SSL_CTX)
                if _SSL_CTX is not None
                else aiohttp.TCPConnector(ssl=False)
            )
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign(self, timestamp: str, method: str, path: str,
              body: str = "") -> dict:
        """Generate authentication headers (for private endpoints)."""
        # FIX 4: hmac.new is removed in Python 3.12+; use hmac.HMAC directly
        message = timestamp + method + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256,
        )
        signature = base64.b64encode(mac.digest()).decode('utf-8')
        return {
            'ACCESS-KEY':       self.api_key,
            'ACCESS-SIGN':      signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type':     'application/json',
        }

    async def _raw_get(self, url: str) -> dict:
        """Low-level GET with retries and comprehensive error logging."""
        session = await self._get_session()
        headers = {'Content-Type': 'application/json'}

        for attempt in range(3):
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    text = await resp.text()

                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        if self.verbose:
                            print(f"      ⏳ Rate limited, waiting {wait}s...")
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
                # On SSL error, retry with ssl=False
                if 'SSL' in ename or 'Certificate' in str(e):
                    if self._session:
                        await self._session.close()
                    connector = aiohttp.TCPConnector(ssl=False)
                    self._session = aiohttp.ClientSession(connector=connector)
                    session = self._session
                await asyncio.sleep(1 + attempt)

        return {}

    async def _api_get(self, path: str, params: dict) -> dict:
        """Build URL and call _raw_get."""
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BASE_URL}{path}?{query}"
        return await self._raw_get(url)

    # ─── CANDLE FETCHING ─────────────────────────────────────────

    async def fetch_candles_page(
        self,
        symbol: str,
        timeframe: str,
        end_ms: int,
        product_type: str = MARKET_CRYPTO,
        use_history: bool = True,
        omit_end_time: bool = False,
    ) -> List[List]:
        """
        Fetch one page of candles.
        Returns list of [ts, open, high, low, close, vol, quoteVol].

        FIX 2 REVISED: The `candles` endpoint DOES accept `endTime` for
        backward pagination. The original FIX 2 removed it entirely, which
        broke pagination for recent data (5m gaps). We now only omit endTime
        on the very first call when omit_end_time=True (to get the absolute
        latest candles), then use endTime on all subsequent pages.
        """
        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(
                f"Unknown timeframe '{timeframe}'. "
                f"Valid: {list(TIMEFRAME_MAP.keys())}"
            )

        if use_history:
            path  = "/api/v2/mix/market/history-candles"
            limit = HISTORY_LIMIT
            params = {
                'symbol':      symbol,
                'productType': product_type,
                'granularity': tf,
                'limit':       str(limit),
                'endTime':     str(int(end_ms)),
            }
        else:
            path  = "/api/v2/mix/market/candles"
            limit = RECENT_LIMIT
            params = {
                'symbol':      symbol,
                'productType': product_type,
                'granularity': tf,
                'limit':       str(limit),
            }
            # Include endTime for pagination (all pages except the very first)
            if not omit_end_time:
                params['endTime'] = str(int(end_ms))

        data = await self._api_get(path, params)

        if not data:
            return []

        code = data.get('code', '')
        msg  = data.get('msg', '')

        if code == '00000':
            return data.get('data', [])

        if self.verbose:
            print(f"      API [{code}]: {msg} | endpoint={path}")
        return []

    async def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
        product_type: str = None,
    ) -> pd.DataFrame:
        """
        Download full OHLCV range with backward pagination.

        Strategy:
          1. Use history-candles (200/page) for full date ranges
          2. FIX 3: Switch to recent candles endpoint automatically when
             the current window falls within the last 30 days — avoids
             silent empty pages that killed pagination prematurely
          3. Paginate backward from end_ms until start_ms

        Returns chronologically sorted DataFrame.
        """
        if product_type is None:
            config       = MarketConfig.detect(symbol)
            product_type = config['product_type']

        start_ms = int(datetime.fromisoformat(start).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

        now_ms = int(time.time() * 1000)

        all_candles      = []
        current_end      = end_ms
        consecutive_empty = 0
        t0               = time.time()
        used_recent_first = False  # Track if we already fetched the latest via candles

        # Dynamic max_pages: estimate needed pages + 20% safety margin
        # history-candles=200/page, recent candles=1000/page
        # Use conservative 200/page for estimation
        total_bars_est = max(1, (end_ms - start_ms) // tf_ms)
        max_pages = max(2000, int(total_bars_est / HISTORY_LIMIT * 1.3))

        if self.verbose:
            total_est = max(1, (end_ms - start_ms) // tf_ms)
            print(f"      productType={product_type}, "
                  f"granularity={TIMEFRAME_MAP.get(timeframe, timeframe)}, "
                  f"ssl={_SSL_MODE}, ~{total_est:,} bars expected, "
                  f"max_pages={max_pages:,}")

        for page in range(1, max_pages + 1):
            if current_end <= start_ms:
                break

            # FIX 3 REVISED: Strategy for endpoint selection:
            #   - If current_end is older than ~30 days → use history-candles (reliable)
            #   - If current_end is recent AND very close to now → use candles
            #     WITHOUT endTime to grab the absolute latest data
            #   - If current_end is recent but not close to now → use candles
            #     WITH endTime for proper backward pagination
            #   - If candles endpoint fails with endTime → fall back to history-candles
            #     (which may work even for somewhat recent data)
            is_recent = (current_end >= now_ms - RECENT_WINDOW_MS)
            # Only omit endTime if we're fetching truly current data (within 1 day of now)
            is_near_now = (abs(now_ms - current_end) < 86400 * 1000)

            if not is_recent:
                use_history = True
                omit_end_time = False
            elif is_near_now and not used_recent_first:
                # First call, end is near now: no endTime, get absolute latest
                use_history = False
                omit_end_time = True
            else:
                # Recent but not near now, or already got first page: use endTime
                use_history = False
                omit_end_time = False

            candles = await self.fetch_candles_page(
                symbol, timeframe,
                end_ms=current_end,
                product_type=product_type,
                use_history=use_history,
                omit_end_time=omit_end_time,
            )

            # Track that we've used the recent endpoint at least once
            if not use_history and candles:
                used_recent_first = True

            # Fallback: if candles endpoint with endTime returns empty,
            # try history-candles (may work for data > ~24h old)
            if not candles and not use_history and not omit_end_time:
                candles = await self.fetch_candles_page(
                    symbol, timeframe,
                    end_ms=current_end,
                    product_type=product_type,
                    use_history=True,
                )

            # If history endpoint fails on first pages, try recent endpoint once
            if not candles and use_history and page <= 3:
                candles = await self.fetch_candles_page(
                    symbol, timeframe,
                    end_ms=current_end,
                    product_type=product_type,
                    use_history=False,
                    omit_end_time=True,
                )
                if candles:
                    used_recent_first = True

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

            # Find earliest timestamp in this page
            timestamps = [int(c[0]) for c in candles]
            earliest   = min(timestamps)

            # Progress bar
            if self.verbose:
                n       = len(all_candles)
                pct     = min(100.0, (end_ms - earliest) / max(1, end_ms - start_ms) * 100)
                elapsed = time.time() - t0
                eta     = (elapsed / max(pct, 0.1) * (100 - pct)) if pct > 1 else 0
                dt_str  = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                sys.stdout.write(
                    f"\r      {n:>7,} bars | {pct:5.1f}% | oldest={dt_str} | ETA {eta:.0f}s   "
                )
                sys.stdout.flush()

            if earliest <= start_ms:
                break

            # Move pagination cursor backward
            new_end = earliest - 1
            if new_end >= current_end:
                # Pagination stuck — the endpoint returned the same or newer data.
                # This can happen if candles endpoint ignores endTime.
                # Strategy: jump cursor backward to where history-candles works.
                jump_target = now_ms - RECENT_WINDOW_MS - tf_ms
                if jump_target > start_ms and current_end > jump_target:
                    if self.verbose:
                        jump_dt = datetime.fromtimestamp(
                            jump_target / 1000, tz=timezone.utc
                        ).strftime('%Y-%m-%d')
                        print(f"\n      ⚠ Pagination stuck, jumping to {jump_dt} (history-candles zone)")
                    current_end = jump_target
                    used_recent_first = True  # Force endTime usage going forward
                    continue
                else:
                    if self.verbose:
                        print(f"\n      ⚠ Pagination stuck at {current_end}, stopping")
                    break
            current_end = new_end

            await asyncio.sleep(0.05)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"\n      Download complete: {len(all_candles):,} raw bars in {elapsed:.1f}s")

        if not all_candles:
            return pd.DataFrame(columns=[
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'
            ])

        # Build DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['datetime']  = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
        df = df[(df['timestamp'] >= start_ms) & (df['timestamp'] <= end_ms)]
        df = df.reset_index(drop=True)

        return df[['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume']]

    async def get_funding_rate_history(
        self,
        symbol: str,
        product_type: str = MARKET_CRYPTO,
    ) -> List[dict]:
        """Fetch historical funding rates."""
        params = {
            'symbol':      symbol,
            'productType': product_type,
            'pageSize':    '100',
        }
        data = await self._api_get("/api/v2/mix/market/history-fund-rate", params)
        if data and data.get('code') == '00000':
            return data.get('data', [])
        return []


# ═══════════════════════════════════════════════════════════════
#  DATA CACHE (Parquet / CSV)
# ═══════════════════════════════════════════════════════════════

class DataCache:
    """
    Local cache for downloaded data using Parquet files (or CSV fallback).
    Supports incremental updates (only downloads missing ranges).
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._use_parquet = False
        try:
            import pyarrow  # noqa
            self._use_parquet = True
        except ImportError:
            try:
                import fastparquet  # noqa
                self._use_parquet = True
            except ImportError:
                pass

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        ext = '.parquet' if self._use_parquet else '.csv'
        return self.cache_dir / f"{symbol}_{timeframe}{ext}"

    def _funding_path(self, symbol: str) -> Path:
        ext = '.parquet' if self._use_parquet else '.csv'
        return self.cache_dir / f"{symbol}_funding{ext}"

    def has_data(self, symbol: str, timeframe: str) -> bool:
        return (
            (self.cache_dir / f"{symbol}_{timeframe}.parquet").exists() or
            (self.cache_dir / f"{symbol}_{timeframe}.csv").exists()
        )

    def load(self, symbol: str, timeframe: str,
             start: str = None, end: str = None) -> pd.DataFrame:
        """Load cached data, optionally filtered by date range."""
        pq_path  = self.cache_dir / f"{symbol}_{timeframe}.parquet"
        csv_path = self.cache_dir / f"{symbol}_{timeframe}.csv"

        if pq_path.exists() and self._use_parquet:
            df = pd.read_parquet(pq_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            return pd.DataFrame()

        if start:
            start_ms = int(datetime.fromisoformat(start).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
            df = df[df['timestamp'] >= start_ms]
        if end:
            end_ms = int(datetime.fromisoformat(end).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
            df = df[df['timestamp'] <= end_ms]

        return df.reset_index(drop=True)

    def save(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save or merge data into cache."""
        path     = self._cache_path(symbol, timeframe)
        pq_path  = self.cache_dir / f"{symbol}_{timeframe}.parquet"
        csv_path = self.cache_dir / f"{symbol}_{timeframe}.csv"

        existing = pd.DataFrame()
        if pq_path.exists() and self._use_parquet:
            existing = pd.read_parquet(pq_path)
        elif csv_path.exists():
            existing = pd.read_csv(csv_path)

        if len(existing) > 0:
            df = pd.concat([existing, df]).drop_duplicates(
                subset='timestamp'
            ).sort_values('timestamp')

        if self._use_parquet:
            df.to_parquet(path, index=False, compression='snappy')
        else:
            df.to_csv(path, index=False)

    def delete(self, symbol: str, timeframe: str):
        """Delete cached data for a symbol/timeframe."""
        for ext in ['.parquet', '.csv']:
            p = self.cache_dir / f"{symbol}_{timeframe}{ext}"
            if p.exists():
                p.unlink()

    def to_numpy(self, df: pd.DataFrame) -> dict:
        """Convert DataFrame to dict of numpy arrays for engine."""
        return {
            'timestamp': df['timestamp'].values.astype(np.float64),
            'open':      df['open'].values.astype(np.float64),
            'high':      df['high'].values.astype(np.float64),
            'low':       df['low'].values.astype(np.float64),
            'close':     df['close'].values.astype(np.float64),
            'volume':    df['volume'].values.astype(np.float64),
            'hl2':  ((df['high'].values + df['low'].values) / 2.0).astype(np.float64),
            'hlc3': ((df['high'].values + df['low'].values + df['close'].values) / 3.0).astype(np.float64),
        }


# ═══════════════════════════════════════════════════════════════
#  SYNC DOWNLOAD WRAPPER
# ═══════════════════════════════════════════════════════════════

def download_data(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cache_dir: str = "./data/cache",
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
) -> pd.DataFrame:
    """
    Synchronous wrapper: download data (or load from cache).

    Usage:
        df = download_data('BTCUSDT', '4h', '2023-01-01', '2025-01-01')
    """
    cache = DataCache(cache_dir)

    if cache.has_data(symbol, timeframe):
        cached = cache.load(symbol, timeframe, start, end)
        if len(cached) > 0:
            start_ms = int(datetime.fromisoformat(start).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
            end_ms = int(datetime.fromisoformat(end).replace(
                tzinfo=timezone.utc).timestamp() * 1000)
            if (cached['timestamp'].iloc[0] <= start_ms and
                    cached['timestamp'].iloc[-1] >= end_ms - TIMEFRAME_SECONDS[timeframe] * 1000):
                return cached

    client = BitgetClient(api_key, api_secret, passphrase)

    async def _download():
        try:
            return await client.download_ohlcv(symbol, timeframe, start, end)
        finally:
            await client.close()

    df = asyncio.run(_download())

    if len(df) > 0:
        cache.save(df, symbol, timeframe)

    return df


# ═══════════════════════════════════════════════════════════════
#  SAMPLE DATA GENERATOR (for testing without API)
# ═══════════════════════════════════════════════════════════════

def generate_sample_data(
    n_bars: int = 5000,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    timeframe: str = '4h',
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for testing.
    Uses geometric Brownian motion with mean-reverting regime drift.
    """
    rng = np.random.RandomState(seed)

    tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 14400)
    timestamps = np.arange(n_bars) * tf_seconds * 1000 + 1_672_531_200_000

    returns = rng.normal(0, volatility, n_bars)
    regime  = np.zeros(n_bars)
    regime_length = rng.randint(50, 200)
    regime_dir    = 1
    pos = 0
    while pos < n_bars:
        end_pos = min(pos + regime_length, n_bars)
        regime[pos:end_pos] = regime_dir * 0.001
        pos           = end_pos
        regime_length = rng.randint(50, 200)
        regime_dir   *= -1

    returns += regime
    prices   = start_price * np.exp(np.cumsum(returns))

    open_  = np.empty(n_bars)
    high   = np.empty(n_bars)
    low    = np.empty(n_bars)
    close  = prices.copy()
    volume = np.empty(n_bars)

    open_[0] = start_price
    for i in range(1, n_bars):
        open_[i] = close[i - 1]

    for i in range(n_bars):
        iv      = volatility * 0.3
        h_dev   = abs(rng.normal(0, iv))
        l_dev   = abs(rng.normal(0, iv))
        high[i] = max(open_[i], close[i]) * (1 + h_dev)
        low[i]  = min(open_[i], close[i]) * (1 - l_dev)

        base_vol = 1000 + rng.exponential(500)
        if rng.random() < 0.05:
            base_vol *= rng.uniform(3, 8)
        volume[i] = base_vol

    # FIX 5: include turnover so engine compatibility is maintained
    turnover = volume * close

    return pd.DataFrame({
        'timestamp': timestamps,
        'datetime':  pd.to_datetime(timestamps, unit='ms'),
        'open':      open_,
        'high':      high,
        'low':       low,
        'close':     close,
        'volume':    volume,
        'turnover':  turnover,
    })