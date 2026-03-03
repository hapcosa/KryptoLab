"""
CryptoLab — Binance Data Provider (USDⓈ-M Futures)
Download OHLCV for perpetual futures via public API.

Endpoints:
  - GET /fapi/v1/klines  (futures, USDⓈ-M)
  - GET /api/v3/klines    (spot, fallback si es necesario)

Documentation: https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
"""
import asyncio
import time
import json
import sys
from datetime import datetime, timezone
from typing import Optional, List, Dict

import pandas as pd
import numpy as np

try:
    import aiohttp
except ImportError:
    aiohttp = None

# SSL (similar a otros clientes)
_SSL_CTX = None
_SSL_MODE = 'disabled'
try:
    import certifi
    import ssl as _ssl
    _SSL_CTX = _ssl.create_default_context(cafile=certifi.where())
    _SSL_MODE = 'certifi'
except ImportError:
    _SSL_MODE = 'no-verify'

# Constantes
BASE_URL_FUTURES = "https://fapi.binance.com"
BASE_URL_SPOT    = "https://api.binance.com"

TIMEFRAME_MAP = {
    '1m':  '1m',   '3m':  '3m',   '5m':  '5m',
    '15m': '15m',  '30m': '30m',
    '1h':  '1h',   '2h':  '2h',   '4h':  '4h',
    '6h':  '6h',   '8h':  '8h',   '12h': '12h',
    '1d':  '1d',   '3d':  '3d',   '1w':  '1w',   '1M':  '1M',
}

TIMEFRAME_SECONDS = {
    '1m': 60,        '3m': 180,      '5m': 300,
    '15m': 900,      '30m': 1800,
    '1h': 3600,      '2h': 7200,     '4h': 14400,
    '6h': 21600,     '8h': 28800,    '12h': 43200,
    '1d': 86400,     '3d': 259200,   '1w': 604800,  '1M': 2592000,  # 30 días aprox
}

KLINES_LIMIT = 1000          # Máximo por petición
RATE_LIMIT_SLEEP = 0.1       # 10 requests/segundo aprox (peso bajo)


class BinanceClient:
    """
    Cliente asíncrono para API de Binance Futures (USDⓈ-M).
    No necesita API key para datos públicos de velas.
    """

    def __init__(self, use_spot: bool = False, verbose: bool = True):
        self.use_spot = use_spot
        self.verbose = verbose
        self._session = None
        self.base_url = BASE_URL_SPOT if use_spot else BASE_URL_FUTURES

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
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, path: str, params: dict) -> dict:
        """Realiza GET con manejo de errores y reintentos."""
        session = await self._get_session()
        url = f"{self.base_url}{path}"

        for attempt in range(3):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        if self.verbose:
                            print(f"      ⏳ Rate limited (Binance), waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        text = await resp.text()
                        if self.verbose:
                            print(f"      ⚠ HTTP {resp.status}: {text[:300]}")
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return {}
                    return await resp.json()
            except Exception as e:
                ename = type(e).__name__
                if self.verbose:
                    print(f"      ❌ {ename} (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    return {}
                await asyncio.sleep(1 + attempt)
        return {}

    async def fetch_klines_page(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int = KLINES_LIMIT
    ) -> List[List]:
        """
        Obtiene una página de velas (hasta 1000) desde start_ms.
        Binance devuelve velas con timestamp de apertura.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ms,
            'limit': limit,
        }
        # Si end_ms se proporciona, lo añadimos (opcional, pero ayuda a limitar)
        if end_ms:
            params['endTime'] = end_ms

        path = "/fapi/v1/klines" if not self.use_spot else "/api/v3/klines"
        data = await self._request(path, params)

        if not data or not isinstance(data, list):
            return []
        return data

    async def download_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Descarga rango completo de velas con paginación hacia adelante.
        """
        interval = TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Timeframe '{timeframe}' no soportado por Binance")

        start_ms = int(datetime.fromisoformat(start).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.fromisoformat(end).replace(
            tzinfo=timezone.utc).timestamp() * 1000)
        tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

        all_candles = []
        current_start = start_ms
        max_pages = 5000
        consecutive_empty = 0
        t0 = time.time()

        if self.verbose:
            total_est = max(1, (end_ms - start_ms) // tf_ms)
            print(f"      Binance {self.base_url} | {symbol} {interval} | "
                  f"~{total_est:,} barras esperadas")

        for page in range(1, max_pages + 1):
            if current_start >= end_ms:
                break

            # Llamada con startTime actual y sin endTime (Binance paginación forward)
            candles = await self.fetch_klines_page(
                symbol, interval, current_start, end_ms, KLINES_LIMIT
            )

            if not candles:
                consecutive_empty += 1
                if consecutive_empty > 3:
                    if self.verbose:
                        print(f"\n      ❌ {consecutive_empty} páginas vacías consecutivas, parando")
                    break
                await asyncio.sleep(0.5)
                continue

            consecutive_empty = 0
            all_candles.extend(candles)

            # Avanzar al siguiente inicio (último timestamp + 1 ms)
            last_ts = candles[-1][0]
            current_start = last_ts + 1

            # Progreso
            if self.verbose:
                n = len(all_candles)
                pct = min(100.0, (last_ts - start_ms) / max(1, end_ms - start_ms) * 100)
                elapsed = time.time() - t0
                eta = (elapsed / max(pct, 0.1) * (100 - pct)) if pct > 1 else 0
                dt_str = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                sys.stdout.write(
                    f"\r      {n:>7,} barras | {pct:5.1f}% | última={dt_str} | ETA {eta:.0f}s   "
                )
                sys.stdout.flush()

            # Pequeña pausa para no exceder rate limit
            await asyncio.sleep(RATE_LIMIT_SLEEP)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"\n      Descarga completa: {len(all_candles):,} barras en {elapsed:.1f}s")

        if not all_candles:
            return pd.DataFrame()

        # Convertir a DataFrame
        # Formato Binance: [openTime, open, high, low, close, volume, closeTime, quoteVolume, ...]
        df = pd.DataFrame(all_candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'closeTime', 'quoteVolume', 'trades', 'takerBase', 'takerQuote', 'ignore'
        ])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.drop_duplicates('timestamp').sort_values('timestamp')
        df = df[(df['timestamp'] >= start_ms) & (df['timestamp'] <= end_ms)]
        return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
#  FUNCIÓN SINCRÓNICA PARA USO DIRECTO
# ═══════════════════════════════════════════════════════════════

def download_data_binance(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    use_spot: bool = False,
    cache_dir: str = "./data/cache/binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Wrapper síncrono para descargar datos de Binance con caché.
    """
    from data.bitget_client import DataCache  # Reutilizamos DataCache (pero con subdirectorio)

    # Normalizar símbolo (Binance usa mayúsculas sin guiones)
    symbol = symbol.upper().replace('-', '').replace('/', '').replace('_', '')

    cache = DataCache(cache_dir)  # Nota: cache_dir específico para Binance

    if use_cache:
        if cache.has_data(symbol, timeframe):
            cached = cache.load(symbol, timeframe, start, end)
            if len(cached) > 0:
                # Verificar cobertura (opcional)
                return cached

    async def _download():
        client = BinanceClient(use_spot=use_spot, verbose=True)
        try:
            return await client.download_ohlcv(symbol, timeframe, start, end)
        finally:
            await client.close()

    df = asyncio.run(_download())

    if use_cache and len(df) > 0:
        cache.save(df, symbol, timeframe)

    return df