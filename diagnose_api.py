#!/usr/bin/env python3
"""
CryptoLab — API Diagnostic Tool
Run this to diagnose download issues.

Usage:
    python diagnose_api.py
"""
import asyncio
import json
import sys
import ssl

print("=" * 60)
print("  CryptoLab API Diagnostic")
print("=" * 60)

# 1. Check aiohttp
print("\n[1] Checking aiohttp...")
try:
    import aiohttp

    print(f"    ✅ aiohttp {aiohttp.__version__}")
except ImportError:
    print("    ❌ aiohttp not installed. Run: pip install aiohttp")
    sys.exit(1)

# 2. Check SSL
print("\n[2] Checking SSL...")
try:
    import certifi

    print(f"    ✅ certifi installed: {certifi.where()}")
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    print(f"    ✅ SSL context created with certifi")
except ImportError:
    print("    ⚠ certifi not installed. Trying ssl=False fallback...")
    ssl_ctx = None

# 3. Test raw HTTPS to Bitget
print("\n[3] Testing HTTPS connection to api.bitget.com...")


async def test_connection():
    connector_args = {}
    if ssl_ctx:
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    else:
        connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Test 1: Simple GET
        url = "https://api.bitget.com/api/v2/public/time"
        try:
            async with session.get(url) as resp:
                text = await resp.text()
                print(f"    Status: {resp.status}")
                print(f"    Response: {text[:200]}")
                if resp.status == 200:
                    print("    ✅ HTTPS connection OK")
                else:
                    print("    ❌ Unexpected status")
        except Exception as e:
            print(f"    ❌ Connection failed: {e}")
            print(f"    Type: {type(e).__name__}")

            # Try without SSL verification
            print("\n    Retrying with ssl=False...")
            connector2 = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector2) as session2:
                try:
                    async with session2.get(url) as resp2:
                        text2 = await resp2.text()
                        print(f"    Status: {resp2.status}")
                        print(f"    Response: {text2[:200]}")
                        print("    ✅ Works with ssl=False")
                except Exception as e2:
                    print(f"    ❌ Still failed: {e2}")
                    return False
            return True

        # Test 2: Candle endpoint
        print("\n[4] Testing candle endpoint...")

        # Test with the EXACT URL from Bitget docs
        url2 = ("https://api.bitget.com/api/v2/mix/market/candles"
                "?symbol=BTCUSDT&granularity=1H&limit=5&productType=usdt-futures")

        try:
            async with session.get(url2) as resp:
                data = await resp.json()
                print(f"    URL: {url2}")
                print(f"    Status: {resp.status}")
                print(f"    Response code: {data.get('code')}")
                print(f"    Response msg: {data.get('msg')}")
                n_candles = len(data.get('data', []))
                print(f"    Candles returned: {n_candles}")
                if n_candles > 0:
                    print(f"    Sample: {data['data'][0]}")
                    print("    ✅ Candle endpoint OK")
                else:
                    print(f"    Full response: {json.dumps(data, indent=2)[:500]}")
                    print("    ❌ No candles returned")
        except Exception as e:
            print(f"    ❌ Candle request failed: {e}")

        # Test 3: History candle endpoint
        print("\n[5] Testing history-candles endpoint...")

        # endTime = Jan 1, 2025 in ms
        end_ms = 1735689600000
        url3 = (f"https://api.bitget.com/api/v2/mix/market/history-candles"
                f"?symbol=BTCUSDT&granularity=1H&limit=5"
                f"&productType=usdt-futures&endTime={end_ms}")

        try:
            async with session.get(url3) as resp:
                data = await resp.json()
                print(f"    URL: ...history-candles?...endTime={end_ms}")
                print(f"    Status: {resp.status}")
                print(f"    Response code: {data.get('code')}")
                print(f"    Response msg: {data.get('msg')}")
                n_candles = len(data.get('data', []))
                print(f"    Candles returned: {n_candles}")
                if n_candles > 0:
                    print(f"    First: {data['data'][0]}")
                    print(f"    Last:  {data['data'][-1]}")
                    print("    ✅ History candle endpoint OK")
                else:
                    print(f"    Full response: {json.dumps(data, indent=2)[:500]}")
                    print("    ❌ No history candles returned")
        except Exception as e:
            print(f"    ❌ History candle request failed: {e}")

        # Test 4: Different granularity formats
        print("\n[6] Testing granularity formats...")
        for gran in ['1h', '1H', '60m', '60']:
            url4 = (f"https://api.bitget.com/api/v2/mix/market/candles"
                    f"?symbol=BTCUSDT&granularity={gran}&limit=2"
                    f"&productType=usdt-futures")
            try:
                async with session.get(url4) as resp:
                    data = await resp.json()
                    n = len(data.get('data', []))
                    code = data.get('code', '?')
                    msg = data.get('msg', '')[:30]
                    print(f"    granularity={gran:>4s} → code={code}, candles={n}, msg={msg}")
            except:
                print(f"    granularity={gran:>4s} → ERROR")

        return True


result = asyncio.run(test_connection())

print("\n" + "=" * 60)
if result:
    print("  Diagnostic complete. Check results above.")
else:
    print("  ❌ Cannot connect to Bitget API")
    print("  Possible fixes:")
    print("    pip install certifi")
    print("    pip install --upgrade certifi aiohttp")
print("=" * 60)