#!/usr/bin/env python3
"""
build_universe.py

Builds universe.txt from Twelve Data:
- Source: /stocks?exchange=NASDAQ and /stocks?exchange=NYSE
- Filter: stocks only (exclude ETFs) using Twelve Data 'type'
- Filter: price > 1.50 using /price endpoint
- Writes: universe.txt (one symbol per line) sorted A-Z

Designed to run in GitHub Actions and commit the updated universe.txt back to the repo.
"""

import os
import sys
import time
import json
import math
from typing import List, Optional
import urllib.parse
import urllib.request


BASE_URL = "https://api.twelvedata.com"


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def http_get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "daily-stock-scanner/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON response from Twelve Data. URL={url} Body={data[:2000]}")


class RateLimiter:
    """Simple per-minute throttler (rolling 60s window)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.window_start = time.time()
        self.count = 0

    def wait(self) -> None:
        if self.max_per_minute <= 0:
            return

        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= 60:
            self.window_start = now
            self.count = 0

        if self.count >= self.max_per_minute:
            sleep_for = max(0.0, 60 - elapsed) + 0.25
            print(f"[pacing] Hit {self.max_per_minute}/min, sleeping {sleep_for:.2f}s")
            time.sleep(sleep_for)
            self.window_start = time.time()
            self.count = 0

        self.count += 1


def build_stocks_catalog(apikey: str, exchange: str) -> List[dict]:
    q = urllib.parse.urlencode({"apikey": apikey, "exchange": exchange})
    url = f"{BASE_URL}/stocks?{q}"
    data = http_get_json(url)
    if data.get("status") != "ok":
        raise RuntimeError(f"/stocks failed for {exchange}: {data}")
    return data.get("data", [])


def is_stock(rec: dict) -> bool:
    """
    Deterministic stock-only filter.

    Keep common equity-like instruments, exclude ETFs by construction.
    Tighten to {"Common Stock"} later if you wish.
    """
    t = (rec.get("type") or "").strip()
    keep = {"Common Stock", "Depositary Receipt", "American Depositary Receipt", "REIT"}
    return t in keep


def fetch_price(apikey: str, symbol: str) -> Optional[float]:
    q = urllib.parse.urlencode({"apikey": apikey, "symbol": symbol})
    url = f"{BASE_URL}/price?{q}"
    data = http_get_json(url)

    if "price" in data and data.get("price") not in (None, ""):
        try:
            return float(data["price"])
        except (TypeError, ValueError):
            return None

    if data.get("status") == "error":
        return None

    return None


def write_universe(path: str, symbols: List[str]) -> None:
    symbols = sorted(set(s.strip().upper() for s in symbols if s and s.strip()))
    with open(path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")


def main() -> int:
    apikey = os.getenv("TWELVE_DATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY") or ""
    if not apikey:
        print("ERROR: Missing Twelve Data API key. Set TWELVE_DATA_API_KEY (or TWELVEDATA_API_KEY).")
        return 2

    min_price = env_float("UNIVERSE_MIN_PRICE", 1.50)
    max_per_minute = env_int("TD_MAX_REQUESTS_PER_MINUTE", 50)
    max_symbols = env_int("UNIVERSE_MAX_SYMBOLS", 0)  # 0 = no cap

    limiter = RateLimiter(max_per_minute)

    exchanges = ["NASDAQ", "NYSE"]

    print("[universe] Fetching symbol catalog from Twelve Data /stocks ...")
    all_recs: List[dict] = []
    for ex in exchanges:
        limiter.wait()
        recs = build_stocks_catalog(apikey, ex)
        print(f"[universe] {ex}: {len(recs)} rows from /stocks")
        all_recs.extend(recs)

    stock_recs = [r for r in all_recs if is_stock(r)]
    print(f"[universe] After stock-only filter: {len(stock_recs)}")

    symbols = sorted(set((r.get("symbol") or "").strip() for r in stock_recs if (r.get("symbol") or "").strip()))
    print(f"[universe] Unique symbols pre-price-filter: {len(symbols)}")

    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]
        print(f"[universe] TEST MODE: limiting to first {max_symbols} symbols")

    kept: List[str] = []
    skipped_no_price = 0
    skipped_below = 0

    print(f"[universe] Applying price filter > {min_price:.2f} using /price (paced at {max_per_minute}/min) ...")
    for i, sym in enumerate(symbols, start=1):
        limiter.wait()
        px = fetch_price(apikey, sym)
        if px is None or (isinstance(px, float) and (math.isnan(px) or math.isinf(px))):
            skipped_no_price += 1
        else:
            if px > min_price:
                kept.append(sym)
            else:
                skipped_below += 1

        if i % 100 == 0:
            print(f"[universe] progress {i}/{len(symbols)} kept={len(kept)} no_price={skipped_no_price} below={skipped_below}")

    out_path = os.getenv("UNIVERSE_OUTFILE", "universe.txt")
    write_universe(out_path, kept)

    print("[universe] Done.")
    print(f"[universe] wrote: {out_path}")
    print(f"[universe] kept: {len(kept)}")
    print(f"[universe] skipped_no_price: {skipped_no_price}")
    print(f"[universe] skipped_below_threshold: {skipped_below}")
    print("[universe] preview:", ", ".join(kept[:20]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
