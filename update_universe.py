import os
import time
import requests

STOCKS_URL = "https://api.twelvedata.com/stocks"


def is_etf_like(row: dict) -> bool:
    """
    Twelve Data fields vary a bit depending on plan/endpoint version.
    We aggressively exclude anything that looks like an ETF.
    """
    blob = " ".join([
        str(row.get("type", "")),
        str(row.get("instrument_type", "")),
        str(row.get("security_type", "")),
        str(row.get("name", "")),
        str(row.get("symbol", "")),
    ]).upper()

    return "ETF" in blob or "EXCHANGE TRADED FUND" in blob or "ETN" in blob


def normalise_exchange(x: str) -> str:
    x = (x or "").strip().upper()
    # Twelve Data sometimes returns "NASDAQ" / "NASDAQ Capital Market" etc.
    if "NASDAQ" in x:
        return "NASDAQ"
    if "NYSE" in x:
        return "NYSE"
    return x


def fetch_stocks(api_key: str) -> list[dict]:
    """
    Fetch all stocks in one call if possible.
    Retries on rate limiting.
    """
    params = {"apikey": api_key, "format": "JSON"}
    backoffs = [3, 10, 30]
    last_err = None

    for i in range(len(backoffs) + 1):
        try:
            r = requests.get(STOCKS_URL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(backoffs[min(i, len(backoffs) - 1)])
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(data.get("message", "Twelve Data stocks endpoint error"))
            # typical shape: {"data":[...], "status":"ok"}
            rows = data.get("data", []) if isinstance(data, dict) else []
            return rows if isinstance(rows, list) else []
        except Exception as e:
            last_err = e
            if i < len(backoffs):
                time.sleep(backoffs[i])
                continue
            raise

    raise last_err


def main():
    api_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing TWELVEDATA_API_KEY")

    exchanges_keep = {"NASDAQ", "NYSE"}

    rows = fetch_stocks(api_key)

    symbols = []
    seen = set()

    for row in rows:
        # Some payloads use "symbol"; some use "ticker"
        sym = (row.get("symbol") or row.get("ticker") or "").strip().upper()
        if not sym:
            continue

        ex = normalise_exchange(row.get("exchange") or row.get("mic_code") or row.get("exchange_name") or "")
        if ex not in exchanges_keep:
            continue

        # Exclude ETFs (and ETF-like instruments)
        if is_etf_like(row):
            continue

        # We store as TICKER:EXCHANGE to match your existing internal format
        out_sym = f"{sym}:{ex}"

        if out_sym not in seen:
            symbols.append(out_sym)
            seen.add(out_sym)

    symbols.sort()

    # Write universe.txt
    out_path = "universe.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from Twelve Data /stocks\n")
        f.write("# Format: TICKER:EXCHANGE (e.g. MSFT:NASDAQ)\n")
        f.write("\n".join(symbols))
        f.write("\n")

    print(f"Wrote {len(symbols)} symbols to {out_path}")


if __name__ == "__main__":
    main()
