import os
import json
import time
from datetime import datetime, timezone

import yaml
import requests
import pandas as pd
import numpy as np
import gspread

TD_BASE = "https://api.twelvedata.com/time_series"
SETUP_NAME = "60D +30% Movers"


# ---------------------------
# Ticker parsing + inputs
# ---------------------------

def parse_symbols_from_text(text: str) -> list[str]:
    """
    Accepts:
      - One ticker per line: AAPL
      - TradingView export: NYSE:AA,NASDAQ:MSFT,...
    Converts EXCHANGE:TICKER -> TICKER:EXCHANGE
    Example: NASDAQ:MSFT -> MSFT:NASDAQ
    """
    tickers: list[str] = []
    seen = set()

    known_exchanges = {"NYSE", "NASDAQ", "AMEX", "NYSEARCA", "ARCA", "BATS", "IEX", "OTC"}

    if not text:
        return []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",") if p.strip()]
        for p in parts:
            if p.startswith("#"):
                continue

            p = p.strip()
            if ":" in p:
                left, right = p.split(":", 1)
                left_u = left.strip().upper()
                right_u = right.strip().upper()

                # TradingView style EXCHANGE:TICKER -> swap it
                if left_u in known_exchanges and right_u:
                    sym = f"{right_u}:{left_u}"
                else:
                    sym = p.strip().upper()
            else:
                sym = p.strip().upper()

            if sym and sym not in seen:
                tickers.append(sym)
                seen.add(sym)

    return tickers


def read_tickers_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_symbols_from_text(f.read())


def read_tickers_from_sheet(sh) -> list[str]:
    """
    Looks for a worksheet called 'Tickers'.
    Preferred:
      - A2 contains the TradingView export string (one long line is fine)
    Also supports:
      - Column A contains one symbol per row (starting A2)
    """
    try:
        ws = sh.worksheet("Tickers")
    except Exception:
        return []

    values = ws.col_values(1)  # column A
    if not values:
        return []

    a2 = values[1].strip() if len(values) >= 2 and values[1] else ""
    if a2:
        return parse_symbols_from_text(a2)

    lines = []
    for i, v in enumerate(values):
        if i == 0:
            continue
        v = (v or "").strip()
        if not v:
            continue
        lines.append(v)

    return parse_symbols_from_text("\n".join(lines))


def display_ticker(sym: str) -> str:
    # Convert MSFT:NASDAQ -> MSFT for display
    if ":" in sym:
        return sym.split(":", 1)[0].strip().upper()
    return sym.strip().upper()


def chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


# ---------------------------
# Twelve Data fetch
# ---------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def fetch_time_series_batch(api_key: str, symbols: list[str], interval: str, outputsize: int) -> dict:
    params = {
        "apikey": api_key,
        "interval": interval,
        "symbol": ",".join(symbols),
        "outputsize": outputsize,
        "format": "JSON",
    }

    # Retry with backoff on rate limiting and transient errors
    backoffs = [3, 10, 30]
    last_err = None

    for i in range(len(backoffs) + 1):
        try:
            r = requests.get(TD_BASE, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(backoffs[min(i, len(backoffs) - 1)])
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i < len(backoffs):
                time.sleep(backoffs[i])
                continue
            raise

    raise last_err


def normalise_timeseries_payload(symbol: str, payload: dict) -> pd.DataFrame:
    if not payload or payload.get("status") == "error":
        return pd.DataFrame()

    values = payload.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].map(safe_float)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime", "close", "high", "low"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["symbol"] = symbol
    return df


# ---------------------------
# Core criterion: 60D return > 30%
# ---------------------------

def pct_change(a: float, b: float) -> float:
    if a == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return 100.0 * (b / a - 1.0)


def return_over_bars(df: pd.DataFrame, bars: int) -> float:
    """
    Return from close[-(bars+1)] to close[-1]
    Example: bars=60 means approx last 60 trading days.
    """
    if df is None or df.empty or len(df) <= bars:
        return np.nan
    a = float(df["close"].iloc[-(bars + 1)])
    b = float(df["close"].iloc[-1])
    return pct_change(a, b)


def detect_60d_mover(df_full: pd.DataFrame, cfg: dict) -> dict:
    """
    ONLY rule:
      - if 60 trading day return > 30% -> WATCH
      - else PASS
    """
    rule = (cfg.get("rule", {}) or {})
    bars = int(rule.get("lookback_bars", 60))
    min_ret = float(rule.get("min_return_pct", 30.0))

    if df_full is None or df_full.empty:
        return {"signal": "PASS", "setup": SETUP_NAME, "score": 0, "entry": "", "stop": "", "ret_60d": "", "reason": "No data"}

    ret = return_over_bars(df_full, bars)
    if np.isnan(ret):
        return {"signal": "PASS", "setup": SETUP_NAME, "score": 0, "entry": "", "stop": "", "ret_60d": "", "reason": f"Not enough data for {bars} bars"}

    if ret >= min_ret:
        score = int(min(100, max(1, 50 + ret)))  # simple sorting score
        return {
            "signal": "WATCH",
            "setup": SETUP_NAME,
            "score": score,
            "entry": "",
            "stop": "",
            "ret_60d": f"{ret:.1f}",
            "reason": f"{bars}D return {ret:.1f}% ≥ {min_ret:.1f}%",
        }

    return {
        "signal": "PASS",
        "setup": SETUP_NAME,
        "score": 0,
        "entry": "",
        "stop": "",
        "ret_60d": f"{ret:.1f}",
        "reason": f"{bars}D return {ret:.1f}% < {min_ret:.1f}%",
    }


# ---------------------------
# Google Sheets helpers
# ---------------------------

def get_gspread_client(sa_json_text: str):
    sa_dict = json.loads(sa_json_text)
    return gspread.service_account_from_dict(sa_dict)


def upsert_worksheet(sh, title: str, rows: int = 1000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def ensure_run_log_header(ws_log):
    header = ["run_time_utc", "tickers", "buy_now", "watch", "errors", "api_calls", "credits_est", "notes"]
    existing = ws_log.get_all_values()

    if not existing:
        ws_log.update("A1", [header])
        return header

    first_row = existing[0]

    if not first_row or first_row[0] != "run_time_utc":
        ws_log.insert_row(header, 1)
        return header

    if first_row[:len(header)] != header:
        ws_log.update("A1", [header])

    return header


def rate_limit_wait(batch_credits: int, max_credits_per_min: int, state: dict) -> None:
    """
    Enforces a simple per-minute credits window.
    Assumes 1 credit per symbol in the batch.
    """
    if max_credits_per_min <= 0:
        return

    now = time.monotonic()
    window_start = state.get("window_start", now)
    used = int(state.get("used", 0))

    elapsed = now - window_start
    if elapsed >= 60.0:
        state["window_start"] = now
        state["used"] = 0
        window_start = now
        used = 0
        elapsed = 0.0

    if used + batch_credits <= max_credits_per_min:
        state["used"] = used + batch_credits
        return

    sleep_s = max(0.0, 60.0 - elapsed) + 0.2
    time.sleep(sleep_s)

    state["window_start"] = time.monotonic()
    state["used"] = batch_credits


# ---------------------------
# Main
# ---------------------------

def main():
    td_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not td_key or not sheet_id or not sa_json:
        raise SystemExit("Missing one or more secrets: TWELVEDATA_API_KEY, SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON")

    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    gc = get_gspread_client(sa_json)
    sh = gc.open_by_key(sheet_id)

    tickers = read_tickers_from_sheet(sh)
    tickers_source = "sheet"
    if not tickers:
        tickers = read_tickers_from_file("tickers.txt")
        tickers_source = "file"

    if not tickers:
        raise SystemExit("No tickers found (Tickers tab empty and tickers.txt empty)")

    interval = cfg.get("api", {}).get("interval", "1day")
    outputsize = int(cfg.get("api", {}).get("outputsize", 200))  # 60D calc needs ~61 bars; keep a buffer

    max_credits_per_min = int(cfg.get("api", {}).get("max_api_credits_per_min", 55))
    if max_credits_per_min < 1:
        max_credits_per_min = 55

    batch_size = int(cfg.get("api", {}).get("batch_size", 55))
    if batch_size < 1:
        batch_size = 55
    batch_size = min(batch_size, max_credits_per_min)

    results: list[dict] = []
    errors = 0
    api_calls = 0
    credits_est = 0

    rl_state = {"window_start": time.monotonic(), "used": 0}

    for sym_batch in chunks(tickers, batch_size):
        batch_credits = len(sym_batch)
        credits_est += batch_credits

        rate_limit_wait(batch_credits, max_credits_per_min, rl_state)

        try:
            data = fetch_time_series_batch(td_key, sym_batch, interval, outputsize)
            api_calls += 1

            # Single-symbol response shape
            if isinstance(data, dict) and "values" in data:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                if df.empty:
                    results.append({"ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "", "ret_60d": "", "reason": data.get("message", "No data")})
                else:
                    r = detect_60d_mover(df, cfg)
                    results.append({"ticker": sym, **r})
            else:
                # Multi-symbol response
                for sym in sym_batch:
                    payload = data.get(sym, {}) if isinstance(data, dict) else {}
                    df = normalise_timeseries_payload(sym, payload)
                    if df.empty:
                        results.append({
                            "ticker": sym,
                            "setup": SETUP_NAME,
                            "signal": "PASS",
                            "score": 0,
                            "entry": "",
                            "stop": "",
                            "ret_60d": "",
                            "reason": payload.get("message", "No data"),
                        })
                        continue

                    r = detect_60d_mover(df, cfg)
                    results.append({"ticker": sym, **r})

        except Exception as e:
            for sym in sym_batch:
                results.append({
                    "ticker": sym,
                    "setup": SETUP_NAME,
                    "signal": "PASS",
                    "score": 0,
                    "entry": "",
                    "stop": "",
                    "ret_60d": "",
                    "reason": f"Fetch error: {type(e).__name__}",
                })
            errors += 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(2000, len(results) + 10), cols=25)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=20)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=25)
    ws_summary = upsert_worksheet(sh, "Summary", rows=80, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "setup", "signal", "score", "ret_60d_pct", "reason", "as_of_utc"]
    signals_rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "ret_60d_pct", "reason", "as_of_utc"]
    watch_rows = [watch_header]

    buy_items = []
    watch_items = []

    for r in results:
        sym = r.get("ticker", "")
        setup = r.get("setup", SETUP_NAME)
        sig = r.get("signal", "PASS")
        score = int(r.get("score", 0) or 0)
        ret60 = r.get("ret_60d", "")
        reason = r.get("reason", "")

        signals_rows.append([sym, setup, sig, score, ret60, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, ret60, reason, now_utc]))

    watch_items.sort(key=lambda x: x[0], reverse=True)
    watch_rows.extend([row for _, row in watch_items])

    ws_signals.clear()
    ws_signals.update("A1", signals_rows)

    ws_buys.clear()
    ws_buys.update("A1", buy_rows)

    ws_watch.clear()
    ws_watch.update("A1", watch_rows)

    buy_count = 0
    watch_count = len(watch_items)
    pass_count = max(0, len(results) - buy_count - watch_count)

    note = f"ok ({tickers_source}) paced at {max_credits_per_min}/min, batch_size={batch_size}"

    summary_rows = [
        ["key", "value"],
        ["last_run_utc", now_utc],
        ["tickers_scanned", str(len(tickers))],
        ["results_rows", str(len(results))],
        ["buy_now_count", str(buy_count)],
        ["watch_count", str(watch_count)],
        ["pass_count", str(pass_count)],
        ["errors", str(errors)],
        ["api_calls", str(api_calls)],
        ["credits_est", str(credits_est)],
        ["source", tickers_source],
        ["note", note],
        ["setup", SETUP_NAME],
        ["outputsize", str(outputsize)],
    ]
    ws_summary.clear()
    ws_summary.update("A1", summary_rows)

    ensure_run_log_header(ws_log)
    ws_log.append_row(
        [now_utc, len(tickers), buy_count, watch_count, errors, api_calls, credits_est, note],
        value_input_option="USER_ENTERED",
    )

    print("WATCH signals (60D movers):")
    for _, row in watch_items:
        print(f"{row[0]} – 60D {row[3]}% – {row[4]}")
    print(f"Done. tickers={len(tickers)} results={len(results)} watch={watch_count} pass={pass_count} errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source}")


if __name__ == "__main__":
    main()
