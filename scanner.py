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


def read_tickers(path: str) -> list[str]:
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().upper()
            if t and not t.startswith("#"):
                tickers.append(t)
    return tickers


def chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i:i + n] for i in range(0, len(items), n)]


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
    r = requests.get(TD_BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


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


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def analyse_symbol(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": ""}

    if df.empty or len(df) < 220:
        out["reason"] = "Not enough daily data"
        return out

    sma50 = df["close"].rolling(cfg["indicators"]["sma_fast"]).mean()
    sma150 = df["close"].rolling(cfg["indicators"]["sma_mid"]).mean()
    sma200 = df["close"].rolling(cfg["indicators"]["sma_slow"]).mean()
    atr = compute_atr(df, cfg["indicators"]["atr_len"])

    df = df.copy()
    df["sma50"] = sma50
    df["sma150"] = sma150
    df["sma200"] = sma200
    df["atr"] = atr
    df["dollar_vol"] = df["close"] * df.get("volume", 0)

    last = df.iloc[-1]
    idx_last = df.index[-1]

    high_52w = df["high"].rolling(252).max().iloc[-1]
    near_52w_ok = last["close"] >= (1 - cfg["filters"]["near_52w_high_pct"] / 100.0) * high_52w

    dv50 = df["dollar_vol"].rolling(50).mean().iloc[-1]
    liquidity_ok = dv50 >= cfg["filters"]["min_dollar_vol_50d"]

    if idx_last - 20 >= 0:
        sma200_up = df["sma200"].iloc[-1] > df["sma200"].iloc[-21]
    else:
        sma200_up = False

    ma_stack_ok = (last["close"] > last["sma50"] > last["sma150"] > last["sma200"])
    trend_ok = (ma_stack_ok and sma200_up and near_52w_ok and liquidity_ok)

    if not trend_ok:
        fails = []
        if not ma_stack_ok:
            fails.append("MA stack")
        if not sma200_up:
            fails.append("200MA not rising")
        if not near_52w_ok:
            fails.append("not near 52w high")
        if not liquidity_ok:
            fails.append("liquidity")
        out["reason"] = "Trend fail: " + ", ".join(fails)
        return out

    base_n = cfg["setup_base_breakout"]["base_lookback"]
    if len(df) < base_n + 2:
        out["signal"] = "WATCH"
        out["reason"] = "Trend ok, base window too small"
        out["score"] = 40
        return out

    base = df.iloc[-(base_n + 1):-1]
    base_high = base["high"].max()
    base_low = base["low"].min()
    depth_pct = 100.0 * (base_high - base_low) / base_high if base_high and base_high > 0 else 999.0
    tight_ok = depth_pct <= cfg["setup_base_breakout"]["base_max_depth_pct"]

    pivot = base_high
    buffer = cfg["setup_base_breakout"]["breakout_buffer_pct"] / 100.0
    entry = pivot * (1 + buffer)

    breakout_ok = last["close"] > entry

    vol_mult = cfg["setup_base_breakout"]["vol_multiplier"]
    vol_ok = True
    if "volume" in df.columns and df["volume"].notna().any():
        base_vol_avg = base["volume"].mean()
        if not np.isnan(base_vol_avg) and base_vol_avg > 0 and not np.isnan(last.get("volume", np.nan)):
            vol_ok = last["volume"] >= base_vol_avg * vol_mult

    if not tight_ok:
        out["signal"] = "WATCH"
        out["setup"] = "Base Breakout"
        out["reason"] = "Trend ok, but base not tight"
        out["score"] = 55
        out["entry"] = f"{entry:.2f}"
        return out

    if not breakout_ok:
        out["signal"] = "WATCH"
        out["setup"] = "Base Breakout"
        out["reason"] = "Trend ok, tight base, no breakout yet"
        out["score"] = 70
        out["entry"] = f"{entry:.2f}"
        return out

    atr_mult = cfg["setup_base_breakout"]["atr_stop_mult"]
    atr_val = last.get("atr", np.nan)
    stop_atr = entry - (atr_mult * atr_val) if not np.isnan(atr_val) else base_low
    stop_swing = base_low
    stop = max(stop_swing, stop_atr)
    if stop >= entry:
        stop = stop_swing

    score = 40 + 30 + (10 if vol_ok else 0)
    tight_bonus = int(max(0, min(20, (cfg["setup_base_breakout"]["base_max_depth_pct"] - depth_pct) * 1.5)))
    score = int(min(100, score + tight_bonus))

    out["signal"] = "BUY_NOW"
    out["setup"] = "Base Breakout"
    out["score"] = score
    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["reason"] = "Trend up, near 52w high, tight base, broke pivot" + (", volume ok" if vol_ok else ", volume weak")
    return out


def get_gspread_client(sa_json_text: str):
    sa_dict = json.loads(sa_json_text)
    return gspread.service_account_from_dict(sa_dict)


def upsert_worksheet(sh, title: str, rows: int = 1000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def main():
    td_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not td_key or not sheet_id or not sa_json:
        raise SystemExit("Missing one or more secrets: TWELVEDATA_API_KEY, SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON")

    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tickers = read_tickers("tickers.txt")
    if not tickers:
        raise SystemExit("tickers.txt is empty")

    interval = cfg["api"]["interval"]
    outputsize = int(cfg["api"]["outputsize"])
    batch_size = int(cfg["api"]["batch_size"])
    max_rpm = max(1, int(cfg["api"]["max_requests_per_min"]))
    sleep_s = 60.0 / max_rpm

    results = []
    errors = 0
    api_calls = 0

    for sym_batch in chunks(tickers, batch_size):
        try:
            data = fetch_time_series_batch(td_key, sym_batch, interval, outputsize)
            api_calls += 1

            if isinstance(data, dict) and "values" in data:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                res = analyse_symbol(df, cfg)
                results.append((sym, res))
            else:
                for sym in sym_batch:
                    payload = data.get(sym, {})
                    df = normalise_timeseries_payload(sym, payload)
                    res = analyse_symbol(df, cfg) if not df.empty else {
                        "signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "",
                        "reason": payload.get("message", "No data"),
                    }
                    results.append((sym, res))

        except Exception as e:
            for sym in sym_batch:
                results.append((sym, {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": f"Fetch error: {type(e).__name__}"}))
            errors += 1

        time.sleep(sleep_s)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    gc = get_gspread_client(sa_json)
    sh = gc.open_by_key(sheet_id)

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(1000, len(results) + 10), cols=12)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "signal", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    rows = [header]
    buy_count = 0

    for sym, res in results:
        if res["signal"] == "BUY_NOW":
            buy_count += 1
        rows.append([
            sym,
            res.get("signal", ""),
            res.get("setup", ""),
            res.get("score", 0),
            res.get("entry", ""),
            res.get("stop", ""),
            res.get("reason", ""),
            now_utc,
        ])

    ws_signals.clear()
    ws_signals.update("A1", rows)

    log_header = ["run_time_utc", "tickers", "buy_now", "errors", "api_calls", "notes"]
    existing = ws_log.get_all_values()
    if not existing:
        ws_log.update("A1", [log_header])

    ws_log.append_row([now_utc, len(tickers), buy_count, errors, api_calls, "ok"], value_input_option="USER_ENTERED")

    buy_lines = []
    for sym, res in results:
        if res["signal"] == "BUY_NOW":
            buy_lines.append(f"{sym} – BUY NOW – Setup: Base Breakout – Entry: {res['entry']} – Stop: {res['stop']} – Reason: {res['reason']}")

    print("BUY_NOW signals:")
    for line in buy_lines:
        print(line)
    print(f"Done. tickers={len(tickers)} buy_now={buy_count} errors={errors} api_calls={api_calls}")


if __name__ == "__main__":
    main()
