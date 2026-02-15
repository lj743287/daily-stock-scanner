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

    # Simple retry on rate limiting
    for attempt in range(2):
        r = requests.get(TD_BASE, params=params, timeout=30)
        if r.status_code == 429 and attempt == 0:
            time.sleep(65)
            continue
        r.raise_for_status()
        return r.json()

    return {}


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


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def compute_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def compute_return(df: pd.DataFrame, lookback: int) -> float:
    # Total return close(today)/close(N days ago) - 1
    if df is None or df.empty:
        return np.nan
    if len(df) < lookback + 2:
        return np.nan
    now = df["close"].iloc[-1]
    past = df["close"].iloc[-1 - lookback]
    if past is None or np.isnan(past) or past == 0:
        return np.nan
    return (now / past) - 1.0


def compute_leader_flags(df_map: dict[str, pd.DataFrame], cfg: dict) -> dict[str, bool]:
    qcfg = cfg.get("setup_qullamaggie_breakout", {}) or {}
    enabled = bool(qcfg.get("enabled", True))
    if not enabled:
        return {sym: False for sym in df_map.keys()}

    top_pct = float(qcfg.get("leader_top_pct", 2.0))
    windows = qcfg.get("momentum_windows", [21, 63, 126])
    windows = [int(x) for x in windows] if isinstance(windows, list) else [21, 63, 126]

    rets_by_sym: dict[str, dict[int, float]] = {}
    for sym, df in df_map.items():
        if df is None or df.empty:
            continue
        rets_by_sym[sym] = {}
        for w in windows:
            rets_by_sym[sym][w] = compute_return(df, w)

    thresholds: dict[int, float] = {}
    for w in windows:
        vals = [rets_by_sym[sym].get(w, np.nan) for sym in rets_by_sym.keys()]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 10:
            thresholds[w] = np.inf  # effectively disables leader test when too few symbols
        else:
            q = 1.0 - (top_pct / 100.0)
            thresholds[w] = float(np.quantile(vals, q))

    flags: dict[str, bool] = {}
    for sym in df_map.keys():
        ok = False
        if sym in rets_by_sym:
            for w in windows:
                v = rets_by_sym[sym].get(w, np.nan)
                thr = thresholds.get(w, np.inf)
                if not np.isnan(v) and v >= thr and v > 0:
                    ok = True
                    break
        flags[sym] = ok

    return flags


def pick_best(results: list[dict]) -> dict:
    # Prefer BUY_NOW > WATCH > PASS, then higher score
    rank = {"PASS": 0, "WATCH": 1, "BUY_NOW": 2}

    best = None
    for r in results:
        if not r:
            continue
        if best is None:
            best = r
            continue
        if rank.get(r.get("signal", "PASS"), 0) > rank.get(best.get("signal", "PASS"), 0):
            best = r
            continue
        if rank.get(r.get("signal", "PASS"), 0) == rank.get(best.get("signal", "PASS"), 0):
            if int(r.get("score", 0) or 0) > int(best.get("score", 0) or 0):
                best = r

    return best or {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": ""}


def eval_base_breakout(df: pd.DataFrame, cfg: dict, last: pd.Series) -> dict:
    out = {"signal": "PASS", "setup": "Base Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

    base_cfg = cfg.get("setup_base_breakout", {}) or {}
    base_n = int(base_cfg.get("base_lookback", 30))
    base_max_depth_pct = float(base_cfg.get("base_max_depth_pct", 15))
    breakout_buffer_pct = float(base_cfg.get("breakout_buffer_pct", 0.2))
    vol_mult = float(base_cfg.get("vol_multiplier", 1.2))
    atr_stop_mult = float(base_cfg.get("atr_stop_mult", 2.0))

    if len(df) < base_n + 2:
        out["signal"] = "WATCH"
        out["reason"] = "Trend ok, base window too small"
        out["score"] = 40
        return out

    base = df.iloc[-(base_n + 1):-1]
    base_high = base["high"].max()
    base_low = base["low"].min()
    depth_pct = 100.0 * (base_high - base_low) / base_high if base_high and base_high > 0 else 999.0
    tight_ok = depth_pct <= base_max_depth_pct

    pivot = base_high
    entry = pivot * (1 + breakout_buffer_pct / 100.0)
    breakout_ok = last["close"] > entry

    vol_ok = True
    if "volume" in df.columns and df["volume"].notna().any():
        base_vol_avg = base["volume"].mean()
        if not np.isnan(base_vol_avg) and base_vol_avg > 0 and not np.isnan(last.get("volume", np.nan)):
            vol_ok = last["volume"] >= base_vol_avg * vol_mult

    atr_val = last.get("atr", np.nan)
    stop_atr = entry - (atr_stop_mult * atr_val) if not np.isnan(atr_val) else base_low
    stop_swing = base_low
    stop = max(stop_swing, stop_atr)
    if stop >= entry:
        stop = stop_swing

    if not tight_ok:
        out["signal"] = "WATCH"
        out["reason"] = "Trend ok, but base not tight"
        out["score"] = 55
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{stop:.2f}"
        return out

    if not breakout_ok:
        out["signal"] = "WATCH"
        out["reason"] = "Trend ok, tight base, no breakout yet"
        out["score"] = 70
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{stop:.2f}"
        return out

    score = 40 + 30 + (10 if vol_ok else 0)
    tight_bonus = int(max(0, min(20, (base_max_depth_pct - depth_pct) * 1.5)))
    score = int(min(100, score + tight_bonus))

    out["signal"] = "BUY_NOW"
    out["score"] = score
    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["reason"] = "Trend up, tight base, broke pivot" + (", volume ok" if vol_ok else ", volume weak")
    return out


def eval_qullamaggie_breakout(df: pd.DataFrame, cfg: dict, last: pd.Series, leader_ok: bool) -> dict:
    out = {"signal": "PASS", "setup": "Qullamaggie Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

    qcfg = cfg.get("setup_qullamaggie_breakout", {}) or {}
    if not bool(qcfg.get("enabled", True)):
        out["reason"] = "Setup disabled"
        return out

    if not leader_ok:
        out["reason"] = "Not a momentum leader (top percentile in 1/3/6M within scan list)"
        return out

    atr_val = last.get("atr", np.nan)
    if np.isnan(atr_val) or atr_val <= 0:
        out["reason"] = "ATR unavailable"
        return out

    min_atr_pct = float(qcfg.get("min_atr_pct", 0.0))
    atr_pct = (atr_val / last["close"]) * 100.0 if last["close"] and last["close"] > 0 else 0.0
    if min_atr_pct > 0 and atr_pct < min_atr_pct:
        out["reason"] = f"ATR% too low ({atr_pct:.2f} < {min_atr_pct:.2f})"
        return out

    impulse_lookback = int(qcfg.get("impulse_lookback", 63))
    impulse_min_pct = float(qcfg.get("impulse_min_pct", 30.0))
    pullback_from_impulse_high_pct = float(qcfg.get("pullback_from_impulse_high_pct", 25.0))

    cons_min = int(qcfg.get("cons_min_bars", 10))
    cons_max = int(qcfg.get("cons_max_bars", 40))
    cons_depth_max_pct = float(qcfg.get("cons_depth_max_pct", 20.0))

    surf_pct = float(qcfg.get("surf_close_above_sma20_pct", 0.7))
    ma_rise_lookback = int(qcfg.get("ma_rise_lookback", 5))

    breakout_buffer_pct = float(qcfg.get("breakout_buffer_pct", 0.1))
    vol_mult = float(qcfg.get("vol_multiplier", 1.2))
    tr_mult = float(qcfg.get("tr_multiplier", 1.2))
    stop_max_atr_mult = float(qcfg.get("stop_max_atr_mult", 1.0))

    best = None

    # We exclude today from consolidation window; today is breakout-check day
    for cons_len in range(cons_min, cons_max + 1):
        need = cons_len + impulse_lookback + 2
        if len(df) < need:
            continue

        cons = df.iloc[-(cons_len + 1):-1]
        if cons.empty:
            continue

        cons_high = cons["high"].max()
        cons_low = cons["low"].min()
        if not cons_high or cons_high <= 0:
            continue

        depth_pct = 100.0 * (cons_high - cons_low) / cons_high
        if depth_pct > cons_depth_max_pct:
            continue

        # Higher lows: last third low > first third low
        n = len(cons)
        one_third = max(1, n // 3)
        early = cons.iloc[:one_third]
        late = cons.iloc[-one_third:]
        if late["low"].min() <= early["low"].min():
            continue

        # Surfing: % of closes above SMA20 in window
        if "sma20" not in cons.columns or cons["sma20"].isna().all():
            continue
        frac_above = float((cons["close"] >= cons["sma20"]).mean())
        if frac_above < surf_pct:
            continue

        # 10/20 rising into the end of the consolidation (yesterday vs N days ago)
        idx_end = len(df) - 2
        idx_prev = idx_end - ma_rise_lookback
        if idx_prev < 0:
            continue
        if df["sma10"].iloc[idx_end] <= df["sma10"].iloc[idx_prev]:
            continue
        if df["sma20"].iloc[idx_end] <= df["sma20"].iloc[idx_prev]:
            continue

        # Impulse window immediately before consolidation
        pre = df.iloc[-(cons_len + impulse_lookback + 1):-(cons_len + 1)]
        if pre.empty:
            continue

        low_idx = int(pre["low"].idxmin())
        high_idx = int(pre["high"].idxmax())
        if high_idx <= low_idx:
            continue

        impulse_low = float(pre["low"].min())
        impulse_high = float(pre["high"].max())
        if impulse_low <= 0:
            continue

        impulse_pct = 100.0 * (impulse_high / impulse_low - 1.0)
        if impulse_pct < impulse_min_pct:
            continue

        # Pullback constraint: consolidation stays within X% of impulse high
        min_cons_high = impulse_high * (1.0 - pullback_from_impulse_high_pct / 100.0)
        if cons_high < min_cons_high:
            continue

        # Score this consolidation candidate (prefer tighter)
        tight_bonus = int(max(0, min(25, (cons_depth_max_pct - depth_pct) * 1.5)))
        surf_bonus = int(max(0, min(10, (frac_above - surf_pct) * 20)))
        cand_score = 60 + tight_bonus + surf_bonus

        cand = {
            "cons_len": cons_len,
            "cons_high": cons_high,
            "cons_low": cons_low,
            "depth_pct": depth_pct,
            "frac_above": frac_above,
            "impulse_pct": impulse_pct,
            "score_base": int(min(85, cand_score)),
        }

        if best is None or cand["score_base"] > best["score_base"]:
            best = cand

    if best is None:
        out["reason"] = "No valid MA-surf consolidation found"
        return out

    pivot = best["cons_high"]
    entry = pivot * (1 + breakout_buffer_pct / 100.0)
    breakout_ok = last["close"] > entry

    # Range/volume expansion (scoring only)
    # TR today:
    prev_close = df["close"].iloc[-2]
    tr_today = max(
        abs(last["high"] - last["low"]),
        abs(last["high"] - prev_close),
        abs(last["low"] - prev_close),
    )
    cons = df.iloc[-(best["cons_len"] + 1):-1]
    tr_cons = (cons["high"] - cons["low"]).abs().mean() if not cons.empty else np.nan
    tr_ok = (not np.isnan(tr_cons)) and tr_today >= tr_cons * tr_mult

    vol_ok = True
    if "volume" in df.columns and df["volume"].notna().any():
        cons_vol = cons["volume"].mean()
        if not np.isnan(cons_vol) and cons_vol > 0 and not np.isnan(last.get("volume", np.nan)):
            vol_ok = last["volume"] >= cons_vol * vol_mult

    if not breakout_ok:
        # WATCH: use conservative support as stop proxy, but rule reminder in reason
        out["signal"] = "WATCH"
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{best['cons_low']:.2f}"
        out["score"] = int(min(100, best["score_base"]))
        out["reason"] = (
            f"Leader ok, impulse {best['impulse_pct']:.0f}%, tight {best['cons_len']}d surf, no breakout yet; "
            "stop on entry is LOD and must be <= 1x ATR"
        )
        return out

    # Breakout day stop rule: LOD, with ATR width constraint
    stop = float(last["low"])
    stop_dist = float(entry - stop)

    if stop_dist <= 0:
        out["reason"] = "Invalid stop (LOD >= entry)"
        return out

    if stop_dist > atr_val * stop_max_atr_mult:
        out["reason"] = "Stop too wide vs ATR (LOD rule fails)"
        return out

    score = best["score_base"] + 10
    if vol_ok:
        score += 5
    if tr_ok:
        score += 5
    score = int(min(100, score))

    out["signal"] = "BUY_NOW"
    out["score"] = score
    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["reason"] = (
        f"Leader ok, impulse {best['impulse_pct']:.0f}%, tight {best['cons_len']}d MA-surf, broke range"
        + (", volume ok" if vol_ok else ", volume weak")
        + (", range expanded" if tr_ok else "")
    )
    return out


def analyse_symbol(df: pd.DataFrame, cfg: dict, leader_ok: bool) -> dict:
    out = {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": ""}

    if df.empty or len(df) < 220:
        out["reason"] = "Not enough daily data"
        return out

    ema10_len = int(cfg.get("indicators", {}).get("ema_fast", 10))
    ema20_len = int(cfg.get("indicators", {}).get("ema_mid", 20))
    ema50_len = int(cfg.get("indicators", {}).get("ema_slow", 50))
    sma200_len = int(cfg.get("indicators", {}).get("sma_slow", 200))
    atr_len = int(cfg.get("indicators", {}).get("atr_len", 14))

    df = df.copy()
    df["ema10"] = compute_ema(df["close"], ema10_len)
    df["ema20"] = compute_ema(df["close"], ema20_len)
    df["ema50"] = compute_ema(df["close"], ema50_len)
    df["sma200"] = df["close"].rolling(sma200_len).mean()
    df["atr"] = compute_atr(df, atr_len)

    # SMAs for Qullamaggie surf rules
    df["sma10"] = compute_sma(df["close"], 10)
    df["sma20"] = compute_sma(df["close"], 20)
    df["sma50"] = compute_sma(df["close"], 50)

    last = df.iloc[-1]
    idx_last = df.index[-1]

    # Trend Template filter (global pre-filter)
    high_52w = df["high"].rolling(252).max().iloc[-1]
    near_pct = float(cfg.get("filters", {}).get("near_52w_high_pct", 25))
    near_52w_ok = last["close"] >= (1 - near_pct / 100.0) * high_52w

    if idx_last - 20 >= 0:
        sma200_up = df["sma200"].iloc[-1] > df["sma200"].iloc[-21]
    else:
        sma200_up = False

    ema_stack_ok = (last["ema10"] > last["ema20"] > last["ema50"])
    trend_ok = (ema_stack_ok and sma200_up and near_52w_ok)

    if not trend_ok:
        fails = []
        if not ema_stack_ok:
            fails.append("EMA stack")
        if not sma200_up:
            fails.append("200MA not rising")
        if not near_52w_ok:
            fails.append("not near 52w high")
        out["reason"] = "Trend fail: " + ", ".join(fails)
        return out

    # Evaluate setups and choose best
    r1 = eval_base_breakout(df, cfg, last)
    r2 = eval_qullamaggie_breakout(df, cfg, last, leader_ok)

    best = pick_best([r1, r2])
    return best


def get_gspread_client(sa_json_text: str):
    sa_dict = json.loads(sa_json_text)
    return gspread.service_account_from_dict(sa_dict)


def upsert_worksheet(sh, title: str, rows: int = 1000, cols: int = 20):
    try:
        return sh.worksheet(title)
    except Exception:
        return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def ensure_run_log_header(ws_log):
    header = ["run_time_utc", "tickers", "buy_now", "errors", "api_calls", "credits_est", "notes"]
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


def main():
    td_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not td_key or not sheet_id or not sa_json:
        raise SystemExit("Missing one or more secrets: TWELVEDATA_API_KEY, SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON")

    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    max_tickers_per_run = int(cfg.get("api", {}).get("max_tickers_per_run", 300))
    if max_tickers_per_run < 1:
        max_tickers_per_run = 300

    gc = get_gspread_client(sa_json)
    sh = gc.open_by_key(sheet_id)

    tickers = read_tickers_from_sheet(sh)
    tickers_source = "sheet"
    if not tickers:
        tickers = read_tickers_from_file("tickers.txt")
        tickers_source = "file"

    if not tickers:
        raise SystemExit("No tickers found (Tickers tab empty and tickers.txt empty)")

    total_before_cap = len(tickers)
    capped = False
    if total_before_cap > max_tickers_per_run:
        tickers = tickers[:max_tickers_per_run]
        capped = True

    interval = cfg.get("api", {}).get("interval", "1day")
    outputsize = int(cfg.get("api", {}).get("outputsize", 260))
    batch_size = int(cfg.get("api", {}).get("batch_size", 8))

    max_credits_per_min = int(cfg.get("api", {}).get("max_api_credits_per_min", 8))
    if max_credits_per_min < 1:
        max_credits_per_min = 1

    df_map: dict[str, pd.DataFrame] = {}
    err_map: dict[str, str] = {}

    errors = 0
    api_calls = 0
    credits_est = 0

    for sym_batch in chunks(tickers, batch_size):
        batch_credits = len(sym_batch)
        credits_est += batch_credits

        try:
            data = fetch_time_series_batch(td_key, sym_batch, interval, outputsize)
            api_calls += 1

            # Single-symbol response shape: {"values":[...], ...}
            if isinstance(data, dict) and "values" in data and len(sym_batch) == 1:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                df_map[sym] = df
                if df.empty:
                    err_map[sym] = data.get("message", "No data")
            else:
                # Multi-symbol response shape: { "AAPL": {...}, "MSFT": {...}, ... }
                for sym in sym_batch:
                    payload = {}
                    if isinstance(data, dict):
                        payload = data.get(sym, {}) or {}
                    df = normalise_timeseries_payload(sym, payload)
                    df_map[sym] = df
                    if df.empty:
                        msg = payload.get("message")
                        if not msg and isinstance(data, dict):
                            msg = data.get("message")
                        err_map[sym] = msg or "No data"

        except Exception as e:
            for sym in sym_batch:
                df_map[sym] = pd.DataFrame()
                err_map[sym] = f"Fetch error: {type(e).__name__}"
            errors += 1

        sleep_s = (batch_credits / max_credits_per_min) * 60.0
        time.sleep(sleep_s)

    leader_flags = compute_leader_flags(df_map, cfg)

    results = []
    for sym in tickers:
        df = df_map.get(sym, pd.DataFrame())
        if df is None or df.empty:
            results.append((sym, {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": err_map.get(sym, "No data")}))
            continue
        res = analyse_symbol(df, cfg, leader_flags.get(sym, False))
        results.append((sym, res))

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(1000, len(results) + 10), cols=12)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=500, cols=12)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=1000, cols=12)
    ws_summary = upsert_worksheet(sh, "Summary", rows=50, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "signal", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    watch_rows = [watch_header]

    buy_count = 0
    buy_items = []
    watch_items = []

    for sym, res in results:
        sig = res.get("signal", "")
        setup = res.get("setup", "")
        score = int(res.get("score", 0) or 0)
        entry = res.get("entry", "")
        stop = res.get("stop", "")
        reason = res.get("reason", "")

        rows.append([sym, sig, setup, score, entry, stop, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "BUY_NOW":
            buy_count += 1
            line = f"{sym_disp} – BUY NOW – Setup: {setup} – Entry: {entry} – Stop: {stop} – Reason: {reason}"
            buy_items.append((score, [line, sym_disp, setup, score, entry, stop, reason, now_utc]))

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, entry, stop, reason, now_utc]))

    watch_count = len(watch_items)
    pass_count = max(0, len(tickers) - buy_count - watch_count)

    ws_signals.clear()
    ws_signals.update("A1", rows)

    buy_items.sort(key=lambda x: x[0], reverse=True)
    buy_rows.extend([row for _, row in buy_items])
    ws_buys.clear()
    ws_buys.update("A1", buy_rows)

    watch_items.sort(key=lambda x: x[0], reverse=True)
    watch_rows.extend([row for _, row in watch_items])
    ws_watch.clear()
    ws_watch.update("A1", watch_rows)

    note = f"ok ({tickers_source})"
    if capped:
        note = f"ok ({tickers_source}) capped {len(tickers)} of {total_before_cap}"

    summary_rows = [
        ["key", "value"],
        ["last_run_utc", now_utc],
        ["tickers_scanned", str(len(tickers))],
        ["buy_now_count", str(buy_count)],
        ["watch_count", str(watch_count)],
        ["pass_count", str(pass_count)],
        ["errors", str(errors)],
        ["api_calls", str(api_calls)],
        ["credits_est", str(credits_est)],
        ["source", tickers_source],
        ["note", note],
    ]
    ws_summary.clear()
    ws_summary.update("A1", summary_rows)

    ensure_run_log_header(ws_log)
    ws_log.append_row(
        [now_utc, len(tickers), buy_count, errors, api_calls, credits_est, note],
        value_input_option="USER_ENTERED",
    )

    print("BUY_NOW signals:")
    for _, r in buy_items:
        print(r[0])
    print(
        f"Done. tickers={len(tickers)} buy_now={buy_count} watch={watch_count} pass={pass_count} "
        f"errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source} capped={capped}"
    )


if __name__ == "__main__":
    main()
