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
SETUP_NAME = "RUN-UP + EMA Touch Triangle (10/20/50)"


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
# Rule helpers
# ---------------------------

def ema_series(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def touches_level(low: pd.Series, high: pd.Series, level: pd.Series) -> pd.Series:
    return (low <= level) & (high >= level)


def pivot_low_centres(low: pd.Series, left: int, right: int) -> np.ndarray:
    """
    Centre-based pivot low test:
      low[i] == min(low[i-left : i+right]) inclusive window.
    """
    n = len(low)
    out = np.zeros(n, dtype=bool)
    l = low.values
    for i in range(left, n - right):
        w = l[i - left:i + right + 1]
        if np.isnan(l[i]) or np.isnan(w).all():
            continue
        if l[i] <= np.nanmin(w):
            out[i] = True
    return out


# ---------------------------
# RUN-UP + EMA Touch Triangle detector
# ---------------------------

def detect_runup_ema_touch_triangle(df: pd.DataFrame, cfg: dict) -> dict:
    """
    WATCH trigger when triangle would be painted, then filter on:
      - Triangle within last watch_last_n_bars (default 5)
      - EMA stack on run-up TOP bar: EMA10 > EMA20 > EMA50
      - Not too new (min_history_days default 90)
      - Min price (min_price default 2.0)
      - AFTER triangle: at least 1 higher low in last higher_low_lookback bars
        higher low definition: low[i] > low[i-1]
    """
    rcfg = (cfg.get("runup_ema_touch", {}) or {})
    baseBars = int(rcfg.get("baseBars", 20))
    minRunUpPct = float(rcfg.get("minRunUpPct", 30.0))
    minPullbackPct = float(rcfg.get("minPullbackPct", 5.0))
    pivotL = int(rcfg.get("pivotL", 3))
    pivotR = int(rcfg.get("pivotR", 3))
    pullbackBasis = str(rcfg.get("pullbackBasis", "Close"))
    priority = str(rcfg.get("priority", "10>20>50"))

    emaSrcCol = str(rcfg.get("emaSrc", "close")).lower()
    len10 = int(rcfg.get("len10", 10))
    len20 = int(rcfg.get("len20", 20))
    len50 = int(rcfg.get("len50", 50))

    min_price = float(rcfg.get("min_price", 2.0))
    min_history_days = int(rcfg.get("min_history_days", 90))

    watch_last_n_bars = int(rcfg.get("watch_last_n_bars", 5))
    if watch_last_n_bars < 1:
        watch_last_n_bars = 1

    hl_lookback = int(rcfg.get("higher_low_lookback", 10))
    if hl_lookback < 2:
        hl_lookback = 2

    out = {
        "signal": "PASS",
        "setup": SETUP_NAME,
        "score": 0,
        "entry": "",
        "stop": "",
        "pivot": "",
        "base_weeks": "",
        "contractions": "",
        "depths": "",
        "risk_pct": "",
        "runup_pct": "",
        "reason": "",
    }

    min_len = max(120, baseBars + pivotL + pivotR + 10, len50 + 10)
    if df is None or df.empty or len(df) < min_len:
        out["reason"] = "Not enough daily data"
        return out

    if "datetime" not in df.columns or df["datetime"].isna().all():
        out["reason"] = "No datetime series"
        return out

    # Filter: age
    first_dt = df["datetime"].iloc[0]
    last_dt = df["datetime"].iloc[-1]
    try:
        age_days = int((last_dt - first_dt).days)
    except Exception:
        age_days = 0

    if age_days < min_history_days:
        out["reason"] = f"Too new (<{min_history_days} days history)"
        return out

    # Filter: min price (latest close)
    last_close = float(df["close"].iloc[-1])
    if np.isnan(last_close) or last_close < min_price:
        out["reason"] = f"Price below minimum (${min_price:.2f})"
        return out

    # Series
    src = df[emaSrcCol] if emaSrcCol in df.columns else df["close"]

    ema10 = ema_series(src, len10)
    ema20 = ema_series(src, len20)
    ema50 = ema_series(src, len50)

    t10 = touches_level(df["low"], df["high"], ema10)
    t20 = touches_level(df["low"], df["high"], ema20)
    t50 = touches_level(df["low"], df["high"], ema50)

    t10_3 = t10 & t10.shift(1).fillna(False) & t10.shift(2).fillna(False)
    t20_3 = t20 & t20.shift(1).fillna(False) & t20.shift(2).fillna(False)
    t50_3 = t50 & t50.shift(1).fillna(False) & t50.shift(2).fillna(False)

    priorBaseHigh = df["high"].rolling(baseBars).max().shift(1)
    breakoutNow = (df["close"] > priorBaseHigh) & (df["close"].shift(1) <= priorBaseHigh)

    piv_centres = pivot_low_centres(df["low"], pivotL, pivotR)

    # State (matches Pine)
    lowIdxArr: list[int] = []
    lowPxArr: list[float] = []

    inRunUp = False
    waitingForTouch = False
    armedFrom = None

    anchorX = None
    anchorPx = None
    topX = None
    topPx = None

    last_triangle_idx = None
    last_triangle_ema = None
    last_runup_pct = None
    last_breakout_idx = None
    last_runup_top_idx = None
    last_trigger_idx = None

    last_fail_top_ema_stack = False

    n = len(df)

    def last_pivot_before(x: int):
        for j in range(len(lowIdxArr) - 1, -1, -1):
            if lowIdxArr[j] < x:
                return lowIdxArr[j], lowPxArr[j]
        return None, None

    for i in range(n):
        # Confirm pivots when we reach centre + pivotR
        centre = i - pivotR
        if centre >= 0 and piv_centres[centre]:
            lowIdxArr.append(centre)
            lowPxArr.append(float(df["low"].iloc[centre]))

        # Breakout starts run-up
        if bool(breakoutNow.iloc[i]):
            ax, ap = last_pivot_before(i)
            if ax is not None and ap is not None and ap > 0:
                inRunUp = True
                waitingForTouch = False
                armedFrom = None

                anchorX = ax
                anchorPx = float(ap)
                topX = i
                topPx = float(df["high"].iloc[i])

                last_breakout_idx = i
                last_runup_top_idx = i
                last_trigger_idx = None
                last_fail_top_ema_stack = False

        # Track top, then trigger run-up label on pullback
        if inRunUp and anchorPx is not None and topPx is not None and topX is not None:
            hi = float(df["high"].iloc[i])
            if hi >= float(topPx):
                topPx = hi
                topX = i
                last_runup_top_idx = i

            basisPx = float(df["low"].iloc[i]) if pullbackBasis.lower() == "low" else float(df["close"].iloc[i])
            pbPct = ((topPx - basisPx) / topPx) * 100.0 if topPx > 0 else np.nan

            pullingBack = (i > topX) and (not np.isnan(pbPct)) and (pbPct >= minPullbackPct) and (hi < topPx)

            chg = float(topPx) - float(anchorPx)
            rupPct = (chg / float(anchorPx)) * 100.0 if anchorPx > 0 else np.nan

            trigger = pullingBack and (not np.isnan(rupPct)) and (rupPct >= minRunUpPct)

            if trigger:
                # EMA stack check on the TOP bar (topX)
                e10_top = float(ema10.iloc[topX])
                e20_top = float(ema20.iloc[topX])
                e50_top = float(ema50.iloc[topX])

                top_stack_ok = (
                    (not np.isnan(e10_top)) and (not np.isnan(e20_top)) and (not np.isnan(e50_top))
                    and (e10_top > e20_top > e50_top)
                )

                if not top_stack_ok:
                    inRunUp = False
                    waitingForTouch = False
                    armedFrom = None
                    last_fail_top_ema_stack = True
                    last_runup_pct = float(rupPct) if not np.isnan(rupPct) else None
                    last_trigger_idx = i
                else:
                    inRunUp = False
                    waitingForTouch = True
                    armedFrom = i
                    last_runup_pct = float(rupPct)
                    last_trigger_idx = i

        # After run-up trigger: FIRST EMA touch for 3 consecutive bars
        if waitingForTouch and armedFrom is not None and i >= armedFrom:
            anyTouch3 = bool(t10_3.iloc[i]) or bool(t20_3.iloc[i]) or bool(t50_3.iloc[i])
            if anyTouch3:
                if priority == "10>20>50":
                    which = 10 if bool(t10_3.iloc[i]) else 20 if bool(t20_3.iloc[i]) else 50
                else:
                    d10 = abs(float(df["close"].iloc[i]) - float(ema10.iloc[i]))
                    d20 = abs(float(df["close"].iloc[i]) - float(ema20.iloc[i]))
                    d50 = abs(float(df["close"].iloc[i]) - float(ema50.iloc[i]))
                    candidates = []
                    if bool(t10_3.iloc[i]):
                        candidates.append((d10, 10))
                    if bool(t20_3.iloc[i]):
                        candidates.append((d20, 20))
                    if bool(t50_3.iloc[i]):
                        candidates.append((d50, 50))
                    which = sorted(candidates, key=lambda x: x[0])[0][1] if candidates else 50

                last_triangle_idx = i
                last_triangle_ema = which
                waitingForTouch = False
                armedFrom = None

    # Helper to format dates
    def fmt_idx(idx):
        if idx is None:
            return ""
        dtx = df["datetime"].iloc[idx]
        return dtx.strftime("%Y-%m-%d") if hasattr(dtx, "strftime") else str(dtx)

    # Triangle recency window
    last_bar = n - 1
    tri_window_start = max(0, last_bar - (watch_last_n_bars - 1))

    if last_triangle_idx is not None and last_triangle_idx >= tri_window_start:
        # Higher lows filter AFTER triangle, within last hl_lookback bars from latest
        hl_window_start = max(1, last_bar - (hl_lookback - 1))  # start at 1 so i-1 exists
        start_i = max(hl_window_start, last_triangle_idx + 1)

        higher_low_found = False
        lows = df["low"].values

        for i in range(start_i, last_bar + 1):
            if i - 1 < 0:
                continue
            lo = lows[i]
            prev = lows[i - 1]
            if np.isnan(lo) or np.isnan(prev):
                continue
            if lo > prev:
                higher_low_found = True
                break

        if not higher_low_found:
            dt = df["datetime"].iloc[last_triangle_idx]
            dt_s = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            out["reason"] = f"Triangle on {dt_s} but no higher low after triangle (last {hl_lookback} bars)"
            out["runup_pct"] = f"{(last_runup_pct or 0.0):.1f}" if last_runup_pct is not None else ""
            out["score"] = int(round(last_runup_pct or 0.0)) if last_runup_pct is not None else 0
            return out

        # WATCH (passes HL filter)
        dt = df["datetime"].iloc[last_triangle_idx]
        dt_s = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)

        out["signal"] = "WATCH"
        out["score"] = int(round(last_runup_pct or 0.0))
        out["runup_pct"] = f"{(last_runup_pct or 0.0):.1f}"
        out["reason"] = (
            f"Triangle on {dt_s} (last {watch_last_n_bars} bars) + higher low after; "
            f"EMA{last_triangle_ema} touched 3 bars; breakout {fmt_idx(last_breakout_idx)}, "
            f"top {fmt_idx(last_runup_top_idx)}, pullback-trigger {fmt_idx(last_trigger_idx)}"
        )
        return out

    # PASS reasons
    if last_breakout_idx is None:
        out["reason"] = "No breakout run-up cycle found"
    elif last_fail_top_ema_stack:
        out["reason"] = "Run-up triggered but EMA stack failed on run-up top (EMA10>EMA20>EMA50)"
    elif last_trigger_idx is None:
        out["reason"] = "Breakout seen but no run-up trigger (pullback/run-up thresholds not met)"
    elif last_triangle_idx is None:
        out["reason"] = "Run-up triggered but no 3-consecutive EMA touch afterwards"
    else:
        tdt = df["datetime"].iloc[last_triangle_idx]
        tdt_s = tdt.strftime("%Y-%m-%d") if hasattr(tdt, "strftime") else str(tdt)
        out["reason"] = f"Triangle occurred on {tdt_s}, not within last {watch_last_n_bars} bars"

    out["runup_pct"] = f"{(last_runup_pct or 0.0):.1f}" if last_runup_pct is not None else ""
    out["score"] = int(round(last_runup_pct or 0.0)) if last_runup_pct is not None else 0
    return out


def detect_setups(df_full: pd.DataFrame, cfg: dict) -> list[dict]:
    return [detect_runup_ema_touch_triangle(df_full, cfg)]


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
    # Accept either name (workflow sets TWELVE_DATA_API_KEY, older setups used TWELVEDATA_API_KEY)
    td_key = (os.environ.get("TWELVE_DATA_API_KEY", "") or os.environ.get("TWELVEDATA_API_KEY", "")).strip()
    sheet_id = os.environ.get("SHEET_ID", "").strip()
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

    if not td_key or not sheet_id or not sa_json:
        raise SystemExit("Missing one or more secrets: TWELVE_DATA_API_KEY/TWELVEDATA_API_KEY, SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON")

    with open("config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    gc = get_gspread_client(sa_json)
    sh = gc.open_by_key(sheet_id)

    # --- NEW: ticker source is universe file in repo ---
    universe_file = os.environ.get("UNIVERSE_FILE", "universe.txt").strip() or "universe.txt"
    tickers = read_tickers_from_file(universe_file)
    tickers_source = f"universe:{universe_file}"

    if not tickers:
        raise SystemExit(f"No tickers found in {universe_file}")

    # Optional: cap tickers for testing scan runtime
    max_tickers = int(os.environ.get("UNIVERSE_MAX_TICKERS", "0") or "0")
    if max_tickers and max_tickers > 0:
        tickers = tickers[:max_tickers]
        tickers_source += f" (capped {max_tickers})"

    interval = cfg.get("api", {}).get("interval", "1day")
    outputsize = int(cfg.get("api", {}).get("outputsize", 520))

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
                    results.append({
                        "ticker": sym,
                        "setup": SETUP_NAME,
                        "signal": "PASS",
                        "score": 0,
                        "entry": "",
                        "stop": "",
                        "pivot": "",
                        "base_weeks": "",
                        "contractions": "",
                        "depths": "",
                        "risk_pct": "",
                        "runup_pct": "",
                        "reason": data.get("message", "No data"),
                    })
                else:
                    for r in detect_setups(df, cfg):
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
                            "pivot": "",
                            "base_weeks": "",
                            "contractions": "",
                            "depths": "",
                            "risk_pct": "",
                            "runup_pct": "",
                            "reason": payload.get("message", "No data"),
                        })
                        continue

                    for r in detect_setups(df, cfg):
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
                    "pivot": "",
                    "base_weeks": "",
                    "contractions": "",
                    "depths": "",
                    "risk_pct": "",
                    "runup_pct": "",
                    "reason": f"Fetch error: {type(e).__name__}",
                })
            errors += 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(2000, len(results) + 10), cols=20)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=20)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=20)
    ws_summary = upsert_worksheet(sh, "Summary", rows=80, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "setup", "signal", "score", "entry", "stop", "pivot", "base_weeks", "contractions", "depths_pct", "risk_pct", "runup_pct", "reason", "as_of_utc"]
    signals_rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "pivot", "base_weeks", "contractions", "risk_pct", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "pivot", "base_weeks", "contractions", "risk_pct", "reason", "as_of_utc"]
    watch_rows = [watch_header]

    buy_items = []
    watch_items = []

    for r in results:
        sym = r.get("ticker", "")
        setup = r.get("setup", SETUP_NAME)
        sig = r.get("signal", "PASS")
        score = int(r.get("score", 0) or 0)
        entry = r.get("entry", "")
        stop = r.get("stop", "")
        pivot = r.get("pivot", "")
        base_weeks = r.get("base_weeks", "")
        contractions = r.get("contractions", "")
        depths = r.get("depths", "")
        risk_pct = r.get("risk_pct", "")
        runup_pct = r.get("runup_pct", "")
        reason = r.get("reason", "")

        signals_rows.append([sym, setup, sig, score, entry, stop, pivot, base_weeks, contractions, depths, risk_pct, runup_pct, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "BUY_NOW":
            line = f"{sym_disp} - BUY NOW - Setup: {setup} - Entry: {entry} - Stop: {stop} - Reason: {reason}"
            buy_items.append((score, [line, sym_disp, setup, score, entry, stop, pivot, base_weeks, contractions, risk_pct, reason, now_utc]))

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, entry, stop, pivot, base_weeks, contractions, risk_pct, reason, now_utc]))

    buy_items.sort(key=lambda x: x[0], reverse=True)
    watch_items.sort(key=lambda x: x[0], reverse=True)

    buy_rows.extend([row for _, row in buy_items])
    watch_rows.extend([row for _, row in watch_items])

    ws_signals.clear()
    ws_signals.update("A1", signals_rows)

    ws_buys.clear()
    ws_buys.update("A1", buy_rows)

    ws_watch.clear()
    ws_watch.update("A1", watch_rows)

    buy_count = len(buy_items)
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

    print("WATCH signals:")
    for _, r in watch_items[:20]:
        print(r[0])
    print(f"Done. tickers={len(tickers)} results={len(results)} watch={watch_count} pass={pass_count} errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source}")


if __name__ == "__main__":
    main()
