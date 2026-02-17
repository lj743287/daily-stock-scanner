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
SETUP_NAME = "Momentum EMA Surf HL (Buy on HL vol)"


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
# Indicators / helpers
# ---------------------------

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def linreg_slope(y: np.ndarray) -> float:
    if y.size < 5:
        return 0.0
    x = np.arange(y.size, dtype=float)
    yv = y.astype(float)
    if np.all(np.isnan(yv)):
        return 0.0
    m = np.polyfit(x, yv, 1)[0]
    return float(m)


def pct_change(a: float, b: float) -> float:
    if a == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return 100.0 * (b / a - 1.0)


def compute_returns(df: pd.DataFrame, bars: int) -> float:
    if df is None or df.empty or len(df) <= bars:
        return np.nan
    a = float(df["close"].iloc[-(bars + 1)])
    b = float(df["close"].iloc[-1])
    return pct_change(a, b)


def touch_ema(low: float, high: float, ema_val: float) -> bool:
    if np.isnan(low) or np.isnan(high) or np.isnan(ema_val):
        return False
    return (low <= ema_val <= high)


# ---------------------------
# Big move detector
# ---------------------------

def detect_big_move(df_form: pd.DataFrame, cfg: dict) -> dict:
    """
    Big move higher in the past 1–3 months:
      - Lookback max 63 bars (configurable)
      - Move lasts min 2 bars (user confirmed), max configurable
      - Min move % configurable (default 30%)

    IMPORTANT: We scan formation data ending yesterday, so the big move must
    have completed by yesterday at the latest.
    """
    pcfg = cfg.get("pattern", {}) or {}
    mc = pcfg.get("big_move", {}) or {}

    lookback = int(mc.get("lookback_bars", 63))
    min_move_pct = float(mc.get("min_move_pct", 30.0))
    max_move_pct = float(mc.get("max_move_pct", 1200.0))
    min_len = int(mc.get("min_move_bars", 2))
    max_len = int(mc.get("max_move_bars", 25))

    if len(df_form) < lookback + max_len + 10:
        return {"ok": False, "reason": "Not enough data for big-move scan"}

    c = df_form["close"].values
    n = len(c)

    end_i = n - 1
    start_i = max(0, n - lookback - max_len - 1)

    best = None  # (ret, i, j)
    for i in range(start_i, end_i - min_len):
        ci = c[i]
        if np.isnan(ci) or ci <= 0:
            continue
        j0 = i + min_len
        j1 = min(end_i, i + max_len)
        for j in range(j0, j1 + 1):
            cj = c[j]
            if np.isnan(cj) or cj <= 0:
                continue
            ret = 100.0 * (cj / ci - 1.0)
            if ret < min_move_pct or ret > max_move_pct:
                continue
            if best is None or ret > best[0]:
                best = (ret, i, j)

    if best is None:
        return {"ok": False, "reason": f"No big move ≥ {min_move_pct:.0f}% found"}

    move_ret, i, j = best
    return {
        "ok": True,
        "move_ret": float(move_ret),
        "move_start_idx": int(i),
        "move_end_idx": int(j),
        "move_start_close": float(c[i]),
        "move_end_close": float(c[j]),
    }


# ---------------------------
# Realtime swing-low / higher-low
# ---------------------------

def swing_low_realtime(low: pd.Series, idx: int, left: int) -> bool:
    """
    Realtime-friendly swing low:
      low[idx] is the lowest of the last (left+1) bars ending at idx.
    No right bars (no future knowledge).
    """
    if idx < left:
        return False
    window = low.iloc[idx - left: idx + 1]
    if window.isna().any():
        return False
    return float(low.iloc[idx]) <= float(window.min())


def find_higher_low_events(low: pd.Series, start_idx: int, end_idx: int, left: int) -> list[dict]:
    """
    Scan [start_idx..end_idx] for realtime swing lows and identify higher lows.
    Returns list of events: {idx, low, prev_low, is_higher_low}
    """
    events = []
    prev_swing_low = None

    for i in range(start_idx, end_idx + 1):
        if not swing_low_realtime(low, i, left):
            continue
        cur = float(low.iloc[i])
        is_hl = (prev_swing_low is not None and cur > prev_swing_low)
        events.append({"idx": i, "low": cur, "prev_low": prev_swing_low, "is_hl": bool(is_hl)})
        prev_swing_low = cur

    return events


# ---------------------------
# Surf detector (your exact spec)
# ---------------------------

def detect_surf_window(df_full: pd.DataFrame, move_end_form_idx: int, cfg: dict) -> dict:
    """
    Your definition:

    - After the big move ends (index in df_form), the first EMA touch marks surf start
    - Touch = low <= EMA <= high (no % threshold)
    - From the touch bar onwards it must stop making lower lows:
        min(low[touch..end]) >= low[touch]
    - Surf length between 2 and 42 bars
    - Must touch an EMA at least 1 in 2 bars in the surf window (>=50% touch rate)
    - Open-to-close bodies tighten (trend down): slope(abs(close-open)) < 0
    - Higher lows start to appear (at least one HL event in the surf window)

    We prefer the latest valid surf end (closest to today).
    """
    pcfg = cfg.get("pattern", {}) or {}
    scfg = pcfg.get("surf", {}) or {}

    ema10_len = int(scfg.get("ema10", 10))
    ema20_len = int(scfg.get("ema20", 20))
    ema50_len = int(scfg.get("ema50", 50))

    min_bars = int(scfg.get("min_surf_bars", 2))
    max_bars = int(scfg.get("max_surf_bars", 42))

    min_touch_frac = float(scfg.get("min_touch_frac", 0.50))

    # Higher low swing definition (realtime)
    hl_left = int(scfg.get("hl_left_bars", 3))

    # Tightening definition
    min_body_slope = float(scfg.get("body_slope_max", -1e-9))  # slope must be < this (negative)

    if df_full is None or df_full.empty or len(df_full) < max(200, ema50_len + 50):
        return {"ok": False, "reason": "Not enough data for surf detection"}

    # df_form ends yesterday; df_full includes today
    df_form = df_full.iloc[:-1].copy()
    if len(df_form) < move_end_form_idx + 3:
        return {"ok": False, "reason": "Formation too short after big move"}

    close = df_full["close"]
    low = df_full["low"]
    high = df_full["high"]
    opn = df_full["open"]

    e10 = ema(close, ema10_len)
    e20 = ema(close, ema20_len)
    e50 = ema(close, ema50_len)

    # Surf start search begins AFTER the move end (in formation indices),
    # but surf can start today as well.
    start_search = move_end_form_idx + 1
    last_idx = len(df_full) - 1  # today index in df_full

    touch_idx = None
    touch_ema_name = None

    for i in range(start_search, last_idx + 1):
        lo = float(low.iloc[i])
        hi = float(high.iloc[i])
        if touch_ema(lo, hi, float(e10.iloc[i])):
            touch_idx, touch_ema_name = i, "EMA10"
            break
        if touch_ema(lo, hi, float(e20.iloc[i])):
            touch_idx, touch_ema_name = i, "EMA20"
            break
        if touch_ema(lo, hi, float(e50.iloc[i])):
            touch_idx, touch_ema_name = i, "EMA50"
            break

    if touch_idx is None:
        return {"ok": False, "reason": "No first EMA touch found after big move"}

    touch_low = float(low.iloc[touch_idx])

    # Evaluate candidate surf ends (prefer latest) with start fixed at touch_idx
    best = None  # (end_idx, details)
    max_end = min(last_idx, touch_idx + max_bars - 1)
    min_end = min(last_idx, touch_idx + min_bars - 1)

    for end_idx in range(min_end, max_end + 1):
        seg = df_full.iloc[touch_idx:end_idx + 1]
        if len(seg) < min_bars:
            continue

        # Must stop making lower lows from touch onwards
        if float(seg["low"].min()) < touch_low:
            continue

        # Touch rate (touching any EMA counts)
        touches = 0
        for k in range(touch_idx, end_idx + 1):
            lo = float(low.iloc[k])
            hi = float(high.iloc[k])
            if touch_ema(lo, hi, float(e10.iloc[k])) or touch_ema(lo, hi, float(e20.iloc[k])) or touch_ema(lo, hi, float(e50.iloc[k])):
                touches += 1
        touch_frac = touches / float(len(seg))

        if touch_frac < min_touch_frac:
            continue

        # Tightening bodies: slope(abs(close-open)) must be negative
        bodies = (seg["close"] - seg["open"]).abs().values.astype(float)
        if np.isnan(bodies).any() or bodies.size < 5:
            continue
        slope = linreg_slope(bodies)
        if not (slope < min_body_slope):
            continue

        # Higher lows must start to appear in the surf window
        hl_events = find_higher_low_events(df_full["low"], touch_idx, end_idx, hl_left)
        has_hl = any(ev["is_hl"] for ev in hl_events)
        if not has_hl:
            continue

        # Keep latest valid end
        best = {
            "touch_idx": int(touch_idx),
            "touch_ema": touch_ema_name,
            "touch_low": float(touch_low),
            "surf_end_idx": int(end_idx),
            "surf_bars": int(len(seg)),
            "touches": int(touches),
            "touch_frac": float(touch_frac),
            "body_slope": float(slope),
            "hl_left_bars": int(hl_left),
            "hl_events": hl_events,
        }

    if best is None:
        return {"ok": False, "reason": "Touch found but no valid surf window meeting all rules"}

    return {"ok": True, **best}


# ---------------------------
# BUY trigger: Higher-low day with vol expansion
# ---------------------------

def buy_trigger_today(df_full: pd.DataFrame, surf: dict, cfg: dict) -> dict:
    """
    BUY is the higher low with expanded volume > 1.4 x the 50dma.

    Implemented as:
      - Today is a realtime swing-low (left bars) AND it is a higher low vs previous swing low
      - Today volume > vol_mult * SMA(vol, 50)

    Entry = today's close
    Stop  = today's low
    """
    pcfg = cfg.get("pattern", {}) or {}
    bcfg = pcfg.get("buy", {}) or {}

    vol_ma = int(bcfg.get("vol_ma", 50))
    vol_mult = float(bcfg.get("vol_mult_vs_ma", 1.40))
    hl_left = int((cfg.get("pattern", {}) or {}).get("surf", {}).get("hl_left_bars", 3))

    last_idx = len(df_full) - 1
    if last_idx < max(80, vol_ma + 5):
        return {"buy": False, "reason": "Not enough data for BUY volume MA"}

    low = df_full["low"]
    vol = df_full.get("volume", pd.Series([np.nan] * len(df_full)))
    opn = df_full["open"]
    cls = df_full["close"]

    # Must be within the surf window
    if not (surf["touch_idx"] <= last_idx <= surf["surf_end_idx"]):
        # If surf_end is before today (rare), treat as no BUY today
        return {"buy": False, "reason": "Today not within detected surf window"}

    # Find swing lows up to today and check if today is a higher-low event
    events = find_higher_low_events(low, surf["touch_idx"], last_idx, hl_left)
    today_ev = None
    for ev in reversed(events):
        if ev["idx"] == last_idx:
            today_ev = ev
            break

    if today_ev is None or not today_ev["is_hl"]:
        return {"buy": False, "reason": "Today is not a higher-low swing day"}

    today_vol = float(vol.iloc[last_idx]) if "volume" in df_full.columns else np.nan
    if np.isnan(today_vol) or today_vol <= 0:
        return {"buy": False, "reason": "Volume not available"}

    hist = vol.iloc[-(vol_ma + 1):-1].dropna()
    if len(hist) < max(20, int(vol_ma * 0.6)):
        return {"buy": False, "reason": "Not enough volume history for MA"}

    vavg = float(hist.tail(vol_ma).mean())
    if vavg <= 0:
        return {"buy": False, "reason": "Invalid volume MA"}

    vol_ok = today_vol > (vavg * vol_mult)
    if not vol_ok:
        return {"buy": False, "reason": f"Volume not expanded (> {vol_mult:.2f}x{vol_ma}dma)"}

    # Optional: require a constructive candle (default true)
    require_green = bool(bcfg.get("require_green_candle", True))
    if require_green:
        if float(cls.iloc[last_idx]) <= float(opn.iloc[last_idx]):
            return {"buy": False, "reason": "Higher low day but not a green candle"}

    return {
        "buy": True,
        "entry": float(cls.iloc[last_idx]),
        "stop": float(low.iloc[last_idx]),
        "today_vol": today_vol,
        "vol_ma": vavg,
        "vol_mult": vol_mult,
        "hl_low": float(today_ev["low"]),
        "prev_swing_low": float(today_ev["prev_low"]) if today_ev["prev_low"] is not None else np.nan,
    }


# ---------------------------
# Full detector per ticker
# ---------------------------

def detect_setup(df_full: pd.DataFrame, cfg: dict) -> dict:
    out = {
        "signal": "PASS",
        "setup": SETUP_NAME,
        "score": 0,
        "entry": "",
        "stop": "",
        "reason": "",
        "r1m": "",
        "r3m": "",
        "r6m": "",
        "move_ret": "",
        "surf_bars": "",
        "touch_frac": "",
        "touch_ema": "",
    }

    if df_full is None or df_full.empty or len(df_full) < 220:
        out["reason"] = "Not enough daily data"
        return out

    # Returns for reference (no filtering)
    r1m = compute_returns(df_full, 21)
    r3m = compute_returns(df_full, 63)
    r6m = compute_returns(df_full, 126)
    out["r1m"] = f"{r1m:.1f}" if not np.isnan(r1m) else ""
    out["r3m"] = f"{r3m:.1f}" if not np.isnan(r3m) else ""
    out["r6m"] = f"{r6m:.1f}" if not np.isnan(r6m) else ""

    # Formation is df_form ending yesterday
    df_form = df_full.iloc[:-1].copy()
    if len(df_form) < 200:
        out["reason"] = "Not enough formation data"
        return out

    # 1) Big move (ends by yesterday at the latest)
    bm = detect_big_move(df_form, cfg)
    if not bm.get("ok"):
        out["reason"] = bm.get("reason", "No big move")
        return out

    out["move_ret"] = f"{bm['move_ret']:.0f}"

    # 2) Surf window (starts at first EMA touch after big move end; includes today if still surfing)
    surf = detect_surf_window(df_full, int(bm["move_end_idx"]), cfg)
    if not surf.get("ok"):
        out["reason"] = surf.get("reason", "No valid surf window")
        return out

    out["surf_bars"] = str(surf["surf_bars"])
    out["touch_frac"] = f"{surf['touch_frac']:.2f}"
    out["touch_ema"] = surf.get("touch_ema", "")

    # 3) WATCH is satisfied once the surf rules are satisfied (your watchlist definition)
    out["signal"] = "WATCH"
    out["score"] = int(min(100, max(1, 55 + min(25, bm["move_ret"] / 4.0) + min(10, surf["surf_bars"] / 5.0))))
    out["reason"] = (
        f"Big move {bm['move_ret']:.0f}% (<=63 bars, min 2 bars); "
        f"first EMA touch at idx {surf['touch_idx']} ({surf['touch_ema']}); "
        f"surf {surf['surf_bars']} bars, touch rate {surf['touch_frac']:.2f}; "
        f"no lower lows since touch; bodies tightening; higher lows started"
    )

    # 4) BUY trigger today: higher-low day with volume expansion
    buy = buy_trigger_today(df_full, surf, cfg)
    if buy.get("buy"):
        out["signal"] = "BUY_NOW"
        out["entry"] = f"{float(buy['entry']):.2f}"
        out["stop"] = f"{float(buy['stop']):.2f}"
        out["score"] = int(min(100, out["score"] + 15))
        out["reason"] = (
            out["reason"]
            + f"; BUY on HL day with vol {buy['today_vol']:.0f} > {buy['vol_mult']:.2f}x{(cfg.get('pattern',{}).get('buy',{}).get('vol_ma',50))}dma"
        )
    else:
        # For WATCH, give a useful stop reference: latest higher-low pivot low in the surf window (if any)
        hl_left = int((cfg.get("pattern", {}) or {}).get("surf", {}).get("hl_left_bars", 3))
        events = find_higher_low_events(df_full["low"], surf["touch_idx"], surf["surf_end_idx"], hl_left)
        hl_events = [ev for ev in events if ev["is_hl"]]
        if hl_events:
            last_hl = hl_events[-1]
            out["stop"] = f"{float(last_hl['low']):.2f}"

    return out


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
                    results.append({"ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": data.get("message", "No data")})
                else:
                    r = detect_setup(df, cfg)
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
                            "reason": payload.get("message", "No data"),
                            "r1m": "",
                            "r3m": "",
                            "r6m": "",
                            "move_ret": "",
                            "surf_bars": "",
                            "touch_frac": "",
                            "touch_ema": "",
                        })
                        continue

                    r = detect_setup(df, cfg)
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
                    "reason": f"Fetch error: {type(e).__name__}",
                    "r1m": "",
                    "r3m": "",
                    "r6m": "",
                    "move_ret": "",
                    "surf_bars": "",
                    "touch_frac": "",
                    "touch_ema": "",
                })
            errors += 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(2000, len(results) + 10), cols=30)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=30)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=30)
    ws_summary = upsert_worksheet(sh, "Summary", rows=80, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "setup", "signal", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "move_ret_pct", "surf_bars", "touch_frac", "touch_ema", "reason", "as_of_utc"]
    signals_rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "move_ret_pct", "surf_bars", "touch_frac", "touch_ema", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "move_ret_pct", "surf_bars", "touch_frac", "touch_ema", "reason", "as_of_utc"]
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
        reason = r.get("reason", "")

        r1m = r.get("r1m", "")
        r3m = r.get("r3m", "")
        r6m = r.get("r6m", "")
        move_ret = r.get("move_ret", "")
        surf_bars = r.get("surf_bars", "")
        touch_frac = r.get("touch_frac", "")
        touch_ema = r.get("touch_ema", "")

        signals_rows.append([sym, setup, sig, score, entry, stop, r1m, r3m, r6m, move_ret, surf_bars, touch_frac, touch_ema, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "BUY_NOW":
            line = f"{sym_disp} – BUY NOW – Entry: {entry} – Stop: {stop} – Move: {move_ret}% – Surf: {surf_bars} – Touch: {touch_frac} – {reason}"
            buy_items.append((score, [line, sym_disp, setup, score, entry, stop, r1m, r3m, r6m, move_ret, surf_bars, touch_frac, touch_ema, reason, now_utc]))

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, entry, stop, r1m, r3m, r6m, move_ret, surf_bars, touch_frac, touch_ema, reason, now_utc]))

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

    print("BUY_NOW signals:")
    for _, r in buy_items:
        print(r[0])
    print(f"Done. tickers={len(tickers)} results={len(results)} buy_now={buy_count} watch={watch_count} pass={pass_count} errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source}")


if __name__ == "__main__":
    main()
