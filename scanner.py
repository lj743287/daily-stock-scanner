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
SETUP_NAME = "Momentum Pullback Breakout"


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


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


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


def pct_range(high_val: float, low_val: float, denom: float | None = None) -> float:
    if denom is None:
        denom = high_val
    if denom <= 0 or np.isnan(denom):
        return np.nan
    return 100.0 * (high_val - low_val) / denom


# ---------------------------
# Pivot lows (for "higher low")
# ---------------------------

def find_pivot_lows(low: pd.Series, left: int, right: int) -> list[tuple[int, float]]:
    """
    Pivot Low: low[i] is min in [i-left, i+right]
    Returns list of (index, price) sorted by index.
    """
    n = len(low)
    out: list[tuple[int, float]] = []
    if n < left + right + 3:
        return out

    lv = low.values
    for i in range(left, n - right):
        w0 = i - left
        w1 = i + right + 1
        li = lv[i]
        if np.isnan(li):
            continue
        win = lv[w0:w1]
        if np.isnan(win).all():
            continue
        if li <= np.nanmin(win):
            out.append((i, float(li)))

    out.sort(key=lambda x: x[0])
    return out


# ---------------------------
# Pattern logic
# ---------------------------

def compute_returns(df: pd.DataFrame, bars: int) -> float:
    if df is None or df.empty or len(df) <= bars:
        return np.nan
    a = float(df["close"].iloc[-(bars + 1)])
    b = float(df["close"].iloc[-1])
    return pct_change(a, b)


def detect_big_move(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Detect a strong up-move within the last 1-3 months (default last 63 bars),
    where the move itself lasts a few days to a few weeks.
    """
    pcfg = cfg.get("pattern", {}) or {}
    mc = pcfg.get("big_move", {}) or {}

    lookback = int(mc.get("lookback_bars", 63))
    min_move_pct = float(mc.get("min_move_pct", 30.0))
    max_move_pct = float(mc.get("max_move_pct", 1200.0))  # effectively no ceiling
    min_len = int(mc.get("min_move_bars", 3))
    max_len = int(mc.get("max_move_bars", 25))

    if len(df) < lookback + max_len + 5:
        return {"ok": False, "reason": "Not enough data for big-move scan"}

    c = df["close"].values
    n = len(c)
    start_i = max(0, n - lookback - max_len - 1)
    end_i = n - 1

    best = None  # (ret, i, j)

    # Brute force within bounded ranges (cheap at these sizes)
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


def consolidation_ok(df: pd.DataFrame, move_end_idx: int, cfg: dict) -> dict:
    """
    First orderly pullback + consolidation:
      - consolidation length between 8 days and 2 months (default 8-42 bars)
      - higher lows (pivot lows rising)
      - tightening range (range compression)
      - "surfs" rising 10/20 EMA (optionally 50 EMA)
    """
    pcfg = cfg.get("pattern", {}) or {}

    ccfg = pcfg.get("consolidation", {}) or {}
    min_bars = int(ccfg.get("min_bars", 8))
    max_bars = int(ccfg.get("max_bars", 42))
    pivot_left = int(ccfg.get("pivot_left", 3))
    pivot_right = int(ccfg.get("pivot_right", 3))

    tighten_mult = float(ccfg.get("tighten_right_to_left_mult", 0.75))
    max_range_pct = float(ccfg.get("max_range_pct", 18.0))

    # EMA surf
    ecfg = pcfg.get("ema_surf", {}) or {}
    ema10 = int(ecfg.get("ema10", 10))
    ema20 = int(ecfg.get("ema20", 20))
    ema50 = int(ecfg.get("ema50", 50))
    use_50 = bool(ecfg.get("allow_50", True))
    surf_dist_pct = float(ecfg.get("max_close_dist_pct", 2.5))
    min_surf_frac = float(ecfg.get("min_surf_frac", 0.45))
    require_rising = bool(ecfg.get("require_rising_emas", True))

    # We evaluate consolidation ending yesterday (formation), breakout today
    form = df.iloc[:-1].copy()
    if len(form) < max_bars + 60:
        return {"ok": False, "reason": "Not enough formation data"}

    # Consolidation must occur AFTER the big move ended
    # Choose the best consolidation window in the last max_bars after move_end_idx
    end_form_idx = len(form) - 1

    # Candidate windows end at end_form_idx and start within [end-max_bars+1, end-min_bars+1]
    best = None  # (score, start, end, details)

    close = form["close"]
    high = form["high"]
    low = form["low"]

    e10 = ema(close, ema10)
    e20 = ema(close, ema20)
    e50 = ema(close, ema50)

    for bars in range(min_bars, max_bars + 1):
        s = end_form_idx - bars + 1
        e = end_form_idx
        if s <= move_end_idx:
            continue  # must be after move end
        seg = form.iloc[s:e + 1]
        if len(seg) < min_bars:
            continue

        hh = float(seg["high"].max())
        ll = float(seg["low"].min())
        mid = (hh + ll) / 2.0 if (hh + ll) != 0 else hh
        r_pct = pct_range(hh, ll, denom=mid)
        if np.isnan(r_pct) or r_pct > max_range_pct:
            continue

        # Tightening: compare left half range to right half range
        half = max(3, int(len(seg) / 2))
        left_seg = seg.iloc[:half]
        right_seg = seg.iloc[-half:]
        lhh, lll = float(left_seg["high"].max()), float(left_seg["low"].min())
        rhh, rll = float(right_seg["high"].max()), float(right_seg["low"].min())
        lmid = (lhh + lll) / 2.0 if (lhh + lll) != 0 else lhh
        rmid = (rhh + rll) / 2.0 if (rhh + rll) != 0 else rhh
        l_rng = pct_range(lhh, lll, denom=lmid)
        r_rng = pct_range(rhh, rll, denom=rmid)
        if np.isnan(l_rng) or np.isnan(r_rng):
            continue
        if r_rng > (l_rng * tighten_mult):
            continue

        # Higher low: last two pivot lows rising within the consolidation
        piv_lows = find_pivot_lows(seg["low"], pivot_left, pivot_right)
        if len(piv_lows) < 2:
            continue
        last2 = piv_lows[-2:]
        if not (last2[1][1] > last2[0][1]):
            continue

        # EMA surf: fraction of bars where close is "near" at least one rising EMA (10/20, optionally 50)
        seg_idx = seg.index
        cseg = close.loc[seg_idx]
        e10s = e10.loc[seg_idx]
        e20s = e20.loc[seg_idx]
        e50s = e50.loc[seg_idx]

        def dist_ok(cval, eval_) -> bool:
            if np.isnan(cval) or np.isnan(eval_) or eval_ <= 0:
                return False
            return abs(100.0 * (cval / eval_ - 1.0)) <= surf_dist_pct

        near10 = [dist_ok(float(cseg.iloc[k]), float(e10s.iloc[k])) for k in range(len(seg))]
        near20 = [dist_ok(float(cseg.iloc[k]), float(e20s.iloc[k])) for k in range(len(seg))]
        if use_50:
            near50 = [dist_ok(float(cseg.iloc[k]), float(e50s.iloc[k])) for k in range(len(seg))]
            near_any = np.array(near10) | np.array(near20) | np.array(near50)
        else:
            near_any = np.array(near10) | np.array(near20)

        surf_frac = float(np.mean(near_any)) if len(seg) else 0.0
        if surf_frac < min_surf_frac:
            continue

        # EMAs rising (simple slope check on last 10 bars of the consolidation)
        if require_rising:
            w = min(10, len(seg))
            s10 = linreg_slope(e10s.tail(w).values)
            s20 = linreg_slope(e20s.tail(w).values)
            if s10 <= 0 or s20 <= 0:
                continue
            if use_50:
                s50 = linreg_slope(e50s.tail(w).values)
                # 50 can be flat-ish, but not sharply down
                if s50 < -1e-6:
                    continue

        # Score: prefer longer, tighter, more surf, and higher-low strength
        hl_strength = 100.0 * (last2[1][1] / last2[0][1] - 1.0)
        score = 0
        score += int(min(25, (bars / max_bars) * 25))
        score += int(min(25, max(0.0, (l_rng - r_rng) / max(1e-6, l_rng)) * 25))
        score += int(min(25, surf_frac * 25))
        score += int(min(25, max(0.0, hl_strength) * 4.0))  # small bonus

        details = {
            "cons_bars": bars,
            "cons_start_idx": int(s),
            "cons_end_idx": int(e),
            "cons_high": float(hh),
            "cons_low": float(ll),
            "cons_range_pct": float(r_pct),
            "left_range_pct": float(l_rng),
            "right_range_pct": float(r_rng),
            "surf_frac": float(surf_frac),
            "hl0_idx": int(s + last2[0][0]),
            "hl0_price": float(last2[0][1]),
            "hl1_idx": int(s + last2[1][0]),
            "hl1_price": float(last2[1][1]),
        }

        if best is None or score > best[0]:
            best = (score, s, e, details)

    if best is None:
        return {"ok": False, "reason": "No valid orderly consolidation found after big move"}

    return {"ok": True, **best[3]}


def breakout_today(df: pd.DataFrame, cons_high: float, cfg: dict) -> dict:
    """
    Breakout / range expansion today:
      - price trigger crosses above consolidation high (+buffer)
      - volume today > vol_mult * 50dma
      - range expansion proxy: TR% > median(TR%) * tr_mult (optional)
    """
    pcfg = cfg.get("pattern", {}) or {}
    bcfg = pcfg.get("breakout", {}) or {}

    price_trigger = str(bcfg.get("price_trigger", "close")).lower()  # close or high
    entry_buffer_pct = float(bcfg.get("entry_buffer_pct", 0.2))

    vol_mult = float(bcfg.get("vol_mult_vs_50dma", 1.4))
    vol_ma = int(bcfg.get("vol_ma", 50))

    require_tr_expansion = bool(bcfg.get("require_tr_expansion", True))
    tr_mult = float(bcfg.get("tr_mult_vs_median", 1.3))
    tr_lookback = int(bcfg.get("tr_median_lookback", 20))

    today = df.iloc[-1]
    prev = df.iloc[-2]

    level = cons_high * (1.0 + entry_buffer_pct / 100.0)

    today_close = float(today["close"])
    today_high = float(today["high"])
    today_open = float(today["open"])
    today_vol = float(today.get("volume", np.nan))

    crossed = (today_close > level) if price_trigger == "close" else (today_high > level)

    # Volume
    vol_ok = True
    if "volume" in df.columns and not np.isnan(today_vol):
        hist = df["volume"].iloc[-(vol_ma + 1):-1].dropna()
        if len(hist) >= max(20, int(vol_ma * 0.6)):
            vavg = float(hist.tail(vol_ma).mean())
            vol_ok = (today_vol > (vavg * vol_mult))
        else:
            vol_ok = False

    # TR expansion (range expansion proxy)
    tr_ok = True
    if require_tr_expansion:
        tr = true_range(df)
        trp = (tr / df["close"]) * 100.0
        med = trp.iloc[-(tr_lookback + 1):-1].dropna()
        if len(med) >= max(10, int(tr_lookback * 0.5)):
            base = float(med.median())
            today_trp = float(trp.iloc[-1])
            tr_ok = (today_trp > base * tr_mult)
        else:
            tr_ok = False

    # A small sanity: breakout bar should be at least not a big red reversal
    not_bad_reversal = not (today_close < today_open and today_close < float(prev["close"]))

    return {
        "crossed": bool(crossed),
        "vol_ok": bool(vol_ok),
        "tr_ok": bool(tr_ok),
        "not_bad_reversal": bool(not_bad_reversal),
        "level": float(level),
        "today_close": today_close,
        "today_vol": today_vol,
    }


# ---------------------------
# Main detector (WATCH / BUY_NOW)
# ---------------------------

def detect_setup(df_full: pd.DataFrame, cfg: dict) -> dict:
    """
    Formation is assessed ending yesterday; breakout is assessed on today's bar.

    Outputs:
      - PASS / WATCH / BUY_NOW
      - entry, stop
      - reasons and details for sheet
    """
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
    }

    if df_full is None or df_full.empty or len(df_full) < 200:
        out["reason"] = "Not enough daily data"
        return out

    # Compute timeframe returns (1m=21, 3m=63, 6m=126 trading days)
    r1m = compute_returns(df_full, 21)
    r3m = compute_returns(df_full, 63)
    r6m = compute_returns(df_full, 126)

    out["r1m"] = f"{r1m:.1f}" if not np.isnan(r1m) else ""
    out["r3m"] = f"{r3m:.1f}" if not np.isnan(r3m) else ""
    out["r6m"] = f"{r6m:.1f}" if not np.isnan(r6m) else ""

    # Big move
    bm = detect_big_move(df_full, cfg)
    if not bm.get("ok"):
        out["reason"] = bm.get("reason", "No big move")
        return out

    # Consolidation after big move
    cons = consolidation_ok(df_full, int(bm["move_end_idx"]), cfg)
    if not cons.get("ok"):
        out["reason"] = cons.get("reason", "No orderly consolidation")
        return out

    # Breakout today?
    bo = breakout_today(df_full, float(cons["cons_high"]), cfg)

    # Stop = most recent higher-low pivot price (hl1_price)
    stop = float(cons["hl1_price"])
    entry_level = float(bo["level"])
    # Risk %
    risk_pct = 100.0 * (entry_level - stop) / entry_level if entry_level > 0 else np.nan

    pcfg = cfg.get("pattern", {}) or {}
    rcfg = pcfg.get("risk", {}) or {}
    max_stop_pct_trade = float(rcfg.get("max_stop_pct_trade", 12.0))
    if not np.isnan(risk_pct) and risk_pct > max_stop_pct_trade:
        out["reason"] = f"Risk too wide ({risk_pct:.1f}%)"
        return out

    # Scoring (purely to sort, not to decide)
    score = 50
    score += int(min(15, max(0.0, bm["move_ret"] / 6.0)))  # big move bonus
    score += int(min(15, max(0.0, (cons["left_range_pct"] - cons["right_range_pct"]) * 1.2)))
    score += int(min(10, cons["surf_frac"] * 10))
    score += int(min(10, max(0.0, (cons["hl1_price"] / cons["hl0_price"] - 1.0) * 200.0)))
    score = int(min(100, max(1, score)))

    # Signal logic
    if bo["crossed"] and bo["vol_ok"] and bo["tr_ok"] and bo["not_bad_reversal"]:
        out["signal"] = "BUY_NOW"
        out["entry"] = f"{entry_level:.2f}"
        out["stop"] = f"{stop:.2f}"
        out["score"] = score
        out["reason"] = (
            f"Top performers filter passed, big move {bm['move_ret']:.0f}%, "
            f"orderly consolidation {cons['cons_bars']}d (tighten {cons['left_range_pct']:.1f}%→{cons['right_range_pct']:.1f}%), "
            f"HL {cons['hl0_price']:.2f}→{cons['hl1_price']:.2f}, "
            f"breakout + vol (> {pcfg.get('breakout',{}).get('vol_mult_vs_50dma',1.4)}×50dma)"
        )
        return out

    # WATCH if setup is valid and either near breakout or already crossed but volume/TR not confirming
    near_pct = float((pcfg.get("breakout", {}) or {}).get("near_breakout_pct", 3.0))
    today_close = float(df_full.iloc[-1]["close"])
    near = today_close >= entry_level * (1.0 - near_pct / 100.0)

    if bo["crossed"] and (not bo["vol_ok"] or not bo["tr_ok"]):
        out["signal"] = "WATCH"
        out["entry"] = f"{entry_level:.2f}"
        out["stop"] = f"{stop:.2f}"
        out["score"] = score
        out["reason"] = (
            f"Valid setup; breakout attempt but "
            f"{'volume weak' if not bo['vol_ok'] else ''}"
            f"{' & ' if (not bo['vol_ok'] and not bo['tr_ok']) else ''}"
            f"{'range not expanding' if not bo['tr_ok'] else ''}"
        )
        return out

    if near:
        out["signal"] = "WATCH"
        out["entry"] = f"{entry_level:.2f}"
        out["stop"] = f"{stop:.2f}"
        out["score"] = score
        out["reason"] = (
            f"Valid setup; near breakout level {entry_level:.2f} "
            f"(cons {cons['cons_bars']}d, HL {cons['hl0_price']:.2f}→{cons['hl1_price']:.2f}, surf {cons['surf_frac']:.2f})"
        )
        return out

    out["reason"] = "Valid setup but not near breakout"
    out["score"] = score
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

    # First pass: fetch all data and compute returns for ranking
    series_by_sym: dict[str, pd.DataFrame] = {}
    ret_rows = []  # (sym, r1m, r3m, r6m)

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
                if not df.empty:
                    series_by_sym[sym] = df
                    r1m = compute_returns(df, 21)
                    r3m = compute_returns(df, 63)
                    r6m = compute_returns(df, 126)
                    ret_rows.append((sym, r1m, r3m, r6m))
            else:
                # Multi-symbol response
                for sym in sym_batch:
                    payload = data.get(sym, {}) if isinstance(data, dict) else {}
                    df = normalise_timeseries_payload(sym, payload)
                    if df.empty:
                        continue
                    series_by_sym[sym] = df
                    r1m = compute_returns(df, 21)
                    r3m = compute_returns(df, 63)
                    r6m = compute_returns(df, 126)
                    ret_rows.append((sym, r1m, r3m, r6m))

        except Exception:
            errors += 1
            continue

    if not series_by_sym:
        raise SystemExit("No price series fetched successfully")

    # Rank filter: top X% composite across 1m/3m/6m
    rs_cfg = (cfg.get("relative_strength", {}) or {})
    top_pct = float(rs_cfg.get("top_percent", 5.0))  # 1-5% typical
    if top_pct <= 0:
        top_pct = 5.0
    if top_pct > 50:
        top_pct = 50.0

    rs = pd.DataFrame(ret_rows, columns=["sym", "r1m", "r3m", "r6m"]).dropna()
    if rs.empty:
        raise SystemExit("Not enough return data to rank the universe")

    # Percentile ranks (higher return => higher rank)
    rs["rk1m"] = rs["r1m"].rank(pct=True)
    rs["rk3m"] = rs["r3m"].rank(pct=True)
    rs["rk6m"] = rs["r6m"].rank(pct=True)
    rs["rk"] = (rs["rk1m"] + rs["rk3m"] + rs["rk6m"]) / 3.0

    # Threshold for top X%
    thr = rs["rk"].quantile(1.0 - top_pct / 100.0)
    top_syms = set(rs.loc[rs["rk"] >= thr, "sym"].tolist())

    # Second pass: run pattern detector only on ranked universe
    results: list[dict] = []
    for sym, df in series_by_sym.items():
        if sym not in top_syms:
            results.append({
                "ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "",
                "reason": "Not in top performers filter", "r1m": "", "r3m": "", "r6m": ""
            })
            continue

        r = detect_setup(df, cfg)
        results.append({"ticker": sym, **r})

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(2000, len(results) + 10), cols=25)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=25)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=25)
    ws_summary = upsert_worksheet(sh, "Summary", rows=80, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "setup", "signal", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "reason", "as_of_utc"]
    signals_rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "r1m_pct", "r3m_pct", "r6m_pct", "reason", "as_of_utc"]
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

        signals_rows.append([sym, setup, sig, score, entry, stop, r1m, r3m, r6m, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "BUY_NOW":
            line = f"{sym_disp} – BUY NOW – Entry: {entry} – Stop: {stop} – 1m/3m/6m: {r1m}/{r3m}/{r6m}% – {reason}"
            buy_items.append((score, [line, sym_disp, setup, score, entry, stop, r1m, r3m, r6m, reason, now_utc]))

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, entry, stop, r1m, r3m, r6m, reason, now_utc]))

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

    note = f"ok ({tickers_source}) paced at {max_credits_per_min}/min, batch_size={batch_size}, RS top {top_pct:.1f}%"

    summary_rows = [
        ["key", "value"],
        ["last_run_utc", now_utc],
        ["tickers_scanned", str(len(tickers))],
        ["fetched_series", str(len(series_by_sym))],
        ["ranked_universe", str(len(top_syms))],
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
