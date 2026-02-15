
import os
import json
import time
import math
from datetime import datetime, timezone

import yaml
import requests
import pandas as pd
import numpy as np
import gspread

TD_BASE = "https://api.twelvedata.com/time_series"


# -----------------------------
# Input parsing
# -----------------------------

def parse_symbols_from_text(text: str) -> list[str]:
    """Accepts:
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
        line = (line or "").strip()
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
    """Looks for a worksheet called 'Tickers'.

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


# -----------------------------
# Data fetch + normalisation
# -----------------------------

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
            data = r.json()

            if isinstance(data, dict) and data.get("status") == "error":
                msg = str(data.get("message", "")).lower()
                if "limit" in msg or "rate" in msg:
                    time.sleep(backoffs[min(i, len(backoffs) - 1)])
                    continue
            return data
        except Exception as e:
            last_err = e
            if i < len(backoffs):
                time.sleep(backoffs[i])
                continue
            raise

    raise last_err


def normalise_timeseries_payload(symbol: str, payload: dict) -> pd.DataFrame:
    if not payload or (isinstance(payload, dict) and payload.get("status") == "error"):
        return pd.DataFrame()

    values = payload.get("values", []) if isinstance(payload, dict) else []
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


# -----------------------------
# Indicators
# -----------------------------

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


def prepare_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = cfg.get("indicators", {}) or {}
    ema10_len = int(ind.get("ema_fast", 10))
    ema20_len = int(ind.get("ema_mid", 20))
    ema50_len = int(ind.get("ema_slow", 50))
    sma50_len = int(ind.get("sma_fast", 50))
    sma150_len = int(ind.get("sma_mid", 150))
    sma200_len = int(ind.get("sma_slow", 200))
    atr_len = int(ind.get("atr_len", 14))

    out = df.copy()
    out["ema10"] = compute_ema(out["close"], ema10_len)
    out["ema20"] = compute_ema(out["close"], ema20_len)
    out["ema50"] = compute_ema(out["close"], ema50_len)
    out["sma50"] = compute_sma(out["close"], sma50_len)
    out["sma150"] = compute_sma(out["close"], sma150_len)
    out["sma200"] = compute_sma(out["close"], sma200_len)
    out["atr"] = compute_atr(out, atr_len)
    out["high_52w"] = out["high"].rolling(252).max()
    out["low_52w"] = out["low"].rolling(252).min()
    out["atrp"] = (out["atr"] / out["close"]) * 100.0
    return out


def pct_range(high_val: float, low_val: float) -> float:
    if high_val <= 0:
        return 999.0
    return 100.0 * (high_val - low_val) / high_val


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -----------------------------
# Minervini trend template
# -----------------------------

def trend_template_minervini(df: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    last = df.iloc[-1]
    near_pct = float(cfg.get("filters", {}).get("near_52w_high_pct", 25))

    needed = ["close", "sma50", "sma150", "sma200"]
    if any(np.isnan(float(last.get(c, np.nan))) for c in needed):
        return False, "Not enough data for SMAs"

    close = float(last["close"])
    sma50 = float(last["sma50"])
    sma150 = float(last["sma150"])
    sma200 = float(last["sma200"])

    ok_stack = (close > sma50 > sma150 > sma200)

    sma150_up = False
    sma200_up = False
    if len(df) >= 221:
        sma150_up = df["sma150"].iloc[-1] > df["sma150"].iloc[-21]
        sma200_up = df["sma200"].iloc[-1] > df["sma200"].iloc[-21]

    high_52w = float(last.get("high_52w", np.nan))
    near_52w_ok = True
    if not np.isnan(high_52w) and high_52w > 0:
        near_52w_ok = close >= (1 - near_pct / 100.0) * high_52w

    fails = []
    if not ok_stack:
        fails.append("SMA stack")
    if not sma150_up:
        fails.append("SMA150 not rising")
    if not sma200_up:
        fails.append("SMA200 not rising")
    if not near_52w_ok:
        fails.append("not near 52w high")

    if fails:
        return False, "Trend template fail: " + ", ".join(fails)
    return True, "Trend template ok"


# -----------------------------
# Pivot helpers (for VCP)
# -----------------------------

def detect_pivots(highs: pd.Series, lows: pd.Series, left: int, right: int) -> list[tuple[int, str, float]]:
    """Return list of pivot points: (index, 'H'|'L', price)."""
    left = max(1, int(left))
    right = max(1, int(right))

    n = len(highs)
    pivots: list[tuple[int, str, float]] = []

    if n < left + right + 3:
        return pivots

    h = highs.to_numpy(dtype=float)
    l = lows.to_numpy(dtype=float)

    for i in range(left, n - right):
        win_h = h[i - left:i + right + 1]
        win_l = l[i - left:i + right + 1]
        if not np.isfinite(h[i]) or not np.isfinite(l[i]):
            continue

        if h[i] == np.nanmax(win_h) and h[i] > np.nanmax(np.concatenate([win_h[:left], win_h[left + 1:]])):
            pivots.append((i, "H", float(h[i])))
        if l[i] == np.nanmin(win_l) and l[i] < np.nanmin(np.concatenate([win_l[:left], win_l[left + 1:]])):
            pivots.append((i, "L", float(l[i])))

    pivots.sort(key=lambda x: x[0])

    cleaned: list[tuple[int, str, float]] = []
    for idx, t, px in pivots:
        if not cleaned:
            cleaned.append((idx, t, px))
            continue
        pidx, pt, ppx = cleaned[-1]
        if t != pt:
            cleaned.append((idx, t, px))
            continue

        if t == "H" and px >= ppx:
            cleaned[-1] = (idx, t, px)
        elif t == "L" and px <= ppx:
            cleaned[-1] = (idx, t, px)

    return cleaned


def build_contractions_from_pivots(pivots: list[tuple[int, str, float]]) -> list[dict]:
    """Build PH->PL contractions from an alternating pivot list."""
    if not pivots:
        return []

    i0 = 0
    while i0 < len(pivots) and pivots[i0][1] != "H":
        i0 += 1
    piv = pivots[i0:]
    if len(piv) < 2:
        return []

    contractions: list[dict] = []
    i = 0
    while i + 1 < len(piv):
        idx_h, t_h, ph = piv[i]
        if t_h != "H":
            i += 1
            continue

        j = i + 1
        while j < len(piv) and piv[j][1] != "L":
            j += 1
        if j >= len(piv):
            break

        idx_l, _, pl = piv[j]
        if ph <= 0:
            break

        depth = 100.0 * (ph - pl) / ph
        contractions.append({"ph_idx": idx_h, "pl_idx": idx_l, "ph": ph, "pl": pl, "depth": depth})
        i = j + 1

    return contractions


# -----------------------------
# Setup 1: Qullamaggie Base Breakout
# -----------------------------

def eval_qullamaggie_base_breakout(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "Qullamaggie Base Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_qullamaggie_breakout", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Disabled"
        return out

    if df.empty or len(df) < 220:
        out["reason"] = "Not enough daily data"
        return out

    last = df.iloc[-1]
    today_close = float(last["close"])
    today_low = float(last["low"])
    atr = float(last.get("atr", np.nan))

    ema10 = float(last.get("ema10", np.nan))
    ema20 = float(last.get("ema20", np.nan))
    ema50 = float(last.get("ema50", np.nan))
    if any(np.isnan(x) for x in [ema10, ema20, ema50]):
        out["reason"] = "Not enough data for EMAs"
        return out
    if not (ema10 > ema20 > ema50):
        out["reason"] = "EMA stack fail (10>20>50)"
        return out

    impulse_lookback = max(10, int(scfg.get("impulse_lookback", 60)))
    impulse_min_pct = float(scfg.get("impulse_min_pct", 30.0))
    impulse_low = df["low"].rolling(impulse_lookback).min().iloc[-1]
    if np.isnan(impulse_low) or impulse_low <= 0:
        out["reason"] = "Impulse calc failed"
        return out
    impulse_pct = 100.0 * (today_close / float(impulse_low) - 1.0)
    if impulse_pct < impulse_min_pct:
        out["reason"] = f"Impulse too small ({impulse_pct:.0f}%)"
        return out

    cons_min = int(scfg.get("cons_min_bars", 10))
    cons_max = int(scfg.get("cons_max_bars", 40))

    enforce_depth = bool(scfg.get("enforce_depth", False))
    cons_max_depth_pct = float(scfg.get("cons_max_depth_pct", 15.0))

    default_drift = float(scfg.get("cons_max_drift_pct", 10.0))
    cons_max_up_drift_pct = float(scfg.get("cons_max_up_drift_pct", default_drift))
    cons_max_down_drift_pct = float(scfg.get("cons_max_down_drift_pct", default_drift))

    higher_low_tol_pct = float(scfg.get("higher_low_tol_pct", 2.0))
    low_must_be_early_pct = float(scfg.get("cons_low_must_be_early_pct", 0.60))

    surf_close_min_pct = float(scfg.get("surf_close_min_pct", 70.0))
    surf_max_dist_pct = float(scfg.get("surf_max_dist_pct", 6.0))
    ma_slope_lookback = int(scfg.get("ma_slope_lookback", 5))

    touch_min_count = int(scfg.get("surf_touch_min_count", 2))
    touch_tol_pct = float(scfg.get("surf_touch_tol_pct", 1.0))

    tightening_ratio_max = float(scfg.get("tightening_ratio_max", 0.90))

    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    pivot_exclude_last_n = int(scfg.get("pivot_exclude_last_n", 2))

    max_ext_pct = float(scfg.get("max_ext_pct_above_surf_ma", 8.0))
    max_ext_atr_mult = float(scfg.get("max_ext_atr_mult_above_surf_ma", 3.0))
    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    stop_width_atr_mult = float(scfg.get("stop_width_atr_mult", 1.0))

    best = None

    for n in range(cons_min, cons_max + 1):
        if len(df) < n + 2:
            continue

        cons = df.iloc[-(n + 1):-1].copy()
        if cons.empty:
            continue

        cons_high = float(cons["high"].max())
        cons_low = float(cons["low"].min())

        depth_pct = pct_range(cons_high, cons_low)
        if enforce_depth and depth_pct > cons_max_depth_pct:
            continue

        c0 = float(cons["close"].iloc[0])
        c1 = float(cons["close"].iloc[-1])
        drift_pct = 100.0 * (c1 / c0 - 1.0) if c0 > 0 else 999.0
        if drift_pct > cons_max_up_drift_pct or drift_pct < -cons_max_down_drift_pct:
            continue

        third = max(2, int(len(cons) / 3))
        low_first = float(cons["low"].iloc[:third].min())
        low_last = float(cons["low"].iloc[-third:].min())
        if low_last < low_first * (1 - higher_low_tol_pct / 100.0):
            continue

        # lowest low must be early-ish (pullback then tighten)
        pos_min_low = int(cons["low"].values.argmin())
        if pos_min_low > int(len(cons) * low_must_be_early_pct):
            continue

        # tightening check left vs right half
        half = max(2, int(len(cons) / 2))
        left_seg = cons.iloc[:half]
        right_seg = cons.iloc[half:]
        r_left = pct_range(float(left_seg["high"].max()), float(left_seg["low"].min()))
        r_right = pct_range(float(right_seg["high"].max()), float(right_seg["low"].min())) if not right_seg.empty else r_left
        if r_right > r_left * tightening_ratio_max:
            continue

        surf_candidates = [("EMA10", cons["ema10"]), ("EMA20", cons["ema20"]), ("EMA50", cons["ema50"])]

        best_surf = None  # (surf_score, name, ma_last, touches)
        for name, ma_series in surf_candidates:
            if ma_series.isna().all():
                continue

            close_above_pct = float((cons["close"] >= ma_series).mean() * 100.0)
            if close_above_pct < surf_close_min_pct:
                continue

            if len(cons) > ma_slope_lookback:
                ma_now = float(ma_series.iloc[-1])
                ma_then = float(ma_series.iloc[-(ma_slope_lookback + 1)])
                if not (ma_now > ma_then):
                    continue

            dist_med_pct = float(((cons["close"] - ma_series).abs() / cons["close"]).median() * 100.0)
            if dist_med_pct > surf_max_dist_pct:
                continue

            touch_line = ma_series * (1 + touch_tol_pct / 100.0)
            touches = int((cons["low"] <= touch_line).sum())
            if touches < touch_min_count:
                continue

            surf_score = close_above_pct - 0.5 * dist_med_pct + min(10.0, touches * 2.0)
            cand = (surf_score, name, float(ma_series.iloc[-1]), touches)
            best_surf = cand if best_surf is None or cand[0] > best_surf[0] else best_surf

        if best_surf is None:
            continue

        excl = max(0, int(pivot_exclude_last_n))
        if excl >= len(cons):
            excl = max(0, len(cons) - 1)

        pivot_slice = cons["high"].iloc[:len(cons) - excl] if excl > 0 else cons["high"]
        if pivot_slice.empty:
            continue

        pivot = float(pivot_slice.max())
        entry = pivot * (1 + breakout_buffer_pct / 100.0)
        breakout = today_close > entry

        ext_pct = 100.0 * (today_close / best_surf[2] - 1.0) if best_surf[2] > 0 else 999.0
        ext_atr = (today_close - best_surf[2]) / atr if (not np.isnan(atr) and atr > 0) else 0.0
        extended = (
            (ext_pct > max_ext_pct)
            or (ext_atr > max_ext_atr_mult)
            or (today_close > entry * (1 + max_breakout_extension_pct / 100.0))
        )

        base_score = 50
        impulse_bonus = int(min(30, max(0, (impulse_pct - impulse_min_pct) / 2.0)))
        tight_bonus = int(max(0, min(20, (max(0.0, 20.0 - depth_pct)) * 0.8)))
        surf_bonus = int(max(0, min(20, best_surf[0] / 5.0)))
        tighten_bonus = int(max(0, min(10, (1.0 - (r_right / max(0.01, r_left))) * 20.0)))
        score = int(min(100, base_score + impulse_bonus + tight_bonus + surf_bonus + tighten_bonus))

        details = {"n": n, "cons_low": cons_low, "entry": entry, "breakout": breakout, "extended": extended,
                   "score": score, "impulse_pct": impulse_pct, "surf_name": best_surf[1], "touches": int(best_surf[3])}
        best = (score, details) if best is None or score > best[0] else best

    if best is None:
        out["reason"] = "No valid base + MA-surf found"
        return out

    d = best[1]
    entry = float(d["entry"])
    cons_low = float(d["cons_low"])

    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{cons_low:.2f}"
    out["score"] = int(d["score"])

    if not d["breakout"]:
        out["signal"] = "WATCH"
        out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, {d['n']}d base surfing {d['surf_name']} (touches {d['touches']}), no breakout"
        return out

    stop_lod = today_low if not np.isnan(today_low) else cons_low
    stop_dist = entry - stop_lod
    if stop_dist <= 0:
        stop_lod = cons_low
        stop_dist = entry - stop_lod

    if not np.isnan(atr) and atr > 0 and stop_dist > atr * stop_width_atr_mult:
        out["signal"] = "PASS"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout but stop too wide vs ATR ({stop_dist:.2f} > {stop_width_atr_mult:.1f}x ATR)"
        return out

    if d["extended"]:
        out["signal"] = "WATCH"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout but extended vs {d['surf_name']} / pivot"
        return out

    out["signal"] = "BUY_NOW"
    out["stop"] = f"{stop_lod:.2f}"
    out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, base surfing {d['surf_name']} (touches {d['touches']}), broke pivot"
    return out


# -----------------------------
# Setup 2: Qullamaggie Staircase Pullback
# -----------------------------

def eval_qullamaggie_staircase(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "Qullamaggie Staircase Pullback", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_qullamaggie_staircase", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Disabled"
        return out

    if df.empty or len(df) < 220:
        out["reason"] = "Not enough daily data"
        return out

    last = df.iloc[-1]
    today_close = float(last["close"])
    today_low = float(last["low"])
    atr = float(last.get("atr", np.nan))

    ema10 = float(last.get("ema10", np.nan))
    ema20 = float(last.get("ema20", np.nan))
    ema50 = float(last.get("ema50", np.nan))
    if any(np.isnan(x) for x in [ema10, ema20, ema50]):
        out["reason"] = "Not enough data for EMAs"
        return out
    if not (ema10 > ema20 > ema50):
        out["reason"] = "EMA stack fail (10>20>50)"
        return out

    pb_min = int(scfg.get("pullback_min_bars", 3))
    pb_max = int(scfg.get("pullback_max_bars", 12))

    impulse_lookback = int(scfg.get("impulse_lookback", 25))
    peak_lookback = int(scfg.get("peak_lookback", 5))
    fib_retrace_max = float(scfg.get("fib_retrace_max", 0.786))

    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    pivot_exclude_last_n = int(scfg.get("pivot_exclude_last_n", 2))

    surf_close_min_pct = float(scfg.get("surf_close_min_pct", 60.0))
    touch_min_count = int(scfg.get("surf_touch_min_count", 1))
    touch_tol_pct = float(scfg.get("surf_touch_tol_pct", 1.0))

    max_pb_range_pct = float(scfg.get("pullback_max_range_pct", 10.0))
    max_pb_drift_pct = float(scfg.get("pullback_max_drift_pct", 6.0))

    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    max_ext_pct_above_surf_ma = float(scfg.get("max_ext_pct_above_surf_ma", 10.0))
    max_ext_atr_mult_above_surf_ma = float(scfg.get("max_ext_atr_mult_above_surf_ma", 4.0))

    stop_width_atr_mult = float(scfg.get("stop_width_atr_mult", 1.2))

    best = None

    for n in range(pb_min, pb_max + 1):
        if len(df) < n + impulse_lookback + 5:
            continue

        pb = df.iloc[-(n + 1):-1].copy()
        if pb.empty:
            continue

        pb_high = float(pb["high"].max())
        pb_low = float(pb["low"].min())
        pb_range_pct = pct_range(pb_high, pb_low)
        if pb_range_pct > max_pb_range_pct:
            continue

        c0 = float(pb["close"].iloc[0])
        c1 = float(pb["close"].iloc[-1])
        pb_drift_pct = 100.0 * (c1 / c0 - 1.0) if c0 > 0 else 999.0
        if abs(pb_drift_pct) > max_pb_drift_pct:
            continue

        pb_start_pos = len(df) - (n + 1)

        peak_start = max(0, pb_start_pos - peak_lookback)
        peak_end = pb_start_pos
        peak_win = df.iloc[peak_start:peak_end]
        if peak_win.empty:
            continue

        impulse_high = float(peak_win["high"].max())

        imp_start = max(0, pb_start_pos - impulse_lookback)
        imp_end = pb_start_pos
        imp_win = df.iloc[imp_start:imp_end]
        if imp_win.empty:
            continue

        impulse_low = float(imp_win["low"].min())
        if impulse_low <= 0 or impulse_high <= impulse_low:
            continue

        impulse_pct = 100.0 * (impulse_high / impulse_low - 1.0)
        min_impulse_pct = float(scfg.get("impulse_min_pct", 15.0))
        if impulse_pct < min_impulse_pct:
            continue

        fib_786 = impulse_high - fib_retrace_max * (impulse_high - impulse_low)
        if pb_low < fib_786:
            continue

        surf_candidates = [("EMA10", pb["ema10"].copy()), ("EMA20", pb["ema20"].copy())]
        best_surf = None  # (score, name, ma_last, touches)

        for name, ma_series in surf_candidates:
            if ma_series.isna().all():
                continue

            close_above_pct = float((pb["close"] >= ma_series).mean() * 100.0)
            if close_above_pct < surf_close_min_pct:
                continue

            touch_line = ma_series * (1 + touch_tol_pct / 100.0)
            touches = int((pb["low"] <= touch_line).sum())
            if touches < touch_min_count:
                continue

            ma_last = float(ma_series.iloc[-1])
            dist_med = float(((pb["close"] - ma_series).abs() / pb["close"]).median() * 100.0)
            surf_score = close_above_pct - 0.6 * dist_med + min(10.0, touches * 3.0)
            cand = (surf_score, name, ma_last, touches)
            best_surf = cand if best_surf is None or cand[0] > best_surf[0] else best_surf

        if best_surf is None:
            continue

        excl = max(0, int(pivot_exclude_last_n))
        if excl >= len(pb):
            excl = max(0, len(pb) - 1)
        pivot_slice = pb["high"].iloc[:len(pb) - excl] if excl > 0 else pb["high"]
        if pivot_slice.empty:
            continue

        pivot = float(pivot_slice.max())
        entry = pivot * (1 + breakout_buffer_pct / 100.0)
        breakout = today_close > entry

        ext_pct = 100.0 * (today_close / best_surf[2] - 1.0) if best_surf[2] > 0 else 999.0
        ext_atr = (today_close - best_surf[2]) / atr if (not np.isnan(atr) and atr > 0) else 0.0
        extended = (
            (today_close > entry * (1 + max_breakout_extension_pct / 100.0))
            or (ext_pct > max_ext_pct_above_surf_ma)
            or (ext_atr > max_ext_atr_mult_above_surf_ma)
        )

        base_score = 55
        impulse_bonus = int(min(25, max(0, (impulse_pct - min_impulse_pct) * 0.6)))
        tight_bonus = int(max(0, min(15, (max(0.0, max_pb_range_pct - pb_range_pct) * 1.2))))
        surf_bonus = int(max(0, min(15, best_surf[0] / 6.0)))
        score = int(min(100, base_score + impulse_bonus + tight_bonus + surf_bonus))

        details = {"n": n, "entry": entry, "breakout": breakout, "extended": extended, "score": score,
                   "impulse_pct": impulse_pct, "surf_name": best_surf[1], "touches": int(best_surf[3]), "pb_low": pb_low}
        best = (score, details) if best is None or score > best[0] else best

    if best is None:
        out["reason"] = "No valid staircase pullback found"
        return out

    d = best[1]
    entry = float(d["entry"])

    out["entry"] = f"{entry:.2f}"
    out["score"] = int(d["score"])

    pb_low = float(d["pb_low"])
    out["stop"] = f"{pb_low:.2f}"

    if not d["breakout"]:
        out["signal"] = "WATCH"
        out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, {d['n']}d pullback surfing {d['surf_name']} (touches {d['touches']}), no breakout"
        return out

    stop_lod = today_low if not np.isnan(today_low) else pb_low
    stop_dist = entry - stop_lod
    if stop_dist <= 0:
        stop_lod = pb_low
        stop_dist = entry - stop_lod

    if not np.isnan(atr) and atr > 0 and stop_dist > atr * stop_width_atr_mult:
        out["signal"] = "PASS"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout but stop too wide vs ATR ({stop_dist:.2f} > {stop_width_atr_mult:.1f}x ATR)"
        return out

    if d["extended"]:
        out["signal"] = "WATCH"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout but extended vs {d['surf_name']} / pivot"
        return out

    out["signal"] = "BUY_NOW"
    out["stop"] = f"{stop_lod:.2f}"
    out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, staircase pullback surfing {d['surf_name']} (touches {d['touches']}), broke pivot"
    return out


# -----------------------------
# Setup B: Minervini VCP (revised)
# -----------------------------

def eval_minervini_vcp(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "Minervini VCP", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_minervini_vcp", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Disabled"
        return out

    ok, reason = trend_template_minervini(df, cfg)
    if not ok:
        out["reason"] = reason
        return out

    if df.empty or len(df) < 260:
        out["reason"] = "Not enough data"
        return out

    weeks_min = int(scfg.get("base_weeks_min", 3))
    weeks_max = int(scfg.get("base_weeks_max", 60))
    weeks_pref_min = int(scfg.get("base_weeks_pref_min", 6))
    weeks_pref_max = int(scfg.get("base_weeks_pref_max", 12))

    overshoot_tol_pct = float(scfg.get("left_high_overshoot_tol_pct", 1.0))
    left_side_max_pos_pct = float(scfg.get("left_side_max_pos_pct", 0.25))

    piv_left = int(scfg.get("pivot_left_bars", 5))
    piv_right = int(scfg.get("pivot_right_bars", 5))

    contractions_min = int(scfg.get("contractions_min", 2))
    contractions_max = int(scfg.get("contractions_max", 6))
    contractions_pref_min = int(scfg.get("contractions_pref_min", 3))
    contractions_pref_max = int(scfg.get("contractions_pref_max", 4))

    max_base_depth_pct = float(scfg.get("max_base_depth_pct", 50.0))  # changed to 50%

    shrink_mult = float(scfg.get("depth_shrink_mult", 0.85))
    shrink_steps_min_frac = float(scfg.get("depth_shrink_min_steps_frac", 0.60))
    allow_one_bump_pct = float(scfg.get("depth_allow_one_bump_pct", 20.0))

    dlast_max_pct = float(scfg.get("d_last_max_pct", 12.0))
    dlast_vs_first_mult = float(scfg.get("d_last_vs_first_mult", 0.60))

    meaningful_pullback_min = float(scfg.get("meaningful_pullback_min_pct", 6.0))

    atrp_compress_req = float(scfg.get("atrp_compress_req", 0.80))
    atrp_compress_pref = float(scfg.get("atrp_compress_pref", 0.70))

    right_bars_min = int(scfg.get("right_segment_bars_min", 10))
    right_bars_max = int(scfg.get("right_segment_bars_max", 20))
    right_range_max_pct = float(scfg.get("right_range_max_pct", 8.0))
    right_close_top_half_min_pct = float(scfg.get("right_close_top_half_min_pct", 60.0))

    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    near_breakout_pct = float(scfg.get("near_breakout_pct", 3.0))

    pivot_lookback_bars = int(scfg.get("pivot_lookback_bars", 15))
    pivot_exclude_last_n = int(scfg.get("pivot_exclude_last_n", 5))

    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    max_ext_above_sma50_pct = float(scfg.get("max_ext_above_sma50_pct", 12.0))

    vol_dryup_mult = float(scfg.get("vol_dryup_mult", 0.85))
    right_vol_expand_bump_pct = float(scfg.get("right_vol_expand_bump_pct", 20.0))

    best = None

    for w in range(weeks_min, weeks_max + 1):
        bars = w * 5
        if len(df) < bars + 2:
            continue

        base = df.iloc[-(bars + 1):-1].copy()
        if base.empty:
            continue

        pivots = detect_pivots(base["high"], base["low"], piv_left, piv_right)
        if not pivots:
            continue

        max_left_pos = max(5, int(len(base) * left_side_max_pos_pct))
        left_candidates = [(i, t, px) for (i, t, px) in pivots if t == "H" and i <= max_left_pos]
        if not left_candidates:
            continue
        left_h_idx, _, left_high = max(left_candidates, key=lambda x: x[2])

        base_high = float(base["high"].max())
        if base_high > left_high * (1 + overshoot_tol_pct / 100.0):
            continue

        base_low = float(base["low"].min())
        depth_pct = pct_range(left_high, base_low)
        if depth_pct > max_base_depth_pct:
            continue

        pivots_after = [(i, t, px) for (i, t, px) in pivots if i >= left_h_idx]
        contractions = build_contractions_from_pivots(pivots_after)
        n_con = len(contractions)
        if n_con < contractions_min or n_con > contractions_max:
            continue

        depths = [float(c["depth"]) for c in contractions if np.isfinite(c["depth"])]
        if len(depths) != n_con:
            continue

        d_first = depths[0]
        d_last = depths[-1]
        if d_last > dlast_max_pct:
            continue
        if d_last > d_first * dlast_vs_first_mult:
            continue

        if max(depths) < meaningful_pullback_min:
            continue

        pass_steps = 0
        bump_used = 0
        for i in range(len(depths) - 1):
            d0 = depths[i]
            d1 = depths[i + 1]
            if d1 <= d0 * shrink_mult:
                pass_steps += 1
            else:
                if bump_used == 0 and d1 <= d0 * (1 + allow_one_bump_pct / 100.0):
                    bump_used = 1

        needed_steps = int(math.ceil((len(depths) - 1) * shrink_steps_min_frac)) if len(depths) > 1 else 0
        if pass_steps < needed_steps:
            continue

        thirds = max(3, int(len(base) / 3))
        left_third = base.iloc[:thirds]
        right_third = base.iloc[-thirds:]
        atrp_left = float(left_third["atrp"].mean()) if left_third["atrp"].notna().any() else np.nan
        atrp_right = float(right_third["atrp"].mean()) if right_third["atrp"].notna().any() else np.nan
        atrp_ok = (np.isfinite(atrp_left) and np.isfinite(atrp_right) and atrp_right <= atrp_left * atrp_compress_req)

        t_right = int(clamp(right_bars_max, right_bars_min, right_bars_max))
        if len(base) < t_right:
            t_right = min(len(base), right_bars_min)
        right_seg = base.iloc[-t_right:]
        rh = float(right_seg["high"].max())
        rl = float(right_seg["low"].min())
        mid = (rh + rl) / 2.0 if (rh + rl) > 0 else np.nan
        right_range_pct = (100.0 * (rh - rl) / mid) if np.isfinite(mid) and mid > 0 else 999.0
        if right_range_pct > right_range_max_pct:
            continue

        top_half_line = rl + 0.5 * (rh - rl)
        close_top_half_pct = float((right_seg["close"] >= top_half_line).mean() * 100.0)
        if close_top_half_pct < right_close_top_half_min_pct:
            continue

        if n_con >= 2:
            d_prev = depths[-2]
            if d_last > d_prev * (1 + right_vol_expand_bump_pct / 100.0):
                if not atrp_ok:
                    continue

        vol_ok = False
        if "volume" in base.columns and base["volume"].notna().any():
            v_left = float(left_third["volume"].mean()) if not left_third.empty else np.nan
            v_right = float(right_third["volume"].mean()) if not right_third.empty else np.nan
            if np.isfinite(v_left) and v_left > 0 and np.isfinite(v_right):
                vol_ok = v_right <= v_left * vol_dryup_mult

        excl = max(0, int(pivot_exclude_last_n))
        pivot_lb = max(8, int(pivot_lookback_bars))
        pivot_lb = min(pivot_lb, len(base))
        pivot_seg = base.iloc[-pivot_lb:]
        if excl >= len(pivot_seg):
            excl = max(0, len(pivot_seg) - 1)
        pivot_slice = pivot_seg["high"].iloc[:len(pivot_seg) - excl] if excl > 0 else pivot_seg["high"]
        if pivot_slice.empty:
            continue

        pivot = float(pivot_slice.max())
        entry = pivot * (1 + breakout_buffer_pct / 100.0)

        today = df.iloc[-1]
        close = float(today["close"])
        breakout = close > entry

        sma50 = float(today.get("sma50", np.nan))
        ext_sma50_pct = 100.0 * (close / sma50 - 1.0) if (np.isfinite(sma50) and sma50 > 0) else 0.0
        extended = (close > entry * (1 + max_breakout_extension_pct / 100.0)) or (ext_sma50_pct > max_ext_above_sma50_pct)

        stop = float(contractions[-1]["pl"]) if contractions else float(base["low"].min())

        score = 55
        if weeks_pref_min <= w <= weeks_pref_max:
            score += 10
        if contractions_pref_min <= n_con <= contractions_pref_max:
            score += 10

        shrink_ratio = pass_steps / max(1, (n_con - 1))
        score += int(15 * shrink_ratio)
        if d_last <= 8.0:
            score += 5

        if atrp_ok:
            score += 10
            if np.isfinite(atrp_left) and np.isfinite(atrp_right) and atrp_right <= atrp_left * atrp_compress_pref:
                score += 5

        score += int(max(0, min(10, (right_range_max_pct - right_range_pct) * 1.5)))
        if vol_ok:
            score += 5

        score = int(min(100, score))

        details = {"weeks": w, "contractions": n_con, "entry": entry, "stop": stop, "breakout": breakout,
                   "extended": extended, "score": score, "vol_ok": vol_ok}
        best = (score, details) if best is None or score > best[0] else best

    if best is None:
        out["reason"] = "No valid VCP base found"
        return out

    d = best[1]
    entry = float(d["entry"])
    stop = float(d["stop"])

    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["score"] = int(d["score"])

    if d["breakout"] and not d["extended"]:
        out["signal"] = "BUY_NOW"
        out["reason"] = f"Trend ok, VCP ({d['weeks']}w, {d['contractions']} contractions), broke pivot" + (", vol dry-up" if d["vol_ok"] else "")
        return out

    if d["breakout"] and d["extended"]:
        out["signal"] = "WATCH"
        out["reason"] = "Broke pivot but extended"
        return out

    today_close = float(df.iloc[-1]["close"])
    if today_close >= entry * (1 - near_breakout_pct / 100.0):
        out["signal"] = "WATCH"
        out["reason"] = f"Trend ok, VCP ({d['weeks']}w, {d['contractions']} contractions), near pivot"
        return out

    out["reason"] = "Trend ok, but not near pivot"
    return out


# -----------------------------
# Multi-setup analysis
# -----------------------------

def analyse_symbol_multi(df: pd.DataFrame, cfg: dict) -> list[dict]:
    if df.empty:
        return [{"setup": "", "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": "No data"}]

    df_i = prepare_indicators(df, cfg)

    out = [
        eval_qullamaggie_base_breakout(df_i, cfg),
        eval_qullamaggie_staircase(df_i, cfg),
        eval_minervini_vcp(df_i, cfg),
    ]

    seen = set()
    uniq = []
    for r in out:
        key = (r.get("setup", ""), r.get("signal", ""), r.get("entry", ""), r.get("stop", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    return uniq


# -----------------------------
# Google Sheets helpers
# -----------------------------

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


# -----------------------------
# Rate limiting
# -----------------------------

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


# -----------------------------
# Main
# -----------------------------

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
    outputsize = int(cfg.get("api", {}).get("outputsize", 260))

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

            # Single-symbol response
            if isinstance(data, dict) and "values" in data:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                if df.empty:
                    msg = data.get("message", "No data") if isinstance(data, dict) else "No data"
                    results.append({"ticker": sym, "signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": msg})
                else:
                    for r in analyse_symbol_multi(df, cfg):
                        results.append({"ticker": sym, **r})
            else:
                if not isinstance(data, dict):
                    raise RuntimeError("Unexpected API payload")

                for sym in sym_batch:
                    payload = data.get(sym, {})
                    df = normalise_timeseries_payload(sym, payload)
                    if df.empty:
                        msg = payload.get("message", "No data") if isinstance(payload, dict) else "No data"
                        results.append({"ticker": sym, "signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": msg})
                        continue

                    for r in analyse_symbol_multi(df, cfg):
                        results.append({"ticker": sym, **r})

        except Exception as e:
            for sym in sym_batch:
                results.append({
                    "ticker": sym,
                    "signal": "PASS",
                    "setup": "",
                    "score": 0,
                    "entry": "",
                    "stop": "",
                    "reason": f"Fetch/analyse error: {type(e).__name__}",
                })
            errors += 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(4000, len(results) + 10), cols=12)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1500, cols=12)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=3000, cols=12)
    ws_summary = upsert_worksheet(sh, "Summary", rows=60, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    header = ["ticker", "setup", "signal", "score", "entry", "stop", "reason", "as_of_utc"]
    signals_rows = [header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    watch_rows = [watch_header]

    buy_items = []
    watch_items = []

    for r in results:
        sym = r.get("ticker", "")
        setup = r.get("setup", "")
        sig = r.get("signal", "")
        score = int(r.get("score", 0) or 0)
        entry = r.get("entry", "")
        stop = r.get("stop", "")
        reason = r.get("reason", "")

        signals_rows.append([sym, setup, sig, score, entry, stop, reason, now_utc])

        sym_disp = display_ticker(sym)

        if sig == "BUY_NOW":
            line = f"{sym_disp} - BUY NOW - Setup: {setup} - Entry: {entry} - Stop: {stop} - Reason: {reason}"
            buy_items.append((score, [line, sym_disp, setup, score, entry, stop, reason, now_utc]))

        if sig == "WATCH":
            watch_items.append((score, [sym_disp, setup, score, entry, stop, reason, now_utc]))

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
    print(
        f"Done. tickers={len(tickers)} results={len(results)} buy_now={buy_count} watch={watch_count} "
        f"pass={pass_count} errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source}"
    )


if __name__ == "__main__":
    main()
