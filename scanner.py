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
    return out


def pct_range(high_val: float, low_val: float) -> float:
    if high_val <= 0:
        return 999.0
    return 100.0 * (high_val - low_val) / high_val


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


def eval_qullamaggie_breakout(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "Qullamaggie Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

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

    impulse_lookback = max(10, int(scfg.get("impulse_lookback", 60)))
    impulse_min_pct = float(scfg.get("impulse_min_pct", 30.0))

    cons_min = int(scfg.get("cons_min_bars", 10))
    cons_max = int(scfg.get("cons_max_bars", 40))
    cons_max_depth_pct = float(scfg.get("cons_max_depth_pct", 15.0))

    default_drift = float(scfg.get("cons_max_drift_pct", 10.0))
    cons_max_up_drift_pct = float(scfg.get("cons_max_up_drift_pct", default_drift))
    cons_max_down_drift_pct = float(scfg.get("cons_max_down_drift_pct", default_drift))

    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    pivot_exclude_last_n = int(scfg.get("pivot_exclude_last_n", 2))

    surf_close_min_pct = float(scfg.get("surf_close_min_pct", 70.0))
    surf_max_dist_pct = float(scfg.get("surf_max_dist_pct", 6.0))
    ma_slope_lookback = int(scfg.get("ma_slope_lookback", 5))

    higher_low_tol_pct = float(scfg.get("higher_low_tol_pct", 2.0))

    max_ext_pct = float(scfg.get("max_ext_pct_above_surf_ma", 8.0))
    max_ext_atr_mult = float(scfg.get("max_ext_atr_mult_above_surf_ma", 3.0))
    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    stop_width_atr_mult = float(scfg.get("stop_width_atr_mult", 1.0))

    impulse_low = df["low"].rolling(impulse_lookback).min().iloc[-1]
    if np.isnan(impulse_low) or impulse_low <= 0:
        out["reason"] = "Impulse calc failed"
        return out

    impulse_pct = 100.0 * (today_close / float(impulse_low) - 1.0)
    if impulse_pct < impulse_min_pct:
        out["reason"] = f"Impulse too small ({impulse_pct:.0f}%)"
        return out

    ema10 = float(last.get("ema10", np.nan))
    ema20 = float(last.get("ema20", np.nan))
    ema50 = float(last.get("ema50", np.nan))
    if any(np.isnan(x) for x in [ema10, ema20, ema50]):
        out["reason"] = "Not enough data for EMAs"
        return out
    if not (ema10 > ema20 > ema50):
        out["reason"] = "EMA stack fail (10>20>50)"
        return out

    best = None  # (score, details)

    for n in range(cons_min, cons_max + 1):
        if len(df) < n + 2:
            continue

        cons = df.iloc[-(n + 1):-1].copy()  # ends yesterday
        if cons.empty:
            continue

        cons_high = float(cons["high"].max())
        cons_low = float(cons["low"].min())
        depth_pct = pct_range(cons_high, cons_low)
        if depth_pct > cons_max_depth_pct:
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

        surf_candidates = [("EMA10", cons["ema10"]), ("EMA20", cons["ema20"]), ("EMA50", cons["ema50"])]

        best_surf = None  # (surf_score, name, ma_last, dist_med_pct, close_above_pct)
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

            surf_score = close_above_pct - 0.5 * dist_med_pct
            candidate = (surf_score, name, float(ma_series.iloc[-1]), dist_med_pct, close_above_pct)
            best_surf = candidate if best_surf is None or candidate[0] > best_surf[0] else best_surf

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
        tight_bonus = int(max(0, min(20, (cons_max_depth_pct - depth_pct) * 1.5)))
        surf_bonus = int(max(0, min(20, best_surf[0] / 5.0)))
        score = int(min(100, base_score + impulse_bonus + tight_bonus + surf_bonus))

        details = {
            "n": n,
            "cons_low": cons_low,
            "entry": entry,
            "breakout": breakout,
            "extended": extended,
            "score": score,
            "impulse_pct": impulse_pct,
            "surf_name": best_surf[1],
        }

        best = (score, details) if best is None or score > best[0] else best

    if best is None:
        out["reason"] = "No valid MA-surf consolidation found"
        return out

    d = best[1]
    entry = float(d["entry"])
    cons_low = float(d["cons_low"])

    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{cons_low:.2f}"
    out["score"] = int(d["score"])

    if not d["breakout"]:
        out["signal"] = "WATCH"
        out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, tight {d['n']}d MA-surf ({d['surf_name']}), no breakout yet"
        return out

    stop_lod = today_low if not np.isnan(today_low) else cons_low
    stop_dist = entry - stop_lod
    if stop_dist <= 0:
        stop_lod = cons_low
        stop_dist = entry - stop_lod

    if not np.isnan(atr) and atr > 0 and stop_dist > atr * stop_width_atr_mult:
        out["signal"] = "PASS"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout, but LOD stop too wide vs ATR (dist {stop_dist:.2f} > {stop_width_atr_mult:.1f}x ATR)"
        return out

    if d["extended"]:
        out["signal"] = "WATCH"
        out["stop"] = f"{stop_lod:.2f}"
        out["reason"] = f"Breakout, but extended vs {d['surf_name']} / pivot"
        return out

    out["signal"] = "BUY_NOW"
    out["stop"] = f"{stop_lod:.2f}"
    out["reason"] = f"Impulse {d['impulse_pct']:.0f}%, tight {d['n']}d MA-surf ({d['surf_name']}), broke pivot"
    return out


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

    lookback = max(40, int(scfg.get("lookback_bars", 65)))
    pivot_excl = max(0, int(scfg.get("pivot_exclude_last_n", 5)))

    vcp_max_depth_pct = float(scfg.get("vcp_max_depth_pct", 35.0))
    vcp_ratio_12 = float(scfg.get("vcp_ratio_12", 0.90))
    vcp_ratio_23 = float(scfg.get("vcp_ratio_23", 0.90))
    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    near_breakout_pct = float(scfg.get("near_breakout_pct", 3.0))

    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    vol_dryup_mult = float(scfg.get("vol_dryup_mult", 0.85))
    max_ext_above_sma50_pct = float(scfg.get("max_ext_above_sma50_pct", 12.0))

    if len(df) < lookback + 2:
        out["reason"] = "Not enough data for VCP window"
        return out

    base = df.iloc[-(lookback + 1):-1].copy()  # ending yesterday
    if base.empty:
        out["reason"] = "VCP window empty"
        return out

    if pivot_excl >= len(base):
        pivot_excl = max(0, len(base) - 1)

    pivot_slice = base["high"].iloc[:len(base) - pivot_excl] if pivot_excl > 0 else base["high"]
    if pivot_slice.empty:
        out["reason"] = "Pivot slice empty"
        return out

    pivot = float(pivot_slice.max())
    low_in_base = float(base["low"].min())
    depth_pct = pct_range(pivot, low_in_base)
    if depth_pct > vcp_max_depth_pct:
        out["reason"] = f"VCP too deep ({depth_pct:.0f}%)"
        return out

    n = len(base)
    seg1 = base.iloc[: int(n * 0.45)]
    seg2 = base.iloc[int(n * 0.45): int(n * 0.75)]
    seg3 = base.iloc[int(n * 0.75):]

    def seg_range(seg: pd.DataFrame) -> float:
        if seg.empty:
            return 999.0
        return pct_range(float(seg["high"].max()), float(seg["low"].min()))

    r1 = seg_range(seg1)
    r2 = seg_range(seg2)
    r3 = seg_range(seg3)

    if not (r2 <= r1 * vcp_ratio_12 and r3 <= r2 * vcp_ratio_23):
        out["reason"] = f"No VCP contraction (r1={r1:.1f} r2={r2:.1f} r3={r3:.1f})"
        return out

    vol_ok = True
    if "volume" in base.columns and base["volume"].notna().any():
        v1 = float(seg1["volume"].mean()) if not seg1.empty else np.nan
        v3 = float(seg3["volume"].mean()) if not seg3.empty else np.nan
        if not np.isnan(v1) and v1 > 0 and not np.isnan(v3):
            vol_ok = v3 <= v1 * vol_dryup_mult

    entry = pivot * (1 + breakout_buffer_pct / 100.0)
    last = df.iloc[-1]
    close = float(last["close"])
    breakout = close > entry

    sma50 = float(last.get("sma50", np.nan))
    ext_sma50_pct = 100.0 * (close / sma50 - 1.0) if (not np.isnan(sma50) and sma50 > 0) else 0.0
    extended = (close > entry * (1 + max_breakout_extension_pct / 100.0)) or (ext_sma50_pct > max_ext_above_sma50_pct)

    stop = float(seg3["low"].min()) if not seg3.empty else float(base["low"].min())

    score = 70
    score += int(max(0, min(15, (vcp_max_depth_pct - depth_pct) * 0.4)))
    if vol_ok:
        score += 10
    score = int(min(100, score))

    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["score"] = score

    if breakout and not extended:
        out["signal"] = "BUY_NOW"
        out["reason"] = "Trend template ok, VCP contraction, broke pivot" + (", volume dry-up" if vol_ok else ", volume not dry")
        return out

    if breakout and extended:
        out["signal"] = "WATCH"
        out["reason"] = "Broke pivot but extended"
        return out

    if close >= entry * (1 - near_breakout_pct / 100.0):
        out["signal"] = "WATCH"
        out["reason"] = "Trend template ok, VCP contraction, near pivot"
        return out

    out["reason"] = "Trend template ok, but not near pivot"
    return out


def eval_minervini_3wt(df: pd.DataFrame, cfg: dict) -> dict:
    out = {"signal": "PASS", "setup": "Minervini 3WT", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_minervini_3wt", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Disabled"
        return out

    ok, reason = trend_template_minervini(df, cfg)
    if not ok:
        out["reason"] = reason
        return out

    bars = max(10, int(scfg.get("bars", 15)))
    max_range_pct = float(scfg.get("max_range_pct", 8.0))
    max_drift_pct = float(scfg.get("max_drift_pct", 8.0))
    breakout_buffer_pct = float(scfg.get("breakout_buffer_pct", 0.2))
    near_breakout_pct = float(scfg.get("near_breakout_pct", 3.0))

    max_breakout_extension_pct = float(scfg.get("max_breakout_extension_pct", 4.0))
    max_ext_above_sma50_pct = float(scfg.get("max_ext_above_sma50_pct", 12.0))

    if len(df) < bars + 2:
        out["reason"] = "Not enough data for 3WT"
        return out

    base = df.iloc[-(bars + 1):-1].copy()  # ending yesterday
    if base.empty:
        out["reason"] = "3WT window empty"
        return out

    base_high = float(base["high"].max())
    base_low = float(base["low"].min())
    range_pct = pct_range(base_high, base_low)
    if range_pct > max_range_pct:
        out["reason"] = f"3WT not tight (range {range_pct:.1f}%)"
        return out

    c0 = float(base["close"].iloc[0])
    c1 = float(base["close"].iloc[-1])
    drift_pct = 100.0 * (c1 / c0 - 1.0) if c0 > 0 else 999.0
    if abs(drift_pct) > max_drift_pct:
        out["reason"] = f"3WT drift too large ({drift_pct:.1f}%)"
        return out

    if len(base) >= 15:
        w1 = base.iloc[:5]
        w2 = base.iloc[5:10]
        w3 = base.iloc[10:15]

        def week_stats(w: pd.DataFrame) -> tuple[float, float, float]:
            wh = float(w["high"].max())
            wl = float(w["low"].min())
            wc = float(w["close"].iloc[-1])
            return wh, wl, wc

        wh1, wl1, wc1 = week_stats(w1)
        wh2, wl2, wc2 = week_stats(w2)
        wh3, wl3, wc3 = week_stats(w3)

        if not (wl3 >= wl2 >= wl1 * 0.99):
            out["reason"] = "3WT weekly lows not rising"
            return out

        def close_pos_ok(wh, wl, wc, min_pos=0.55):
            if wh <= wl:
                return False
            return ((wc - wl) / (wh - wl)) >= min_pos

        if not (close_pos_ok(wh1, wl1, wc1) and close_pos_ok(wh2, wl2, wc2) and close_pos_ok(wh3, wl3, wc3)):
            out["reason"] = "3WT weekly closes not strong"
            return out

    entry = base_high * (1 + breakout_buffer_pct / 100.0)
    last = df.iloc[-1]
    close = float(last["close"])
    breakout = close > entry

    sma50 = float(last.get("sma50", np.nan))
    ext_sma50_pct = 100.0 * (close / sma50 - 1.0) if (not np.isnan(sma50) and sma50 > 0) else 0.0
    extended = (close > entry * (1 + max_breakout_extension_pct / 100.0)) or (ext_sma50_pct > max_ext_above_sma50_pct)

    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{base_low:.2f}"

    score = 75 + int(max(0, min(15, (max_range_pct - range_pct) * 1.5)))
    out["score"] = int(min(100, score))

    if breakout and not extended:
        out["signal"] = "BUY_NOW"
        out["reason"] = "Trend template ok, 3 weeks tight, broke 3WT pivot"
        return out

    if breakout and extended:
        out["signal"] = "WATCH"
        out["reason"] = "Broke 3WT pivot but extended"
        return out

    if close >= entry * (1 - near_breakout_pct / 100.0):
        out["signal"] = "WATCH"
        out["reason"] = "Trend template ok, 3 weeks tight, near pivot"
        return out

    out["reason"] = "Trend template ok, but not near 3WT pivot"
    return out


def analyse_symbol_multi(df: pd.DataFrame, cfg: dict) -> list[dict]:
    if df.empty:
        return [{"setup": "", "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": "No data"}]

    df_i = prepare_indicators(df, cfg)

    return [
        eval_qullamaggie_breakout(df_i, cfg),
        eval_minervini_vcp(df_i, cfg),
        eval_minervini_3wt(df_i, cfg),
    ]


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
    We assume 1 credit per symbol in the batch.

    state keys:
      - window_start (monotonic seconds)
      - used (credits used in current window)
    """
    if max_credits_per_min <= 0:
        return

    now = time.monotonic()
    window_start = state.get("window_start", now)
    used = int(state.get("used", 0))

    elapsed = now - window_start
    if elapsed >= 60.0:
        # reset window
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

    # after sleep, reset window and book the credits
    state["window_start"] = time.monotonic()
    state["used"] = batch_credits


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

    batch_size = int(cfg.get("api", {}).get("batch_size", 50))
    if batch_size < 1:
        batch_size = 50
    # Never allow a batch bigger than the per-minute credit cap, otherwise you cannot be compliant.
    batch_size = min(batch_size, max_credits_per_min)

    results: list[dict] = []
    errors = 0
    api_calls = 0
    credits_est = 0

    rl_state = {"window_start": time.monotonic(), "used": 0}

    for sym_batch in chunks(tickers, batch_size):
        batch_credits = len(sym_batch)
        credits_est += batch_credits

        # Respect provider limits (Grow 55 = 55 credits/min).
        rate_limit_wait(batch_credits, max_credits_per_min, rl_state)

        try:
            data = fetch_time_series_batch(td_key, sym_batch, interval, outputsize)
            api_calls += 1

            # Single-symbol response
            if isinstance(data, dict) and "values" in data:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                res_list = analyse_symbol_multi(df, cfg) if not df.empty else [{
                    "signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": data.get("message", "No data")
                }]
                for r in res_list:
                    results.append({"ticker": sym, **r})
            else:
                for sym in sym_batch:
                    payload = data.get(sym, {}) if isinstance(data, dict) else {}
                    df = normalise_timeseries_payload(sym, payload)
                    if df.empty:
                        results.append({
                            "ticker": sym,
                            "signal": "PASS",
                            "setup": "",
                            "score": 0,
                            "entry": "",
                            "stop": "",
                            "reason": payload.get("message", "No data"),
                        })
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
                    "reason": f"Fetch error: {type(e).__name__}",
                })
            errors += 1

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(2000, len(results) + 10), cols=12)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=12)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=12)
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
            line = f"{sym_disp} – BUY NOW – Setup: {setup} – Entry: {entry} – Stop: {stop} – Reason: {reason}"
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
    print(f"Done. tickers={len(tickers)} results={len(results)} buy_now={buy_count} watch={watch_count} pass={pass_count} errors={errors} api_calls={api_calls} credits_est={credits_est} source={tickers_source}")


if __name__ == "__main__":
    main()
