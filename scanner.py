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
    if df is None or df.empty:
        return np.nan
    if len(df) < lookback + 2:
        return np.nan
    now = df["close"].iloc[-1]
    past = df["close"].iloc[-1 - lookback]
    if past is None or np.isnan(past) or past == 0:
        return np.nan
    return (now / past) - 1.0


def compute_percentile_flags(df_map: dict[str, pd.DataFrame], windows: list[int], top_pct: float) -> dict[str, bool]:
    """
    Flags symbols that are in the top X% by return for ANY of the windows.
    Percentiles are computed within the scanned list (df_map).
    If too few symbols have valid returns, thresholds become +inf (no one qualifies).
    """
    windows = [int(w) for w in windows]
    top_pct = float(top_pct)

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
            thresholds[w] = np.inf
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

        r_rank = rank.get(r.get("signal", "PASS"), 0)
        b_rank = rank.get(best.get("signal", "PASS"), 0)

        if r_rank > b_rank:
            best = r
            continue
        if r_rank == b_rank and int(r.get("score", 0) or 0) > int(best.get("score", 0) or 0):
            best = r

    return best or {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": ""}


def build_debug_summary(setup_results: list[dict], chosen_setup: str, max_items: int = 3) -> str:
    watch_items: list[str] = []
    late_items: list[str] = []
    early_items: list[str] = []

    def is_late_failure(reason: str) -> bool:
        markers = (
            "breakout fail",
            "stop too wide",
            "lod rule fails",
            "invalid stop",
            "volume not strong enough",
        )
        r = (reason or "").lower()
        return any(m in r for m in markers)

    def is_early_failure(reason: str) -> bool:
        markers = (
            "trend fail",
            "atr unavailable",
            "atr% too low",
            "no valid",
            "vcp fail",
            "3wt fail",
            "not enough data",
            "window too small",
            "segments too small",
            "invalid price data",
            "volume unavailable",
            "ma-surf",
        )
        r = (reason or "").lower()
        return any(m in r for m in markers)

    for r in setup_results:
        if not r:
            continue
        setup = (r.get("setup") or "").strip()
        sig = (r.get("signal") or "").strip()
        reason = (r.get("reason") or "").strip()

        if not setup or setup == chosen_setup:
            continue
        if reason.lower() in {"setup disabled", "disabled"}:
            continue

        if sig == "WATCH":
            watch_items.append(f"{setup}: WATCH ({reason})")
        elif sig == "PASS":
            if is_late_failure(reason):
                late_items.append(f"{setup}: {reason}")
            elif is_early_failure(reason):
                early_items.append(f"{setup}: {reason}")

    items: list[str] = []
    for bucket in (watch_items, late_items, early_items):
        for it in bucket:
            if it not in items:
                items.append(it)
            if len(items) >= max_items:
                break
        if len(items) >= max_items:
            break

    return " | ".join(items) if items else ""


def eval_base_breakout(df: pd.DataFrame, cfg: dict, last: pd.Series) -> dict:
    out = {"signal": "PASS", "setup": "Base Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

    base_cfg = cfg.get("setup_base_breakout", {}) or {}
    if not bool(base_cfg.get("enabled", True)):
        out["reason"] = "Setup disabled"
        return out

    base_n = int(base_cfg.get("base_lookback", 30))
    base_max_depth_pct = float(base_cfg.get("base_max_depth_pct", 15))
    breakout_buffer_pct = float(base_cfg.get("breakout_buffer_pct", 0.2))
    vol_mult = float(base_cfg.get("vol_multiplier", 1.2))
    atr_stop_mult = float(base_cfg.get("atr_stop_mult", 2.0))

    if len(df) < base_n + 2:
        out["signal"] = "WATCH"
        out["reason"] = "Base window too small"
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
        out["reason"] = "Base not tight"
        out["score"] = 55
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{stop:.2f}"
        return out

    if not breakout_ok:
        out["signal"] = "WATCH"
        out["reason"] = "Tight base, no breakout yet"
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
    out["reason"] = "Tight base, broke pivot" + (", volume ok" if vol_ok else ", volume weak")
    return out


def eval_qullamaggie_breakout(df: pd.DataFrame, cfg: dict, last: pd.Series) -> dict:
    """
    Qullamaggie Breakout (daily-only):
      - Impulse leg within lookback: impulse_min_pct rise from swing low to swing high
      - Consolidation:
          - depth <= cons_depth_max_pct
          - higher lows (last third low > first third low)
          - surfing: % closes above ONE of EMA10/EMA20/EMA50 >= surf_pct
          - MA(s) rising into end of consolidation:
              - EMA20 must be rising
              - and the surf EMA must be rising (EMA10 or EMA20 or EMA50)
      - Entry: break above consolidation high + buffer
      - Stop on breakout day: Low of Day; must not exceed ATR * stop_max_atr_mult
    Momentum leader filter is REMOVED (you will screen that).
    """
    out = {"signal": "PASS", "setup": "Qullamaggie Breakout", "score": 0, "entry": "", "stop": "", "reason": ""}

    qcfg = cfg.get("setup_qullamaggie_breakout", {}) or {}
    if not bool(qcfg.get("enabled", True)):
        out["reason"] = "Setup disabled"
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
    impulse_min_pct = float(qcfg.get("impulse_min_pct", 25.0))
    pullback_from_impulse_high_pct = float(qcfg.get("pullback_from_impulse_high_pct", 35.0))

    cons_min = int(qcfg.get("cons_min_bars", 8))
    cons_max = int(qcfg.get("cons_max_bars", 45))
    cons_depth_max_pct = float(qcfg.get("cons_depth_max_pct", 25.0))

    # Supports old key surf_close_above_ema20_pct and new generic surf_close_above_ema_pct
    surf_pct = float(qcfg.get("surf_close_above_ema_pct", qcfg.get("surf_close_above_ema20_pct", 0.6)))
    ma_rise_lookback = int(qcfg.get("ma_rise_lookback", 3))

    breakout_buffer_pct = float(qcfg.get("breakout_buffer_pct", 0.05))
    vol_mult = float(qcfg.get("vol_multiplier", 1.1))
    tr_mult = float(qcfg.get("tr_multiplier", 1.1))
    stop_max_atr_mult = float(qcfg.get("stop_max_atr_mult", 1.2))

    # Allow surf on ANY of these EMAs
    surf_mas = [
        ("EMA10", "ema10", 3),  # prefer faster MA if equal
        ("EMA20", "ema20", 2),
        ("EMA50", "ema50", 1),
    ]

    best = None
    fail_counts = {"depth": 0, "higher_lows": 0, "surf": 0, "ma_rise": 0, "impulse": 0, "pullback": 0, "data": 0}

    for cons_len in range(cons_min, cons_max + 1):
        need = cons_len + impulse_lookback + 2
        if len(df) < need:
            fail_counts["data"] += 1
            continue

        cons = df.iloc[-(cons_len + 1):-1]
        if cons.empty:
            fail_counts["data"] += 1
            continue

        cons_high = cons["high"].max()
        cons_low = cons["low"].min()
        if not cons_high or cons_high <= 0:
            fail_counts["data"] += 1
            continue

        depth_pct = 100.0 * (cons_high - cons_low) / cons_high
        if depth_pct > cons_depth_max_pct:
            fail_counts["depth"] += 1
            continue

        n = len(cons)
        one_third = max(1, n // 3)
        early = cons.iloc[:one_third]
        late = cons.iloc[-one_third:]
        if late["low"].min() <= early["low"].min():
            fail_counts["higher_lows"] += 1
            continue

        # Pick the best surf MA that meets the surf threshold
        surf_candidates = []
        for ma_name, col, pref in surf_mas:
            if col not in cons.columns or cons[col].isna().all():
                continue
            frac_above = float((cons["close"] >= cons[col]).mean())
            if frac_above >= surf_pct:
                surf_candidates.append((frac_above, pref, ma_name, col))

        if not surf_candidates:
            fail_counts["surf"] += 1
            continue

        # Choose: highest frac_above, then prefer EMA10 > EMA20 > EMA50
        surf_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_frac_above, _, best_ma_name, best_ma_col = surf_candidates[0]

        # MA rise check: EMA20 must be rising AND surf MA must be rising (could be EMA20)
        idx_end = len(df) - 2  # yesterday
        idx_prev = idx_end - ma_rise_lookback
        if idx_prev < 0:
            fail_counts["data"] += 1
            continue

        # EMA20 rising (anchor)
        if df["ema20"].iloc[idx_end] <= df["ema20"].iloc[idx_prev]:
            fail_counts["ma_rise"] += 1
            continue

        # Surf MA rising
        if df[best_ma_col].iloc[idx_end] <= df[best_ma_col].iloc[idx_prev]:
            fail_counts["ma_rise"] += 1
            continue

        # Impulse window immediately before consolidation
        pre = df.iloc[-(cons_len + impulse_lookback + 1):-(cons_len + 1)]
        if pre.empty:
            fail_counts["data"] += 1
            continue

        low_idx = int(pre["low"].idxmin())
        high_idx = int(pre["high"].idxmax())
        if high_idx <= low_idx:
            fail_counts["impulse"] += 1
            continue

        impulse_low = float(pre["low"].min())
        impulse_high = float(pre["high"].max())
        if impulse_low <= 0:
            fail_counts["data"] += 1
            continue

        impulse_pct = 100.0 * (impulse_high / impulse_low - 1.0)
        if impulse_pct < impulse_min_pct:
            fail_counts["impulse"] += 1
            continue

        # Pullback constraint: consolidation stays within X% of impulse high
        min_cons_high = impulse_high * (1.0 - pullback_from_impulse_high_pct / 100.0)
        if cons_high < min_cons_high:
            fail_counts["pullback"] += 1
            continue

        tight_bonus = int(max(0, min(25, (cons_depth_max_pct - depth_pct) * 1.5)))
        surf_bonus = int(max(0, min(10, (best_frac_above - surf_pct) * 20)))
        cand_score = 60 + tight_bonus + surf_bonus

        cand = {
            "cons_len": cons_len,
            "cons_high": float(cons_high),
            "cons_low": float(cons_low),
            "depth_pct": float(depth_pct),
            "frac_above": float(best_frac_above),
            "surf_ma_name": best_ma_name,
            "surf_ma_col": best_ma_col,
            "impulse_pct": float(impulse_pct),
            "score_base": int(min(85, cand_score)),
        }

        if best is None or cand["score_base"] > best["score_base"]:
            best = cand

    if best is None:
        top_fail = max(fail_counts.items(), key=lambda kv: kv[1])[0]
        reason_map = {
            "depth": "consolidation too deep",
            "higher_lows": "higher-lows test failed",
            "surf": "not surfing EMA10/20/50 enough",
            "ma_rise": "EMA rise test failed",
            "impulse": "impulse leg rule not met",
            "pullback": "pullback too deep vs impulse high",
            "data": "insufficient/invalid data in windows",
        }
        out["reason"] = "No valid MA-surf consolidation found (" + reason_map.get(top_fail, top_fail) + ")"
        return out

    pivot = best["cons_high"]
    entry = pivot * (1 + breakout_buffer_pct / 100.0)
    breakout_ok = last["close"] > entry

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
        out["signal"] = "WATCH"
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{best['cons_low']:.2f}"
        out["score"] = int(min(100, best["score_base"]))
        out["reason"] = f"Impulse {best['impulse_pct']:.0f}%, tight {best['cons_len']}d MA-surf ({best['surf_ma_name']}), no breakout yet"
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
        f"Impulse {best['impulse_pct']:.0f}%, tight {best['cons_len']}d MA-surf ({best['surf_ma_name']}), broke range"
        + (", volume ok" if vol_ok else ", volume weak")
        + (", range expanded" if tr_ok else "")
    )
    return out


def minervini_trend_template_ok(df: pd.DataFrame, cfg: dict, last: pd.Series, rs_ok: bool) -> tuple[bool, str]:
    """
    Deterministic Minervini Trend Template (strict):
      - Close > SMA50
      - SMA50 > SMA150 > SMA200
      - SMA200 rising (today > 20 trading days ago)
      - Near 52w high (uses filters.near_52w_high_pct)
      - Off lows: close >= 52w low * min_above_52w_low_mult (default 1.3)
      - Optional RS proxy (default OFF unless enabled in config)
    """
    mcfg = cfg.get("minervini", {}) or {}
    if not bool(mcfg.get("enabled", True)):
        return False, "Minervini disabled"

    fails = []

    if np.isnan(last.get("sma50", np.nan)) or np.isnan(last.get("sma150", np.nan)) or np.isnan(last.get("sma200", np.nan)):
        return False, "Minervini trend fail: SMA unavailable"

    if not (last["close"] > last["sma50"]):
        fails.append("close <= SMA50")

    if not (last["sma50"] > last["sma150"] > last["sma200"]):
        fails.append("SMA stack")

    idx_last = df.index[-1]
    sma200_up = False
    if idx_last - 20 >= 0 and not np.isnan(df["sma200"].iloc[-21]) and not np.isnan(df["sma200"].iloc[-1]):
        sma200_up = df["sma200"].iloc[-1] > df["sma200"].iloc[-21]
    if not sma200_up:
        fails.append("SMA200 not rising")

    high_52w = df["high"].rolling(252).max().iloc[-1]
    low_52w = df["low"].rolling(252).min().iloc[-1]
    near_pct = float(cfg.get("filters", {}).get("near_52w_high_pct", 25))
    if not (last["close"] >= (1 - near_pct / 100.0) * high_52w):
        fails.append("not near 52w high")

    min_above_low = float(mcfg.get("min_above_52w_low_mult", 1.3))
    if not (last["close"] >= low_52w * min_above_low):
        fails.append("not far enough off 52w low")

    # Default RS proxy OFF unless explicitly enabled
    rs_enabled = bool(mcfg.get("rs_enabled", False))
    if rs_enabled and not rs_ok:
        fails.append("RS proxy not strong (top percentile)")

    if fails:
        return False, "Minervini trend fail: " + ", ".join(fails)

    return True, ""


def eval_minervini_vcp(df: pd.DataFrame, cfg: dict, last: pd.Series, rs_ok: bool) -> dict:
    out = {"signal": "PASS", "setup": "Minervini VCP", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_minervini_vcp", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Setup disabled"
        return out

    ok, reason = minervini_trend_template_ok(df, cfg, last, rs_ok)
    if not ok:
        out["reason"] = reason
        return out

    if "volume" not in df.columns or df["volume"].isna().all():
        out["reason"] = "Volume unavailable"
        return out

    lookback = int(scfg.get("vcp_lookback", 50))
    if len(df) < lookback + 2:
        out["reason"] = "Not enough data for VCP window"
        return out

    win = df.iloc[-(lookback + 1):-1]
    if len(win) < 30:
        out["reason"] = "VCP window too small"
        return out

    n = len(win)
    seg_len = n // 3
    if seg_len < 5:
        out["reason"] = "VCP segments too small"
        return out

    early = win.iloc[:seg_len]
    mid = win.iloc[seg_len:2 * seg_len]
    late = win.iloc[2 * seg_len:]

    def range_pct(seg: pd.DataFrame) -> float:
        h = float(seg["high"].max())
        l = float(seg["low"].min())
        if h <= 0:
            return np.inf
        return 100.0 * (h - l) / h

    r1 = range_pct(early)
    r2 = range_pct(mid)
    r3 = range_pct(late)

    final_max = float(scfg.get("vcp_final_range_max_pct", 10.0))
    if not (r1 > r2 > r3):
        out["reason"] = "VCP fail: no volatility contraction"
        return out
    if r3 > final_max:
        out["reason"] = f"VCP fail: final range too wide ({r3:.1f}% > {final_max:.1f}%)"
        return out

    dry_mult = float(scfg.get("vcp_vol_dryup_mult", 0.8))
    v1 = float(early["volume"].mean())
    v3 = float(late["volume"].mean())
    if v1 > 0 and not (v3 <= v1 * dry_mult):
        out["reason"] = "VCP fail: no volume dry-up"
        return out

    pivot = float(win["high"].max())
    buffer_pct = float(scfg.get("breakout_buffer_pct", 0.10))
    entry = pivot * (1 + buffer_pct / 100.0)
    breakout_ok = last["close"] > entry

    vol_mult = float(scfg.get("vcp_breakout_vol_mult", 1.5))
    vol20 = float(df["volume"].rolling(20).mean().iloc[-1])
    vol_ok = False
    if not np.isnan(last.get("volume", np.nan)) and vol20 > 0:
        vol_ok = float(last["volume"]) >= vol20 * vol_mult

    atr_mult = float(scfg.get("atr_stop_mult", 2.0))
    atr_val = last.get("atr", np.nan)
    stop_swing = float(late["low"].min())
    stop_atr = entry - (atr_mult * atr_val) if not np.isnan(atr_val) else stop_swing
    stop = max(stop_swing, stop_atr)
    if stop >= entry:
        stop = stop_swing

    if not breakout_ok:
        out["signal"] = "WATCH"
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{stop:.2f}"
        out["score"] = 70
        out["reason"] = f"Trend ok, VCP contracting (ranges {r1:.1f}>{r2:.1f}>{r3:.1f}), volume drying, no breakout yet"
        return out

    if not vol_ok:
        out["reason"] = "VCP breakout fail: volume not strong enough"
        return out

    score = 80 + int(max(0, min(10, (final_max - r3) * 1.5)))
    score = int(min(100, score))

    out["signal"] = "BUY_NOW"
    out["score"] = score
    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["reason"] = "VCP breakout: contraction + volume dry-up, broke pivot with volume"
    return out


def eval_minervini_3wt(df: pd.DataFrame, cfg: dict, last: pd.Series, rs_ok: bool) -> dict:
    out = {"signal": "PASS", "setup": "Minervini 3-Week Tight", "score": 0, "entry": "", "stop": "", "reason": ""}

    scfg = cfg.get("setup_minervini_3wt", {}) or {}
    if not bool(scfg.get("enabled", True)):
        out["reason"] = "Setup disabled"
        return out

    ok, reason = minervini_trend_template_ok(df, cfg, last, rs_ok)
    if not ok:
        out["reason"] = reason
        return out

    twt_bars = int(scfg.get("twt_bars", 15))
    if len(df) < twt_bars + 2:
        out["reason"] = "Not enough data for 3-week window"
        return out

    win = df.iloc[-(twt_bars + 1):-1]
    if len(win) != twt_bars:
        out["reason"] = "3-week window invalid"
        return out

    h = float(win["high"].max())
    l = float(win["low"].min())
    if h <= 0:
        out["reason"] = "Invalid price data"
        return out

    max_range = float(scfg.get("twt_max_range_pct", 8.0))
    rng_pct = 100.0 * (h - l) / h
    if rng_pct > max_range:
        out["reason"] = f"3WT fail: range too wide ({rng_pct:.1f}% > {max_range:.1f}%)"
        return out

    close_pos = float(scfg.get("twt_week_close_pos", 0.6))
    for i in range(3):
        wk = win.iloc[i * 5:(i + 1) * 5]
        wk_high = float(wk["high"].max())
        wk_low = float(wk["low"].min())
        wk_close = float(wk["close"].iloc[-1])
        if wk_high <= wk_low:
            if wk_close != wk_high:
                out["reason"] = "3WT fail: weekly close position"
                return out
        else:
            thresh = wk_low + close_pos * (wk_high - wk_low)
            if wk_close < thresh:
                out["reason"] = "3WT fail: weekly close not strong enough"
                return out

    pivot = float(win["high"].max())
    buffer_pct = float(scfg.get("breakout_buffer_pct", 0.10))
    entry = pivot * (1 + buffer_pct / 100.0)
    breakout_ok = last["close"] > entry

    vol_confirm = bool(scfg.get("vol_confirm_enabled", True))
    vol_ok = True
    if vol_confirm:
        if "volume" not in df.columns or df["volume"].isna().all():
            out["reason"] = "Volume unavailable"
            return out
        vol_mult = float(scfg.get("twt_breakout_vol_mult", 1.2))
        vol20 = float(df["volume"].rolling(20).mean().iloc[-1])
        vol_ok = False
        if not np.isnan(last.get("volume", np.nan)) and vol20 > 0:
            vol_ok = float(last["volume"]) >= vol20 * vol_mult

    atr_mult = float(scfg.get("atr_stop_mult", 2.0))
    atr_val = last.get("atr", np.nan)
    last_week = win.iloc[-5:]
    stop_swing = float(last_week["low"].min())
    stop_atr = entry - (atr_mult * atr_val) if not np.isnan(atr_val) else stop_swing
    stop = max(stop_swing, stop_atr)
    if stop >= entry:
        stop = stop_swing

    if not breakout_ok:
        out["signal"] = "WATCH"
        out["entry"] = f"{entry:.2f}"
        out["stop"] = f"{stop:.2f}"
        out["score"] = 72
        out["reason"] = f"Trend ok, 3WT tight ({rng_pct:.1f}%), no breakout yet"
        return out

    if vol_confirm and not vol_ok:
        out["reason"] = "3WT breakout fail: volume not strong enough"
        return out

    score = 82 + int(max(0, min(10, (max_range - rng_pct) * 1.5)))
    score = int(min(100, score))

    out["signal"] = "BUY_NOW"
    out["score"] = score
    out["entry"] = f"{entry:.2f}"
    out["stop"] = f"{stop:.2f}"
    out["reason"] = "3WT breakout: tight 3-week range, broke pivot" + (", volume ok" if vol_confirm else "")
    return out


def analyse_symbol(df: pd.DataFrame, cfg: dict, min_rs_ok: bool) -> tuple[dict, list[dict]]:
    """
    Returns:
      - best_result: single 'winner' for Signals tab
      - all_results: list of setup results (BUY_NOW/WATCH union across setups)
    """
    if df.empty or len(df) < 260:
        res = {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": "Not enough daily data", "debug": ""}
        return res, [res]

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

    # Minervini SMAs
    df["sma50"] = compute_sma(df["close"], 50)
    df["sma150"] = compute_sma(df["close"], 150)

    last = df.iloc[-1]

    all_results = [
        eval_qullamaggie_breakout(df, cfg, last),
        eval_minervini_vcp(df, cfg, last, min_rs_ok),
        eval_minervini_3wt(df, cfg, last, min_rs_ok),
        eval_base_breakout(df, cfg, last),
    ]

    best = pick_best(all_results)
    best_setup = (best.get("setup") or "").strip()
    best["debug"] = build_debug_summary(all_results, best_setup, max_items=3)
    return best, all_results


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

            if isinstance(data, dict) and "values" not in data:
                for sym in sym_batch:
                    payload = data.get(sym, {}) or {}
                    df = normalise_timeseries_payload(sym, payload)
                    df_map[sym] = df
                    if df.empty:
                        err_map[sym] = payload.get("message", "No data")
            else:
                sym = sym_batch[0]
                df = normalise_timeseries_payload(sym, data)
                df_map[sym] = df
                if df.empty:
                    err_map[sym] = data.get("message", "No data")
                for other in sym_batch[1:]:
                    df_map[other] = pd.DataFrame()
                    err_map[other] = "No data (batch response mismatch)"

        except Exception as e:
            for sym in sym_batch:
                df_map[sym] = pd.DataFrame()
                err_map[sym] = f"Fetch error: {type(e).__name__}"
            errors += 1

        sleep_s = (batch_credits / max_credits_per_min) * 60.0
        time.sleep(sleep_s)

    # RS proxy flags: Minervini (default OFF unless enabled)
    mcfg = cfg.get("minervini", {}) or {}
    rs_enabled = bool(mcfg.get("rs_enabled", False))
    rs_flags = {sym: True for sym in df_map.keys()}
    if rs_enabled:
        rs_windows = mcfg.get("rs_windows", [63, 126])
        rs_top = float(mcfg.get("rs_top_pct", 20))
        rs_windows = [int(x) for x in rs_windows] if isinstance(rs_windows, list) else [63, 126]
        rs_flags = compute_percentile_flags(df_map, rs_windows, rs_top)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ws_signals = upsert_worksheet(sh, "Signals", rows=max(1000, len(tickers) + 10), cols=14)
    ws_buys = upsert_worksheet(sh, "BUY_NOW", rows=1000, cols=12)
    ws_watch = upsert_worksheet(sh, "WATCH", rows=2000, cols=12)
    ws_summary = upsert_worksheet(sh, "Summary", rows=50, cols=4)
    ws_log = upsert_worksheet(sh, "Run_Log", rows=1000, cols=12)

    sig_header = ["ticker", "signal", "setup", "score", "entry", "stop", "reason", "debug", "as_of_utc"]
    sig_rows = [sig_header]

    buy_header = ["line", "ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    buy_rows = [buy_header]

    watch_header = ["ticker", "setup", "score", "entry", "stop", "reason", "as_of_utc"]
    watch_rows = [watch_header]

    buy_items = []
    watch_items = []
    buy_tickers = set()
    watch_tickers = set()

    for sym in tickers:
        df = df_map.get(sym, pd.DataFrame())
        sym_disp = display_ticker(sym)

        if df is None or df.empty:
            best = {"signal": "PASS", "setup": "", "score": 0, "entry": "", "stop": "", "reason": err_map.get(sym, "No data"), "debug": ""}
            all_res = [best]
        else:
            best, all_res = analyse_symbol(df, cfg, rs_flags.get(sym, True))

        sig_rows.append([
            sym,
            best.get("signal", ""),
            best.get("setup", ""),
            int(best.get("score", 0) or 0),
            best.get("entry", ""),
            best.get("stop", ""),
            best.get("reason", ""),
            best.get("debug", ""),
            now_utc,
        ])

        # BUY_NOW/WATCH as UNION of all setups
        for r in all_res:
            sig = (r.get("signal") or "").strip()
            setup = (r.get("setup") or "").strip()
            score = int(r.get("score", 0) or 0)
            entry = r.get("entry", "")
            stop = r.get("stop", "")
            reason = r.get("reason", "")

            if sig == "BUY_NOW":
                buy_tickers.add(sym_disp)
                line = f"{sym_disp} – BUY NOW – Setup: {setup} – Entry: {entry} – Stop: {stop} – Reason: {reason}"
                buy_items.append((score, [line, sym_disp, setup, score, entry, stop, reason, now_utc]))

            if sig == "WATCH":
                watch_tickers.add(sym_disp)
                watch_items.append((score, [sym_disp, setup, score, entry, stop, reason, now_utc]))

    ws_signals.clear()
    ws_signals.update("A1", sig_rows)

    buy_items.sort(key=lambda x: x[0], reverse=True)
    buy_rows.extend([row for _, row in buy_items])
    ws_buys.clear()
    ws_buys.update("A1", buy_rows)

    watch_items.sort(key=lambda x: x[0], reverse=True)
    watch_rows.extend([row for _, row in watch_items])
    ws_watch.clear()
    ws_watch.update("A1", watch_rows)

    buy_count = len(buy_tickers)
    watch_count = len(watch_tickers)
    pass_count = max(0, len(tickers) - len(buy_tickers.union(watch_tickers)))

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
