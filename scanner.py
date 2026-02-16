# scanner.py
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
SETUP_NAME = "Minervini VCP"


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


def get_exchange(sym: str) -> str:
    # Convert MSFT:NASDAQ -> NASDAQ (empty if none)
    if ":" in sym:
        return sym.split(":", 1)[1].strip().upper()
    return ""


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
# Helpers: ATR, TR%, slope
# ---------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    return true_range(df).rolling(length).mean()


def atr_percent(df: pd.DataFrame, atr_len: int = 14) -> pd.Series:
    atr = compute_atr(df, atr_len)
    return (atr / df["close"]) * 100.0


def linreg_slope(y: np.ndarray) -> float:
    """
    Simple linear regression slope, returns slope per bar in y-units.
    """
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
        return 0.0
    return 100.0 * (b / a - 1.0)


def pct_depth(peak: float, trough: float) -> float:
    if peak <= 0 or np.isnan(peak) or np.isnan(trough):
        return 0.0
    return 100.0 * (peak - trough) / peak


def pct_range(high_val: float, low_val: float, denom: float | None = None) -> float:
    if denom is None:
        denom = high_val
    if denom <= 0:
        return 999.0
    return 100.0 * (high_val - low_val) / denom


# ---------------------------
# Pivot / contraction logic
# ---------------------------

def find_pivots(high: pd.Series, low: pd.Series, left: int, right: int) -> list[tuple[int, str, float]]:
    """
    Returns pivots: (index, 'H' or 'L', price)
    Pivot High: high[i] is max in [i-left, i+right]
    Pivot Low : low[i] is min in [i-left, i+right]
    """
    n = len(high)
    pivots: list[tuple[int, str, float]] = []
    if n < left + right + 3:
        return pivots

    h = high.values
    l = low.values

    for i in range(left, n - right):
        w0 = i - left
        w1 = i + right + 1

        hi = h[i]
        lo = l[i]

        if np.isnan(hi) or np.isnan(lo):
            continue

        win_h = h[w0:w1]
        win_l = l[w0:w1]
        if np.isnan(win_h).all() or np.isnan(win_l).all():
            continue

        if hi >= np.nanmax(win_h):
            pivots.append((i, "H", float(hi)))
        if lo <= np.nanmin(win_l):
            pivots.append((i, "L", float(lo)))

    pivots.sort(key=lambda x: x[0])
    return pivots


def enforce_alternation(pivots: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """
    Forces H/L alternation. If same type repeats, keep the more extreme one.
    Ensures sequence starts with H (otherwise drops leading L).
    """
    if not pivots:
        return []

    out = [pivots[0]]
    for idx, typ, price in pivots[1:]:
        last_idx, last_typ, last_price = out[-1]
        if typ != last_typ:
            out.append((idx, typ, price))
            continue

        if typ == "H":
            if price >= last_price:
                out[-1] = (idx, typ, price)
        else:
            if price <= last_price:
                out[-1] = (idx, typ, price)

    if out and out[0][1] == "L":
        out = out[1:]

    return out


def build_contractions(pivots: list[tuple[int, str, float]]) -> list[dict]:
    """
    Build contraction legs from alternating pivots:
      H1 -> L1 -> H2 -> L2 -> H3 ...
    Each contraction i uses:
      peak = H_i
      trough = L_i
      rebound_peak = H_{i+1} (if present)
    """
    piv = enforce_alternation(pivots)
    contractions: list[dict] = []

    i = 0
    while i + 1 < len(piv):
        if piv[i][1] != "H" or piv[i + 1][1] != "L":
            i += 1
            continue

        peak_idx, _, peak_price = piv[i]
        low_idx, _, low_price = piv[i + 1]

        rebound_idx = None
        rebound_price = None
        if i + 2 < len(piv) and piv[i + 2][1] == "H":
            rebound_idx = piv[i + 2][0]
            rebound_price = piv[i + 2][2]

        contractions.append({
            "peak_idx": peak_idx,
            "peak": float(peak_price),
            "trough_idx": low_idx,
            "trough": float(low_price),
            "rebound_idx": rebound_idx,
            "rebound": float(rebound_price) if rebound_price is not None else np.nan,
        })

        i += 2

    return contractions


# ---------------------------
# Tightening + fail filters (no volume gating)
# ---------------------------

def tightening_ok(df: pd.DataFrame, cfg: dict) -> tuple[bool, str, dict]:
    """
    Right-side tightening checks (either can pass; both is best):
      A) ATR% compression: ATRp_right <= ATRp_left * X
      B) Final segment tightness:
           - range_right <= % threshold
           - closes in top half >= % threshold
    """
    tcfg = cfg.get("vcp", {}).get("tightening", {}) or {}

    atr_len = int(tcfg.get("atr_len", 14))
    left_right_ratio_req = float(tcfg.get("atrp_right_to_left_max", 0.8))
    left_right_ratio_pref = float(tcfg.get("atrp_right_to_left_pref", 0.7))

    right_bars = int(tcfg.get("right_segment_bars", 15))
    right_range_max = float(tcfg.get("right_range_max_pct", 8.0))
    right_range_max_soft = float(tcfg.get("right_range_max_pct_soft", 10.0))
    top_half_min = float(tcfg.get("right_close_top_half_min_pct", 60.0))

    if len(df) < max(60, right_bars + 5):
        return False, "Not enough data for tightening checks", {}

    atrp = atr_percent(df, atr_len=atr_len)
    if atrp.isna().all():
        return False, "ATR% not available", {}

    n = len(df)
    third = max(10, int(n / 3))
    left = atrp.iloc[:third].dropna()
    right = atrp.iloc[-third:].dropna()
    if left.empty or right.empty:
        return False, "ATR% segments empty", {}

    atrp_left = float(left.mean())
    atrp_right = float(right.mean())
    ratio = (atrp_right / atrp_left) if atrp_left > 0 else 999.0

    seg = df.iloc[-right_bars:]
    hh = float(seg["high"].max())
    ll = float(seg["low"].min())
    mid = (hh + ll) / 2.0 if (hh + ll) != 0 else hh
    r_range = pct_range(hh, ll, denom=mid)

    half = ll + 0.5 * (hh - ll)
    top_half_pct = float((seg["close"] >= half).mean() * 100.0)

    stats = {
        "atrp_left": atrp_left,
        "atrp_right": atrp_right,
        "atrp_ratio": ratio,
        "right_range_pct": r_range,
        "right_top_half_pct": top_half_pct,
    }

    atr_ok = ratio <= left_right_ratio_req
    range_ok = (r_range <= right_range_max_soft) and (top_half_pct >= (top_half_min - 5.0))

    if not atr_ok and not range_ok:
        return False, "No right-side tightening (ATR% and final range both weak)", stats

    if ratio <= left_right_ratio_pref and r_range <= right_range_max and top_half_pct >= top_half_min:
        return True, "Strong right-side tightening (ATR% + tight final segment)", stats
    if ratio <= left_right_ratio_req:
        return True, "ATR% compression on right side", stats
    return True, "Tight final segment on right side", stats


def is_grind_up_not_base(contractions: list[dict], cfg: dict) -> bool:
    fcfg = cfg.get("vcp", {}).get("fail_filters", {}) or {}
    min_meaningful = float(fcfg.get("min_meaningful_contraction_pct", 7.0))

    depths = [pct_depth(c["peak"], c["trough"]) for c in contractions if c["peak"] > 0]
    if not depths:
        return True
    return max(depths) < min_meaningful


def right_side_vol_expansion_fail(df: pd.DataFrame, depths: list[float], cfg: dict) -> bool:
    """
    Reject if volatility expands on the right:
      - last contraction > prev contraction by > mult AND ATR% rising
      - multiple wide-range down bars in last segment
    """
    fcfg = cfg.get("vcp", {}).get("fail_filters", {}) or {}
    max_last_vs_prev = float(fcfg.get("max_last_depth_vs_prev_mult", 1.2))
    wide_down_bars_min = int(fcfg.get("wide_down_bars_min", 2))
    wide_down_tr_mult = float(fcfg.get("wide_down_tr_mult", 1.5))
    lookback = int(fcfg.get("right_fail_lookback_bars", 15))
    atr_len = int(cfg.get("vcp", {}).get("tightening", {}).get("atr_len", 14))

    if len(df) < max(60, lookback + 5):
        return False

    depth_bad = False
    if len(depths) >= 2:
        d_prev = depths[-2]
        d_last = depths[-1]
        if d_prev > 0 and d_last > (d_prev * max_last_vs_prev):
            atrp = atr_percent(df, atr_len=atr_len)
            seg = atrp.iloc[-(lookback + 5):].dropna()
            if len(seg) >= 10:
                slope = linreg_slope(seg.values)
                depth_bad = slope > 0

    seg = df.iloc[-lookback:]
    trp = (true_range(seg) / seg["close"]) * 100.0
    base_med = float(trp.median()) if trp.notna().any() else np.nan
    if np.isnan(base_med) or base_med <= 0:
        return depth_bad

    down = seg["close"] < seg["open"]
    wide = trp > (base_med * wide_down_tr_mult)
    wide_down = int((down & wide).sum())

    return depth_bad or (wide_down >= wide_down_bars_min)


def contraction_shrink_ok(depths: list[float], cfg: dict) -> tuple[bool, str, dict]:
    vcfg = cfg.get("vcp", {}) or {}
    min_c = int(vcfg.get("min_contractions", 2))
    max_c = int(vcfg.get("max_contractions", 6))

    shrink_mult = float(vcfg.get("shrink_step_mult", 0.92))
    shrink_steps_min_frac = float(vcfg.get("shrink_steps_min_frac", 0.45))
    allow_one_exception = bool(vcfg.get("allow_one_exception", True))
    exception_mult = float(vcfg.get("exception_step_mult", 1.2))

    last_vs_first_mult = float(vcfg.get("last_vs_first_max_mult", 0.7))
    max_last_depth = float(vcfg.get("max_last_contraction_pct", 15.0))

    if len(depths) < min_c or len(depths) > max_c:
        return False, f"Contraction count {len(depths)} not in range {min_c}-{max_c}", {"N": len(depths)}

    steps = []
    exceptions = 0
    ok_steps = 0

    for i in range(len(depths) - 1):
        d0 = depths[i]
        d1 = depths[i + 1]
        if d1 <= d0 * shrink_mult:
            ok_steps += 1
            steps.append("ok")
        elif allow_one_exception and d1 <= d0 * exception_mult:
            exceptions += 1
            steps.append("exc")
        else:
            steps.append("bad")

    need_ok = int(np.ceil((len(depths) - 1) * shrink_steps_min_frac))

    if ok_steps < need_ok:
        return False, "Not enough shrinking steps", {"ok_steps": ok_steps, "need_ok": need_ok, "steps": steps, "depths": depths}

    if exceptions > 1:
        return False, "Too many contraction exceptions", {"exceptions": exceptions, "steps": steps, "depths": depths}

    d_first = depths[0]
    d_last = depths[-1]

    if d_last > d_first * last_vs_first_mult:
        return False, "Last contraction not tight enough vs first", {"d_first": d_first, "d_last": d_last, "mult": last_vs_first_mult}

    if d_last > max_last_depth:
        return False, "Last contraction too deep", {"d_last": d_last, "max_last_depth": max_last_depth}

    return True, "Contractions shrinking", {"ok_steps": ok_steps, "exceptions": exceptions, "steps": steps, "depths": depths}


def dist_from_52w_high_pct(df: pd.DataFrame, lookback: int = 252) -> float:
    if df is None or df.empty:
        return 999.0
    sub = df.iloc[-min(len(df), lookback):]
    hh = float(sub["high"].max())
    last = float(df["close"].iloc[-1])
    if hh <= 0:
        return 999.0
    return 100.0 * (hh - last) / hh


# ---------------------------
# Full VCP detector (base ending yesterday, breakout today)
# ---------------------------

def detect_vcp(df_full: pd.DataFrame, cfg: dict) -> dict:
    """
    Formation assessed ending yesterday; breakout assessed on today's bar.
    Volume gating is REMOVED (no volume character, no breakout volume confirm).
    """
    out = {"signal": "PASS", "setup": SETUP_NAME, "score": 0, "entry": "", "stop": "", "reason": ""}

    if df_full is None or df_full.empty or len(df_full) < 120:
        out["reason"] = "Not enough daily data"
        return out

    vcfg = cfg.get("vcp", {}) or {}

    # Base window (weeks)
    min_weeks = int(vcfg.get("min_base_weeks", 3))
    max_weeks = int(vcfg.get("max_base_weeks", 60))
    prefer_min_w = int(vcfg.get("prefer_base_weeks_min", 6))
    prefer_max_w = int(vcfg.get("prefer_base_weeks_max", 12))

    left_high_max_pos = float(vcfg.get("left_high_max_pos_frac", 0.4))
    overshoot_tol = float(vcfg.get("base_top_overshoot_tol_pct", 0.5))

    # Pivot / entry rules
    entry_buffer_pct = float(vcfg.get("entry_buffer_pct", 0.2))
    near_breakout_pct = float(vcfg.get("near_breakout_pct", 3.0))

    # Pretrend context (no MA filters)
    pretrend_bars = int(vcfg.get("pretrend_bars", 100))
    min_pretrend_return = float(vcfg.get("min_pretrend_return_pct", 15.0))
    require_positive_slope = bool(vcfg.get("require_positive_pretrend_slope", True))

    # Pivot selection
    pivot_use_last_rebound = bool(vcfg.get("pivot_use_last_rebound_peak", True))
    pivot_exclude_last_n = int(vcfg.get("pivot_exclude_last_n_bars", 2))

    # Breakout behaviour
    bcfg = vcfg.get("breakout", {}) or {}
    breakout_on = str(bcfg.get("price_trigger", "close")).lower()  # close or high
    require_fresh_cross = bool(bcfg.get("require_fresh_cross", False))  # changed default: false

    # Context (52w high proximity)
    ctx = vcfg.get("context", {}) or {}
    max_dist_52w = float(ctx.get("max_dist_from_52w_high_pct", 40.0))
    if max_dist_52w > 0:
        d52 = dist_from_52w_high_pct(df_full, lookback=int(ctx.get("lookback_52w_bars", 252)))
        if d52 > max_dist_52w:
            out["reason"] = f"Too far from 52w high ({d52:.0f}% > {max_dist_52w:.0f}%)"
            return out

    piv_cfg = vcfg.get("pivots", {}) or {}
    left_opts = piv_cfg.get("left_bars_options", [3, 4, 5, 6, 7])
    right_opts = piv_cfg.get("right_bars_options", [3, 4, 5, 6, 7])

    today = df_full.iloc[-1]
    form = df_full.iloc[:-1].copy()
    if len(form) < 100:
        out["reason"] = "Not enough formation data"
        return out

    best = None  # (priority, score, result_dict)

    def book_best(priority: int, score: int, d: dict):
        nonlocal best
        key = (priority, score)
        if best is None or key > (best[0], best[1]):
            best = (priority, score, d)

    max_w_possible = int((len(form) - 30) / 5)
    max_w = max(min_weeks, min(max_weeks, max_w_possible))

    for W in range(min_weeks, max_w + 1):
        bars = int(W * 5)
        if len(form) < bars + 10:
            continue

        base = form.iloc[-bars:].copy()

        left_pos = int(np.nanargmax(base["high"].values))
        if left_pos > int(len(base) * left_high_max_pos):
            continue

        left_high = float(base["high"].iloc[left_pos])
        if np.isnan(left_high) or left_high <= 0:
            continue

        later_max = float(base["high"].iloc[left_pos:].max())
        if later_max > left_high * (1 + overshoot_tol / 100.0):
            continue

        # Uptrend into base
        full_left_pos = (len(df_full) - 1 - bars) + left_pos
        start_pre = max(0, full_left_pos - pretrend_bars)
        if full_left_pos - start_pre < max(30, int(pretrend_bars * 0.6)):
            continue

        pre = df_full.iloc[start_pre:full_left_pos + 1]
        pre_ret = pct_change(float(pre["close"].iloc[0]), float(pre["close"].iloc[-1]))
        if pre_ret < min_pretrend_return:
            continue
        if require_positive_slope:
            slope = linreg_slope(pre["close"].values)
            if slope <= 0:
                continue

        sub = base.iloc[left_pos:].copy()
        if len(sub) < 25:
            continue

        best_local = None  # (local_score, details)

        for lb in left_opts:
            for rb in right_opts:
                pivots = find_pivots(sub["high"], sub["low"], int(lb), int(rb))
                pivots = enforce_alternation(pivots)
                if len(pivots) < 5:
                    continue

                contractions = build_contractions(pivots)
                if not contractions:
                    continue

                if len(contractions) > int(vcfg.get("reject_if_contractions_gt", 8)):
                    continue

                if is_grind_up_not_base(contractions, cfg):
                    continue

                depths = [pct_depth(c["peak"], c["trough"]) for c in contractions]
                depths = [float(d) for d in depths if not np.isnan(d)]
                if len(depths) < 2:
                    continue

                max_base_depth = float(vcfg.get("max_base_depth_pct", 50.0))
                base_low = float(sub["low"].min())
                base_depth = pct_depth(left_high, base_low)
                if base_depth > max_base_depth:
                    continue

                ok_shrink, _, shrink_stats = contraction_shrink_ok(depths, cfg)
                if not ok_shrink:
                    continue

                ok_tight, _, tight_stats = tightening_ok(sub, cfg)
                if not ok_tight:
                    continue

                if right_side_vol_expansion_fail(sub, depths, cfg):
                    continue

                # Optional handle sweet spot
                handle_ok = True
                hcfg = vcfg.get("handle", {}) or {}
                if bool(hcfg.get("enabled", True)):
                    last_low_idx = int(contractions[-1]["trough_idx"])
                    handle_ok = last_low_idx >= int(len(sub) * float(hcfg.get("last_low_min_pos_frac", 0.6)))
                    chop_bars = int(hcfg.get("min_chop_bars", 5))
                    if handle_ok and len(sub) >= chop_bars:
                        recent = sub.iloc[-chop_bars:]
                        rh = float(recent["high"].max())
                        rl = float(recent["low"].min())
                        mid = (rh + rl) / 2.0 if (rh + rl) != 0 else rh
                        recent_range = pct_range(rh, rl, denom=mid)
                        if recent_range > float(hcfg.get("max_chop_range_pct", 6.0)):
                            handle_ok = False

                # Pivot (buy point)
                pivot = np.nan
                if pivot_use_last_rebound and contractions[-1].get("rebound_idx") is not None and not np.isnan(contractions[-1].get("rebound_idx", np.nan)):
                    pivot = float(contractions[-1].get("rebound", np.nan))

                if np.isnan(pivot) or pivot <= 0:
                    excl = max(0, pivot_exclude_last_n)
                    hh_slice = sub["high"].iloc[:max(1, len(sub) - excl)] if excl > 0 else sub["high"]
                    pivot = float(hh_slice.max()) if not hh_slice.empty else left_high

                entry = pivot * (1 + entry_buffer_pct / 100.0)
                stop = float(contractions[-1]["trough"])

                # NOTE: risk cap intentionally removed per your request

                # Score
                score = 50
                if prefer_min_w <= W <= prefer_max_w:
                    score += 10
                else:
                    score += 4

                N = len(depths)
                if 3 <= N <= 4:
                    score += 12
                elif N == 2:
                    score += 6
                elif 5 <= N <= 6:
                    score += 8

                ok_steps = int(shrink_stats.get("ok_steps", 0))
                steps_total = max(1, N - 1)
                score += int(min(15, (ok_steps / steps_total) * 15))

                d_last = float(depths[-1])
                if d_last <= 8:
                    score += 10
                elif d_last <= float(vcfg.get("max_last_contraction_pct", 15.0)):
                    score += 6

                atr_ratio = float(tight_stats.get("atrp_ratio", 999.0))
                right_range = float(tight_stats.get("right_range_pct", 999.0))
                top_half_pct = float(tight_stats.get("right_top_half_pct", 0.0))

                if atr_ratio <= float(cfg.get("vcp", {}).get("tightening", {}).get("atrp_right_to_left_pref", 0.7)):
                    score += 10
                elif atr_ratio <= float(cfg.get("vcp", {}).get("tightening", {}).get("atrp_right_to_left_max", 0.8)):
                    score += 6

                if right_range <= float(cfg.get("vcp", {}).get("tightening", {}).get("right_range_max_pct", 8.0)) and top_half_pct >= float(cfg.get("vcp", {}).get("tightening", {}).get("right_close_top_half_min_pct", 60.0)):
                    score += 8
                elif right_range <= float(cfg.get("vcp", {}).get("tightening", {}).get("right_range_max_pct_soft", 10.0)):
                    score += 4

                pre_low = float(pre["low"].min()) if "low" in pre.columns else float(pre["close"].min())
                runup = pct_change(pre_low, left_high)
                score += int(min(10, max(0, runup / 10.0)))

                if handle_ok:
                    score += 4

                score = int(min(100, max(1, score)))

                details = {
                    "W": W,
                    "contractions": N,
                    "depths": depths,
                    "pivot": pivot,
                    "entry": entry,
                    "stop": stop,
                    "runup_pct": runup,
                    "handle_ok": handle_ok,
                    "score": score,
                }

                local_score = score + (2 if handle_ok else 0)
                if best_local is None or local_score > best_local[0]:
                    best_local = (local_score, details)

        if best_local is None:
            continue

        d = best_local[1]
        entry = float(d["entry"])
        stop = float(d["stop"])
        pivot = float(d["pivot"])
        score = int(d["score"])

        today_close = float(today["close"])
        today_high = float(today["high"])

        prev = form.iloc[-1]
        prev_close = float(prev["close"])
        prev_high = float(prev["high"])

        price_cross = (today_close > entry) if breakout_on == "close" else (today_high > entry)

        if require_fresh_cross:
            prev_cross = (prev_close > entry) if breakout_on == "close" else (prev_high > entry)
            if prev_cross:
                price_cross = False

        near = today_close >= entry * (1 - near_breakout_pct / 100.0)

        reason_parts = [
            f"{d['contractions']}T",
            "depths " + "→".join([f"{x:.0f}%" for x in d["depths"]]),
            f"base {d['W']}w",
            f"pivot {pivot:.2f}",
        ]

        if price_cross:
            signal = "BUY_NOW"
            reason_parts.append("breakout")
            priority = 2
        elif near:
            signal = "WATCH"
            reason_parts.append("near pivot")
            priority = 1
        else:
            signal = "PASS"
            reason_parts.append("valid VCP but not near pivot")
            priority = 0

        result = {
            "signal": signal,
            "setup": SETUP_NAME,
            "score": score,
            "entry": f"{entry:.2f}",
            "stop": f"{stop:.2f}",
            "reason": ", ".join(reason_parts),
            "pivot": f"{pivot:.2f}",
            "base_weeks": d["W"],
            "contractions": d["contractions"],
            "depths": "->".join([f"{x:.1f}" for x in d["depths"]]),
            "risk_pct": f"{pct_depth(entry, stop):.1f}",
            "runup_pct": f"{d['runup_pct']:.0f}",
        }

        book_best(priority, score, result)

    if best is None:
        out["reason"] = "No VCP base found"
        return out

    out.update(best[2])
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

    # Exchange filtering: NO whitelist, only exclude list (to avoid nuking unknown formats)
    fcfg = cfg.get("filters", {}) or {}
    exclude_exchanges = [str(x).upper() for x in (fcfg.get("exclude_exchanges", []) or [])]

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
                payload = data
                df = normalise_timeseries_payload(sym, payload)

                # Exchange exclude
                if get_exchange(sym) in exclude_exchanges:
                    results.append({"ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": "Excluded exchange"})
                    continue

                if df.empty:
                    results.append({"ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": payload.get("message", "No data")})
                else:
                    r = detect_vcp(df, cfg)
                    results.append({"ticker": sym, **r})
            else:
                # Multi-symbol response
                for sym in sym_batch:
                    payload = data.get(sym, {}) if isinstance(data, dict) else {}
                    df = normalise_timeseries_payload(sym, payload)

                    # Exchange exclude
                    if get_exchange(sym) in exclude_exchanges:
                        results.append({"ticker": sym, "setup": SETUP_NAME, "signal": "PASS", "score": 0, "entry": "", "stop": "", "reason": "Excluded exchange"})
                        continue

                    if df.empty:
                        results.append({
                            "ticker": sym,
                            "setup": SETUP_NAME,
                            "signal": "PASS",
                            "score": 0,
                            "entry": "",
                            "stop": "",
                            "reason": payload.get("message", "No data"),
                        })
                        continue

                    r = detect_vcp(df, cfg)
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
            line = f"{sym_disp} – BUY NOW – Setup: {setup} – Entry: {entry} – Stop: {stop} – Reason: {reason}"
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

    note = f"ok ({tickers_source}) paced at {max_credits_per_min}/min, batch_size={batch_size} (no volume gating)"

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
