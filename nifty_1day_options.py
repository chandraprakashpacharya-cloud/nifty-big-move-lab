#!/usr/bin/env python3
"""
NIFTY 50 1-DAY OPTIONS ENGINE
=============================

Purpose
-------
One-day intraday NIFTY CALL/PUT strategy using the previous 1, 2, 3, 5 and
7 trading days as the context. The engine generates a NIFTY spot signal and
maps it to an ATM/ITM option.

IMPORTANT
---------
Yahoo/yfinance does NOT provide a reliable historical NIFTY option-premium
series for arbitrary past dates. Therefore the default backtest is an INDEX
SIGNAL backtest. It reports the option to buy (CE/PE, strike selection), but
its P&L is based on NIFTY points, not historical option premium.

For TRUE OPTION-PREMIUM BACKTESTING, supply a CSV with:
Date,Expiry,Strike,OptionType,Open,High,Low,Close,Volume
The optional option backtest will then use actual option OHLC.

Strategy
--------
1. Daily context from the previous 1/2/3/5/7 completed sessions.
2. 09:15-09:30 opening range.
3. Identify major support/resistance and confirmed candle swing trendlines.
4. Require a real candle CLOSE beyond the level.
5. Require a retest of the broken level and a successful hold.
6. Require a strong continuation candle before entry.
7. CALL requires bullish trendline/EMA/VWAP alignment; PUT is the inverse.
8. Entry on the next candle OPEN to avoid look-ahead.
9. Stop = 0.9 ATR; target = 1.8R by default.
10. One trade per day; no trade if the structure sequence is absent.
11. Option selection = ATM by default, configurable ITM steps.

Examples
--------
python nifty_1day_options.py --mode backtest --period 60d --interval 15m
python nifty_1day_options.py --mode scan --period 10d --interval 5m
python nifty_1day_options.py --mode option-backtest --spot-csv nifty.csv --options-csv options.csv
python nifty_1day_options.py --mode web
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
IST = ZoneInfo("Asia/Kolkata")
SYMBOL = "^NSEI"

DEFAULT_PERIOD = "60d"
DEFAULT_INTERVAL = "15m"
OR_MINUTES = 15
LAST_ENTRY = "14:30"
FORCE_EXIT = "15:15"

ADX_MIN = 22.0
REL_VOL_MIN = 1.15
RSI_LONG_MIN, RSI_LONG_MAX = 52.0, 72.0
RSI_SHORT_MIN, RSI_SHORT_MAX = 28.0, 48.0
ATR_EXPANSION_MIN = 1.05
BODY_RATIO_MIN = 0.55
CLOSE_LOCATION_MIN = 0.70
BREAK_BUFFER_ATR = 0.10
STOP_ATR = 0.90
TARGET_R = 1.80
MAX_BARS_IN_TRADE = 12
MAX_TRADES_PER_DAY = 1

# Strict price-action rules: trendline + breakout + retest + confirmation.
SWING_WINDOW = 2
RETEST_LOOKBACK = 5
RETEST_TOLERANCE_ATR = 0.25
MIN_SETUP_SCORE = 9

# 0 = ATM, 1 = one strike ITM, 2 = two strikes ITM.
# For a CALL ITM means lower strike; for a PUT ITM means higher strike.
OPTION_ITM_STEPS = 0
NIFTY_STRIKE_STEP = 50

app = Flask(__name__)
CORS(app)


def json_safe(value):
    """Recursively convert NumPy/Pandas scalar types to JSON-native values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def download_data(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    df = yf.download(SYMBOL, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No NIFTY data returned by Yahoo Finance.")
    df = _flat_columns(df.copy())
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    df = df[required].dropna().copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert(IST)
    else:
        idx = idx.tz_convert(IST)
    df.index = idx
    df = df.between_time("09:15", "15:30")
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(high, low, close, n=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift(1)
    tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atrv = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atrv
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atrv
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def session_vwap(df):
    day = df.index.date
    tp = (df.High + df.Low + df.Close) / 3.0
    pv = tp * df.Volume
    return pv.groupby(day).cumsum() / df.Volume.groupby(day).cumsum().replace(0, np.nan)


def add_features(df):
    x = df.copy()
    x["EMA20"] = ema(x.Close, 20)
    x["EMA50"] = ema(x.Close, 50)
    x["RSI"] = rsi(x.Close)
    x["ATR"] = atr(x.High, x.Low, x.Close)
    x["ADX"] = adx(x.High, x.Low, x.Close)
    x["VWAP"] = session_vwap(x)
    x["BodyRatio"] = (x.Close-x.Open).abs() / (x.High-x.Low).replace(0, np.nan)
    x["RelVol"] = x.Volume / x.Volume.rolling(20).median().replace(0, np.nan)
    x["ATRExpansion"] = x.ATR / x.ATR.rolling(20).median().replace(0, np.nan)
    x["Day"] = pd.Series(x.index.date, index=x.index)
    x["Date"] = pd.to_datetime(x["Day"])

    # Daily aggregates and lagged context: ONLY previous completed days.
    daily = x.groupby("Day").agg(
        DOpen=("Open", "first"),
        DHigh=("High", "max"),
        DLow=("Low", "min"),
        DClose=("Close", "last"),
        DVolume=("Volume", "sum"),
    )
    daily["DRange"] = daily.DHigh - daily.DLow
    daily["DATR5"] = daily.DRange.rolling(5).mean()
    daily["DTrend3"] = daily.DClose.pct_change(3)
    daily["DTrend5"] = daily.DClose.pct_change(5)
    for n in [1, 2, 3, 5, 7]:
        daily[f"H{n}"] = daily.DHigh.shift(n)
        daily[f"L{n}"] = daily.DLow.shift(n)
        daily[f"C{n}"] = daily.DClose.shift(n)
        daily[f"R{n}"] = daily.DRange.shift(n)
    daily["H1_7"] = daily[[f"H{n}" for n in [1,2,3,5,7]]].max(axis=1)
    daily["L1_7"] = daily[[f"L{n}" for n in [1,2,3,5,7]]].min(axis=1)
    daily["C1"] = daily.C1

    # Map previous completed-day context onto every intraday bar.
    ctx = daily.shift(0)
    x = x.join(ctx, on="Day", how="left", rsuffix="_D")
    return x


def opening_range_for_day(day_df):
    first = day_df.between_time("09:15", "09:30", inclusive="left")
    if first.empty:
        return np.nan, np.nan
    return float(first.High.max()), float(first.Low.min())



def candle_quality(row, side):
    rng = float(row.High - row.Low)
    if rng <= 0:
        return False

    body_ratio = abs(float(row.Close - row.Open)) / rng
    close_location = (
        float(row.Close - row.Low) / rng
        if side == "CALL"
        else float(row.High - row.Close) / rng
    )

    if side == "CALL":
        return (
            row.Close > row.Open
            and body_ratio >= BODY_RATIO_MIN
            and close_location >= CLOSE_LOCATION_MIN
        )

    return (
        row.Close < row.Open
        and body_ratio >= BODY_RATIO_MIN
        and close_location >= CLOSE_LOCATION_MIN
    )


def pivot_points(df, window=SWING_WINDOW):
    """Confirmed candle swing highs/lows using only candles available so far."""
    highs, lows = [], []
    if len(df) < 2 * window + 1:
        return highs, lows

    h = df.High.to_numpy()
    l = df.Low.to_numpy()

    for i in range(window, len(df) - window):
        if h[i] == max(h[i-window:i+window+1]):
            highs.append((i, float(h[i])))
        if l[i] == min(l[i-window:i+window+1]):
            lows.append((i, float(l[i])))

    return highs, lows


def trendline_value(points, x):
    """Value of the line through the latest two confirmed pivots."""
    if len(points) < 2:
        return np.nan

    (x1, y1), (x2, y2) = points[-2], points[-1]
    if x2 == x1:
        return np.nan

    return y1 + (y2 - y1) * ((x - x2) / (x2 - x1))


def structure_levels(day_df, i, row):
    """
    Causal support/resistance from previous completed 1-7 sessions, opening
    range and confirmed intraday swings.
    """
    prior = day_df.iloc[:i]
    if prior.empty:
        return np.nan, np.nan, np.nan, np.nan

    resistance_candidates = []
    support_candidates = []

    for c in ("H1_7", "H1", "H2", "H3", "H5", "H7"):
        if c in row.index and pd.notna(row[c]):
            resistance_candidates.append(float(row[c]))

    for c in ("L1_7", "L1", "L2", "L3", "L5", "L7"):
        if c in row.index and pd.notna(row[c]):
            support_candidates.append(float(row[c]))

    orh, orl = opening_range_for_day(day_df)
    if np.isfinite(orh):
        resistance_candidates.append(orh)
    if np.isfinite(orl):
        support_candidates.append(orl)

    highs, lows = pivot_points(prior)

    if highs:
        resistance_candidates.append(highs[-1][1])
    if lows:
        support_candidates.append(lows[-1][1])

    resistance = max(resistance_candidates) if resistance_candidates else np.nan
    support = min(support_candidates) if support_candidates else np.nan

    bullish_tl = trendline_value(lows, len(prior) - 1)
    bearish_tl = trendline_value(highs, len(prior) - 1)

    return resistance, support, bullish_tl, bearish_tl


def breakout_retest_confirmed(day_df, i, side, level):
    """
    Current candle i is the confirmation candle.

    Before it, the engine requires:
      1. candle CLOSE beyond the level,
      2. later retest of the level,
      3. retest holds,
      4. current candle confirms continuation.
    """
    if not np.isfinite(level) or i < 3:
        return False, None

    atr_now = float(day_df.iloc[i].ATR)
    if not np.isfinite(atr_now) or atr_now <= 0:
        return False, None

    tolerance = atr_now * RETEST_TOLERANCE_ATR
    start = max(1, i - RETEST_LOOKBACK)

    for b in range(i - 1, start - 1, -1):
        breakout = day_df.iloc[b]

        if side == "CALL":
            breakout_ok = (
                float(breakout.Close)
                > level + BREAK_BUFFER_ATR * atr_now
            )
        else:
            breakout_ok = (
                float(breakout.Close)
                < level - BREAK_BUFFER_ATR * atr_now
            )

        if not breakout_ok:
            continue

        for r in range(b + 1, i):
            retest = day_df.iloc[r]

            if side == "CALL":
                touched = float(retest.Low) <= level + tolerance
                held = float(retest.Close) > level
            else:
                touched = float(retest.High) >= level - tolerance
                held = float(retest.Close) < level

            if touched and held and candle_quality(day_df.iloc[i], side):
                current = day_df.iloc[i]

                if side == "CALL":
                    continuation = float(current.Close) > float(retest.High)
                else:
                    continuation = float(current.Close) < float(retest.Low)

                if continuation:
                    return True, {
                        "breakout_bar": day_df.index[b].isoformat(),
                        "retest_bar": day_df.index[r].isoformat(),
                        "level": round(float(level), 2),
                    }

    return False, None


def score_row(row, side, or_high, or_low, structure=None):
    """
    Strict confluence score.

    Price-action structure is mandatory; indicators cannot override a missing
    breakout/retest sequence.
    """
    if structure is None:
        structure = {}

    close = float(row.Close)
    resistance = structure.get("resistance", np.nan)
    support = structure.get("support", np.nan)
    bullish_tl = structure.get("bullish_tl", np.nan)
    bearish_tl = structure.get("bearish_tl", np.nan)
    break_retest = bool(structure.get("break_retest", False))

    if side == "CALL":
        checks = {
            "Breakout+Retest": break_retest,
            "Trendline": np.isfinite(bullish_tl) and close > bullish_tl,
            "VWAP": pd.notna(row.VWAP) and close > row.VWAP,
            "EMA": row.EMA20 > row.EMA50 and close > row.EMA20,
            "ADX": row.ADX >= ADX_MIN,
            "RSI": RSI_LONG_MIN <= row.RSI <= RSI_LONG_MAX,
            "Volume": row.RelVol >= REL_VOL_MIN,
            "ATR": row.ATRExpansion >= ATR_EXPANSION_MIN,
            "Candle": candle_quality(row, "CALL"),
            "DailyContext": (
                (pd.notna(row.DTrend3) and row.DTrend3 > 0)
                or (pd.notna(row.DTrend5) and row.DTrend5 > 0)
            ),
            "Structure": (
                np.isfinite(resistance) and close > resistance
            ),
        }
    else:
        checks = {
            "Breakout+Retest": break_retest,
            "Trendline": np.isfinite(bearish_tl) and close < bearish_tl,
            "VWAP": pd.notna(row.VWAP) and close < row.VWAP,
            "EMA": row.EMA20 < row.EMA50 and close < row.EMA20,
            "ADX": row.ADX >= ADX_MIN,
            "RSI": RSI_SHORT_MIN <= row.RSI <= RSI_SHORT_MAX,
            "Volume": row.RelVol >= REL_VOL_MIN,
            "ATR": row.ATRExpansion >= ATR_EXPANSION_MIN,
            "Candle": candle_quality(row, "PUT"),
            "DailyContext": (
                (pd.notna(row.DTrend3) and row.DTrend3 < 0)
                or (pd.notna(row.DTrend5) and row.DTrend5 < 0)
            ),
            "Structure": (
                np.isfinite(support) and close < support
            ),
        }

    # Convert NumPy/Pandas boolean scalars to native Python bools so Flask
    # jsonify() can serialize the response correctly.
    checks = {k: bool(v) for k, v in checks.items()}
    return sum(checks.values()), checks

def choose_option(spot, side, itm_steps=OPTION_ITM_STEPS):
    atm = int(round(float(spot) / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)
    if side == "CALL":
        strike = atm - itm_steps * NIFTY_STRIKE_STEP
        option = "CE"
    else:
        strike = atm + itm_steps * NIFTY_STRIKE_STEP
        option = "PE"
    return option, int(strike), int(atm)


def simulate_index_trade(day_df, entry_i, side, entry, atr_value):
    risk = max(float(atr_value) * STOP_ATR, 1.0)
    stop = entry - risk if side == "CALL" else entry + risk
    target = entry + risk * TARGET_R if side == "CALL" else entry - risk * TARGET_R
    end_i = min(entry_i + MAX_BARS_IN_TRADE, len(day_df)-1)
    exit_price = None
    exit_reason = None
    exit_time = None
    for j in range(entry_i, end_i+1):
        bar = day_df.iloc[j]
        if side == "CALL":
            # Conservative ordering when both are touched: stop first.
            if bar.Low <= stop:
                exit_price, exit_reason = stop, "SL"
                exit_time = day_df.index[j]
                break
            if bar.High >= target:
                exit_price, exit_reason = target, "TARGET"
                exit_time = day_df.index[j]
                break
        else:
            if bar.High >= stop:
                exit_price, exit_reason = stop, "SL"
                exit_time = day_df.index[j]
                break
            if bar.Low <= target:
                exit_price, exit_reason = target, "TARGET"
                exit_time = day_df.index[j]
                break
    if exit_price is None:
        bar = day_df.iloc[end_i]
        exit_price = float(bar.Close)
        exit_reason = "TIME/EOD"
        exit_time = day_df.index[end_i]
    points = exit_price-entry if side == "CALL" else entry-exit_price
    return {
        "entry_price": round(float(entry), 2),
        "stop": round(float(stop), 2),
        "target": round(float(target), 2),
        "exit_price": round(float(exit_price), 2),
        "exit_time": exit_time.isoformat(),
        "exit_reason": exit_reason,
        "points": round(float(points), 2),
        "R": round(float(points/risk), 3),
    }



def backtest(df, itm_steps=OPTION_ITM_STEPS):
    x = add_features(df)
    trades = []

    for day, day_df in x.groupby("Day"):
        if len(day_df) < 20:
            continue

        or_high, or_low = opening_range_for_day(day_df)
        if pd.isna(or_high):
            continue

        traded = False

        # Candle i = confirmation; entry = next candle OPEN.
        for i in range(5, len(day_df) - 1):
            if traded:
                break

            ts = day_df.index[i]
            hhmm = ts.strftime("%H:%M")

            if hhmm < "09:45" or hhmm > LAST_ENTRY:
                continue

            row = day_df.iloc[i]
            needed = [
                "ATR", "ADX", "RSI", "EMA20", "EMA50",
                "ATRExpansion", "RelVol"
            ]

            if any(not np.isfinite(row[c]) for c in needed):
                continue

            resistance, support, bullish_tl, bearish_tl = structure_levels(
                day_df, i, row
            )

            for side in ["CALL", "PUT"]:
                level = resistance if side == "CALL" else support

                confirmed, br_info = breakout_retest_confirmed(
                    day_df, i, side, level
                )

                structure = {
                    "resistance": resistance,
                    "support": support,
                    "bullish_tl": bullish_tl,
                    "bearish_tl": bearish_tl,
                    "break_retest": confirmed,
                }

                score, checks = score_row(
                    row, side, or_high, or_low, structure
                )

                # Mandatory structural gate.
                if not confirmed or score < MIN_SETUP_SCORE:
                    continue

                entry_i = i + 1
                entry = float(day_df.iloc[entry_i].Open)

                # Reject large gaps/chases after confirmation.
                gap_atr = abs(entry - float(row.Close)) / float(row.ATR)
                if gap_atr > 0.75:
                    continue

                sim = simulate_index_trade(
                    day_df, entry_i, side, entry, row.ATR
                )
                option, strike, atm = choose_option(
                    entry, side, itm_steps
                )

                tl = bullish_tl if side == "CALL" else bearish_tl

                trades.append({
                    "date": str(day),
                    "signal_time": ts.isoformat(),
                    "side": side,
                    "option": option,
                    "strike": strike,
                    "atm": atm,
                    "score": score,
                    "entry_time": day_df.index[entry_i].isoformat(),
                    "spot_entry": round(entry, 2),
                    "level": round(float(level), 2)
                        if np.isfinite(level) else None,
                    "trendline": round(float(tl), 2)
                        if np.isfinite(tl) else None,
                    "or_high": round(or_high, 2),
                    "or_low": round(or_low, 2),
                    "prev_1d_high": round(float(row.H1), 2)
                        if pd.notna(row.H1) else None,
                    "prev_1d_low": round(float(row.L1), 2)
                        if pd.notna(row.L1) else None,
                    "prev_7d_high": round(float(row.H1_7), 2)
                        if pd.notna(row.H1_7) else None,
                    "prev_7d_low": round(float(row.L1_7), 2)
                        if pd.notna(row.L1_7) else None,
                    "breakout_time": (
                        br_info["breakout_bar"] if br_info else None
                    ),
                    "retest_time": (
                        br_info["retest_bar"] if br_info else None
                    ),
                    "signal_checks": ",".join(
                        k for k, v in checks.items() if v
                    ),
                    **sim,
                    "note": (
                        "Spot price-action backtest. "
                        "Option premium requires historical option OHLC."
                    ),
                })

                traded = True
                break

    t = pd.DataFrame(trades)

    if t.empty:
        return t, {
            "trades": 0,
            "note": "No high-conviction breakout/retest setups found.",
        }

    t["cum_points"] = t.points.cumsum()

    wins = int((t.points > 0).sum())
    losses = int((t.points <= 0).sum())
    gross_win = t.loc[t.points > 0, "points"].sum()
    gross_loss = -t.loc[t.points < 0, "points"].sum()
    pf = gross_win / gross_loss if gross_loss else float("inf")
    dd = (t.cum_points - t.cum_points.cummax()).min()

    metrics = {
        "trades": int(len(t)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / len(t), 2),
        "net_points": round(float(t.points.sum()), 2),
        "avg_points": round(float(t.points.mean()), 2),
        "profit_factor": round(float(pf), 2)
            if math.isfinite(pf) else None,
        "max_drawdown_points": round(float(dd), 2),
        "target_R": TARGET_R,
        "min_setup_score": MIN_SETUP_SCORE,
        "strategy": (
            "Trendline + breakout + retest + confirmation candle"
        ),
        "note": (
            "NIFTY spot signal backtest mapped to CE/PE; "
            "not historical option-premium P&L."
        ),
    }

    return t, metrics

def load_option_csv(path):
    o = pd.read_csv(path)
    required = ["Date","Expiry","Strike","OptionType","Open","High","Low","Close"]
    missing = [c for c in required if c not in o.columns]
    if missing:
        raise ValueError(f"Options CSV missing columns: {missing}")
    o["Date"] = pd.to_datetime(o["Date"]).dt.date
    o["Expiry"] = pd.to_datetime(o["Expiry"]).dt.date
    o["OptionType"] = o["OptionType"].str.upper().str.replace("CALL","CE").str.replace("PUT","PE")
    return o


def option_backtest(spot_df, options_csv, itm_steps=0):
    # Generate spot signals first. Then look up the matching contract by date,
    # strike and option type. The CSV is expected to contain intraday OHLC.
    signals, _ = backtest(spot_df, itm_steps)
    if signals.empty:
        return pd.DataFrame(), {"trades":0, "note":"No spot signals."}
    opt = load_option_csv(options_csv)
    opt["DateTime"] = pd.to_datetime(opt["Date"].astype(str) + " " + opt.get("Time", "09:15:00").astype(str), errors="coerce")
    # If Time is absent, DateTime becomes the session date; exact intraday matching then uses first bar.
    results=[]
    for _, s in signals.iterrows():
        d = s["date"]
        subset = opt[(opt.Date == pd.to_datetime(d).date()) & (opt.Strike == s.strike) & (opt.OptionType == s.option)]
        if subset.empty:
            continue
        # Approximate entry/exit using first matching bar after signal entry time.
        entry_dt = pd.to_datetime(s.entry_time).tz_localize(None)
        subset = subset.copy()
        if "Time" in opt.columns:
            subset["dt"] = pd.to_datetime(subset["Date"].astype(str)+" "+subset["Time"].astype(str), errors="coerce")
            subset = subset[subset.dt >= entry_dt].sort_values("dt")
        else:
            subset = subset.sort_values("Expiry")
        if subset.empty: continue
        e = subset.iloc[0]
        entry=float(e.Open)
        # Use same risk logic in option premium space.
        risk=max(entry*0.25, 1.0)
        sl=entry-risk
        tgt=entry+risk*TARGET_R
        exit_row=None; reason="EOD"
        for _, b in subset.iloc[:MAX_BARS_IN_TRADE].iterrows():
            if b.Low <= sl:
                exit_row=b; reason="SL"; break
            if b.High >= tgt:
                exit_row=b; reason="TARGET"; break
        if exit_row is None: exit_row=subset.iloc[min(MAX_BARS_IN_TRADE-1,len(subset)-1)]
        pnl=float(exit_row.Close-entry)
        results.append({**s.to_dict(),"option_entry":entry,"option_stop":sl,"option_target":tgt,"option_exit":float(exit_row.Close),"option_pnl":round(pnl,2),"option_exit_reason":reason})
    out=pd.DataFrame(results)
    if out.empty:return out,{"trades":0,"note":"No matching historical option contracts found."}
    out["cum_option_pnl"]=out.option_pnl.cumsum()
    metrics={"trades":len(out),"wins":int((out.option_pnl>0).sum()),"win_rate_pct":round(100*(out.option_pnl>0).mean(),2),"net_option_points":round(float(out.option_pnl.sum()),2),"note":"Actual option OHLC backtest from supplied CSV."}
    return out,metrics



def live_scan(df, itm_steps=0):
    x = add_features(df)
    today = x[x.Day == x.Day.iloc[-1]]

    if today.empty:
        return {"signal": "NO DATA"}

    orh, orl = opening_range_for_day(today)
    row = today.iloc[-1]

    if pd.isna(orh):
        return {
            "signal": "WAIT",
            "reason": "Opening range not complete",
        }

    i = len(today) - 1

    if i < 5:
        return {
            "signal": "WAIT",
            "reason": (
                "Not enough candles for confirmed swing/trendline analysis"
            ),
        }

    resistance, support, bullish_tl, bearish_tl = structure_levels(
        today, i, row
    )

    results = {}

    for side in ["CALL", "PUT"]:
        level = resistance if side == "CALL" else support

        confirmed, br_info = breakout_retest_confirmed(
            today, i, side, level
        )

        structure = {
            "resistance": resistance,
            "support": support,
            "bullish_tl": bullish_tl,
            "bearish_tl": bearish_tl,
            "break_retest": confirmed,
        }

        score, checks = score_row(
            row, side, orh, orl, structure
        )

        results[side] = {
            "score": score,
            "checks": checks,
            "level": level,
            "trendline": (
                bullish_tl if side == "CALL" else bearish_tl
            ),
            "break_retest": br_info,
        }

    call = results["CALL"]
    put = results["PUT"]

    if (
        call["break_retest"]
        and call["score"] >= MIN_SETUP_SCORE
        and call["score"] > put["score"]
    ):
        side = "CALL"
        selected = call

    elif (
        put["break_retest"]
        and put["score"] >= MIN_SETUP_SCORE
        and put["score"] > call["score"]
    ):
        side = "PUT"
        selected = put

    else:
        side = "NO TRADE"
        selected = call if call["score"] >= put["score"] else put

    option = strike = atm = None

    if side in ("CALL", "PUT"):
        option, strike, atm = choose_option(
            row.Close, side, itm_steps
        )

    return {
        "timestamp": row.name.isoformat(),
        "spot": round(float(row.Close), 2),
        "vwap": (
            round(float(row.VWAP), 2)
            if pd.notna(row.VWAP) else None
        ),
        "ema20": round(float(row.EMA20), 2),
        "ema50": round(float(row.EMA50), 2),
        "adx": round(float(row.ADX), 2),
        "rsi": round(float(row.RSI), 2),
        "rel_volume": (
            round(float(row.RelVol), 2)
            if pd.notna(row.RelVol) else None
        ),
        "atr": round(float(row.ATR), 2),
        "or_high": round(orh, 2),
        "or_low": round(orl, 2),
        "resistance": (
            round(float(resistance), 2)
            if np.isfinite(resistance) else None
        ),
        "support": (
            round(float(support), 2)
            if np.isfinite(support) else None
        ),
        "bullish_trendline": (
            round(float(bullish_tl), 2)
            if np.isfinite(bullish_tl) else None
        ),
        "bearish_trendline": (
            round(float(bearish_tl), 2)
            if np.isfinite(bearish_tl) else None
        ),
        "signal": side,
        "score": selected["score"],
        "max_score": 11,
        "min_setup_score": MIN_SETUP_SCORE,
        "option": option,
        "strike": strike,
        "atm": atm,
        "checks": selected["checks"],
        "call": call,
        "put": put,
        "reason": (
            "Confirmed breakout → retest → continuation candle "
            "with trendline/confluence."
            if side in ("CALL", "PUT")
            else (
                "No qualifying breakout/retest/trendline setup. "
                "Wait for confirmation."
            )
        ),
    }

def save_results(trades, metrics, prefix="nifty_1day_options"):
    stamp=datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    if not trades.empty: trades.to_excel(RESULTS_DIR/f"{prefix}_{stamp}.xlsx",index=False)
    (RESULTS_DIR/f"{prefix}_{stamp}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")


@app.get("/")
def home():
    from flask import send_from_directory
    return send_from_directory("static", "index.html")

@app.get("/api/scan")
def api_scan():
    try:
        period=request.args.get("period","10d"); interval=request.args.get("interval","5m"); itm=int(request.args.get("itm","0"))
        return jsonify(json_safe(live_scan(download_data(period,interval),itm)))
    except Exception as e:return jsonify({"status":"error","error":str(e)}),500

@app.get("/api/backtest")
def api_backtest():
    try:
        period=request.args.get("period",DEFAULT_PERIOD); interval=request.args.get("interval",DEFAULT_INTERVAL); itm=int(request.args.get("itm","0"))
        trades,metrics=backtest(download_data(period,interval),itm); save_results(trades,metrics)
        return jsonify(json_safe({"status":"ok","metrics":metrics,"trades":trades.to_dict("records") if not trades.empty else []}))
    except Exception as e:return jsonify({"status":"error","error":str(e)}),500


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["backtest","scan","option-backtest","web"],default="backtest")
    p.add_argument("--period",default=DEFAULT_PERIOD); p.add_argument("--interval",default=DEFAULT_INTERVAL)
    p.add_argument("--itm",type=int,default=0); p.add_argument("--spot-csv"); p.add_argument("--options-csv"); p.add_argument("--port",type=int,default=5000)
    a=p.parse_args()
    if a.mode=="web": app.run(host="0.0.0.0",port=a.port,debug=False); return
    spot=download_data(a.period,a.interval) if not a.spot_csv else pd.read_csv(a.spot_csv)
    if a.spot_csv:
        spot["Datetime"]=pd.to_datetime(spot["Datetime"] if "Datetime" in spot.columns else spot["Date"])
        spot=spot.set_index("Datetime")
    if a.mode=="scan": print(json.dumps(live_scan(spot,a.itm),indent=2)); return
    if a.mode=="option-backtest":
        if not a.options_csv: raise SystemExit("--options-csv is required")
        trades,metrics=option_backtest(spot,a.options_csv,a.itm)
    else: trades,metrics=backtest(spot,a.itm)
    save_results(trades,metrics)
    print(json.dumps(metrics,indent=2))
    if not trades.empty: print(trades.to_string(index=False))

if __name__=="__main__": main()