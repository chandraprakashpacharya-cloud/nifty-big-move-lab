#!/usr/bin/env python3
"""
NIFTY 50 Big-Move Scanner + Backtester
=======================================

Built from the same core stack as the supplied scanner:
    numpy, pandas, yfinance, Flask, flask-cors, openpyxl

What it does
------------
1. Downloads NIFTY 50 (^NSEI) intraday candles.
2. Builds a "big move" signal using:
   - Opening Range Breakout (ORB)
   - VWAP
   - EMA 20 / 50
   - ADX
   - RSI
   - ATR / ATR expansion
   - Relative volume
   - Previous-day high/low
   - Volatility compression -> expansion
   - Candle/body strength
   - Higher-timeframe trend filter
3. Enters on the NEXT candle open after a completed signal candle.
   This avoids look-ahead bias.
4. Uses ATR-based stop and a 2R / 3R target.
5. Has a time-based exit and end-of-day square-off.
6. Produces:
      results/nifty_backtest_*.json
      results/nifty_backtest_*.xlsx
      results/equity_curve_*.csv
7. Can also run a live scanner for today's completed candles.
8. Includes a small Flask API/UI backend.

IMPORTANT
---------
This backtests NIFTY index movement, NOT historical option premium.
For NIFTY options, use the generated signal to select an option.
Option-premium backtesting needs historical option-chain/contract data,
which yfinance does not provide reliably.

INSTALL
-------
pip install flask flask-cors yfinance pandas openpyxl numpy apscheduler

BACKTEST
--------
python nifty_big_move.py --mode backtest --period 60d --interval 15m

LIVE SCAN
---------
python nifty_big_move.py --mode scan --period 60d --interval 15m

WEB
---
python nifty_big_move.py --mode web
Then open http://localhost:5000

NOTES
-----
Yahoo/yfinance intraday history is limited. If Yahoo returns a shorter
history than requested, the program backtests whatever valid candles it
receives and reports the actual date range.
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOL = "^NSEI"
IST = ZoneInfo("Asia/Kolkata")

DEFAULT_PERIOD = "60d"
DEFAULT_INTERVAL = "15m"

# Trading session
SESSION_START = "09:15"
OR_START = "09:15"
OR_END = "09:45"
LAST_ENTRY = "14:45"
FORCE_EXIT = "15:15"

# Signal thresholds
ADX_MIN = 20.0
RSI_LONG_MIN = 55.0
RSI_LONG_MAX = 78.0
RSI_SHORT_MIN = 22.0
RSI_SHORT_MAX = 45.0

REL_VOL_MIN = 1.15
ATR_EXPANSION_MIN = 1.05
BODY_RATIO_MIN = 0.55
BREAK_BUFFER_ATR = 0.05

# Compression / expansion
COMPRESSION_LOOKBACK = 20
COMPRESSION_FACTOR = 0.85

# Risk model
STOP_ATR = 1.20
TARGET_R = 2.50
MAX_BARS_IN_TRADE = 10

# One trade at a time
ONE_TRADE_PER_SESSION = False

# Costs in index points for a realistic index backtest.
# Change these to your assumptions.
SLIPPAGE_POINTS = 1.0
ROUND_TRIP_COST_POINTS = 1.0

# ---------------------------------------------------------------------
# FLASK
# ---------------------------------------------------------------------

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, window=14):
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / window, adjust=False).mean()


def adx(high, low, close, window=14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=high.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=high.index,
    )

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_s = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)

    return dx.ewm(alpha=1 / window, adjust=False).mean()


def session_vwap(df):
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical * df["Volume"]

    day = df.index.date
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["Volume"].groupby(day).cumsum()

    return cum_pv / cum_vol.replace(0, np.nan)


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def download_data(period=DEFAULT_PERIOD, interval=DEFAULT_INTERVAL):
    """
    Download NIFTY index candles.

    auto_adjust=True is harmless for the index and keeps behavior aligned
    with the supplied scanner.
    """
    data = yf.download(
        SYMBOL,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise RuntimeError("No NIFTY data returned by yfinance.")

    # yfinance can return MultiIndex columns even for one symbol.
    if isinstance(data.columns, pd.MultiIndex):
        if SYMBOL in data.columns.get_level_values(-1):
            data = data.xs(SYMBOL, axis=1, level=-1)
        elif SYMBOL in data.columns.get_level_values(0):
            data = data.xs(SYMBOL, axis=1, level=0)
        else:
            data.columns = data.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    for col in needed:
        if col not in data.columns:
            raise RuntimeError(f"Missing column from yfinance: {col}")

    data = data[needed].copy()
    data = data.dropna()

    # Make index timezone-aware in IST.
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC").tz_convert(IST)
    else:
        data.index = data.index.tz_convert(IST)

    # Keep NSE regular session.
    data = data.between_time("09:15", "15:30")

    if len(data) < 100:
        raise RuntimeError(
            f"Only {len(data)} valid candles received. "
            "Yahoo intraday history may be unavailable/too short."
        )

    return data


# ---------------------------------------------------------------------
# FEATURE ENGINE
# ---------------------------------------------------------------------

def add_features(df):
    x = df.copy()

    x["EMA20"] = ema(x["Close"], 20)
    x["EMA50"] = ema(x["Close"], 50)

    x["RSI"] = rsi(x["Close"], 14)
    x["ATR"] = atr(x["High"], x["Low"], x["Close"], 14)
    x["ATR_pct"] = x["ATR"] / x["Close"]

    x["ADX"] = adx(x["High"], x["Low"], x["Close"], 14)

    x["VWAP"] = session_vwap(x)

    x["VolAvg20"] = x["Volume"].rolling(20).mean()
    x["RelVol"] = x["Volume"] / x["VolAvg20"]

    x["ATRAvg20"] = x["ATR"].rolling(20).mean()
    x["ATRExpansion"] = x["ATR"] / x["ATRAvg20"]

    # Candle quality
    x["Range"] = x["High"] - x["Low"]
    x["Body"] = (x["Close"] - x["Open"]).abs()
    x["BodyRatio"] = x["Body"] / x["Range"].replace(0, np.nan)

    x["BullBody"] = (
        (x["Close"] > x["Open"])
        & (x["BodyRatio"] >= BODY_RATIO_MIN)
    )

    x["BearBody"] = (
        (x["Close"] < x["Open"])
        & (x["BodyRatio"] >= BODY_RATIO_MIN)
    )

    # Compression: recent ATR was below its longer average.
    x["ATRCompression"] = (
        x["ATR"].rolling(COMPRESSION_LOOKBACK).mean()
        < x["ATR"].rolling(COMPRESSION_LOOKBACK * 2).mean()
        * COMPRESSION_FACTOR
    )

    # Previous session levels.
    dates = pd.Series(x.index.date, index=x.index)
    daily_high = x["High"].groupby(dates).max()
    daily_low = x["Low"].groupby(dates).min()

    prev_high_map = daily_high.shift(1)
    prev_low_map = daily_low.shift(1)

    x["PrevDayHigh"] = dates.map(prev_high_map)
    x["PrevDayLow"] = dates.map(prev_low_map)

    # Opening range.
    x["ORHigh"] = np.nan
    x["ORLow"] = np.nan

    for d in pd.unique(dates):
        mask = dates == d
        day_index = x.index[mask]

        if len(day_index) == 0:
            continue

        or_mask = (
            (day_index.strftime("%H:%M") >= OR_START)
            & (day_index.strftime("%H:%M") < OR_END)
        )

        or_idx = day_index[or_mask]

        if len(or_idx):
            or_high = x.loc[or_idx, "High"].max()
            or_low = x.loc[or_idx, "Low"].min()

            x.loc[day_index, "ORHigh"] = or_high
            x.loc[day_index, "ORLow"] = or_low

    # Trend alignment
    x["BullTrend"] = (
        (x["Close"] > x["VWAP"])
        & (x["Close"] > x["EMA20"])
        & (x["EMA20"] > x["EMA50"])
    )

    x["BearTrend"] = (
        (x["Close"] < x["VWAP"])
        & (x["Close"] < x["EMA20"])
        & (x["EMA20"] < x["EMA50"])
    )

    return x


# ---------------------------------------------------------------------
# SIGNAL LOGIC
# ---------------------------------------------------------------------

def long_signal(row):
    """
    Designed to find an expansion day rather than merely an uptrend.

    Conditions:
      1. OR breakout
      2. Price above VWAP/EMA stack
      3. ADX trend strength
      4. RSI momentum
      5. Relative volume
      6. ATR expansion
      7. Strong candle body
      8. Previous-day high confirmation OR fresh OR breakout
      9. Prior compression OR unusually strong expansion
    """
    required = [
        row["ORHigh"],
        row["VWAP"],
        row["EMA20"],
        row["EMA50"],
        row["ADX"],
        row["RSI"],
        row["RelVol"],
        row["ATRExpansion"],
        row["BodyRatio"],
    ]

    if any(pd.isna(v) for v in required):
        return False

    breakout = row["Close"] > row["ORHigh"] + BREAK_BUFFER_ATR * row["ATR"]

    trend = row["BullTrend"]
    strength = row["ADX"] >= ADX_MIN
    momentum = RSI_LONG_MIN <= row["RSI"] <= RSI_LONG_MAX
    volume = row["RelVol"] >= REL_VOL_MIN
    expansion = row["ATRExpansion"] >= ATR_EXPANSION_MIN
    candle = row["BullBody"]

    # Either prior-day high is broken OR ORB itself is a clean breakout.
    level_break = (
        pd.notna(row["PrevDayHigh"])
        and row["Close"] > row["PrevDayHigh"]
    ) or breakout

    # Compression is a bonus, not a mandatory condition.
    expansion_context = (
        bool(row["ATRCompression"])
        or row["ATRExpansion"] >= 1.20
    )

    score = sum(
        [
            bool(breakout),
            bool(trend),
            bool(strength),
            bool(momentum),
            bool(volume),
            bool(expansion),
            bool(candle),
            bool(level_break),
            bool(expansion_context),
        ]
    )

    return score >= 8


def short_signal(row):
    required = [
        row["ORLow"],
        row["VWAP"],
        row["EMA20"],
        row["EMA50"],
        row["ADX"],
        row["RSI"],
        row["RelVol"],
        row["ATRExpansion"],
        row["BodyRatio"],
    ]

    if any(pd.isna(v) for v in required):
        return False

    breakout = row["Close"] < row["ORLow"] - BREAK_BUFFER_ATR * row["ATR"]

    trend = row["BearTrend"]
    strength = row["ADX"] >= ADX_MIN
    momentum = RSI_SHORT_MIN <= row["RSI"] <= RSI_SHORT_MAX
    volume = row["RelVol"] >= REL_VOL_MIN
    expansion = row["ATRExpansion"] >= ATR_EXPANSION_MIN
    candle = row["BearBody"]

    level_break = (
        pd.notna(row["PrevDayLow"])
        and row["Close"] < row["PrevDayLow"]
    ) or breakout

    expansion_context = (
        bool(row["ATRCompression"])
        or row["ATRExpansion"] >= 1.20
    )

    score = sum(
        [
            bool(breakout),
            bool(trend),
            bool(strength),
            bool(momentum),
            bool(volume),
            bool(expansion),
            bool(candle),
            bool(level_break),
            bool(expansion_context),
        ]
    )

    return score >= 8


# ---------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------

def market_time(ts):
    return ts.strftime("%H:%M")


def backtest(df):
    x = add_features(df)

    trades = []
    in_trade = None
    used_today = set()

    dates = pd.Series(x.index.date, index=x.index)

    for i in range(1, len(x)):
        ts = x.index[i]
        row = x.iloc[i]

        # Manage an existing position using the current candle.
        if in_trade is not None:
            direction = in_trade["direction"]
            stop = in_trade["stop"]
            target = in_trade["target"]

            candle_high = float(row["High"])
            candle_low = float(row["Low"])

            exit_price = None
            exit_reason = None

            if direction == "LONG":
                hit_stop = candle_low <= stop
                hit_target = candle_high >= target

                # Conservative rule when both occur in same candle:
                # assume stop happened first.
                if hit_stop:
                    exit_price = stop
                    exit_reason = "STOP"
                elif hit_target:
                    exit_price = target
                    exit_reason = "TARGET"

            else:
                hit_stop = candle_high >= stop
                hit_target = candle_low <= target

                if hit_stop:
                    exit_price = stop
                    exit_reason = "STOP"
                elif hit_target:
                    exit_price = target
                    exit_reason = "TARGET"

            bars_held = i - in_trade["entry_i"]

            # Time exit
            if exit_price is None and bars_held >= MAX_BARS_IN_TRADE:
                exit_price = float(row["Close"])
                exit_reason = "TIME"

            # EOD exit
            if (
                exit_price is None
                and market_time(ts) >= FORCE_EXIT
            ):
                exit_price = float(row["Close"])
                exit_reason = "EOD"

            if exit_price is not None:
                if direction == "LONG":
                    gross_points = exit_price - in_trade["entry_price"]
                else:
                    gross_points = in_trade["entry_price"] - exit_price

                net_points = (
                    gross_points
                    - SLIPPAGE_POINTS
                    - ROUND_TRIP_COST_POINTS
                )

                risk_points = in_trade["risk_points"]

                r_multiple = (
                    net_points / risk_points
                    if risk_points > 0
                    else np.nan
                )

                trades.append(
                    {
                        "entry_time": in_trade["entry_time"],
                        "exit_time": ts,
                        "direction": direction,
                        "entry_price": round(in_trade["entry_price"], 2),
                        "exit_price": round(exit_price, 2),
                        "stop": round(stop, 2),
                        "target": round(target, 2),
                        "risk_points": round(risk_points, 2),
                        "gross_points": round(gross_points, 2),
                        "net_points": round(net_points, 2),
                        "R": round(r_multiple, 3),
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                        "signal_score": in_trade["signal_score"],
                    }
                )

                in_trade = None

            continue

        # No open position: look for a fresh signal.
        # Only trade after the OR is complete.
        if market_time(ts) < OR_END:
            continue

        if market_time(ts) > LAST_ENTRY:
            continue

        day = ts.date()

        if ONE_TRADE_PER_SESSION and day in used_today:
            continue

        is_long = long_signal(row)
        is_short = short_signal(row)

        if not is_long and not is_short:
            continue

        # If both are somehow true, skip rather than invent direction.
        if is_long and is_short:
            continue

        direction = "LONG" if is_long else "SHORT"

        # IMPORTANT:
        # Signal is known only after this candle closes.
        # Enter at NEXT candle open.
        if i + 1 >= len(x):
            continue

        next_ts = x.index[i + 1]

        # Do not carry a signal into another trading day.
        if next_ts.date() != day:
            continue

        entry_price = float(x.iloc[i + 1]["Open"])

        signal_atr = float(row["ATR"])

        if not np.isfinite(signal_atr) or signal_atr <= 0:
            continue

        risk_points = STOP_ATR * signal_atr

        if direction == "LONG":
            stop = entry_price - risk_points
            target = entry_price + TARGET_R * risk_points
        else:
            stop = entry_price + risk_points
            target = entry_price - TARGET_R * risk_points

        in_trade = {
            "entry_i": i + 1,
            "entry_time": next_ts,
            "entry_price": entry_price,
            "direction": direction,
            "stop": stop,
            "target": target,
            "risk_points": risk_points,
            "signal_score": 8,
        }

        used_today.add(day)

    # Close an unfinished trade at final available close.
    if in_trade is not None:
        final_ts = x.index[-1]
        final_close = float(x.iloc[-1]["Close"])

        if in_trade["direction"] == "LONG":
            gross = final_close - in_trade["entry_price"]
        else:
            gross = in_trade["entry_price"] - final_close

        net = gross - SLIPPAGE_POINTS - ROUND_TRIP_COST_POINTS
        r = net / in_trade["risk_points"]

        trades.append(
            {
                "entry_time": in_trade["entry_time"],
                "exit_time": final_ts,
                "direction": in_trade["direction"],
                "entry_price": round(in_trade["entry_price"], 2),
                "exit_price": round(final_close, 2),
                "stop": round(in_trade["stop"], 2),
                "target": round(in_trade["target"], 2),
                "risk_points": round(in_trade["risk_points"], 2),
                "gross_points": round(gross, 2),
                "net_points": round(net, 2),
                "R": round(r, 3),
                "exit_reason": "DATA_END",
                "bars_held": len(x) - in_trade["entry_i"],
                "signal_score": in_trade["signal_score"],
            }
        )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, {}

    trades_df["cum_points"] = trades_df["net_points"].cumsum()

    wins = trades_df[trades_df["net_points"] > 0]
    losses = trades_df[trades_df["net_points"] < 0]

    gross_profit = wins["net_points"].sum()
    gross_loss = abs(losses["net_points"].sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = trades_df["cum_points"]
    peak = equity.cummax()
    drawdown = equity - peak

    metrics = {
        "trades": int(len(trades_df)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": round(100 * len(wins) / len(trades_df), 2),
        "net_points": round(float(trades_df["net_points"].sum()), 2),
        "gross_points": round(float(trades_df["gross_points"].sum()), 2),
        "profit_factor": round(float(profit_factor), 3)
        if np.isfinite(profit_factor)
        else "INF",
        "avg_trade_points": round(
            float(trades_df["net_points"].mean()), 2
        ),
        "avg_R": round(float(trades_df["R"].mean()), 3),
        "max_drawdown_points": round(
            float(drawdown.min()), 2
        ),
        "best_trade_points": round(
            float(trades_df["net_points"].max()), 2
        ),
        "worst_trade_points": round(
            float(trades_df["net_points"].min()), 2
        ),
        "avg_bars_held": round(
            float(trades_df["bars_held"].mean()), 2
        ),
        "target_hit_rate_pct": round(
            100
            * (trades_df["exit_reason"] == "TARGET").mean(),
            2,
        ),
        "stop_hit_rate_pct": round(
            100
            * (trades_df["exit_reason"] == "STOP").mean(),
            2,
        ),
    }

    return trades_df, metrics


# ---------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------

def save_backtest(trades, metrics, data):
    now = datetime.now(IST)
    stamp = now.strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"nifty_backtest_{stamp}.json"
    xlsx_path = RESULTS_DIR / f"nifty_backtest_{stamp}.xlsx"
    csv_path = RESULTS_DIR / f"equity_curve_{stamp}.csv"

    meta = {
        "symbol": SYMBOL,
        "generated_at": now.isoformat(),
        "data_start": str(data.index.min()),
        "data_end": str(data.index.max()),
        "bars": int(len(data)),
        "strategy": {
            "OR": f"{OR_START}-{OR_END}",
            "ADX_MIN": ADX_MIN,
            "REL_VOL_MIN": REL_VOL_MIN,
            "ATR_EXPANSION_MIN": ATR_EXPANSION_MIN,
            "STOP_ATR": STOP_ATR,
            "TARGET_R": TARGET_R,
            "MAX_BARS_IN_TRADE": MAX_BARS_IN_TRADE,
            "SLIPPAGE_POINTS": SLIPPAGE_POINTS,
            "ROUND_TRIP_COST_POINTS": ROUND_TRIP_COST_POINTS,
        },
        "metrics": metrics,
        "trades": json.loads(
            trades.to_json(orient="records", date_format="iso")
        )
        if not trades.empty
        else [],
    }

    json_path.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    if not trades.empty:
        trades.to_excel(xlsx_path, index=False)
        trades[["exit_time", "cum_points"]].to_csv(
            csv_path, index=False
        )

    return json_path, xlsx_path if not trades.empty else None, csv_path if not trades.empty else None


def print_report(metrics, data):
    print("\n" + "=" * 65)
    print("NIFTY 50 BIG-MOVE BACKTEST")
    print("=" * 65)

    print(
        f"Data      : {data.index.min()} -> {data.index.max()}"
    )
    print(f"Candles   : {len(data)}")

    if not metrics:
        print("\nNo trades generated.")
        return

    for k, v in metrics.items():
        print(f"{k:25s}: {v}")

    print("=" * 65)


# ---------------------------------------------------------------------
# LIVE SCANNER
# ---------------------------------------------------------------------

def live_scan(df):
    x = add_features(df)

    last = x.iloc[-1]
    ts = x.index[-1]

    long_ok = long_signal(last)
    short_ok = short_signal(last)

    result = {
        "timestamp": ts.isoformat(),
        "close": round(float(last["Close"]), 2),
        "vwap": round(float(last["VWAP"]), 2)
        if pd.notna(last["VWAP"]) else None,
        "ema20": round(float(last["EMA20"]), 2)
        if pd.notna(last["EMA20"]) else None,
        "ema50": round(float(last["EMA50"]), 2)
        if pd.notna(last["EMA50"]) else None,
        "rsi": round(float(last["RSI"]), 2)
        if pd.notna(last["RSI"]) else None,
        "adx": round(float(last["ADX"]), 2)
        if pd.notna(last["ADX"]) else None,
        "rel_volume": round(float(last["RelVol"]), 2)
        if pd.notna(last["RelVol"]) else None,
        "atr_expansion": round(
            float(last["ATRExpansion"]), 2
        )
        if pd.notna(last["ATRExpansion"]) else None,
        "or_high": round(float(last["ORHigh"]), 2)
        if pd.notna(last["ORHigh"]) else None,
        "or_low": round(float(last["ORLow"]), 2)
        if pd.notna(last["ORLow"]) else None,
        "signal": (
            "BUY CALL / LONG"
            if long_ok
            else "BUY PUT / SHORT"
            if short_ok
            else "NO TRADE"
        ),
    }

    return result


# ---------------------------------------------------------------------
# FLASK API
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return """
    <html>
    <head>
      <title>NIFTY Big Move Scanner</title>
      <style>
        body { font-family: Arial; max-width: 900px; margin: 40px auto; }
        pre { background:#f4f4f4; padding:20px; overflow:auto; }
        button { padding:10px 18px; cursor:pointer; }
      </style>
    </head>
    <body>
      <h1>NIFTY 50 Big-Move Scanner</h1>
      <p>Use /api/backtest or /api/scan.</p>
      <button onclick="run()">Run Backtest</button>
      <pre id="out"></pre>
      <script>
        async function run() {
          const r = await fetch('/api/backtest');
          const j = await r.json();
          document.getElementById('out').textContent =
            JSON.stringify(j, null, 2);
        }
      </script>
    </body>
    </html>
    """


@app.route("/api/backtest")
def api_backtest():
    try:
        period = os.getenv("NIFTY_PERIOD", DEFAULT_PERIOD)
        interval = os.getenv("NIFTY_INTERVAL", DEFAULT_INTERVAL)

        data = download_data(period, interval)
        trades, metrics = backtest(data)

        save_backtest(trades, metrics, data)

        return jsonify(
            {
                "status": "ok",
                "symbol": SYMBOL,
                "period": period,
                "interval": interval,
                "metrics": metrics,
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "status": "error",
                "error": str(exc),
            }
        ), 500


@app.route("/api/scan")
def api_scan():
    try:
        period = os.getenv("NIFTY_PERIOD", DEFAULT_PERIOD)
        interval = os.getenv("NIFTY_INTERVAL", DEFAULT_INTERVAL)

        data = download_data(period, interval)
        return jsonify(live_scan(data))

    except Exception as exc:
        return jsonify(
            {
                "status": "error",
                "error": str(exc),
            }
        ), 500


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NIFTY 50 Big-Move Scanner + Backtester"
    )

    parser.add_argument(
        "--mode",
        choices=["backtest", "scan", "web"],
        default="backtest",
    )

    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
    )

    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=["5m", "15m", "30m", "60m"],
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    if args.mode == "web":
        os.environ["NIFTY_PERIOD"] = args.period
        os.environ["NIFTY_INTERVAL"] = args.interval
        print(
            f"Starting NIFTY Big-Move Web App on "
            f"http://localhost:{args.port}"
        )
        app.run(
            host="0.0.0.0",
            port=args.port,
            debug=False,
        )
        return

    data = download_data(args.period, args.interval)

    if args.mode == "scan":
        result = live_scan(data)
        print("\n" + json.dumps(result, indent=2))
        return

    trades, metrics = backtest(data)
    print_report(metrics, data)

    json_path, xlsx_path, csv_path = save_backtest(
        trades, metrics, data
    )

    print("\nFiles:")
    print(f"  JSON : {json_path}")
    if xlsx_path:
        print(f"  XLSX : {xlsx_path}")
        print(f"  CSV  : {csv_path}")


if __name__ == "__main__":
    main()
