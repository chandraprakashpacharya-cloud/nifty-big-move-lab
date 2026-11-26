import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")

IST = ZoneInfo("Asia/Kolkata")
NIFTY = "^NSEI"
STRIKE_STEP = 50


def download_nifty(period="30d", interval="15m"):
    # Yahoo has limited history for intraday intervals. 5m is safest with 5d;
    # 15m/30m/60m can use longer windows.
    if interval == "5m" and period not in ("1d", "5d"):
        period = "5d"

    df = yf.download(
        NIFTY,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no NIFTY data.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise RuntimeError(f"Yahoo Finance response is missing {c}.")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(IST)
    else:
        df.index = df.index.tz_localize(IST)

    return df


def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean()
    al = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()

    plus_dm = pd.Series(
        np.where((up > dn) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index
    )

    prev = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev).abs(), (l - prev).abs()], axis=1
    ).max(axis=1)

    atrv = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atrv.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atrv.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def indicators(df):
    x = df.copy()

    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["ATR_PCT"] = x["ATR"] / x["Close"]
    x["ADX"] = adx(x)

    # IMPORTANT:
    # NSE NIFTY index candles from Yahoo often have Volume=0.
    # A volume-weighted VWAP/RVOL therefore becomes NaN and used to remove
    # nearly every candle. We use a price-only session VWAP proxy and a
    # range-expansion proxy instead. This keeps the signal engine usable
    # while clearly labeling the data limitation.
    typical = (x["High"] + x["Low"] + x["Close"]) / 3
    day = pd.Series(x.index.date, index=x.index)

    x["VWAP"] = typical.groupby(day).transform("mean")

    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)
    x["RANGE_VOL"] = candle_range / candle_range.rolling(20).median().replace(0, np.nan)

    # Keep RVOL as a compatibility field for the frontend.
    # It is a range-expansion proxy, NOT exchange volume.
    x["RVOL"] = x["RANGE_VOL"]

    x["BODY_RATIO"] = (
        (x["Close"] - x["Open"]).abs() / candle_range
    )
    x["HIGH20"] = x["High"].rolling(20).max().shift(1)
    x["LOW20"] = x["Low"].rolling(20).min().shift(1)

    # Opening range: first 15 minutes for 5m data, first 30 minutes for 15m+.
    x["SESSION_DATE"] = day

    return x


def daily_context(df, target):
    d = (
        df.assign(day=df.index.date)
        .groupby("day")
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
        )
    )
    d.index = pd.to_datetime(d.index)

    d["EMA5"] = d["Close"].ewm(span=5, adjust=False).mean()
    d["EMA10"] = d["Close"].ewm(span=10, adjust=False).mean()
    d["RANGE"] = (d["High"] - d["Low"]) / d["Open"].replace(0, np.nan)

    target_ts = pd.Timestamp(target)
    prev = d[d.index < target_ts].tail(7)

    if len(prev) < 3:
        return {
            "score": 0,
            "direction": "WAIT",
            "reasons": ["Need at least 3 previous trading sessions for context."],
        }

    last = prev.iloc[-1]
    score = 0
    reasons = []

    if last["Close"] > last["EMA5"]:
        score += 1
        reasons.append("previous day above 5D EMA")
    elif last["Close"] < last["EMA5"]:
        score -= 1
        reasons.append("previous day below 5D EMA")

    if last["EMA5"] > last["EMA10"]:
        score += 1
        reasons.append("short-term daily trend positive")
    elif last["EMA5"] < last["EMA10"]:
        score -= 1
        reasons.append("short-term daily trend negative")

    if prev["Close"].iloc[-1] > prev["Close"].iloc[0]:
        score += 1
        reasons.append("7-day momentum positive")
    elif prev["Close"].iloc[-1] < prev["Close"].iloc[0]:
        score -= 1
        reasons.append("7-day momentum negative")

    if prev["RANGE"].tail(3).mean() > prev["RANGE"].mean():
        score += 1 if last["Close"] >= last["Open"] else -1
        reasons.append("recent daily range expansion")

    if last["Close"] > last["Open"]:
        score += 1
        reasons.append("previous day green")
    else:
        score -= 1
        reasons.append("previous day red")

    if score >= 2:
        direction = "CALL"
    elif score <= -2:
        direction = "PUT"
    else:
        direction = "WAIT"

    return {"score": int(score), "direction": direction, "reasons": reasons}


def signal(row):
    call = 0
    put = 0
    call_reasons = []
    put_reasons = []

    if row["Close"] > row["VWAP"]:
        call += 1
        call_reasons.append("above session VWAP proxy")
    elif row["Close"] < row["VWAP"]:
        put += 1
        put_reasons.append("below session VWAP proxy")

    if row["EMA20"] > row["EMA50"]:
        call += 1
        call_reasons.append("EMA20 > EMA50")
    elif row["EMA20"] < row["EMA50"]:
        put += 1
        put_reasons.append("EMA20 < EMA50")

    if row["ADX"] >= 20:
        if row["EMA20"] > row["EMA50"]:
            call += 1
            call_reasons.append("ADX trend strength")
        elif row["EMA20"] < row["EMA50"]:
            put += 1
            put_reasons.append("ADX trend strength")

    if row["RANGE_VOL"] >= 1.15:
        if row["Close"] > row["Open"]:
            call += 1
            call_reasons.append("range expansion")
        elif row["Close"] < row["Open"]:
            put += 1
            put_reasons.append("range expansion")

    if row["BODY_RATIO"] >= 0.60:
        if row["Close"] > row["Open"]:
            call += 1
            call_reasons.append("strong bullish candle")
        elif row["Close"] < row["Open"]:
            put += 1
            put_reasons.append("strong bearish candle")

    if pd.notna(row["HIGH20"]) and row["Close"] > row["HIGH20"]:
        call += 2
        call_reasons.append("20-bar breakout")

    if pd.notna(row["LOW20"]) and row["Close"] < row["LOW20"]:
        put += 2
        put_reasons.append("20-bar breakdown")

    if call >= 5 and call > put:
        return "CALL", call, call_reasons, put
    if put >= 5 and put > call:
        return "PUT", put, put_reasons, call
    return "WAIT", max(call, put), [], min(call, put)


def normcdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(spot, strike, t, iv=0.18, rate=0.06, call=True):
    if t <= 0:
        return max(spot - strike, 0) if call else max(strike - spot, 0)

    v = max(iv, 0.01)
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * v * v) * t
    ) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)

    if call:
        return spot * normcdf(d1) - strike * math.exp(-rate * t) * normcdf(d2)

    return strike * math.exp(-rate * t) * normcdf(-d2) - spot * normcdf(-d1)


def backtest(df, lot=75):
    x = indicators(df)
    trades = []
    equity = 100000.0
    peak = equity
    max_dd = 0.0

    available_days = sorted(set(x.index.date))

    for day in available_days:
        ctx = daily_context(df, day)
        if not ctx or ctx["direction"] == "WAIT":
            continue

        z = x[x.index.date == day].copy()
        z = z[
            (z.index.hour > 9)
            | ((z.index.hour == 9) & (z.index.minute >= 45))
        ]

        if len(z) < 3:
            continue

        for i in range(len(z) - 1):
            r = z.iloc[i]

            if r[["EMA20", "EMA50", "VWAP", "ADX", "RANGE_VOL", "ATR"]].isna().any():
                continue

            side, score, reasons, _ = signal(r)

            if side != ctx["direction"] or score < 5:
                continue

            ent = z.iloc[i + 1]
            ep = float(ent["Open"])
            strike = int(round(ep / STRIKE_STEP) * STRIKE_STEP)

            pe = bs(ep, strike, 1 / 252, call=side == "CALL")

            stop = max(float(r["ATR"]), ep * 0.002)
            target = 2 * stop

            xp = float(z.iloc[-1]["Close"])
            xt = z.index[-1]
            why = "EOD"

            for j in range(i + 1, len(z)):
                b = z.iloc[j]

                if side == "CALL":
                    if b["Low"] <= ep - stop:
                        xp = ep - stop
                        xt = z.index[j]
                        why = "SL"
                        break
                    if b["High"] >= ep + target:
                        xp = ep + target
                        xt = z.index[j]
                        why = "TARGET"
                        break
                else:
                    if b["High"] >= ep + stop:
                        xp = ep + stop
                        xt = z.index[j]
                        why = "SL"
                        break
                    if b["Low"] <= ep - target:
                        xp = ep - target
                        xt = z.index[j]
                        why = "TARGET"
                        break

            remaining = max(
                (xt - ent.name).total_seconds() / (365 * 86400),
                1 / 31536000,
            )
            px = bs(xp, strike, remaining, call=side == "CALL")

            pnl = (px - pe) * lot
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)

            trades.append(
                {
                    "date": str(day),
                    "entry_time": str(ent.name),
                    "exit_time": str(xt),
                    "side": side,
                    "strike": strike,
                    "spot_entry": round(ep, 2),
                    "spot_exit": round(xp, 2),
                    "premium_entry": round(pe, 2),
                    "premium_exit": round(px, 2),
                    "pnl": round(pnl, 2),
                    "exit_reason": why,
                    "signal_score": int(score),
                    "lookback_score": int(ctx["score"]),
                    "reasons": ", ".join(reasons),
                    "equity": round(equity, 2),
                }
            )

            # One trade per day.
            break

    wins = sum(t["pnl"] > 0 for t in trades)

    metrics = {
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round(100 * wins / len(trades), 2) if trades else 0,
        "net_pnl": round(sum(t["pnl"] for t in trades), 2),
        "return_pct": round((equity / 100000 - 1) * 100, 2),
        "max_drawdown": round(max_dd, 2),
        "final_equity": round(equity, 2),
    }

    return trades, metrics


def nearest_strike(spot):
    return int(round(spot / STRIKE_STEP) * STRIKE_STEP)


def nse_option_chain():
    import requests

    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )
    s.get("https://www.nseindia.com/", timeout=10)
    r = s.get(
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def select_option(spot, direction):
    data = nse_option_chain()
    records = data.get("records", {})
    expiry_dates = records.get("expiryDates", [])
    rows = records.get("data", [])

    if not expiry_dates:
        raise RuntimeError("NSE did not return an expiry date.")

    expiry = expiry_dates[0]
    atm = nearest_strike(spot)

    # Slightly ITM option: CALL ATM-50, PUT ATM+50.
    desired = atm - STRIKE_STEP if direction == "CALL" else atm + STRIKE_STEP

    choices = []

    for row in rows:
        if row.get("expiryDate") != expiry:
            continue

        strike = int(row.get("strikePrice", 0))
        leg = row.get("CE") if direction == "CALL" else row.get("PE")

        if leg:
            choices.append((abs(strike - desired), strike, leg))

    if not choices:
        raise RuntimeError("No suitable option contract was returned.")

    _, strike, leg = min(choices, key=lambda z: z[0])
    ltp = leg.get("lastPrice")

    if ltp is None:
        raise RuntimeError("Option LTP unavailable.")

    ltp = float(ltp)

    return {
        "strike": strike,
        "type": "CE" if direction == "CALL" else "PE",
        "expiry": expiry,
        "ltp": round(ltp, 2),
        "buy_zone_low": round(ltp * 0.99, 2),
        "buy_zone_high": round(ltp * 1.02, 2),
        "stop": round(ltp * 0.80, 2),
        "target1": round(ltp * 1.20, 2),
        "target2": round(ltp * 1.35, 2),
        "oi": leg.get("openInterest"),
        "volume": leg.get("totalTradedVolume"),
        "change_pct": leg.get("percentChange"),
    }


def live_scan(period="30d", interval="5m"):
    df = download_nifty(period, interval)
    x = indicators(df)

    usable = x.dropna(
        subset=["EMA20", "EMA50", "VWAP", "RSI", "ATR", "ADX", "RANGE_VOL"]
    )

    if len(usable) < 30:
        raise RuntimeError(
            f"Only {len(usable)} usable candles are available. "
            "Try 15m or wait for more market data."
        )

    r = usable.iloc[-1]
    side, score, reasons, other_score = signal(r)

    result = {
        "status": "ok",
        "signal": side,
        "score": int(score),
        "other_score": int(other_score),
        "price": round(float(r["Close"]), 2),
        "adx": round(float(r["ADX"]), 2),
        "rvol": round(float(r["RANGE_VOL"]), 2),
        "rvol_label": "Range expansion",
        "vwap": round(float(r["VWAP"]), 2),
        "vwap_label": "Session VWAP proxy",
        "ema20": round(float(r["EMA20"]), 2),
        "ema50": round(float(r["EMA50"]), 2),
        "rsi": round(float(r["RSI"]), 2),
        "atr": round(float(r["ATR"]), 2),
        "reasons": reasons,
        "timestamp": r.name.strftime("%d %b %Y, %I:%M %p"),
        "data_note": (
            "NIFTY index volume from Yahoo is often zero; "
            "the dashboard therefore uses price-only VWAP and range-expansion "
            "instead of exchange-volume RVOL."
        ),
    }

    if side in ("CALL", "PUT"):
        try:
            result["option"] = select_option(float(r["Close"]), side)
        except Exception as exc:
            result["option"] = None
            result["option_error"] = str(exc)
    else:
        result["option"] = None

    return result


@app.route("/")
def home():
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
        "index.html",
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/scan")
def api_scan():
    try:
        return jsonify(
            live_scan(
                request.args.get("period", "30d"),
                request.args.get("interval", "5m"),
            )
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.get("/api/option-backtest")
def option_backtest():
    try:
        df = download_nifty(
            request.args.get("period", "60d"),
            request.args.get("interval", "15m"),
        )
        trades, metrics = backtest(df)
        return jsonify(
            {
                "status": "ok",
                "metrics": metrics,
                "trades": trades,
                "note": (
                    "Option premiums are modeled with Black-Scholes from "
                    "NIFTY spot history; this is not a historical NSE option-premium feed."
                ),
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.get("/api/backtest")
def api_backtest():
    return option_backtest()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
