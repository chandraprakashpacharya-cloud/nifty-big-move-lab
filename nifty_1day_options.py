import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NIFTY = "^NSEI"
STRIKE_STEP = 50


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
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
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    av = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / av
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / av
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def session_vwap(df):
    dates = df.index.date
    pv = df["Close"] * df["Volume"]
    return pv.groupby(dates).cumsum() / df["Volume"].groupby(dates).cumsum()


def prepare(df):
    x = df.copy()
    x["EMA20"] = ema(x["Close"], 20)
    x["EMA50"] = ema(x["Close"], 50)
    x["EMA200"] = ema(x["Close"], 200)
    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["ADX"] = adx(x)
    x["VWAP"] = session_vwap(x)
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["RVOL"] = x["Volume"] / x["VOL_MA20"].replace(0, np.nan)
    x["BODY"] = (x["Close"] - x["Open"]).abs()
    x["RANGE"] = (x["High"] - x["Low"]).replace(0, np.nan)
    x["BODY_RATIO"] = x["BODY"] / x["RANGE"]
    x["UPPER_WICK"] = x["High"] - x[["Open", "Close"]].max(axis=1)
    x["LOWER_WICK"] = x[["Open", "Close"]].min(axis=1) - x["Low"]
    x["PREV_HIGH_20"] = x["High"].shift(1).rolling(20).max()
    x["PREV_LOW_20"] = x["Low"].shift(1).rolling(20).min()
    return x


def download_nifty(interval="5m", period="5d"):
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

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)
    return prepare(df)


def _local_extrema_levels(df, window=2):
    highs = df["High"].values
    lows = df["Low"].values
    hi = []
    lo = []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            hi.append(float(highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            lo.append(float(lows[i]))
    return hi, lo


def support_resistance(df, lookback=80, tolerance_atr=0.25):
    """Return nearest meaningful horizontal levels using confirmed swing points.

    Levels are clustered so several nearby swing points become one line.
    Only candles before the latest candle are used, preventing look-ahead in
    the live signal.
    """
    x = df.iloc[-lookback:].copy()
    if len(x) < 15:
        return None, None, [], []

    atrv = float(x["ATR"].dropna().iloc[-1]) if x["ATR"].notna().any() else 0
    tolerance = max(2.0, atrv * tolerance_atr)

    highs, lows = _local_extrema_levels(x, window=2)

    def cluster(values):
        if not values:
            return []
        values = sorted(values)
        groups = [[values[0]]]
        for v in values[1:]:
            if abs(v - np.mean(groups[-1])) <= tolerance:
                groups[-1].append(v)
            else:
                groups.append([v])
        return sorted(
            [
                {
                    "level": round(float(np.mean(g)), 2),
                    "touches": len(g),
                }
                for g in groups
            ],
            key=lambda z: (z["touches"], z["level"]),
            reverse=True,
        )

    resistances = cluster(highs)
    supports = cluster(lows)
    price = float(x["Close"].iloc[-1])

    above = [z for z in resistances if z["level"] > price]
    below = [z for z in supports if z["level"] < price]

    nearest_r = min(above, key=lambda z: z["level"]) if above else None
    nearest_s = max(below, key=lambda z: z["level"]) if below else None

    return (
        nearest_s["level"] if nearest_s else None,
        nearest_r["level"] if nearest_r else None,
        supports,
        resistances,
    )


def _break_retest_state(x, support, resistance, atrv):
    """Detect breakout/breakdown and retest confirmation from recent candles."""
    if len(x) < 5:
        return {
            "call_breakout": False,
            "call_retest": False,
            "put_breakdown": False,
            "put_retest": False,
        }

    c = x.iloc[-1]
    p = x.iloc[-2]
    tol = max(2.0, atrv * 0.15)

    call_breakout = (
        resistance is not None
        and p["Close"] <= resistance + tol
        and c["Close"] > resistance
        and c["Close"] > c["Open"]
        and c["BODY_RATIO"] >= 0.50
    )
    call_retest = (
        resistance is not None
        and min(c["Low"], p["Low"]) <= resistance + tol
        and c["Close"] >= resistance
    )

    put_breakdown = (
        support is not None
        and p["Close"] >= support - tol
        and c["Close"] < support
        and c["Close"] < c["Open"]
        and c["BODY_RATIO"] >= 0.50
    )
    put_retest = (
        support is not None
        and max(c["High"], p["High"]) >= support - tol
        and c["Close"] <= support
    )

    return {
        "call_breakout": bool(call_breakout),
        "call_retest": bool(call_retest),
        "put_breakdown": bool(put_breakdown),
        "put_retest": bool(put_retest),
    }


def market_signal(df):
    required = ["EMA20", "EMA50", "VWAP", "RSI", "ATR", "ADX", "RVOL"]
    x = df.dropna(subset=required).copy()
    if len(x) < 60:
        raise RuntimeError(
            f"Not enough candles to calculate the signal. Got {len(x)}, need at least 60."
        )

    r = x.iloc[-1]
    price = float(r.Close)
    atrv = float(r.ATR)

    support, resistance, support_levels, resistance_levels = support_resistance(
        x, lookback=min(120, len(x)), tolerance_atr=0.25
    )
    state = _break_retest_state(x, support, resistance, atrv)

    today = r.name.date()
    day = x[x.index.date == today]
    opening = day.between_time("09:15", "09:30")
    orb_hi = float(opening.High.max()) if not opening.empty else float("nan")
    orb_lo = float(opening.Low.min()) if not opening.empty else float("nan")

    call = {
        "Above VWAP": bool(price > r.VWAP),
        "EMA20 > EMA50": bool(r.EMA20 > r.EMA50),
        "ADX >= 20": bool(r.ADX >= 20),
        "RSI 52-72": bool(52 <= r.RSI <= 72),
        "RVOL >= 1.15": bool(r.RVOL >= 1.15),
        "Strong green candle": bool(r.Close > r.Open and r.BODY_RATIO >= 0.50),
        "Resistance breakout": bool(state["call_breakout"]),
        "Resistance retest holds": bool(state["call_retest"]),
        "Opening-range break": bool(np.isnan(orb_hi) or price > orb_hi),
    }

    put = {
        "Below VWAP": bool(price < r.VWAP),
        "EMA20 < EMA50": bool(r.EMA20 < r.EMA50),
        "ADX >= 20": bool(r.ADX >= 20),
        "RSI 28-48": bool(28 <= r.RSI <= 48),
        "RVOL >= 1.15": bool(r.RVOL >= 1.15),
        "Strong red candle": bool(r.Close < r.Open and r.BODY_RATIO >= 0.50),
        "Support breakdown": bool(state["put_breakdown"]),
        "Support retest fails": bool(state["put_retest"]),
        "Opening-range break": bool(np.isnan(orb_lo) or price < orb_lo),
    }

    cs, ps = int(sum(call.values())), int(sum(put.values()))

    # A breakout/retest is deliberately weighted more heavily than generic
    # oscillator conditions. This avoids CALL/PUT signals in the middle of a range.
    call_structure = state["call_breakout"] and state["call_retest"]
    put_structure = state["put_breakdown"] and state["put_retest"]

    extension = abs(price - float(r.EMA20)) / atrv if atrv else 0
    chase = extension > 1.5

    if call_structure and cs >= 7 and cs > ps and not chase:
        signal = "CALL"
        confidence = min(95, 55 + cs * 4)
        stop = min(float(r.EMA20), price - atrv)
        target = price + 2.0 * atrv
        reason = "Resistance breakout + retest hold with trend, momentum and volume confirmation."
        checks = call
    elif put_structure and ps >= 7 and ps > cs and not chase:
        signal = "PUT"
        confidence = min(95, 55 + ps * 4)
        stop = max(float(r.EMA20), price + atrv)
        target = price - 2.0 * atrv
        reason = "Support breakdown + retest failure with trend, momentum and volume confirmation."
        checks = put
    else:
        signal = "WAIT"
        confidence = min(80, max(cs, ps) * 8)
        stop = target = None
        if resistance and price < resistance and support and price > support:
            reason = "Price is inside support/resistance range. Wait for a confirmed breakout or breakdown."
        elif chase:
            reason = "Price is extended more than 1.5 ATR from EMA20. Avoid chasing."
        else:
            reason = "No fully confirmed breakout/retest setup."
        checks = call if cs >= ps else put

    return {
        "timestamp": r.name.strftime("%d %b %Y, %I:%M %p"),
        "signal": signal,
        "confidence": int(confidence),
        "nifty": round(price, 2),
        "nifty_entry": round(price, 2) if signal != "WAIT" else None,
        "nifty_stop": round(stop, 2) if stop is not None else None,
        "nifty_target": round(target, 2) if target is not None else None,
        "atr": round(atrv, 2),
        "adx": round(float(r.ADX), 2),
        "rsi": round(float(r.RSI), 2),
        "vwap": round(float(r.VWAP), 2),
        "ema20": round(float(r.EMA20), 2),
        "ema50": round(float(r.EMA50), 2),
        "rvol": round(float(r.RVOL), 2),
        "support": round(support, 2) if support else None,
        "resistance": round(resistance, 2) if resistance else None,
        "support_levels": support_levels[:5],
        "resistance_levels": resistance_levels[:5],
        "call_score": cs,
        "put_score": ps,
        "checks": checks,
        "reason": reason,
        "breakout_state": state,
        "risk_reward": 2.0 if signal != "WAIT" else None,
    }


def nse_option_chain():
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


def nearest_strike(spot):
    return int(round(spot / STRIKE_STEP) * STRIKE_STEP)


def select_option(spot, direction):
    data = nse_option_chain()
    records = data.get("records", {})
    expiry_dates = records.get("expiryDates", [])
    rows = records.get("data", [])
    if not expiry_dates:
        raise RuntimeError("NSE did not return an expiry date.")

    expiry = expiry_dates[0]
    atm = nearest_strike(spot)
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


def run_scan(interval="5m"):
    df = download_nifty(interval)
    market = market_signal(df)
    option = None
    option_error = None

    if market["signal"] in ("CALL", "PUT"):
        try:
            option = select_option(market["nifty"], market["signal"])
        except Exception as e:
            option_error = str(e)

    return {
        "market": market,
        "option": option,
        "option_error": option_error,
        "data_source": "Yahoo Finance (NIFTY) + NSE option chain",
    }


if __name__ == "__main__":
    result = run_scan("5m")
    print(result)
