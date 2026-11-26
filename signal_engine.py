
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
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    av = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / av
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / av
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def session_vwap(df):
    dates = df.index.date
    pv = df["Close"] * df["Volume"]
    return pv.groupby(dates).cumsum() / df["Volume"].groupby(dates).cumsum()


def prepare(df):
    x = df.copy()
    x["EMA20"] = ema(x["Close"], 20)
    x["EMA50"] = ema(x["Close"], 50)
    x["RSI"] = rsi(x["Close"])
    x["ATR"] = atr(x)
    x["ADX"] = adx(x)
    x["VWAP"] = session_vwap(x)
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["RVOL"] = x["Volume"] / x["VOL_MA20"].replace(0, np.nan)
    x["BODY"] = (x["Close"] - x["Open"]).abs()
    x["RANGE"] = (x["High"] - x["Low"]).replace(0, np.nan)
    x["BODY_RATIO"] = x["BODY"] / x["RANGE"]
    x["PREV_HIGH_20"] = x["High"].shift(1).rolling(20).max()
    x["PREV_LOW_20"] = x["Low"].shift(1).rolling(20).min()
    return x


def download_nifty(interval="5m"):
    df = yf.download(
        NIFTY, period="5d", interval=interval,
        auto_adjust=False, progress=False, threads=False
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


def market_signal(df):
    x = df.dropna(subset=["EMA20","EMA50","VWAP","RSI","ATR","ADX","RVOL"])
    if len(x) < 30:
        raise RuntimeError("Not enough candles to calculate the signal.")

    r = x.iloc[-1]
    price = float(r.Close)
    atrv = float(r.ATR)

    today = r.name.date()
    day = x[x.index.date == today]
    opening = day.between_time("09:15", "09:30")
    orb_hi = float(opening.High.max()) if not opening.empty else float("nan")
    orb_lo = float(opening.Low.min()) if not opening.empty else float("nan")

    call = {
        "Above VWAP": price > r.VWAP,
        "EMA20 > EMA50": r.EMA20 > r.EMA50,
        "ADX >= 20": r.ADX >= 20,
        "RSI 52-72": 52 <= r.RSI <= 72,
        "RVOL >= 1.15": r.RVOL >= 1.15,
        "Strong green candle": r.Close > r.Open and r.BODY_RATIO >= .50,
        "20-bar breakout": price > r.PREV_HIGH_20,
        "Opening-range break": np.isnan(orb_hi) or price > orb_hi,
    }
    put = {
        "Below VWAP": price < r.VWAP,
        "EMA20 < EMA50": r.EMA20 < r.EMA50,
        "ADX >= 20": r.ADX >= 20,
        "RSI 28-48": 28 <= r.RSI <= 48,
        "RVOL >= 1.15": r.RVOL >= 1.15,
        "Strong red candle": r.Close < r.Open and r.BODY_RATIO >= .50,
        "20-bar breakdown": price < r.PREV_LOW_20,
        "Opening-range break": np.isnan(orb_lo) or price < orb_lo,
    }

    cs, ps = sum(call.values()), sum(put.values())
    extension = abs(price - r.EMA20) / atrv if atrv else 0
    chase = extension > 1.5

    if cs >= 6 and cs > ps:
        signal = "CALL"
        confidence = min(95, 55 + cs*5)
        stop = min(float(r.EMA20), price - atrv)
        target = price + 2*atrv
        reason = "Bullish trend, momentum and breakout alignment."
        checks = call
    elif ps >= 6 and ps > cs:
        signal = "PUT"
        confidence = min(95, 55 + ps*5)
        stop = max(float(r.EMA20), price + atrv)
        target = price - 2*atrv
        reason = "Bearish trend, momentum and breakdown alignment."
        checks = put
    else:
        signal = "WAIT"
        confidence = max(cs, ps) * 10
        stop = target = None
        reason = "Conditions are mixed. Wait for confirmation."
        checks = call if cs >= ps else put

    if chase and signal != "WAIT":
        reason += " Price is extended >1.5 ATR from EMA20; avoid chasing."

    return {
        "timestamp": r.name.strftime("%d %b %Y, %I:%M %p"),
        "signal": signal,
        "confidence": int(confidence),
        "nifty": round(price, 2),
        "nifty_entry": round(price, 2),
        "nifty_stop": round(stop, 2) if stop is not None else None,
        "nifty_target": round(target, 2) if target is not None else None,
        "atr": round(atrv, 2),
        "adx": round(float(r.ADX), 2),
        "rsi": round(float(r.RSI), 2),
        "vwap": round(float(r.VWAP), 2),
        "ema20": round(float(r.EMA20), 2),
        "ema50": round(float(r.EMA50), 2),
        "rvol": round(float(r.RVOL), 2),
        "call_score": cs,
        "put_score": ps,
        "checks": checks,
        "reason": reason,
    }


def nse_option_chain():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/"
    })
    s.get("https://www.nseindia.com/", timeout=10)
    r = s.get(
        "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
        timeout=15
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
            choices.append((abs(strike-desired), strike, leg))

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
        "buy_zone_low": round(ltp * .99, 2),
        "buy_zone_high": round(ltp * 1.02, 2),
        "stop": round(ltp * .80, 2),
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
