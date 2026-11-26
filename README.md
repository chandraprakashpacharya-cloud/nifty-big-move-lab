# NIFTY 50 1-Day Options Engine — Fixed

## Fix included
The previous version could show:

`Not enough candles to calculate the signal.`

The main issue was that Yahoo's NIFTY index data commonly has zero volume. The old code calculated volume-based VWAP/RVOL, produced NaN values, dropped those rows, and ended up with too few usable candles.

This version:
- uses a session price VWAP proxy when index volume is unavailable
- uses candle-range expansion as a volume substitute
- keeps the signal logic based on EMA20/EMA50, ADX, RSI, breakout/breakdown, candle strength and session VWAP proxy
- works with 5m/15m/30m/60m data
- gives a clear data-quality note in the API
- keeps the option-chain lookup and 1-day option backtest

## Render
Build command:
`pip install -r requirements.txt`

Start command:
`gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 server:app`

## Local
`pip install -r requirements.txt`
`python server.py`

Open:
`http://localhost:5000`

## Important
The option backtest uses a Black-Scholes model from NIFTY spot history. It is not a historical NSE option-premium feed, so use it for research/backtesting only.
