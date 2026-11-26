# NIFTY 1-Day Options Engine

## Render

Use the GitHub repo as the source.

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn server:app`
- **Branch:** `main`

The wrapper `server.py` imports the Flask app from `nifty_1day_options.py`, so this fixes the earlier `ModuleNotFoundError: No module named 'app'` caused by Render using `gunicorn app:app`.

## Strategy

Uses the previous completed 1, 2, 3, 5 and 7 trading sessions as context. It then trades only today's session:

- 09:15–09:30 opening range
- Entry after 09:30 and before 14:30
- 10-factor score: ORB, VWAP, EMA20/50, ADX, RSI, volume, ATR expansion, candle strength, previous 1–7 day levels, 3/5-day trend
- Minimum score 8/10
- Entry on next candle open
- 0.8 ATR stop
- 2.2R target
- Maximum one trade per day
- Force exit by 15:15
- CALL for bullish signal, PUT for bearish signal
- ATM by default; 1/2 strikes ITM available

## Backtest limitation

The default backtest is a NIFTY spot/index movement backtest mapped to an option. It is **not** historical option-premium P&L because yfinance does not provide a reliable historical NIFTY option-chain/premium series for arbitrary past sessions.

For actual option P&L, provide intraday option OHLC CSV with:

`Date,Expiry,Strike,OptionType,Open,High,Low,Close,Volume,Time`

and run:

```bash
python nifty_1day_options.py --mode option-backtest --spot-csv nifty.csv --options-csv options.csv
```
