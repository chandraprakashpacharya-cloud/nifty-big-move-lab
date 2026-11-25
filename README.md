# NIFTY 50 Big Move Lab

## Run locally

```bash
pip install -r requirements.txt
python server.py
```

Open http://localhost:5000

## Features
- Live NIFTY big-move signal
- 15m / 5m / 30m / 60m interval selector
- 30/60/90 day selector
- ORB + VWAP + EMA20/50 + ADX + RSI + relative volume + ATR expansion
- Backtest metrics
- Equity curve
- Recent trades
- JSON/XLSX/CSV backtest files under `results/`

The signal is generated from NIFTY index data. It is not an historical option-premium backtest.
