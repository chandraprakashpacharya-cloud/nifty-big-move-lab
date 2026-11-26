# NIFTY 50 CALL / PUT Render Web App

## Render setup

Create a new **Web Service** from this GitHub repository.

Build Command:
`pip install -r requirements.txt`

Start Command:
`gunicorn server:app`

Health Check Path:
`/health`

The application serves the dashboard from `/` and the live scanner from `/api/scan`.

## Features

- 5m/15m/30m/60m NIFTY signal scan
- CALL / PUT / WAIT
- EMA20/EMA50
- VWAP
- RSI
- ADX
- Relative volume
- Candle strength
- 20-bar breakout/breakdown
- Opening-range confirmation
- ATR-based NIFTY levels
- NSE nearest-expiry option selection
- Option buy zone, stop and two targets
- Mobile-friendly dashboard

The option premium is fetched from the public NSE option-chain endpoint when available. NSE can rate-limit/block automated requests, so option data may occasionally be unavailable. For production trading, use a broker API for reliable market data and order execution.

This is educational decision-support software and does not guarantee profitable trades.
