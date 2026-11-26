# NIFTY 50 Enhanced Breakout Engine

Render-ready Flask app.

## Logic
- 5-minute NIFTY data
- EMA20 / EMA50
- VWAP
- RSI / ADX / ATR
- Relative volume
- Confirmed swing support/resistance
- Resistance breakout + retest confirmation for CALL
- Support breakdown + retest failure for PUT
- Strong candle-body confirmation
- Avoids chasing moves > 1.5 ATR from EMA20
- NSE option-chain selection only after a confirmed signal

## Run
pip install -r requirements.txt
gunicorn server:app

Open the Render URL.

## Important
This is a research/signal engine, not a guaranteed-profit system. The 80% win-rate target should be validated out-of-sample; do not assume it is achievable live.
