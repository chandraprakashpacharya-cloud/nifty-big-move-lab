# NIFTY Big Move Lab — Render Fixed

## Render
Build Command:
`pip install -r requirements.txt`

Start Command:
`gunicorn server:app`

Do not use `gunicorn app:app`.

## Structure
static/index.html
nifty_1day_options.py
server.py
requirements.txt

The Flask root route exists only in nifty_1day_options.py. server.py only imports the app, preventing the duplicate `home` endpoint error.

## Routes
/
 /health
 /api/scan
 /api/backtest
 /api/option-backtest

The options backtest uses NIFTY spot history and Black-Scholes modeled premiums. It is not a historical NSE option-chain premium backtest.
