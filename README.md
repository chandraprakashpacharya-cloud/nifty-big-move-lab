# NIFTY 50 Enhanced Breakout Engine — Render Fixed

## Render
Build:
```bash
pip install -r requirements.txt
```

Start:
```bash
gunicorn server:app
```

## Yahoo data handling
The engine now tries:
1. yfinance
2. Yahoo Chart API query1
3. Yahoo Chart API query2
4. 15-minute Yahoo candles when a 5-minute request fails

The app now reports a real Yahoo/network error instead of the misleading
"Got 0, need at least 60" calculation error.

If every Yahoo endpoint is unreachable from the Render instance, live candles
cannot be generated locally; the Render logs will show the underlying network
or Yahoo response error.
