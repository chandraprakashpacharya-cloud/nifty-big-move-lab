from pathlib import Path
import json, os, sys
from flask import Flask, jsonify, send_from_directory, request

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import nifty_big_move as engine

app = Flask(__name__, static_folder="static")
LAST = {"trades": [], "metrics": {}, "data": None}

@app.get("/")
def home():
    return send_from_directory(BASE / "static", "index.html")

@app.get("/api/scan")
def scan():
    period = request.args.get("period", "60d")
    interval = request.args.get("interval", "15m")
    try:
        data = engine.download_data(period, interval)
        result = engine.live_scan(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

@app.get("/api/backtest")
def backtest():
    period = request.args.get("period", "60d")
    interval = request.args.get("interval", "15m")
    try:
        data = engine.download_data(period, interval)
        trades, metrics = engine.backtest(data)
        LAST["trades"] = trades.to_dict("records") if not trades.empty else []
        LAST["metrics"] = metrics
        LAST["data"] = data
        engine.save_backtest(trades, metrics, data)
        return jsonify({"status":"ok","metrics":metrics})
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500

@app.get("/api/trades")
def trades():
    return jsonify({"trades": LAST["trades"]})

@app.get("/api/equity")
def equity():
    if not LAST["trades"]:
        return jsonify({"points":[]})
    return jsonify({"points":[t.get("cum_points",0) for t in LAST["trades"]]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","5000")), debug=False)
