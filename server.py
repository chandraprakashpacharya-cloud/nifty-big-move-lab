
import os
from flask import Flask, jsonify, send_from_directory, request
from signal_engine import run_scan

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/scan")
def scan():
    interval = request.args.get("interval", "5m")
    if interval not in {"5m", "15m", "30m", "60m"}:
        interval = "5m"
    try:
        return jsonify(run_scan(interval=interval))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
