from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import nifty_1day_options as engine

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/scan")
def scan():
    try:
        return jsonify(engine.run_scan("5m"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "name": "NIFTY 50 Enhanced Breakout Engine"})


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
