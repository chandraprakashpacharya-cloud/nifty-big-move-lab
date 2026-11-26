import os
from flask import Flask, jsonify, send_from_directory

from nifty_1day_options import app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


@app.route("/")
def home():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
