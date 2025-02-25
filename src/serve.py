#!/usr/bin/env python3
# serve.py

import os
import sys
import json
import time
import logging
from flask import Flask, jsonify
from inference import get_prediction

# --- Load Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DEBUG_MODE = config.get("debug", False)

# --- Setup Logging ---
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prediction_service.log')
    ]
)
logger = logging.getLogger("vodka-serve")

if not config.get("serve", False):
    logger.error("Serve mode is disabled. Set 'serve' to true in vodka.json.")
    sys.exit(1)

PORT = config.get("port", 5000)

PREDICTION_HISTORY = []
latest_history = 50

# --- Flask Application ---
www_dir = os.path.join(BASE_DIR, "www")
app = Flask(__name__, static_folder=www_dir, static_url_path="")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/")
def prediction_api():
    prediction = get_prediction()
    PREDICTION_HISTORY.append(prediction)
    if len(PREDICTION_HISTORY) > latest_history:
        del PREDICTION_HISTORY[0]
    return jsonify(prediction)

@app.route("/api/history")
def history_api():
    """Return up to the last 30 prediction results."""
    return jsonify(PREDICTION_HISTORY)

@app.route("/api/status")
def status_api():
    return jsonify({
        "status": "running",
        "version": "1.0.0"
    })

@app.route("/api/refresh")
def refresh_api():
    prediction = get_prediction(use_cache=False)
    # Also record refreshed prediction in history
    PREDICTION_HISTORY.append(prediction)
    if len(PREDICTION_HISTORY) > 30:
        del PREDICTION_HISTORY[0]
    return jsonify(prediction)

@app.route("/api/config")
def config_api():
    safe_config = {
        "symbol": config.get("symbol"),
        "interval": config.get("interval"),
        "window_size": config.get("limit"),
        "prediction_offsets": config.get("prediction", [1, 5, 10]),
        "cache_duration": config.get("cache_duration", 30)
    }
    return jsonify(safe_config)

if __name__ == "__main__":
    logger.info(f"Starting prediction service on port {PORT}")
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
