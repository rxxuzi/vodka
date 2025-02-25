#!/usr/bin/env python3
# inference.py

import os
import time
import requests
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from lime import PositionalEncoding

# --- Load Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- Setup Logging ---
DEBUG_MODE = config.get("debug", False)
logger = logging.getLogger("inference")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.ERROR)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Configuration parameters
SYMBOL = config["symbol"]
INTERVAL = config["interval"]
WINDOW_SIZE = config["limit"]
PREDICTION_OFFSETS = config.get("prediction", [1, 5, 10])
MODEL_NAME = config.get("model", "vodka_btc_1m")
MOVING_AVERAGE_WINDOW = config.get("ma_window", 3)
PREDICTION_ADJUSTMENT = config.get("prediction_adjustment", 0.0)
CACHE_DURATION = config.get("cache_duration", 30)

# --- Model & Scaler Loading ---
models_dir = os.path.join(BASE_DIR, "models")

def load_models_and_scalers():
    models = {}
    scalers = {}
    model_filename = f"{MODEL_NAME}.h5"
    model_path = os.path.join(models_dir, model_filename)
    scaler_filename = f"{MODEL_NAME}_scaler.pkl"
    scaler_path = os.path.join(models_dir, scaler_filename)
    try:
        models["primary"] = load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        with open(scaler_path, "rb") as f:
            scalers["primary"] = pickle.load(f)
        logger.debug("Primary model and scaler loaded.")
    except Exception as e:
        logger.error(f"Error loading primary model or scaler: {e}")
        raise e
    return models, scalers

models, scalers = load_models_and_scalers()

# --- Cache for predictions ---
_cache = {"prediction": None, "timestamp": 0}

# --- Helper Functions ---

def fetch_klines(symbol, interval, limit):
    """Fetch candlestick data from Binance API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def extract_features(klines):
    """
    Extract 5 features (Open, High, Low, Close, Volume) from kline data.
    Binance kline: index 1: Open, 2: High, 3: Low, 4: Close, 5: Volume.
    """
    features = []
    for kline in klines:
        features.append([
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5])
        ])
    return np.array(features)  # shape: (num_candles, 5)

def moving_average_smoothing(values, window=MOVING_AVERAGE_WINDOW):
    """Apply moving average smoothing to predictions."""
    if window <= 1 or window > len(values):
        return values
    result = np.copy(values)
    for i in range(len(values) - window + 1):
        result[i + window - 1] = np.mean(values[i:i+window])
    return result

def inverse_close(scaled_values, scaler):
    """
    Inverse normalization for predicted Close value.
    Assumes that the scaler was fit on 5 features and Close is at index 3.
    """
    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]
    return scaled_values * close_std + close_mean

# --- Main Inference Function ---

def get_prediction(use_cache=True):
    """
    Generate price predictions.
    1. Fetch the latest (WINDOW_SIZE + max(PREDICTION_OFFSETS)) candles.
    2. Use the most recent WINDOW_SIZE candles (all 5 features) as model input.
    3. The model outputs scaled Close values for each offset.
    4. Inverse transform using the Close parameters.
    """
    current_time = time.time()
    if use_cache and _cache["prediction"] is not None and (current_time - _cache["timestamp"]) < CACHE_DURATION:
        return _cache["prediction"]

    try:
        required_length = WINDOW_SIZE + max(PREDICTION_OFFSETS)
        klines = fetch_klines(SYMBOL, INTERVAL, required_length)
        features_arr = extract_features(klines)  # shape: (required_length, 5)
        if features_arr.shape[0] < WINDOW_SIZE:
            logger.error(f"Insufficient historical data: got {features_arr.shape[0]}, need {WINDOW_SIZE}")
            return {"error": "Insufficient historical data."}
        current_price = features_arr[-1, 3]  # Close of the last candle
        input_window = features_arr[-WINDOW_SIZE:]  # shape: (WINDOW_SIZE, 5)
        scaled_input = scalers["primary"].transform(input_window)  # shape: (WINDOW_SIZE, 5)
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 5)

        primary_pred_scaled = models["primary"].predict(X_input, verbose=0)
        primary_preds = primary_pred_scaled[0]  # shape: (len(PREDICTION_OFFSETS),)
        primary_preds_inv = inverse_close(primary_preds, scalers["primary"])
        
        final_preds = moving_average_smoothing(primary_preds_inv)
        adjusted_preds = final_preds + PREDICTION_ADJUSTMENT

        prediction_keys = ["x", "y", "z"]  # Corresponding to each offset
        pred_dict = {}
        for i, key in enumerate(prediction_keys):
            if i < len(adjusted_preds):
                pred_dict[key] = {
                    "after": PREDICTION_OFFSETS[i],
                    "price": float(round(adjusted_preds[i], 2))
                }
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "current_price": float(round(current_price, 2)),
            "pred": pred_dict
        }
        _cache["prediction"] = result
        _cache["timestamp"] = current_time
        logger.debug(f"Generated prediction: {pred_dict}")
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        if _cache["prediction"] is not None:
            logger.warning("Returning stale prediction")
            return _cache["prediction"]
        return {"error": str(e)}
