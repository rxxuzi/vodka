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
    if window <= 1 or len(values) < window:
        return values
    result = np.copy(values)
    for i in range(len(values) - window + 1):
        result[i + window - 1] = np.mean(values[i:i+window])
    return result

# --- Main Inference Function ---

def get_prediction(use_cache=True):
    """
    Generate price predictions.
    1. Fetch the latest WINDOW_SIZE candles.
    2. Use these candles as model input.
    3. The model outputs percentage changes for each offset.
    4. Calculate the predicted prices based on the current price.
    """
    current_time = time.time()
    if use_cache and _cache["prediction"] is not None and (current_time - _cache["timestamp"]) < CACHE_DURATION:
        return _cache["prediction"]

    try:
        # Fetch just enough data for the window size
        klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE)
        features_arr = extract_features(klines)  # shape: (WINDOW_SIZE, 5)
        
        if features_arr.shape[0] < WINDOW_SIZE:
            logger.error(f"Insufficient historical data: got {features_arr.shape[0]}, need {WINDOW_SIZE}")
            return {"error": "Insufficient historical data."}
            
        current_price = features_arr[-1, 3]  # Close of the last candle
        
        # Scale the input data
        scaled_input = scalers["primary"].transform(features_arr)  # shape: (WINDOW_SIZE, 5)
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 5)
        
        # Get raw predictions (these are already percentage changes as trained in train.py)
        primary_pred = models["primary"].predict(X_input, verbose=0)
        predicted_rates = primary_pred[0]  # 直接パーセンテージ値が出力される
        
        # Apply smoothing and adjustment if needed
        adjusted_preds = moving_average_smoothing(predicted_rates) + PREDICTION_ADJUSTMENT
        
        # Debug logs to check predictions
        logger.debug(f"Raw predictions: {predicted_rates}")
        logger.debug(f"Adjusted predictions: {adjusted_preds}")
        
        # Format predictions for response
        prediction_keys = ["x", "y", "z"]  # 対応するオフセット
        pred_dict = {}
        for i, key in enumerate(prediction_keys):
            if i < len(adjusted_preds):
                # 予測された変動率（%）をもとに将来価格を計算
                rate_pct = float(round(adjusted_preds[i], 2))
                predicted_price = current_price * (1 + rate_pct/100)
                
                pred_dict[key] = {
                    "after": PREDICTION_OFFSETS[i],
                    "rate": rate_pct,
                    "predicted_price": float(round(predicted_price, 2))
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
        logger.debug(f"Generated prediction: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        if _cache["prediction"] is not None:
            logger.warning("Returning stale prediction")
            return _cache["prediction"]
        return {"error": str(e)}