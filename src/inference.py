# inference.py

import os
import time
import requests
import numpy as np
import pickle
import json
import logging
import pandas as pd
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
    """
    Load the trained model, scaler, and any additional configuration needed for inference.
    """
    models = {}
    scalers = {}
    model_config = {}
    
    model_filename = f"{MODEL_NAME}.h5"
    model_path = os.path.join(models_dir, model_filename)
    
    scaler_filename = f"{MODEL_NAME}_scaler.pkl"
    scaler_path = os.path.join(models_dir, scaler_filename)
    
    # Try to load adjustment configuration (added in new training script)
    adj_filename = f"{MODEL_NAME}_adjustment.json"
    adj_path = os.path.join(models_dir, adj_filename)
    
    try:
        # Load model
        models["primary"] = load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
        logger.debug("Primary model loaded.")
        
        # Load scaler
        with open(scaler_path, "rb") as f:
            scalers["primary"] = pickle.load(f)
        logger.debug("Primary scaler loaded.")
        
        # Try to load adjustment config if it exists
        if os.path.exists(adj_path):
            with open(adj_path, "r") as f:
                model_config = json.load(f)
            logger.debug(f"Found adjustment config: {model_config}")
        else:
            # Default values if file doesn't exist
            model_config = {
                "target_amplification": 1.0,
                "use_log_returns": False
            }
            logger.debug("No adjustment config found, using defaults.")
        
        # Try to load metadata if it exists
        meta_path = os.path.join(models_dir, f"{MODEL_NAME}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            model_config.update(meta)
            logger.debug(f"Loaded model metadata: {meta}")
        
    except Exception as e:
        logger.error(f"Error loading model, scaler, or config: {e}")
        raise e
    
    return models, scalers, model_config

models, scalers, model_config = load_models_and_scalers()

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

def extract_features(klines, model_config):
    """
    Extract features from kline data based on model configuration.
    
    Args:
        klines (list): Raw kline data from Binance API.
        model_config (dict): Model configuration with feature information.
        
    Returns:
        np.ndarray: Extracted features.
    """
    # Default features (OHLCV)
    default_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Get features from config if available
    features = model_config.get("features", default_features)
    
    # Map between feature names and kline indices
    feature_map = {
        'Open': 1,
        'High': 2,
        'Low': 3,
        'Close': 4,
        'Volume': 5
    }
    
    # Prepare basic features
    basic_features = []
    for kline in klines:
        row = []
        for feature in features:
            if feature in feature_map:
                row.append(float(kline[feature_map[feature]]))
            else:
                # For technical indicators, we'll calculate them separately
                pass
        basic_features.append(row)
    
    basic_features = np.array(basic_features)
    
    # Calculate technical indicators if needed
    if any(f not in feature_map for f in features):
        # For real-time inference, we need to calculate technical indicators
        # This is a simplified version - you may need to expand based on which indicators you use
        df = pd.DataFrame(basic_features, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Calculate technical indicators
        if 'SMA_5' in features:
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
        if 'SMA_10' in features:
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
        if 'SMA_20' in features:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        if 'EMA_5' in features:
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        if 'EMA_10' in features:
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        if 'EMA_20' in features:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        if 'TP' in features:
            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        if 'VWAP_5' in features and 'TP' in df.columns:
            df['VWAP_5'] = (df['TP'] * df['Volume']).rolling(window=5).sum() / df['Volume'].rolling(window=5).sum()
        
        # Fill NaN values that occur during indicator calculation
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # Extract all features in the correct order
        all_features = []
        for i in range(len(df)):
            row = []
            for feature in features:
                row.append(df[feature].iloc[i])
            all_features.append(row)
        
        return np.array(all_features)
    
    return basic_featuress