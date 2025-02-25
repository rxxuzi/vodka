#!/usr/bin/env python3
# serve.py

import os
import sys
import requests
import numpy as np
import pickle
import json
from datetime import datetime
from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from lime import PositionalEncoding  # カスタムレイヤー

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# serveモードが有効でなければ終了
if not config.get("serve", False):
    print("Serve mode is disabled. Please set 'serve' to true in vodka.json.")
    sys.exit(0)

# --- 設定パラメータ ---
SYMBOL = config["symbol"]
INTERVAL = config["interval"]
WINDOW_SIZE = config["limit"]
PREDICTION_OFFSETS = config.get("prediction", [1, 5, 10])

# --- モデル・スケーラーの読み込み ---
models_dir = os.path.join(BASE_DIR, "models")
model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
try:
    # PositionalEncoding を custom_objects に渡す
    model = load_model(model_path, custom_objects={"PositionalEncoding": PositionalEncoding})
except Exception as e:
    print(f"Model loading error: {e}")
    sys.exit(1)

scaler_filename = f"{config['model']}_scaler.pkl"
scaler_path = os.path.join(models_dir, scaler_filename)
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Scaler loading error: {e}")
    sys.exit(1)

def fetch_klines(symbol, interval, limit):
    """Binance APIからローソク足データを取得"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_close(klines):
    """ローソク足データからClose値のみを抽出（インデックス4）"""
    closes = []
    for kline in klines:
        closes.append(float(kline[4]))
    return np.array(closes)

def inverse_transform(values, scaler):
    """逆正規化（Closeのみの場合）"""
    close_mean = scaler.mean_[0]
    close_std = scaler.scale_[0]
    return values * close_std + close_mean

def get_prediction():
    try:
        # 過去の確定済みローソク足をWINDOW_SIZE本取得（最新分を含む）
        klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE)
        closes = extract_close(klines)  # shape: (WINDOW_SIZE,)
        
        if closes.shape[0] < WINDOW_SIZE:
            return {"error": "Insufficient historical data."}
        
        # 現在の価格 = 取得した最後のローソク足のClose値
        current_price = closes[-1]
        
        # (WINDOW_SIZE, 1) に整形してスケーリング
        closes_2d = closes.reshape(-1, 1)  
        scaled_input = scaler.transform(closes_2d)
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 1)
        
        # 推論
        pred_scaled = model.predict(X_input)
        preds = pred_scaled[0]  # shape: (len(PREDICTION_OFFSETS),)
        preds_inv = inverse_transform(preds, scaler)
        
        # 結果の整形
        offset_keys = ["x", "y", "z"]  # 3つの場合
        pred_dict = {}
        for i, offset_key in enumerate(offset_keys):
            if i < len(preds_inv):
                pred_dict[offset_key] = {
                    "after": PREDICTION_OFFSETS[i],
                    "price": float(round(preds_inv[i], 2))
                }
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "current_price": float(round(current_price, 2)),
            "pred": pred_dict
        }
    except Exception as e:
        return {"error": f"Prediction error: {e}"}

# --- Flaskアプリの構築 ---
www_dir = os.path.join(BASE_DIR, "www")
app = Flask(__name__, static_folder=www_dir, static_url_path="")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/")
def prediction_api():
    prediction = get_prediction()
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
