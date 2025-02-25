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
from lime import PositionalEncoding  # 共通のカスタムレイヤー

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
INTERVAL = config["interval"]  # 例: "1m", "10s", "1h" など
WINDOW_SIZE = config["limit"]  # 例: 11 (過去10本＋現在)
PREDICTION_OFFSETS = config.get("prediction", [1, 5, 10])

# --- モデル・スケーラーの読み込み ---
models_dir = os.path.join(BASE_DIR, "models")
model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
try:
    # カスタムレイヤーを custom_objects に渡してモデルをロード
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

# --- Binance APIからローソク足データを取得 ---
def fetch_klines(symbol, interval, limit):
    """
    Binance APIからローソク足データを取得する

    Args:
        symbol (str): Trading pair (例: BTCUSDT)
        interval (str): 時間足（例: "1m"）
        limit (int): 取得件数

    Returns:
        list: ローソク足データ
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_features(klines):
    """
    Binance APIのローソク足データから、必要な特徴量(Open, High, Low, Close, Volume)を抽出する

    Args:
        klines (list): Binance APIから取得したローソク足データ

    Returns:
        ndarray: 特徴量データ、形状は (WINDOW_SIZE, 5)
    """
    features = []
    # kline の各項目は次の順番:
    # 0: Open Time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, ...
    for kline in klines:
        features.append([
            float(kline[1]),  # Open
            float(kline[2]),  # High
            float(kline[3]),  # Low
            float(kline[4]),  # Close
            float(kline[5])   # Volume
        ])
    return np.array(features)

# --- 逆正規化関数 ---
def inverse_transform(values):
    """
    逆正規化処理（Close値用）
    
    Args:
        values (ndarray): 正規化された値

    Returns:
        ndarray: 逆正規化された値
    """
    # scalerは5列のデータに対してfitされているので、Closeは4番目（index 3）
    return values * scaler.scale_[3] + scaler.mean_[3]

# --- 予測処理 ---
def get_prediction():
    try:
        klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE)
        features_array = extract_features(klines)
        if features_array.shape[0] < WINDOW_SIZE:
            return {"error": "Insufficient data."}
        current_price = float(features_array[-1, 3])  # Close値を現在価格として使用

        # scalerは5次元データを前提としている
        scaled_input = scaler.transform(features_array)
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 5)
        pred_scaled = model.predict(X_input)
        preds = pred_scaled[0]
        preds_inv = inverse_transform(preds)
        pred_dict = {
            "x": {"after": PREDICTION_OFFSETS[0], "price": float(round(preds_inv[0], 2))},
            "y": {"after": PREDICTION_OFFSETS[1], "price": float(round(preds_inv[1], 2))},
            "z": {"after": PREDICTION_OFFSETS[2], "price": float(round(preds_inv[2], 2))}
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

# --- Flask アプリケーションの構築 ---
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
