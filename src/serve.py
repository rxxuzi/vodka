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
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# serveモードが有効でなければ終了
if not config.get("serve", False):
    print("serveモードが無効です。vodka.json の 'serve' を true に設定してください。")
    sys.exit(0)

# --- 設定パラメータ ---
SYMBOL = config["symbol"]
INTERVAL = config["interval"]
WINDOW_SIZE = config["limit"]           # 例: 11 (過去10本＋現在)
PREDICTION_OFFSETS = config.get("prediction", [1, 5, 10])  # 予測対象。出力は固定で pred_x, pred_y, pred_z に対応

# --- モデル・スケーラーの読み込み ---
models_dir = os.path.join(BASE_DIR, "models")
model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
try:
    model = load_model(model_path)
except Exception as e:
    print(f"モデルの読み込みエラー: {e}")
    sys.exit(1)

scaler_filename = f"{config['model']}_scaler.pkl"
scaler_path = os.path.join(models_dir, scaler_filename)
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"スケーラーの読み込みエラー: {e}")
    sys.exit(1)

# --- Binance APIからローソク足データを取得 ---
def fetch_klines(symbol, interval, limit):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_close_prices(klines):
    closes = [float(kline[4]) for kline in klines]
    return np.array(closes).reshape(-1, 1)  # shape: (WINDOW_SIZE, 1)

# --- 逆正規化関数 ---
def inverse_transform(values):
    return values * scaler.scale_[0] + scaler.mean_[0]

# --- 予測処理 ---
def get_prediction():
    try:
        klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE)
    except Exception as e:
        return {"error": f"データ取得エラー: {e}"}
    
    closes = extract_close_prices(klines)
    if closes.shape[0] < WINDOW_SIZE:
        return {"error": "十分なデータが取得できませんでした。"}
    
    current_price = closes[-1, 0]
    scaled_input = scaler.transform(closes)
    X_input = scaled_input.reshape(1, WINDOW_SIZE, 1)
    
    pred_scaled = model.predict(X_input)
    preds = pred_scaled[0]  # 例: [pred_for_offset1, pred_for_offset2, pred_for_offset3]
    preds_inv = inverse_transform(preds)
    
    # 固定キー 'x', 'y', 'z' に対応（拡張性のため）
    pred_dict = {
        "x": round(preds_inv[0], 2),
        "y": round(preds_inv[1], 2),
        "z": round(preds_inv[2], 2)
    }
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "current_price": round(current_price, 2),
        "pred": pred_dict
    }

# --- Flask アプリケーションの構築 ---
# 静的ファイルはプロジェクトルートの "www" ディレクトリに配置
www_dir = os.path.join(BASE_DIR, "www")
app = Flask(__name__, static_folder=www_dir, static_url_path="")

@app.route("/")
def index():
    # www/index.html を返す
    return app.send_static_file("index.html")

@app.route("/api/")
def prediction_api():
    prediction = get_prediction()
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
