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
from lime import PositionalEncoding  # カスタムレイヤーをインポート

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
INTERVAL = config["interval"]   # 例: "1m", "5m", "1h" など
WINDOW_SIZE = config["limit"]   # 例: 11 (過去10本＋最新)
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
    """
    Binance APIからローソク足データを取得する

    Args:
        symbol (str): 取引ペア (例: BTCUSDT)
        interval (str): 時間足 (例: "1m")
        limit (int): 取得するローソク足の本数

    Returns:
        list: ローソク足データ（JSON形式）
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_current_price(symbol):
    """
    最新の実際の価格を取得する

    Args:
        symbol (str): 取引ペア (例: BTCUSDT)

    Returns:
        float: 現在の価格
    """
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return float(data["price"])

def extract_features(klines):
    """
    ローソク足データから (Open, High, Low, Close, Volume) を抽出

    Args:
        klines (list): Binance APIから取得したローソク足

    Returns:
        ndarray: 形状 (N, 5) の特徴量配列
    """
    features = []
    for kline in klines:
        open_ = float(kline[1])
        high_ = float(kline[2])
        low_  = float(kline[3])
        close_ = float(kline[4])
        volume_ = float(kline[5])
        features.append([open_, high_, low_, close_, volume_])
    return np.array(features)

def inverse_transform(values):
    """
    逆正規化処理（Close値用）
    学習時に StandardScaler を fit しているため、
    scaler.mean_[3], scaler.scale_[3] が Close列に対応する。

    Args:
        values (ndarray): 正規化された予測値（Closeのみ）

    Returns:
        ndarray: 元のスケールに戻した予測値
    """
    return values * scaler.scale_[3] + scaler.mean_[3]

def get_prediction():
    """
    最新のOHLCVを使って、過去 (WINDOW_SIZE-1) 本 + 最新1本 のデータを作成し、
    学習済みモデルで推論を行う。
    """
    try:
        # 1) 過去 (WINDOW_SIZE - 1) 本のローソク足を取得
        historical_klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE - 1)
        historical_features = extract_features(historical_klines)
        if historical_features.shape[0] < (WINDOW_SIZE - 1):
            return {"error": "Insufficient historical data."}

        # 2) 最新の実際価格を取得
        current_price = fetch_current_price(SYMBOL)
        # 直前のVolumeを再利用 (または 0 にする)
        last_volume = historical_features[-1, 4]

        # 3) 最新の行を作成 (Open,High,Low,Close = current_price, Volume = last_volume)
        current_row = np.array([current_price, current_price, current_price, current_price, last_volume])

        # 4) historical_features + current_row → (WINDOW_SIZE, 5)
        window_features = np.vstack([historical_features, current_row])

        # デバッグ用にログを出す
        print("Window features (last row):", window_features[-1])

        # 5) スケーリングしてモデル入力を作成
        scaled_input = scaler.transform(window_features)  # shape: (WINDOW_SIZE, 5)
        if scaled_input.shape[0] != WINDOW_SIZE:
            return {"error": f"Unexpected window size: got {scaled_input.shape[0]}, expected {WINDOW_SIZE}"}

        # (1, WINDOW_SIZE, 5) に reshape
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 5)

        # 6) 推論
        pred_scaled = model.predict(X_input)  # shape: (1, len(PREDICTION_OFFSETS))
        preds = pred_scaled[0]               # shape: (len(PREDICTION_OFFSETS),)

        # 7) Close値のみを逆正規化
        preds_inv = inverse_transform(preds)  # shape: (len(PREDICTION_OFFSETS),)

        # 8) JSONで返す形式に整形
        pred_dict = {}
        offset_keys = ["x", "y", "z"]
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
            "current_price": float(round(window_features[-1, 3], 2)),  # 最新Close
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
