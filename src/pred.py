#!/usr/bin/env python3
import os
import sys
import time
import select
import requests
import numpy as np
import pickle
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- 設定パラメータ ---
SYMBOL = config["symbol"]
INTERVAL = config["interval"]
WINDOW_SIZE = config["limit"]  # 例: 11本
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, config["pred_output"])

# CSVがなければヘッダー作成
if not os.path.exists(CSV_OUTPUT_PATH):
    with open(CSV_OUTPUT_PATH, "w") as f:
        f.write("timestamp,current_price,pred_1min,pred_5min,pred_10min\n")

console = Console()
console.print(Panel("Vodka : Starts a multi-step forecast. \ߋn exit with 'q' or Ctrl+C",
                     title="Multi-Step Prediction Bot", style="bold green"))

# --- モデル・スケーラーの読み込み ---
models_dir = os.path.join(BASE_DIR, "models")
model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
try:
    model = load_model(model_path)
except Exception as e:
    console.print(f"[red]モデルの読み込みエラー:[/red] {e}")
    sys.exit(1)

scaler_filename = f"{config['model']}_scaler.pkl"
scaler_path = os.path.join(models_dir, scaler_filename)
try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    console.print(f"[red]スケーラーの読み込みエラー:[/red] {e}")
    sys.exit(1)

# --- Binance APIからローソク足データを取得する関数 ---
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

# --- 予測ループ ---
try:
    while True:
        try:
            klines = fetch_klines(SYMBOL, INTERVAL, WINDOW_SIZE)
        except Exception as e:
            console.print(f"[red]データ取得エラー:[/red] {e}")
            time.sleep(60)
            continue
        
        closes = extract_close_prices(klines)
        if closes.shape[0] < WINDOW_SIZE:
            console.print("[red]十分なデータが取得できませんでした。[/red]")
            time.sleep(60)
            continue
        
        # 現在の価格は最新の終値
        current_price = closes[-1, 0]
        
        # 前処理：正規化と形状調整
        scaled_input = scaler.transform(closes)
        X_input = scaled_input.reshape(1, WINDOW_SIZE, 1)
        
        # マルチステップ予測（出力 shape: (1, 3)）
        pred_scaled = model.predict(X_input)
        preds = pred_scaled[0]  # 3値
        preds_inv = inverse_transform(preds)
        pred_1min, pred_5min, pred_10min = preds_inv
        
        # カラー設定：予測値が現在より上なら緑、下なら赤
        color_1min = "green" if pred_1min > current_price else "red"
        color_5min = "green" if pred_5min > current_price else "red"
        color_10min = "green" if pred_10min > current_price else "red"
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        table = Table(title="マルチステップ予測結果", box=box.ROUNDED)
        table.add_column("項目", style="cyan", no_wrap=True)
        table.add_column("値", style="magenta")
        table.add_row("現在の価格", f"{current_price:.2f} USD")
        table.add_row("1分後予測", f"[{color_1min}]{pred_1min:.2f} USD[/{color_1min}]")
        table.add_row("5分後予測", f"[{color_5min}]{pred_5min:.2f} USD[/{color_5min}]")
        table.add_row("10分後予測", f"[{color_10min}]{pred_10min:.2f} USD[/{color_10min}]")
        table.add_row("実行時刻", now)
        
        console.clear()
        console.print(table)
        
        # CSVに結果を追記
        with open(CSV_OUTPUT_PATH, "a") as f:
            f.write(f"{now},{current_price:.2f},{pred_1min:.2f},{pred_5min:.2f},{pred_10min:.2f}\n")
        
        console.print("[yellow]次回予測まで 60 秒待機中... ('q' + Enter で終了) [/yellow]")
        for _ in range(60):
            if sys.stdin in select.select([sys.stdin], [], [], 1)[0]:
                line = sys.stdin.readline().strip()
                if line.lower() == "q":
                    console.print("[red]ユーザーにより終了要求を受けました。[/red]")
                    sys.exit(0)
except KeyboardInterrupt:
    console.print("\n[red]Ctrl+C により終了します。[/red]")
