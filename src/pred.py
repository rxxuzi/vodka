#!/usr/bin/env python3
# pred.py
import os
import sys
import time
import select
import requests
import numpy as np
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# 設定値
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 61  # 最新のローソク足61本：直近60本を入力、61本目を実際の価格とする

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, "pred.csv")

# CSVヘッダーが無ければ作成
if not os.path.exists(CSV_OUTPUT_PATH):
    with open(CSV_OUTPUT_PATH, "w") as f:
        f.write("timestamp,actual_price,predicted_price,error_rate\n")

def fetch_klines(symbol: str, interval: str, limit: int) -> list:
    """
    Binance APIからローソク足データを取得する。
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def extract_close_prices(klines: list) -> np.array:
    """
    klineデータから「Close」価格を抽出し、float型のnumpy配列に変換する。
    """
    closes = [float(kline[4]) for kline in klines]
    return np.array(closes)

def main():
    console = Console()
    console.print(Panel("Binance APIから価格予測を開始します。\n終了は 'q' または Ctrl+C で", title="Prediction Job", style="bold green"))

    # モデル・スケーラーのパス設定（プロジェクトルート/models）
    model_path = os.path.join(BASE_DIR, "models", "vodka_model.h5")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    # 学習済みモデルの読み込み
    try:
        model = load_model(model_path)
    except Exception as e:
        console.print(f"[red]モデルの読み込みエラー:[/red] {e}")
        return

    # 学習時に使用したスケーラーの読み込み
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        console.print(f"[red]スケーラーの読み込みエラー:[/red] {e}")
        return

    console.print("[green]予測処理を開始します。[/green]")

    try:
        while True:
            # Binance APIから最新のローソク足データ（61本）を取得
            try:
                klines = fetch_klines(SYMBOL, INTERVAL, LIMIT)
            except Exception as e:
                console.print(f"[red]データ取得エラー:[/red] {e}")
                time.sleep(60)
                continue

            closes = extract_close_prices(klines)
            if closes.shape[0] < LIMIT:
                console.print("[red]十分なデータが取得できませんでした。[/red]")
                time.sleep(60)
                continue

            # 最新60本を入力データ、61本目を実際の価格とする
            input_data = closes[:60]
            actual_price = closes[60]

            # 前処理：正規化および形状変換
            input_data = input_data.reshape(-1, 1)
            scaled_input = scaler.transform(input_data)
            X_input = scaled_input.reshape(1, 60, 1)

            # 予測実行
            predicted_scaled = model.predict(X_input)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

            # 誤差率（％）： (予測 - 実際) / 実際 × 100
            error_rate = (predicted_price - actual_price) / actual_price * 100

            # 現在時刻
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Richテーブルで結果を表示
            table = Table(title="価格予測結果", box=box.ROUNDED)
            table.add_column("項目", style="cyan", no_wrap=True)
            table.add_column("値", style="magenta")
            table.add_row("実際の価格", f"{actual_price:.2f} USD")
            table.add_row("予測価格", f"{predicted_price:.2f} USD")
            table.add_row("誤差率", f"{error_rate:.2f} %")
            table.add_row("実行時刻", now)
            console.clear()
            console.print(table)

            # CSVに結果を追記
            with open(CSV_OUTPUT_PATH, "a") as f:
                f.write(f"{now},{actual_price:.2f},{predicted_price:.2f},{error_rate:.2f}\n")

            console.print("[yellow]次回の予測まで 60 秒待機中... ('q' + Enter で終了) [/yellow]")

            # 60秒間、qキーが押されたかをチェック（1秒刻みで確認）
            for _ in range(60):
                # 非ブロッキングで標準入力をチェック
                if sys.stdin in select.select([sys.stdin], [], [], 1)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() == "q":
                        console.print("[red]ユーザーにより終了要求を受けました。[/red]")
                        return
            # 1分後に次回予測ループへ
    except KeyboardInterrupt:
        console.print("\n[red]Ctrl+C により終了します。[/red]")

if __name__ == "__main__":
    main()
