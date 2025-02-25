#!/usr/bin/env python3
# fetch.py

import os
import time
import requests
import pandas as pd
import json
from dotenv import load_dotenv

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# .envファイルが存在すれば読み込む
load_dotenv()

# --- 設定パラメータ ---
symbol = config.get("symbol", "BTCUSDT")
interval = config.get("interval", "1m")
days = config.get("fetch_days", 21)  # 取得する日数
limit = 1000  # APIの制限

def get_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    """
    Binance APIからkline（ローソク足）データを取得する。

    引数:
        symbol (str): 取引ペア（例: BTCUSDT）。
        interval (str): 時間間隔（例: 1m）。
        start_time (int): 開始時間（ミリ秒）。
        end_time (int): 終了時間（ミリ秒）。
        limit (int): API呼び出しごとの最大取得数。

    戻り値:
        list: Binance APIからのJSONデータ。
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_time,
        "endTime": end_time
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Binance API エラー: {response.status_code} - {response.text}")
    return response.json()

def process_klines_to_df(data):
    """
    Binance APIから取得したklineデータをPandas DataFrameに変換する。

    引数:
        data (list): Binance APIから取得した生データ。

    戻り値:
        DataFrame: 変換されたデータ（モデル学習向けに適切なカラムを含む）。
    """
    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    df = pd.DataFrame(data, columns=columns)
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    # 数値カラムを適切なデータ型に変換
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote Asset Volume', 'Number of Trades', 
                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # 'Ignore'カラムは不要なため削除
    if 'Ignore' in df.columns:
        df = df.drop(columns=['Ignore'])
    return df

def fetch_historical_data(symbol="BTCUSDT", interval="1m", days=7):
    """
    Binance APIから過去のローソク足データを取得し、CSVファイルとして保存する。
    取得するデータには複数の特徴量（Open, High, Low, Close など）が含まれ、
    モデルの学習精度向上に活用できる。

    引数:
        symbol (str): 取引ペア。
        interval (str): 時間間隔。
        days (int): 取得する日数。
    """
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    all_data = []
    while start_time < end_time:
        try:
            raw_data = get_binance_klines(symbol, interval, start_time, end_time, limit)
            if not raw_data:
                break
            df = process_klines_to_df(raw_data)
            all_data.append(df)
            # 次回取得開始時間を更新（最後のClose Time + 1ミリ秒）
            start_time = int(df['Close Time'].iloc[-1].timestamp() * 1000) + 1
            print(f"データ取得完了: {df['Open Time'].iloc[0]} - {df['Open Time'].iloc[-1]}")
            time.sleep(1)
        except Exception as e:
            print(f"エラー: {e}")
            break
    if all_data:
        final_df = pd.concat(all_data)
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"crypto_data_{symbol}_{interval}.csv")
        final_df.to_csv(filename, index=False)
        print(f"CSVファイル '{filename}' を作成しました。")
    else:
        print("データを取得できませんでした。")

if __name__ == "__main__":
    fetch_historical_data(symbol=symbol, interval=interval, days=days)
