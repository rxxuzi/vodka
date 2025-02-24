import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

def get_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    """
    Binance APIから指定のシンボル・インターバルのローソク足データを取得する。
    start_time, end_time を指定して過去データを取得可能。
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
        raise Exception(f"Binance APIエラー: {response.status_code} - {response.text}")
    return response.json()

def process_klines_to_df(data):
    """
    取得したローソク足データをDataFrameに変換し、前処理を行う。
    """
    columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
    ]
    df = pd.DataFrame(data, columns=columns)

    # ミリ秒のタイムスタンプを日時に変換
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    # 数値カラムの型変換
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote Asset Volume', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def fetch_historical_data(symbol="BTCUSDT", interval="1m", days=7):
    """
    指定した期間（days）の過去データを取得し、CSVに保存する。
    """
    end_time = int(time.time() * 1000)  # 現在時刻（ミリ秒）
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # 過去N日分の開始時間（ミリ秒）

    all_data = []

    while start_time < end_time:
        try:
            raw_data = get_binance_klines(symbol, interval, start_time, end_time)
            if not raw_data:
                break  # 取得データが空なら終了

            df = process_klines_to_df(raw_data)
            all_data.append(df)

            # 次の開始時間を更新（最後のタイムスタンプ + 1ms）
            start_time = int(df['Close Time'].iloc[-1].timestamp() * 1000) + 1
            print(f"取得完了: {df['Open Time'].iloc[0]} - {df['Open Time'].iloc[-1]}")

            time.sleep(1)  # API制限対策（1秒スリープ）

        except Exception as e:
            print(f"エラー: {e}")
            break

    # すべてのデータを結合し、プロジェクトルートの data フォルダに保存
    if all_data:
        final_df = pd.concat(all_data)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"crypto_data_{symbol}_{interval}.csv")
        final_df.to_csv(filename, index=False)
        print(f"CSVファイル '{filename}' を生成しました。")
    else:
        print("データが取得できませんでした。")

if __name__ == "__main__":
    load_dotenv()
    fetch_historical_data(symbol="BTCUSDT", interval="1m", days=7)  # 過去7日分の1分足を取得
