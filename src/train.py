#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import pickle

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- Debug モード設定 ---
if not config.get("debug", False):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

# --- データの準備 ---
data_dir = os.path.join(BASE_DIR, "data")
csv_filename = f"crypto_data_{config['symbol']}_{config['interval']}.csv"
csv_path = os.path.join(data_dir, csv_filename)

df = pd.read_csv(csv_path)

# マルチステップ予測用は「Close」のみを使用
data = df[['Close']].values  # shape: (n, 1)

# --- 正規化処理 ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# --- マルチステップデータセットの作成 ---
# 入力ウィンドウサイズは config["limit"]（例：11 = 過去10本＋現在）
window_size = config["limit"]
# 予測対象：現在の次（1分後）、5分後、10分後
# ※入力ウィンドウの最後の値が「現在」の値とし、ターゲットはそれぞれ
#    index: i, i+4, i+9  （※各ローソク足は1分間隔）
prediction_offsets = [0, 4, 9]
max_offset = max(prediction_offsets)

X, y = [], []
for i in range(window_size, len(scaled_data) - max_offset):
    X.append(scaled_data[i-window_size:i])  # shape: (window_size, 1)
    # 各ターゲットは、入力終了直後（1分後）～の価格を取得
    y.append([scaled_data[i + offset][0] for offset in prediction_offsets])
X = np.array(X)  # shape: (samples, window_size, 1)
y = np.array(y)  # shape: (samples, 3)

# --- 訓練・テストデータの分割 ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- モデルの構築 ---
model = Sequential()
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(window_size, 1)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(3))  # 出力ユニット数 = 3（1分後、5分後、10分後）
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- 予測と評価 ---
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆正規化（StandardScaler の inverse_transform: value * std + mean）
close_mean = scaler.mean_[0]
close_std = scaler.scale_[0]
def inverse_transform(values):
    return values * close_std + close_mean

train_predict_inv = inverse_transform(train_predict)
y_train_inv = inverse_transform(y_train)
test_predict_inv = inverse_transform(test_predict)
y_test_inv = inverse_transform(y_test)

plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:,0], label='Actual price (1min later)')
plt.plot(test_predict_inv[:,0], label='Predicted price (1min later)')
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.legend()
plt.title('Multi-step Crypto Price Forecasting (1min prediction)')
plot_path = os.path.join(BASE_DIR, f"{config['name']}_prediction.png")
plt.savefig(plot_path)
plt.show()

# --- 学習済みモデルとスケーラーの保存 ---
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
model.save(model_path)
print(f"学習済みモデルを '{model_path}' に保存しました。")

scaler_filename = f"{config['model']}_scaler.pkl"
scaler_path = os.path.join(models_dir, scaler_filename)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"スケーラーを '{scaler_path}' に保存しました。")
