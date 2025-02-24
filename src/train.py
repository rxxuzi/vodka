#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle

# =============================
# Debug モード設定
# =============================
debug = False  # False にするとデバッグメッセージ非表示

if not debug:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # TensorFlowの警告や情報を非表示
    tf.get_logger().setLevel("ERROR")         # TensorFlowのログレベルをERRORに設定

# =============================
# データの準備
# =============================
# プロジェクトルートからdataフォルダのパスを取得
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
csv_path = os.path.join(data_dir, "crypto_data_BTCUSDT_1m.csv")

# CSVファイルの読み込み
df = pd.read_csv(csv_path)
data = df[['Close']].values

# =============================
# 正規化処理
# =============================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# =============================
# 時系列データの作成
# =============================
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_dataset(scaled_data, window_size)
# LSTMの入力形式に合わせてリシェイプ
X = X.reshape(X.shape[0], X.shape[1], 1)

# =============================
# 訓練・テストデータの分割
# =============================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =============================
# LSTMモデルの構築
# =============================
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stopping の設定（val_lossが改善しなければ学習を早期終了）
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# =============================
# モデルの学習
# =============================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1  # 進捗バーやloss、epochのみ表示
)

# =============================
# 予測と評価
# =============================
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆正規化して元のスケールに戻す
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# =============================
# 予測結果の可視化
# =============================
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual closing price')
plt.plot(test_predict, label='Estimated Closing Price')
plt.xlabel('Timestep')
plt.ylabel('Price')
plt.legend()
plt.title('Crypto Price Forecasting')
plt.savefig(os.path.join(base_dir, "lstm_crypto_prediction.png"))
plt.show()

# =============================
# 学習済みモデルとスケーラーの保存
# =============================
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# モデルの保存
model_path = os.path.join(models_dir, "vodka_model.h5")
model.save(model_path)
print(f"学習済みモデルを '{model_path}' に保存しました。")

# スケーラーの保存（学習時に使用したパラメータを保存）
scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"スケーラーを '{scaler_path}' に保存しました。")
