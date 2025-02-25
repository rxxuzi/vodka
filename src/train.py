#!/usr/bin/env python3
# train.py

import os
import json
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# lime.py内の関数をインポート（PositionalEncoding, transformer_encoder）
from lime import PositionalEncoding, transformer_encoder
from logger import save_model_info  # 学習済みモデルの情報保存用

# --- 設定ファイルの読み込み ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# --- Debugモード設定 ---
if not config.get("debug", False):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")

# --- データの読み込み ---
data_dir = os.path.join(BASE_DIR, "data")
csv_filename = f"crypto_data_{config['symbol']}_{config['interval']}.csv"
csv_path = os.path.join(data_dir, csv_filename)
df = pd.read_csv(csv_path)

# 使用するデータは複数の特徴量（Open, High, Low, Close, Volume）
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features].values  # shape: (num_samples, 5)

# --- 正規化 ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# --- マルチステップデータセットの作成 ---
window_size = config["limit"]  # 例: 16（過去15本＋現在）
prediction_offsets = config["prediction"]  # 例: [1, 5, 10]
max_offset = max(prediction_offsets)

X, y = [], []
for i in range(window_size, len(scaled_data) - max_offset):
    # 入力: 過去window_size本の全特徴量
    X.append(scaled_data[i - window_size:i])
    # 出力: 各オフセット先のClose値（featuresのうち index 3 がClose）
    y.append([scaled_data[i + offset][3] for offset in prediction_offsets])
X = np.array(X)  # shape: (samples, window_size, 5)
y = np.array(y)  # shape: (samples, len(prediction_offsets))

# --- 訓練・テストデータの分割 ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Transformerモデルの構築 ---
d_model = 128
num_heads = 4
num_transformer_blocks = 2
ff_dim = 256
dropout_rate = 0.2

input_tensor = Input(shape=(window_size, len(features)))  # shape: (window_size, 5)
x = Dense(d_model)(input_tensor)
x = PositionalEncoding(sequence_length=window_size, d_model=d_model)(x)

for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size=32, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)

x = GlobalAveragePooling1D()(x)

# 隠れ層
x = Dense(512, activation="relu")(x)
x = Dropout(dropout_rate)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(dropout_rate)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(dropout_rate)(x)

# 出力層: 予測対象の3つの時間オフセットに対するClose値
output_tensor = Dense(len(prediction_offsets))(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['mae'])
model.summary()

# --- コールバックの設定 ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# --- 学習開始 ---
train_start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
train_end_time = time.time()
elapsed_time = train_end_time - train_start_time

# --- 予測と評価 ---
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 予測値（およびラベル）はスケーリング時、Close列（index 3）のみを対象としているので、
# 手動で逆正規化を行う。
def inverse_close(scaled_values, scaler):
    # scaler.mean_[3] と scaler.scale_[3] がCloseの平均・標準偏差
    return scaled_values * scaler.scale_[3] + scaler.mean_[3]

train_predict_inv = inverse_close(train_predict, scaler)
y_train_inv = inverse_close(y_train, scaler)
test_predict_inv = inverse_close(test_predict, scaler)
y_test_inv = inverse_close(y_test, scaler)

# --- 学習曲線の可視化 ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
loss_plot_path = os.path.join(BASE_DIR, f"{config['name']}_loss_curve.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"Loss curve saved to '{loss_plot_path}'.")

# --- 学習済みモデルとスケーラーの保存 ---
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_filename = f"{config['model']}.h5"
model_path = os.path.join(models_dir, model_filename)
model.save(model_path)
print(f"Trained model saved to '{model_path}'.")

scaler_filename = f"{config['model']}_scaler.pkl"
scaler_path = os.path.join(models_dir, scaler_filename)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to '{scaler_path}'.")

# --- モデル情報の保存 ---
save_model_info(
    model_name=config['model'],
    training_time=elapsed_time,
    total_epochs=len(history.history['loss']),
    history=history.history,
    model_size=model.count_params(),
    dataset_name=csv_filename,
    input_shape=X_train.shape[1:],
    output_shape=y_train.shape[1:],
    features=features
)
