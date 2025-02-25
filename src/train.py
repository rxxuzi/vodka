# train.py
import os
import json
import time
import pickle
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from lime import PositionalEncoding, transformer_encoder, inverse_transform
from logger import save_model_info

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

# 使用するデータ
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features].values  # shape: (num_samples, 5)

# --- 正規化 ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# --- マルチステップデータセットの作成 ---
window_size = config["limit"]
# 予測対象は config["prediction"] で指定された分後のClose値（例: [1, 5, 10]）
prediction_offsets = config["prediction"]
max_offset = max(prediction_offsets)

X, y = [], []
for i in range(window_size, len(scaled_data) - max_offset):
    # 入力：各サンプルは複数の特徴量を持つ (window_size, 5)
    X.append(scaled_data[i - window_size:i])
    # ターゲット：未来の各時点のClose値（Closeは5列中4番目: index 3）
    y.append([scaled_data[i + offset][3] for offset in prediction_offsets])
X = np.array(X)  # shape: (samples, window_size, 5)
y = np.array(y)  # shape: (samples, len(prediction_offsets))

# --- 訓練・テストデータの分割 ---
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# --- Transformerモデルの構築 ---
d_model = 256
num_heads = 16
num_transformer_blocks = 6
ff_dim = 1024

input_tensor = Input(shape=(window_size, data.shape[1]))  # shape: (window_size, 5)
x = Dense(d_model)(input_tensor)
x = PositionalEncoding(sequence_length=window_size, d_model=d_model)(x)

# 6層の Transformer ブロックを適用
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size=64, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)

# シーケンス全体の特徴を集約
x = GlobalAveragePooling1D()(x)

# 隠れ層
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

# 出力層
output_tensor = Dense(len(prediction_offsets))(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# --- コールバックの設定 ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# --- 学習開始 ---
train_start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
train_end_time = time.time()
elapsed_time = train_end_time - train_start_time

# --- 予測と評価 ---
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆正規化処理（ターゲットはClose値）
train_predict_inv = inverse_transform(train_predict, scaler)
y_train_inv = inverse_transform(y_train, scaler)
test_predict_inv = inverse_transform(test_predict, scaler)
y_test_inv = inverse_transform(y_test, scaler)

plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:, 0], label='Actual Price (1min later)')
plt.plot(test_predict_inv[:, 0], label='Predicted Price (1min later)')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.title('Multi-step Crypto Price Forecasting (1min Prediction)')
plot_path = os.path.join(BASE_DIR, f"{config['name']}_prediction.png")
plt.savefig(plot_path)
plt.show()

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
model_file_size = os.path.getsize(model_path)

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
