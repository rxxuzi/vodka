# train.py
# 
# This script trains a Transformer-based model for cryptocurrency price prediction.
# It reads historical market data, processes it, and trains a model to predict future price changes.
#
# Version : 2.0.0

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

from lime import PositionalEncoding, transformer_encoder
from logger import save_model_info

def run():
    # --- 設定ファイルの読み込み ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # --- 使用する Transformer の設定を取得 ---
    transformer_config = config["transformer"]
    d_model = transformer_config["d_model"]
    num_heads = transformer_config["num_heads"]
    num_transformer_blocks = transformer_config["num_transformer_blocks"]
    ff_dim = transformer_config["ff_dim"]
    dropout_rate = transformer_config["dropout_rate"]
    lr = transformer_config["lr"]
    hidden_layer = transformer_config["hidden_layer"]
    epochs = transformer_config["epochs"]
    batch_size = transformer_config["batch_size"]
    optimizer = transformer_config["optimizer"]
    min_lr = 1e-5

    # --- Debugモード設定 ---
    if not config.get("debug", False):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")

    # --- データの読み込み ---
    data_dir = os.path.join(BASE_DIR, "data")
    csv_filename = f"crypto_data_{config['symbol']}_{config['interval']}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    df = pd.read_csv(csv_path)

    # 使用するデータ（ここではシンプルにOHLCVのみを使用）
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values  # shape: (num_samples, 5)

    # --- 正規化 ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # --- マルチステップデータセットの作成 ---
    window_size = config["limit"]
    prediction_offsets = config["prediction"]
    max_offset = max(prediction_offsets)

    X, y = [], []
    # 最大のオフセットに合わせてループの終了条件を変更
    for i in range(window_size, len(data) - max_offset + 1):
        X.append(scaled_data[i - window_size:i])
        current_close = data[i - 1][3]
        # ターゲットは、直近終値を基準に未来の終値との差分を百分率で算出
        y.append([ ((data[i + offset - 1][3] - current_close) / current_close) * 100 for offset in prediction_offsets ])

    X = np.array(X)
    y = np.array(y)

    # --- 訓練・テストデータの分割 ---
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # --- Transformerモデルの構築 ---
    input_tensor = Input(shape=(window_size, len(features)))
    x = Dense(d_model)(input_tensor)
    x = PositionalEncoding(sequence_length=window_size, d_model=d_model)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size=32, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate)

    x = GlobalAveragePooling1D()(x)

    # 隠れ層
    x = Dense(hidden_layer, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_layer // 2, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_layer // 4, activation="relu")(x)
    x = Dropout(dropout_rate)(x)

    # 出力層（各オフセットごとに1つの予測値を出力）
    output_tensor = Dense(len(prediction_offsets))(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])
    model.summary()

    # --- コールバックの設定 ---
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=min_lr)

    # --- 学習開始 ---
    train_start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    train_end_time = time.time()
    elapsed_time = train_end_time - train_start_time

    # --- 予測と評価 ---
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # （学習時、ターゲットは既に百分率になっているので追加の掛け算は不要）

    # --- 学習履歴の保存 ---
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    train_log_path = os.path.join(log_dir, "train_log.json")
    train_log = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_time_sec": elapsed_time,
        "epochs_completed": len(history.history["loss"])
    }
    with open(train_log_path, "w") as f:
        json.dump(train_log, f, indent=4)
    print(f"Training log saved to '{train_log_path}'.")

    # --- モデル設定の保存 ---
    model_summary_path = os.path.join(log_dir, "model_summary.json")
    model_summary = {
        "model_name": config["model"],
        "parameters": model.count_params(),
        "training_time": elapsed_time,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer,
        "learning_rate": lr,
        "transformer_config": transformer_config
    }
    with open(model_summary_path, "w") as f:
        json.dump(model_summary, f, indent=4)
    print(f"Model summary saved to '{model_summary_path}'.")

    # --- vodka.json のバックアップ ---
    config_backup_path = os.path.join(log_dir, "vodka_config_used.json")
    with open(config_backup_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration backup saved to '{config_backup_path}'.")

    # --- 学習曲線の可視化 ---
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    loss_plot_path = os.path.join(log_dir, f"{config['name']}_loss_curve.png")
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curve saved to '{loss_plot_path}'")

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

if __name__ == "__main__":
    run()
