# logger.py
import os
import platform
import time
import tensorflow as tf

def get_device_info():
    """
    利用可能なハードウェア情報（CPU / GPU）を取得

    Returns:
        str: デバイス情報
    """
    device_name = tf.config.list_physical_devices('GPU')
    if device_name:
        return f"Device: {device_name[0].name}"
    return f"Device: CPU ({platform.processor()})"

def save_model_info(model_name, training_time, total_epochs, history, model_size, dataset_name, input_shape, output_shape, features):
    """
    学習済みモデルの情報を `models/{model_name}.info` に保存する。

    Args:
        model_name (str): モデル名
        training_time (float): 学習時間（秒）
        total_epochs (int): 実際に学習したエポック数
        history (dict): 訓練履歴 (`history.history`)
        model_size (int): モデルファイルのサイズ（バイト）
        dataset_name (str): 使用したデータセットのファイル名
        input_shape (tuple): 入力データの形状
        output_shape (tuple): 出力データの形状
        features (list): 使用した特徴量のリスト
    """
    models_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    info_path = os.path.join(models_dir, f"{model_name}.info")

    # 最終的な損失値
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]

    # ハードウェア情報
    device_info = get_device_info()

    # ログフォーマット
    info_text = f"""
Vodka Project 2025.
Model Training Information
===========================
Model Name: {model_name}
Training Time (seconds): {training_time:.2f}
Total Epochs: {total_epochs}
Final Training Loss: {final_train_loss:.6f}
Final Validation Loss: {final_val_loss:.6f}
Number of Parameters: {model_size}

Dataset Information
-------------------
Dataset: {dataset_name}
Input Shape: {input_shape}
Output Shape: {output_shape}
Features Used: {features}
Normalization: StandardScaler

Training Configuration
----------------------
Optimizer: Adam (learning_rate=0.001)
Loss Function: Mean Squared Error (MSE)
EarlyStopping: patience=10, restore_best_weights=True
ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6

Hardware Information
--------------------
{device_info}
""".strip()

    # ファイルに保存
    with open(info_path, "w") as f:
        f.write(info_text)

    print(f"Model information saved to '{info_path}'.")
