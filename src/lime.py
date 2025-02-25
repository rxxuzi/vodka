# lime.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add

class PositionalEncoding(tf.keras.layers.Layer):
    """
    位置エンコーディングレイヤー

    入力系列に対して、各時刻の位置情報をsinとcosを用いてエンコードし、加算する。

    Attributes:
        sequence_length (int): 入力系列の長さ。
        d_model (int): 埋め込み次元数。
        pos_encoding (Tensor): 事前計算された位置エンコーディング。
    """
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config

    def _positional_encoding(self, position, d_model):
        """
        位置エンコーディングを計算する

        Args:
            position (int): 系列長。
            d_model (int): 埋め込み次元数。

        Returns:
            Tensor: 形状 (1, position, d_model) の位置エンコーディング。
        """
        angle_rads = self._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        # 偶数インデックスにsin、奇数インデックスにcosを適用
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def _get_angles(self, pos, i, d_model):
        """
        角度の計算を行うヘルパー関数

        Args:
            pos (ndarray): 位置情報。
            i (ndarray): 次元情報。
            d_model (int): 埋め込み次元数。

        Returns:
            ndarray: 各位置・次元ごとの角度。
        """
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        """
        入力に位置エンコーディングを加算する

        Args:
            inputs (Tensor): 入力テンソル。

        Returns:
            Tensor: 位置エンコーディングが加算されたテンソル。
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """
    Transformerのエンコーダブロック

    自己注意機構とフィードフォワードネットワークをResidual ConnectionとLayerNormalizationで補強する。

    Args:
        inputs (Tensor): 入力テンソル。
        head_size (int): 各ヘッドの次元数。
        num_heads (int): マルチヘッドアテンションのヘッド数。
        ff_dim (int): フィードフォワードネットワーク中間層の次元数。
        dropout (float): ドロップアウト率。

    Returns:
        Tensor: エンコーダブロックの出力テンソル。
    """
    # 自己注意機構
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    # フィードフォワードネットワーク
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def inverse_transform(values, scaler):
    """
    逆正規化を実施する

    StandardScalerで正規化された値を、元のスケールに戻す。
    ここではターゲットとなるClose値（5列中4番目）の平均と標準偏差を用いる。

    Args:
        values (ndarray): 正規化された値。
        scaler (StandardScaler): 学習済みスケーラー。

    Returns:
        ndarray: 逆正規化された値。
    """
    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]
    return values * close_std + close_mean

