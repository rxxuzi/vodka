import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# プロジェクトのルートディレクトリを基準に data フォルダのパスを設定
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
csv_path = os.path.join(data_dir, "crypto_data_BTCUSDT_1m.csv")

# 1. データの読み込み
df = pd.read_csv(csv_path)

# 必要なカラム（ここでは終値）を抽出
data = df[['Close']].values

# 2. 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. 時系列ウィンドウの作成（例：過去60分分のデータで次の1分の終値を予測）
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_dataset(scaled_data, window_size)
# LSTMの入力は [サンプル数, タイムステップ, 特徴量数] となるようにリシェイプ
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. データの分割（80%を訓練、20%をテスト）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. LSTMモデルの構築（ユニット数を増やし、Dropoutで正則化）
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early Stoppingで過学習防止
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 6. モデルの学習
history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# 7. 予測と評価
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 逆正規化して元のスケールに戻す
train_predict = scaler.inverse_transform(train_predict)
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# テストデータの予測結果の可視化
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='実際の終値')
plt.plot(test_predict, label='予測終値')
plt.xlabel('タイムステップ')
plt.ylabel('価格')
plt.legend()
plt.title('LSTMによる仮想通貨価格予測')
plt.savefig(os.path.join(base_dir, "lstm_crypto_prediction.png"))
plt.show()

# 8. 学習済みモデルの保存（models フォルダに保存）
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "vodka_model.h5")
model.save(model_path)
print(f"学習済みモデルを '{model_path}' に保存しました。")
