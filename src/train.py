# train.py
# 
# This script trains a Transformer-based model for cryptocurrency price prediction.
# It reads historical market data, processes it, and trains a model to predict future price changes.
# 
# Version : 3.0.0

import os
import json
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.regularizers import l1_l2
import datetime

from lime import PositionalEncoding, transformer_encoder
from logger import save_model_info

def create_sequences(data, window_size, prediction_offsets, features, target_amplification=1.0, use_log_returns=False):
    """
    Create sequence data for time series forecasting.
    
    Args:
        data (np.ndarray): Original data array.
        window_size (int): Size of the lookback window.
        prediction_offsets (list): List of future offsets to predict.
        features (list): List of feature names.
        target_amplification (float): Factor to amplify target values.
        use_log_returns (bool): Whether to use log returns instead of percentage changes.
        
    Returns:
        tuple: X and y arrays for training.
    """
    max_offset = max(prediction_offsets)
    close_idx = features.index('Close') if 'Close' in features else 3  # Default to index 3 if not found
    
    X, y = [], []
    
    # For feature importance analysis, create structure for tracking actual changes
    price_changes = {offset: [] for offset in prediction_offsets}
    
    for i in range(window_size, len(data) - max_offset):
        # Input sequence
        X.append(data[i - window_size:i])
        
        # Current close price (reference price for calculating changes)
        current_close = data[i - 1][close_idx]
        
        # Calculate future price changes
        future_changes = []
        for j, offset in enumerate(prediction_offsets):
            future_close = data[i + offset - 1][close_idx]
            
            if use_log_returns:
                # Use log returns for more stable learning
                change = np.log(future_close / current_close) * 100 * target_amplification
            else:
                # Use percentage change
                change = ((future_close - current_close) / current_close) * 100 * target_amplification
                
            future_changes.append(change)
            price_changes[offset].append(change)
        
        y.append(future_changes)
    
    # Print statistics about target values
    print("\n=== Target Value Statistics ===")
    for offset in prediction_offsets:
        changes = price_changes[offset]
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        min_change = np.min(changes)
        max_change = np.max(changes)
        zero_pct = np.sum(np.abs(changes) < 0.01) / len(changes) * 100
        
        print(f"Offset {offset}:")
        print(f"  Mean change: {mean_change:.4f}%")
        print(f"  Std dev: {std_change:.4f}%")
        print(f"  Min: {min_change:.4f}%, Max: {max_change:.4f}%")
        print(f"  Near-zero values (<0.01%): {zero_pct:.2f}%\n")
    
    return np.array(X), np.array(y)

def build_transformer_model(input_shape, output_size, config):
    """
    Build a Transformer-based model for time series forecasting.
    
    Args:
        input_shape (tuple): Shape of input data (window_size, features).
        output_size (int): Number of outputs (prediction horizons).
        config (dict): Model configuration parameters.
        
    Returns:
        Model: Compiled Keras model.
    """
    # Extract parameters from config
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    num_transformer_blocks = config["num_transformer_blocks"]
    ff_dim = config["ff_dim"]
    dropout_rate = config["dropout_rate"]
    hidden_layer = config["hidden_layer"]
    lr = config["lr"]
    
    # Create model
    input_tensor = Input(shape=input_shape)
    x = Dense(d_model)(input_tensor)
    x = BatchNormalization()(x)
    x = PositionalEncoding(sequence_length=input_shape[0], d_model=d_model)(x)
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            head_size=d_model // num_heads,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Hidden layers with regularization
    x = Dense(hidden_layer, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    x = Dense(hidden_layer // 2, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    # Output layer
    output_tensor = Dense(output_size)(x)
    
    # Create and compile model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=[MeanAbsoluteError(name="mae"), MeanSquaredError(name="mse")]
    )
    
    return model

def build_hybrid_model(input_shape, output_size, config):
    """
    Build a hybrid model combining CNN, LSTM, and Dense layers.
    This model can capture both local patterns (CNN) and temporal dependencies (LSTM).
    
    Args:
        input_shape (tuple): Shape of input data (window_size, features).
        output_size (int): Number of outputs (prediction horizons).
        config (dict): Model configuration parameters.
        
    Returns:
        Model: Compiled Keras model.
    """
    dropout_rate = config["dropout_rate"]
    hidden_layer = config["hidden_layer"]
    lr = config["lr"]
    
    # Model input
    input_tensor = Input(shape=input_shape)
    
    # CNN for feature extraction
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM for temporal patterns
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layers for final prediction
    x = Dense(hidden_layer, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(hidden_layer // 2, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    output_tensor = Dense(output_size)(x)
    
    # Create and compile model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=[MeanAbsoluteError(name="mae"), MeanSquaredError(name="mse")]
    )
    
    return model

def plot_prediction_distribution(predictions, y_test, prediction_offsets):
    """
    Plot the distribution of predictions vs actual values.
    
    Args:
        predictions (np.ndarray): Model predictions.
        y_test (np.ndarray): Actual target values.
        prediction_offsets (list): List of prediction offsets.
    """
    fig, axes = plt.subplots(len(prediction_offsets), 1, figsize=(10, 4*len(prediction_offsets)))
    
    if len(prediction_offsets) == 1:
        axes = [axes]
    
    for i, offset in enumerate(prediction_offsets):
        ax = axes[i]
        
        # Plot histogram of actual values
        ax.hist(y_test[:, i], bins=50, alpha=0.5, label='Actual')
        
        # Plot histogram of predicted values
        ax.hist(predictions[:, i], bins=50, alpha=0.5, label='Predicted')
        
        ax.set_title(f'Distribution for {offset}-minute prediction')
        ax.set_xlabel('Percentage change')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_sample_predictions(predictions, y_test, prediction_offsets, n_samples=100):
    """
    Plot sample predictions vs actual values.
    
    Args:
        predictions (np.ndarray): Model predictions.
        y_test (np.ndarray): Actual target values.
        prediction_offsets (list): List of prediction offsets.
        n_samples (int): Number of samples to plot.
    """
    # Limit to n_samples
    predictions = predictions[:n_samples]
    y_test = y_test[:n_samples]
    
    fig, axes = plt.subplots(len(prediction_offsets), 1, figsize=(10, 4*len(prediction_offsets)))
    
    if len(prediction_offsets) == 1:
        axes = [axes]
    
    for i, offset in enumerate(prediction_offsets):
        ax = axes[i]
        
        # Plot actual values
        ax.plot(y_test[:, i], label='Actual', marker='o', markersize=3)
        
        # Plot predicted values
        ax.plot(predictions[:, i], label='Predicted', marker='x', markersize=3)
        
        ax.set_title(f'Predictions for {offset}-minute horizon')
        ax.set_xlabel('Sample index')
        ax.set_ylabel('Percentage change')
        ax.legend()
    
    plt.tight_layout()
    return fig

def run():
    # --- Load configuration ---
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(BASE_DIR, "vodka.json")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # --- Set debug mode ---
    if not config.get("debug", False):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.get_logger().setLevel("ERROR")
    else:
        print("Running in DEBUG mode")

    # --- Load data ---
    data_dir = os.path.join(BASE_DIR, "data")
    csv_filename = f"crypto_data_{config['symbol']}_{config['interval']}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_filename}")
    print(f"Columns: {df.columns.tolist()}")
    
    # --- Check for missing values ---
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values in the dataset.")
        df = df.fillna(method='ffill')
        
    # --- Data preparation ---
    # Use more features, including technical indicators
    # Choose features based on what's available in your CSV
    basic_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    technical_features = []
    
    # Check which technical indicators are available
    for indicator in ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20', 'TP', 'VWAP_5']:
        if indicator in df.columns:
            technical_features.append(indicator)
    
    # Select features for model
    features = basic_features + technical_features
    print(f"Using features: {features}")
    
    # Extract data
    data = df[features].values
    
    # Apply target amplification to make small changes more significant for the model
    target_amplification = 5.0  # Amplify percentage changes by 5x
    use_log_returns = True      # Use log returns instead of percentage changes
    
    # --- Normalization ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # --- Create sequences ---
    window_size = config["limit"]
    prediction_offsets = config["prediction"]
    
    X, y = create_sequences(
        data=data,
        window_size=window_size,
        prediction_offsets=prediction_offsets,
        features=features,
        target_amplification=target_amplification,
        use_log_returns=use_log_returns
    )

    # --- Train/validation/test split ---
    # Use proper time series split (no shuffling)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # --- Define model type ---
    model_type = config.get("model_type", "transformer")  # Default to transformer
    print(f"Using model type: {model_type}")

    # --- Build model ---
    if model_type == "hybrid":
        model = build_hybrid_model(
            input_shape=(window_size, len(features)),
            output_size=len(prediction_offsets),
            config=config["transformer"]  # Reuse transformer config
        )
    else:  # Default to transformer
        model = build_transformer_model(
            input_shape=(window_size, len(features)),
            output_size=len(prediction_offsets),
            config=config["transformer"]
        )

    model.summary()

    # --- Setup callbacks ---
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    tensorboard_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(log_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1
        )
    ]

    # --- Train model ---
    print(f"\n=== Starting training ===")
    train_start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["transformer"]["epochs"],
        batch_size=config["transformer"]["batch_size"],
        callbacks=callbacks,
        verbose=1
    )
    
    train_end_time = time.time()
    elapsed_time = train_end_time - train_start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # --- Evaluate on test set ---
    print("\n=== Model Evaluation ===")
    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    metric_names = ['loss'] + [m.name for m in model.metrics]
    
    for name, value in zip(metric_names, test_metrics):
        print(f"Test {name}: {value:.6f}")

    # --- Make predictions ---
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    test_predictions = model.predict(X_test)
    
    # --- Analyze predictions ---
    print("\n=== Prediction Analysis ===")
    for i, offset in enumerate(prediction_offsets):
        train_mean = np.mean(train_predictions[:, i])
        train_std = np.std(train_predictions[:, i])
        test_mean = np.mean(test_predictions[:, i])
        test_std = np.std(test_predictions[:, i])
        
        print(f"Offset {offset}:")
        print(f"  Training predictions - Mean: {train_mean:.4f}, Std: {train_std:.4f}")
        print(f"  Test predictions - Mean: {test_mean:.4f}, Std: {test_std:.4f}")
        
        # Check for zero predictions
        zero_pct = np.sum(np.abs(test_predictions[:, i]) < 0.01) / len(test_predictions) * 100
        print(f"  Near-zero predictions (<0.01%): {zero_pct:.2f}%")
        
        # Mean absolute error
        mae = np.mean(np.abs(test_predictions[:, i] - y_test[:, i]))
        print(f"  Test MAE: {mae:.4f}\n")

    # --- Save plots ---
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color='blue')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training MAE Curve')
    plt.legend()
    
    plt.tight_layout()
    loss_plot_path = os.path.join(plots_dir, f"{config['name']}_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot prediction distributions
    dist_fig = plot_prediction_distribution(test_predictions, y_test, prediction_offsets)
    dist_plot_path = os.path.join(plots_dir, f"{config['name']}_prediction_distribution.png")
    dist_fig.savefig(dist_plot_path)
    plt.close(dist_fig)
    
    # Plot sample predictions
    sample_fig = plot_sample_predictions(test_predictions, y_test, prediction_offsets, n_samples=100)
    sample_plot_path = os.path.join(plots_dir, f"{config['name']}_sample_predictions.png")
    sample_fig.savefig(sample_plot_path)
    plt.close(sample_fig)

    # --- Save model ---
    print("\n=== Saving Model and Artifacts ===")
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_filename = f"{config['model']}.h5"
    model_path = os.path.join(models_dir, model_filename)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata
    model_meta = {
        "target_amplification": target_amplification,
        "use_log_returns": use_log_returns,
        "features": features,
        "model_type": model_type
    }
    
    meta_filename = f"{config['model']}_meta.json"
    meta_path = os.path.join(models_dir, meta_filename)
    with open(meta_path, 'w') as f:
        json.dump(model_meta, f, indent=4)
    print(f"Model metadata saved to {meta_path}")
    
    # Save scaler
    scaler_filename = f"{config['model']}_scaler.pkl"
    scaler_path = os.path.join(models_dir, scaler_filename)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Save target amplification factor in prediction.py-friendly format
    adjustment_config = {
        "target_amplification": target_amplification,
        "use_log_returns": use_log_returns
    }
    
    adj_filename = f"{config['model']}_adjustment.json"
    adj_path = os.path.join(models_dir, adj_filename)
    with open(adj_path, 'w') as f:
        json.dump(adjustment_config, f, indent=4)
    print(f"Target adjustment config saved to {adj_path}")
    
    # --- Log training information ---
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
    
    # --- Save full training log ---
    train_log = {
        "config": config,
        "model_type": model_type,
        "features": features,
        "target_amplification": target_amplification,
        "use_log_returns": use_log_returns,
        "training_time": elapsed_time,
        "epochs_completed": len(history.history['loss']),
        "final_metrics": {name: value for name, value in zip(metric_names, test_metrics)},
        "train_history": {k: [float(val) for val in v] for k, v in history.history.items()}
    }
    
    log_filename = f"{config['model']}_training_log.json"
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=4)
    print(f"Complete training log saved to {log_path}")

if __name__ == "__main__":
    run()