import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent if 'src' in str(Path(__file__).resolve().parent) else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import src.compat
"""
train_lstm.py
Builds, compiles, and trains the baseline LSTM neural network
using early stopping and checkpointing. Evaluates learning curves
and saves models/lstm_energy_model.keras and reports/figures/11_lstm_training_history.png.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Isolate PyArrow and configure backend
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['KERAS_HOME'] = os.path.abspath('.keras')
# sys.modules['pyarrow'] = None
# sys.modules['pyarrow._generated_version'] = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks

# Reproducibility seeds
np.random.seed(42)
tf.random.set_seed(42)

def build_lstm_model(input_shape):
    """
    Constructs the exact baseline LSTM architecture specified in requirements:
    Input -> LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, ReLU) -> Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="input_layer"),
        layers.LSTM(64, return_sequences=True, name="lstm_layer_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.LSTM(32, return_sequences=False, name="lstm_layer_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(16, activation="relu", name="dense_intermediate"),
        layers.Dense(1, name="output_layer")
    ], name="LSTM_Energy_Predictor")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

def plot_and_save_history(history, output_path="reports/figures/11_lstm_training_history.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # 1. Loss Curves
    ax1.plot(epochs, hist['loss'], color='#1f77b4', marker='o', markersize=3, label='Train MSE Loss')
    ax1.plot(epochs, hist['val_loss'], color='#d62728', marker='s', markersize=3, linestyle='--', label='Val MSE Loss')
    best_loss_idx = np.argmin(hist['val_loss'])
    ax1.scatter([best_loss_idx + 1], [hist['val_loss'][best_loss_idx]], color='gold', edgecolor='black', s=120, zorder=5, label=f'Best Val Loss ({hist["val_loss"][best_loss_idx]:.5f})')
    ax1.set_title('LSTM Training vs. Validation Loss (MSE)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Mean Squared Error (Scaled)', fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. MAE Curves
    ax1_mae = ax2.plot(epochs, hist['mae'], color='#2ca02c', marker='o', markersize=3, label='Train MAE')
    ax2.plot(epochs, hist['val_mae'], color='#ff7f0e', marker='s', markersize=3, linestyle='--', label='Val MAE')
    best_mae_idx = np.argmin(hist['val_mae'])
    ax2.scatter([best_mae_idx + 1], [hist['val_mae'][best_mae_idx]], color='gold', edgecolor='black', s=120, zorder=5, label=f'Best Val MAE ({hist["val_mae"][best_mae_idx]:.5f})')
    ax2.set_title('LSTM Training vs. Validation MAE', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Mean Absolute Error (Scaled)', fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved LSTM training history curves to: {output_path}")

def train_lstm(data, model_save_path="models/lstm_energy_model.keras", epochs=40, batch_size=32):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    print("\n--- LSTM Architecture Summary ---")
    model.summary()
    
    cb_early = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )
    
    print(f"\nStarting LSTM training ({epochs} max epochs, batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early, cb_checkpoint, cb_reduce_lr],
        verbose=1
    )
    
    # Save best model to ensure persistence
    model.save(model_save_path)
    print(f"Model saved successfully to: {model_save_path}")
    
    # Plot history
    plot_and_save_history(history)
    
    return model, history

if __name__ == "__main__":
    from src.sequence_dataset import prepare_stage2_data
    data = prepare_stage2_data()
    model, history = train_lstm(data)
