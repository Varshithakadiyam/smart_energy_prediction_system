import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent if 'src' in str(Path(__file__).resolve().parent) else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure macOS ARM compatibility workaround is applied first
import src.compat

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import layers, callbacks

# Reproducibility seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def create_univariate_sequences(series_scaled, lookback=24, horizon=1):
    X, y = [], []
    num_samples = len(series_scaled) - lookback - horizon + 1
    for i in range(num_samples):
        X.append(series_scaled[i : i + lookback])
        y.append(series_scaled[i + lookback + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def calculate_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"MAE_kWh": mae, "RMSE_kWh": rmse, "R2": r2}

def train_univariate_model(
    csv_path="data/processed/hourly_energy_consumption.csv",
    model_save_path="models/univariate_energy_model.keras",
    scaler_save_path="models/univariate_scaler.pkl",
    lookback=24,
    epochs=20,
    batch_size=32
):
    print("=" * 70)
    print("   TRAINING DEDICATED UNIVARIATE GRU MODEL (STRATEGY B)")
    print("=" * 70)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"Loading hourly dataset from: {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if 'energy_consumption_kwh' not in df.columns:
        if 'energy_consumption' in df.columns:
            df['energy_consumption_kwh'] = df['energy_consumption'] / 1000.0
        else:
            raise KeyError("Neither 'energy_consumption_kwh' nor 'energy_consumption' found in dataset")
            
    series_kwh = df[['energy_consumption_kwh']].copy()
    n = len(series_kwh)
    print(f"Total observations: {n} hourly time-steps")
    
    # Chronological Split (70% Train, 15% Val, 15% Test)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = series_kwh.iloc[:train_end]
    val_df = series_kwh.iloc[train_end:val_end]
    test_df = series_kwh.iloc[val_end:]
    
    assert train_df.index.max() < val_df.index.min(), "Leakage Error: Train overlaps Val"
    assert val_df.index.max() < test_df.index.min(), "Leakage Error: Val overlaps Test"
    
    print(f"Train split: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Val split:   {len(val_df)} rows ({val_df.index.min()} to {val_df.index.max()})")
    print(f"Test split:  {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    # Fit Scaler strictly on Train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df.values)
    
    train_scaled = scaler.transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)
    
    # Create sequences: (N, 24, 1)
    X_train, y_train = create_univariate_sequences(train_scaled, lookback=lookback)
    X_val, y_val = create_univariate_sequences(val_scaled, lookback=lookback)
    X_test, y_test = create_univariate_sequences(test_scaled, lookback=lookback)
    
    raw_test_target = test_df.values[lookback:].flatten()
    
    print(f"Tensors: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    
    # Build Univariate GRU Architecture
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1), name="univariate_input"),
        layers.GRU(32, return_sequences=True, name="gru_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.GRU(16, return_sequences=False, name="gru_2"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(8, activation="relu", name="dense_intermediate"),
        layers.Dense(1, activation="linear", name="output_layer")
    ], name="Univariate_GRU_Predictor")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
    ]
    
    print(f"Training Univariate GRU for max {epochs} epochs...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )
    
    # Save Model & Scaler Bundle
    model.save(model_save_path)
    scaler_bundle = {
        'scaler': scaler,
        'feature_col': 'energy_consumption_kwh',
        'target_col': 'energy_consumption_kwh',
        'unit': 'kWh',
        'lookback': lookback,
        'horizon': 1,
        'data_min': float(scaler.data_min_[0]),
        'data_max': float(scaler.data_max_[0])
    }
    joblib.dump(scaler_bundle, scaler_save_path)
    print(f"Saved model to:  {model_save_path}")
    print(f"Saved scaler to: {scaler_save_path}")
    
    # Evaluate strictly on unseen test set
    pred_scaled = model.predict(X_test, verbose=0)
    pred_kwh = scaler.inverse_transform(pred_scaled).flatten()
    
    # Persistence Baseline: y_t (last observation in each sequence)
    base_scaled = X_test[:, -1, 0]
    base_kwh = scaler.inverse_transform(base_scaled.reshape(-1, 1)).flatten()
    
    metrics_gru = calculate_metrics(raw_test_target, pred_kwh)
    metrics_base = calculate_metrics(raw_test_target, base_kwh)
    
    print("\n" + "=" * 60)
    print("      UNSEEN TEST SET EVALUATION (kWh)")
    print("=" * 60)
    print(f"Persistence Baseline (Previous Hour):")
    print(f"  MAE  : {metrics_base['MAE_kWh']:.4f} kWh")
    print(f"  RMSE : {metrics_base['RMSE_kWh']:.4f} kWh")
    print(f"  R²   : {metrics_base['R2']:.4f}")
    print("-" * 60)
    print(f"Univariate GRU Model:")
    print(f"  MAE  : {metrics_gru['MAE_kWh']:.4f} kWh  ({metrics_gru['MAE_kWh']*1000:.1f} Wh)")
    print(f"  RMSE : {metrics_gru['RMSE_kWh']:.4f} kWh  ({metrics_gru['RMSE_kWh']*1000:.1f} Wh)")
    print(f"  R²   : {metrics_gru['R2']:.4f}")
    print("=" * 60)
    
    return model, scaler_bundle, metrics_gru

if __name__ == "__main__":
    train_univariate_model()
