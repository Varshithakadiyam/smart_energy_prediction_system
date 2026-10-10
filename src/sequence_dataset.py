import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent if 'src' in str(Path(__file__).resolve().parent) else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import src.compat
"""
sequence_dataset.py
Prepares multivariate LSTM sequences with 24-hour lookback and 1-hour forecast horizon.
Performs strictly chronological 70/15/15 splitting, fits MinMaxScaler solely on Train,
and exports scaled 3D tensors along with target timestamps and unscaled test data.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SELECTED_FEATURES = [
    'energy_consumption',          # Auto-regressive target (Wh)
    'hour',                        # Diurnal cycle
    'day_of_week',                 # Weekly periodicity
    'is_weekend',                  # Behavioral shift indicator
    'sin_hour', 'cos_hour',        # Continuous circular diurnal encoding
    'sin_day_of_week', 'cos_day_of_week', # Continuous circular weekly encoding
    'lag_1',                       # Immediate preceding consumption
    'lag_24',                      # Same hour previous day (24h seasonality)
    'lag_168',                     # Same hour previous week (168h seasonality)
    'rolling_mean_24h',            # Smoothed 24h baseline demand
    'rolling_std_24h',             # 24h consumption volatility / appliance bursts
    'average_voltage',             # Grid operating voltage
    'average_global_reactive_power',# Inductive load signature
    'average_global_intensity',    # Circuit current intensity
    'sub_metering_1_wh',           # Kitchen zone active energy
    'sub_metering_2_wh',           # Laundry zone active energy
    'sub_metering_3_wh'            # Climate/water heater active energy
]

TARGET_COL = 'energy_consumption'

def create_sequences(features, targets, timestamps, lookback=24, horizon=1):
    """
    Sliding window tensor generator:
    X[i] = features[i : i + lookback]                  (24 hours)
    y[i] = targets[i + lookback + horizon - 1]         (next hour, t+1)
    target_ts[i] = timestamps[i + lookback + horizon - 1]
    """
    X, y, ts = [], [], []
    num_samples = len(features) - lookback - horizon + 1
    for i in range(num_samples):
        X.append(features[i : i + lookback])
        if horizon == 1:
            y.append(targets[i + lookback])
        else:
            y.append(targets[i + lookback : i + lookback + horizon])
        ts.append(timestamps[i + lookback + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(ts)

def prepare_stage2_data(csv_path="data/processed/hourly_energy_consumption.csv",
                        lookback=24, horizon=1,
                        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                        model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    print(f"Loading hourly dataset from: {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Verify features exist
    for col in SELECTED_FEATURES:
        if col not in df.columns:
            raise KeyError(f"Feature {col} not found in {csv_path}")
            
    print(f"Total dataset shape: {df.shape}")
    print(f"Selected {len(SELECTED_FEATURES)} multivariate features for LSTM training.")
    
    # 1. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print("\n--- Chronological Split Date Ranges ---")
    print(f"Train split:      {len(train_df):>5} samples ({len(train_df)/n*100:.1f}%) | {train_df.index.min()} to {train_df.index.max()}")
    print(f"Validation split: {len(val_df):>5} samples ({len(val_df)/n*100:.1f}%) | {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test split:       {len(test_df):>5} samples ({len(test_df)/n*100:.1f}%) | {test_df.index.min()} to {test_df.index.max()}")
    
    # Temporal assertions
    assert train_df.index.max() < val_df.index.min(), "Leakage Error: Train overlaps with Validation!"
    assert val_df.index.max() < test_df.index.min(), "Leakage Error: Validation overlaps with Test!"
    
    # 2. Fit Scaler STRICTLY on Training Split
    print("\nFitting MinMaxScaler(0, 1) exclusively on Training split...")
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    feature_scaler.fit(train_df[SELECTED_FEATURES])
    target_scaler.fit(train_df[[TARGET_COL]])
    
    # Transform partitions
    train_X_scaled = feature_scaler.transform(train_df[SELECTED_FEATURES])
    train_y_scaled = target_scaler.transform(train_df[[TARGET_COL]]).flatten()
    
    val_X_scaled = feature_scaler.transform(val_df[SELECTED_FEATURES])
    val_y_scaled = target_scaler.transform(val_df[[TARGET_COL]]).flatten()
    
    test_X_scaled = feature_scaler.transform(test_df[SELECTED_FEATURES])
    test_y_scaled = target_scaler.transform(test_df[[TARGET_COL]]).flatten()
    
    # Save Scaler Bundle
    scaler_bundle = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': SELECTED_FEATURES,
        'target_col': TARGET_COL,
        'scaler_type': 'MinMaxScaler'
    }
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler_bundle, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")
    
    # 3. Create LSTM Sequences
    print(f"\nGenerating sliding sequences (Lookback={lookback}h, Horizon={horizon}h)...")
    X_train, y_train, ts_train = create_sequences(train_X_scaled, train_y_scaled, train_df.index, lookback, horizon)
    X_val, y_val, ts_val = create_sequences(val_X_scaled, val_y_scaled, val_df.index, lookback, horizon)
    X_test, y_test, ts_test = create_sequences(test_X_scaled, test_y_scaled, test_df.index, lookback, horizon)
    
    # Unscaled raw target arrays for baselines and exact test evaluation
    _, raw_y_test, _ = create_sequences(test_X_scaled, test_df[TARGET_COL].values, test_df.index, lookback, horizon)
    
    # Reshape 1D targets to (N, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        raw_y_test = raw_y_test.reshape(-1, 1)
        
    print(f"Train tensor shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Val tensor shapes:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test tensor shapes:  X={X_test.shape}, y={y_test.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train, 'ts_train': ts_train,
        'X_val': X_val, 'y_val': y_val, 'ts_val': ts_val,
        'X_test': X_test, 'y_test': y_test, 'ts_test': ts_test,
        'raw_y_test': raw_y_test,
        'test_df': test_df,
        'feature_cols': SELECTED_FEATURES,
        'target_col': TARGET_COL,
        'scaler_bundle': scaler_bundle
    }

if __name__ == "__main__":
    data = prepare_stage2_data()
