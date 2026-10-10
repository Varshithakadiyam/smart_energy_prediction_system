"""
prepare_lstm.py
Chronological time-series split, leak-free feature scaling,
and 3D tensor sequence generation for LSTM forecasting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def split_time_series(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Splits time-series chronologically into Train, Validation, and Test sets.
    Strictly preserves temporal order: Train < Validation < Test.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1.0"
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print("\n--- Chronological Split Summary ---")
    print(f"Total observations: {n}")
    print(f"Train split:      {len(train_df):>5} rows ({len(train_df)/n*100:.1f}%) | {train_df.index.min()} to {train_df.index.max()}")
    print(f"Validation split: {len(val_df):>5} rows ({len(val_df)/n*100:.1f}%) | {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test split:       {len(test_df):>5} rows ({len(test_df)/n*100:.1f}%) | {test_df.index.min()} to {test_df.index.max()}")
    
    # Verify temporal isolation
    assert train_df.index.max() < val_df.index.min(), "Leakage detected: Train overlaps with Validation!"
    assert val_df.index.max() < test_df.index.min(), "Leakage detected: Validation overlaps with Test!"
    
    return train_df, val_df, test_df

def fit_and_apply_scalers(train_df, val_df, test_df, feature_cols, target_col='average_global_active_power', scaler_type='minmax'):
    """
    Fits scaler ONLY on the training data to prevent future information leakage.
    Scales Validation and Test using parameters learned strictly from Train.
    
    Rationale for MinMaxScaler:
    1. LSTMs utilize saturating activation functions (tanh in cell states, sigmoid in gates).
       MinMaxScaler maps inputs cleanly into [0, 1], preventing vanishing/exploding gradients.
    2. Power consumption is inherently non-negative; MinMaxScaler preserves the natural lower bound of zero.
    """
    print(f"\nFitting {scaler_type.upper()} scaler strictly on TRAINING data...")
    if scaler_type.lower() == 'minmax':
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
    # Fit strictly on train
    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])
    
    # Transform splits
    train_feats_scaled = feature_scaler.transform(train_df[feature_cols])
    val_feats_scaled = feature_scaler.transform(val_df[feature_cols])
    test_feats_scaled = feature_scaler.transform(test_df[feature_cols])
    
    train_target_scaled = target_scaler.transform(train_df[[target_col]]).flatten()
    val_target_scaled = target_scaler.transform(val_df[[target_col]]).flatten()
    test_target_scaled = target_scaler.transform(test_df[[target_col]]).flatten()
    
    scaler_bundle = {
        'scaler_type': scaler_type,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_cols': feature_cols,
        'target_col': target_col
    }
    
    scaled_data = {
        'train_X': train_feats_scaled,
        'train_y': train_target_scaled,
        'val_X': val_feats_scaled,
        'val_y': val_target_scaled,
        'test_X': test_feats_scaled,
        'test_y': test_target_scaled
    }
    
    return scaled_data, scaler_bundle

def create_lstm_sequences(features, targets, lookback=24, horizon=1):
    """
    Generates sliding window 3D tensors for LSTM models:
    X shape: (samples, lookback, num_features)  --> [t - lookback + 1, ..., t]
    y shape: (samples, horizon) or (samples, 1) --> [t + horizon]
    
    Parameters:
    - lookback: Number of past hourly steps to use as input (default: 24)
    - horizon: Number of future hourly steps to forecast (default: 1, next hour)
    """
    X, y = [], []
    num_samples = len(features) - lookback - horizon + 1
    
    for i in range(num_samples):
        # Input: previous 24 hours [i, ..., i + lookback - 1]
        X.append(features[i : i + lookback])
        # Target: next hour [i + lookback + horizon - 1] or multi-step horizon
        if horizon == 1:
            y.append(targets[i + lookback])
        else:
            y.append(targets[i + lookback : i + lookback + horizon])
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def generate_and_save_all(feat_df, output_dir="data/processed", model_dir="models", lookback=24, horizon=1):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    target_col = 'average_global_active_power'
    # Features exclude timestamp (which is index)
    feature_cols = [c for c in feat_df.columns]
    
    # 1. Save full hourly dataset
    hourly_path = os.path.join(output_dir, "hourly_energy_consumption.csv")
    feat_df.to_csv(hourly_path)
    print(f"Saved full hourly dataset to: {hourly_path}")
    
    # 2. Chronological Split
    train_df, val_df, test_df = split_time_series(feat_df)
    train_df.to_csv(os.path.join(output_dir, "train.csv"))
    val_df.to_csv(os.path.join(output_dir, "validation.csv"))
    test_df.to_csv(os.path.join(output_dir, "test.csv"))
    print("Saved train.csv, validation.csv, test.csv")
    
    # 3. Leak-free Scaling
    scaled_data, scaler_bundle = fit_and_apply_scalers(train_df, val_df, test_df, feature_cols, target_col=target_col)
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler_bundle, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")
    
    # 4. Generate LSTM Sequences
    print(f"\nGenerating LSTM sequences (lookback={lookback}h, horizon={horizon}h)...")
    X_train, y_train = create_lstm_sequences(scaled_data['train_X'], scaled_data['train_y'], lookback, horizon)
    X_val, y_val = create_lstm_sequences(scaled_data['val_X'], scaled_data['val_y'], lookback, horizon)
    X_test, y_test = create_lstm_sequences(scaled_data['test_X'], scaled_data['test_y'], lookback, horizon)
    
    # Reshape 1D targets to (N, 1) if horizon == 1
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    
    npz_path = os.path.join(output_dir, "lstm_sequences.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_cols=np.array(feature_cols),
        target_col=np.array([target_col])
    )
    print(f"Saved LSTM sequences to compressed archive: {npz_path}")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler_bundle': scaler_bundle,
        'feature_cols': feature_cols
    }

if __name__ == "__main__":
    from src.clean_and_aggregate import load_and_clean_minute_data, aggregate_to_hourly
    from src.feature_engineering import build_feature_engineered_dataset
    
    df_clean, _, _ = load_and_clean_minute_data()
    hourly_df = aggregate_to_hourly(df_clean)
    feat_df = build_feature_engineered_dataset(hourly_df)
    results = generate_and_save_all(feat_df)
