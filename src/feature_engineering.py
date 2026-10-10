"""
feature_engineering.py
Generates temporal, calendar, cyclical, lag, and rolling features
for the Smart Energy Consumption Prediction project.
Strictly prevents data leakage using past-only observations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

def add_time_features(df):
    """
    Extracts calendar and temporal indicators from the DatetimeIndex.
    Includes cyclical sine/cosine encodings for periodic continuity.
    """
    df = df.copy()
    idx = df.index
    
    # Standard calendar features
    df['hour'] = idx.hour
    df['day'] = idx.day
    df['day_of_week'] = idx.dayofweek          # 0=Monday, 6=Sunday
    df['day_of_month'] = idx.day
    df['month'] = idx.month
    df['week_of_year'] = idx.isocalendar().week.astype(int)
    df['is_weekend'] = (idx.dayofweek >= 5).astype(int)
    
    # Behavioral period indicators
    df['is_morning'] = ((idx.hour >= 6) & (idx.hour < 12)).astype(int)
    df['is_evening'] = ((idx.hour >= 17) & (idx.hour < 22)).astype(int)
    # Peak hours: evening peak typically occurs between 18:00 and 21:00 in residential power
    df['is_peak_hour'] = ((idx.hour >= 18) & (idx.hour <= 21)).astype(int)
    
    # Cyclical encodings (smooth cyclic transitions for neural networks)
    df['sin_hour'] = np.sin(2 * np.pi * idx.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * idx.hour / 24.0)
    df['sin_day_of_week'] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    df['cos_day_of_week'] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    df['sin_month'] = np.sin(2 * np.pi * (idx.month - 1) / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * (idx.month - 1) / 12.0)
    
    return df

def add_lag_features(df, target_col='average_global_active_power', lags=(1, 2, 3, 6, 12, 24, 48, 168)):
    """
    Creates historical lag features for the target series.
    lag_k at timestamp t corresponds to the value observed at timestamp (t - k hours).
    - lag_24: consumption exactly 24 hours earlier (same hour previous day)
    - lag_168: consumption exactly 168 hours earlier (same hour previous week, 7 * 24 = 168)
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col='average_global_active_power'):
    """
    Creates rolling statistics using strictly past observations.
    Using .shift(1) ensures the rolling window at time t covers [t-W, ..., t-1],
    preventing any target leakage from time t or future steps.
    """
    df = df.copy()
    shifted = df[target_col].shift(1)
    
    df['rolling_mean_6h'] = shifted.rolling(window=6).mean()
    df['rolling_mean_12h'] = shifted.rolling(window=12).mean()
    df['rolling_mean_24h'] = shifted.rolling(window=24).mean()
    df['rolling_std_24h'] = shifted.rolling(window=24).std()
    df['rolling_mean_168h'] = shifted.rolling(window=168).mean()
    
    return df

def build_feature_engineered_dataset(hourly_df, target_col='average_global_active_power', drop_initial_nan=True):
    """
    Executes full feature engineering pipeline:
    1. Time features
    2. Lag features
    3. Rolling statistics
    4. Drops the warm-up period (first 168 hours) to ensure zero missing values.
    """
    print("Engineering features: Temporal, Lag, and Rolling...")
    df = add_time_features(hourly_df)
    df = add_lag_features(df, target_col=target_col)
    df = add_rolling_features(df, target_col=target_col)
    
    if drop_initial_nan:
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        print(f"Dropped {dropped} initial warm-up rows (required for 168h lookback).")
        print(f"Remaining complete rows: {len(df)}")
        
    return df

if __name__ == "__main__":
    from src.clean_and_aggregate import load_and_clean_minute_data, aggregate_to_hourly
    df_clean, _, _ = load_and_clean_minute_data()
    hourly_df = aggregate_to_hourly(df_clean)
    feat_df = build_feature_engineered_dataset(hourly_df)
    print("Feature dataset shape:", feat_df.shape)
    print("Feature columns:", list(feat_df.columns))
