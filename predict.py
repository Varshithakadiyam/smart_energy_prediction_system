import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent if 'src' in str(Path(__file__).resolve().parent) else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import src.compat
"""
predict.py
Inference CLI script for the Smart Energy Consumption Prediction System.
Accepts a 24-hour sequence of observations (or defaults to the latest available sequence)
and returns:
- Predicted next-hour consumption (in kWh and Wh)
- Expected consumption level (LOW, NORMAL, HIGH, UNUSUALLY HIGH)

Example usage:
    python3 predict.py
    python3 predict.py --csv path/to/24h_data.csv
"""

import sys
import os
import argparse
from pathlib import Path

# Backend setup to avoid macOS ARM PyArrow conflict
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['KERAS_HOME'] = os.path.abspath('.keras')
# sys.modules['pyarrow'] = None
# sys.modules['pyarrow._generated_version'] = None

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Empirical distribution thresholds (kWh) based on training dataset
# Low: < 25th percentile (0.30 kWh)
# Normal: 25th - 75th percentile (0.30 - 1.65 kWh)
# High: 75th - 95th percentile (1.65 - 3.20 kWh)
# Unusually High: > 95th percentile (> 3.20 kWh)
TIER_LOW = 0.30
TIER_NORMAL = 1.65
TIER_HIGH = 3.20

def classify_consumption_level(kwh_val):
    if kwh_val < TIER_LOW:
        return "LOW", "Baseline standby load (appliances idle / night hours)"
    elif kwh_val <= TIER_NORMAL:
        return "NORMAL", "Typical diurnal consumption (standard household activity)"
    elif kwh_val <= TIER_HIGH:
        return "HIGH", "Heavy appliance usage (e.g., cooking, heating, laundry active)"
    else:
        return "UNUSUALLY HIGH", "Extreme demand surge (concurrent high-draw appliance usage)"

def run_prediction(input_csv=None, model_path="models/best_energy_model.keras", scaler_path="models/scaler.pkl"):
    if not os.path.exists(model_path):
        # Fallback to lstm_energy_model if best model not copied yet
        fallback = "models/lstm_energy_model.keras"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler bundle not found at {scaler_path}")
        
    scaler_bundle = joblib.load(scaler_path)
    feat_scaler = scaler_bundle['feature_scaler']
    target_scaler = scaler_bundle['target_scaler']
    feat_cols = scaler_bundle['feature_cols']
    
    # Load model
    model = keras.models.load_model(model_path)
    
    # Obtain 24-hour input sequence
    if input_csv is not None and os.path.exists(input_csv):
        df_in = pd.read_csv(input_csv, index_col=0, parse_dates=True)
        if len(df_in) < 24:
            raise ValueError(f"Input CSV must contain at least 24 hourly rows, found {len(df_in)}")
        seq_df = df_in[feat_cols].iloc[-24:]
        last_timestamp = seq_df.index[-1]
    else:
        # Default: Use the last 24 hours of the processed test set
        default_csv = "data/processed/hourly_energy_consumption.csv"
        df_all = pd.read_csv(default_csv, index_col=0, parse_dates=True)
        seq_df = df_all[feat_cols].iloc[-24:]
        last_timestamp = seq_df.index[-1]
        
    # Scale sequence
    seq_scaled = feat_scaler.transform(seq_df)
    tensor_in = np.expand_dims(seq_scaled, axis=0) # shape (1, 24, num_features)
    
    # Predict
    pred_scaled = model.predict(tensor_in, verbose=0)
    pred_wh = float(target_scaler.inverse_transform(pred_scaled)[0, 0])
    pred_kwh = max(0.0, pred_wh / 1000.0) # Ensure non-negative
    
    level, description = classify_consumption_level(pred_kwh)
    next_timestamp = pd.to_datetime(last_timestamp) + pd.Timedelta(hours=1)
    
    print("\n" + "=" * 60)
    print("      SMART ENERGY CONSUMPTION INFERENCE SYSTEM")
    print("=" * 60)
    print(f"Observation Window : 24 Hours ending at {last_timestamp}")
    print(f"Forecast Timestamp : Next Hour -> {next_timestamp}")
    print(f"Active DL Model    : {os.path.basename(model_path)}")
    print("-" * 60)
    print(f"Predicted next-hour consumption: {pred_kwh:.2f} kWh  ({pred_wh:.1f} Wh)")
    print(f"Consumption level: {level}")
    print(f"Assessment       : {description}")
    print("=" * 60 + "\n")
    
    return {
        'predicted_kwh': pred_kwh,
        'predicted_wh': pred_wh,
        'consumption_level': level,
        'timestamp': next_timestamp
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy Consumption Predictor")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV containing at least 24 hours of data")
    args = parser.parse_args()
    run_prediction(args.csv)
