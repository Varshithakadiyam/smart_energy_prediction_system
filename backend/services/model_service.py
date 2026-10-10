"""
model_service.py
Singleton service managing the trained deep learning model,
fitted scaler, and historical reference dataset.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure macOS ARM compatibility
import src.compat

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

class ModelService:
    _instance = None
    
    def __init__(self):
        self.model_path = os.path.join(ROOT_DIR, "models", "best_energy_model.keras")
        self.scaler_path = os.path.join(ROOT_DIR, "models", "scaler.pkl")
        self.data_path = os.path.join(ROOT_DIR, "data", "processed", "hourly_energy_consumption.csv")
        
        self.model = None
        self.scaler_bundle = None
        self.feature_cols = None
        self.target_col = None
        self.feature_scaler = None
        self.target_scaler = None
        self.sample_df = None
        self.is_loaded = False
        self.load_error = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load_artifacts()
        return cls._instance

    def load_artifacts(self):
        if self.is_loaded:
            return
            
        try:
            active_model_path = self.model_path
            if not os.path.exists(active_model_path):
                fallback = os.path.join(ROOT_DIR, "models", "lstm_energy_model.keras")
                if os.path.exists(fallback):
                    active_model_path = fallback
                else:
                    self.load_error = f"Model file not found at {self.model_path} or {fallback}"
                    print(self.load_error)
                    return
            
            print(f"Loading DL model from {active_model_path}...")
            self.model = keras.models.load_model(active_model_path)
            self.model_path = active_model_path
            
            print(f"Loading Scaler bundle from {self.scaler_path}...")
            if not os.path.exists(self.scaler_path):
                self.load_error = f"Scaler file not found at {self.scaler_path}"
                print(self.load_error)
                return
            self.scaler_bundle = joblib.load(self.scaler_path)
            
            self.feature_cols = self.scaler_bundle['feature_cols']
            self.target_col = self.scaler_bundle['target_col']
            self.feature_scaler = self.scaler_bundle['feature_scaler']
            self.target_scaler = self.scaler_bundle['target_scaler']
            
            print(f"Loading reference dataset from {self.data_path}...")
            if os.path.exists(self.data_path):
                self.sample_df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
                
            self.is_loaded = True
            self.load_error = None
            print("ModelService artifacts successfully loaded!")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"Error loading ModelService artifacts: {e}")

    def get_latest_sample(self, hours=24):
        """Returns the most recent N hours from the dataset."""
        if self.sample_df is None or len(self.sample_df) == 0:
            return None
        return self.sample_df.iloc[-hours:]

    def get_model_metadata(self):
        """Returns architecture information and baseline comparisons."""
        return {
            "model_name": "GRU_Energy_Predictor (Best Model)",
            "model_type": "GRU (Gated Recurrent Unit)",
            "input_window": "24 hours",
            "prediction_horizon": "1 hour (Next Hour)",
            "features_used": self.feature_cols,
            "total_features": len(self.feature_cols) if self.feature_cols else 19,
            "loss_function": "Mean Squared Error (MSE)",
            "optimizer": "Adam (lr=0.001)",
            "metrics": {
                "MAE_Wh": 406.86,
                "RMSE_Wh": 607.26,
                "MAPE_pct": 44.80,
                "R2_score": 0.6489,
                "MAE_kWh": 0.407,
                "RMSE_kWh": 0.607
            },
            "baseline_comparison": [
                {"model": "Previous Hour (Persistence)", "mae_wh": 496.30, "rmse_wh": 766.10, "mape_pct": 48.11, "r2": 0.4411},
                {"model": "Previous Day (24h Persistence)", "mae_wh": 732.42, "rmse_wh": 1086.26, "mape_pct": 92.06, "r2": -0.1236},
                {"model": "Moving Average (24h)", "mae_wh": 785.88, "rmse_wh": 1013.10, "mape_pct": 125.25, "r2": 0.0227},
                {"model": "LSTM (Baseline)", "mae_wh": 407.03, "rmse_wh": 614.46, "mape_pct": 44.96, "r2": 0.6405},
                {"model": "GRU (Best Selected)", "mae_wh": 406.86, "rmse_wh": 607.26, "mape_pct": 44.80, "r2": 0.6489}
            ]
        }
