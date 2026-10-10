"""
univariate_model_service.py
Singleton service managing the trained Univariate GRU model and scaler (Strategy B).
Provides single-step (next-hour) and 24-step autoregressive rolling forecasts
for user-uploaded 2-column CSV consumption data.
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure macOS ARM compatibility is applied
import src.compat

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

TIER_LOW = 0.30
TIER_NORMAL = 1.65
TIER_HIGH = 3.20

def classify_level(kwh: float) -> str:
    if kwh < TIER_LOW:
        return "Low"
    elif kwh <= TIER_NORMAL:
        return "Normal"
    elif kwh <= TIER_HIGH:
        return "High"
    else:
        return "Unusually High"

class UnivariateModelService:
    _instance = None

    def __init__(self):
        self.model_path = os.path.join(ROOT_DIR, "models", "univariate_energy_model.keras")
        self.scaler_path = os.path.join(ROOT_DIR, "models", "univariate_scaler.pkl")
        self.model = None
        self.scaler_bundle = None
        self.scaler = None
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
            if not os.path.exists(self.model_path):
                self.load_error = f"Univariate model file not found at {self.model_path}"
                print(self.load_error)
                return

            if not os.path.exists(self.scaler_path):
                self.load_error = f"Univariate scaler bundle not found at {self.scaler_path}"
                print(self.load_error)
                return

            print(f"Loading Univariate GRU model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)

            print(f"Loading Univariate Scaler bundle from {self.scaler_path}...")
            self.scaler_bundle = joblib.load(self.scaler_path)
            self.scaler = self.scaler_bundle['scaler']

            self.is_loaded = True
            self.load_error = None
            print("UnivariateModelService artifacts successfully loaded!")
        except Exception as e:
            self.is_loaded = False
            self.load_error = str(e)
            print(f"Error loading UnivariateModelService artifacts: {e}")

    def predict_next_hour(self, df_user: pd.DataFrame) -> dict:
        """
        Predicts next-hour consumption ($t+1$) from the last 24 hours of user data.
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            raise RuntimeError(f"Univariate model service unavailable: {self.load_error}")

        if len(df_user) < 24:
            raise ValueError(f"Expected at least 24 hourly rows, received {len(df_user)}")

        # Extract last 24 values
        series_24 = df_user['energy_consumption_kwh'].iloc[-24:].values.reshape(-1, 1)
        last_ts = pd.to_datetime(df_user.index[-1])
        next_ts = last_ts + pd.Timedelta(hours=1)

        # Scale
        scaled_seq = self.scaler.transform(series_24)
        input_tensor = np.expand_dims(scaled_seq, axis=0) # shape (1, 24, 1)

        # Predict
        pred_scaled = self.model.predict(input_tensor, verbose=0)
        pred_kwh_raw = float(self.scaler.inverse_transform(pred_scaled)[0, 0])
        pred_kwh = max(0.0, round(pred_kwh_raw, 3))
        pred_wh = round(pred_kwh * 1000.0, 1)

        level = classify_level(pred_kwh)

        return {
            "predicted_consumption": pred_kwh,
            "predicted_wh": pred_wh,
            "unit": "kWh",
            "level": level,
            "timestamp": next_ts.isoformat(),
            "reference_timestamp": last_ts.isoformat(),
            "model_used": "Univariate GRU (Strategy B)"
        }

    def predict_next_24_hours(self, df_user: pd.DataFrame) -> dict:
        """
        Autoregressively rolls the 24-hour sequence forward for 24 steps ahead.
        """
        if not self.is_loaded or self.model is None or self.scaler is None:
            raise RuntimeError(f"Univariate model service unavailable: {self.load_error}")

        if len(df_user) < 24:
            raise ValueError(f"Expected at least 24 hourly rows, received {len(df_user)}")

        current_window = df_user['energy_consumption_kwh'].iloc[-24:].tolist()
        last_ts = pd.to_datetime(df_user.index[-1])
        
        predictions = []

        for step in range(1, 25):
            step_ts = last_ts + pd.Timedelta(hours=step)
            
            # Prepare tensor (1, 24, 1)
            recent_24 = np.array(current_window[-24:]).reshape(-1, 1)
            scaled_seq = self.scaler.transform(recent_24)
            input_tensor = np.expand_dims(scaled_seq, axis=0)

            # Predict
            pred_scaled = self.model.predict(input_tensor, verbose=0)
            pred_kwh_raw = float(self.scaler.inverse_transform(pred_scaled)[0, 0])
            pred_kwh = max(0.0, round(pred_kwh_raw, 3))
            pred_wh = round(pred_kwh * 1000.0, 1)

            level = classify_level(pred_kwh)

            predictions.append({
                "step": step,
                "timestamp": step_ts.isoformat(),
                "hour_label": step_ts.strftime("%H:00"),
                "predicted_consumption": pred_kwh,
                "predicted_wh": pred_wh,
                "unit": "kWh",
                "level": level
            })

            # Append prediction to window for next step
            current_window.append(pred_kwh)

        total_kwh = round(sum(p['predicted_consumption'] for p in predictions), 2)
        peak_item = max(predictions, key=lambda x: x['predicted_consumption'])
        lowest_item = min(predictions, key=lambda x: x['predicted_consumption'])

        return {
            "predictions": predictions,
            "total_predicted_kwh": total_kwh,
            "peak_hour": peak_item['hour_label'],
            "peak_consumption": peak_item['predicted_consumption'],
            "lowest_hour": lowest_item['hour_label'],
            "lowest_consumption": lowest_item['predicted_consumption'],
            "model_used": "Univariate GRU (Strategy B)"
        }
