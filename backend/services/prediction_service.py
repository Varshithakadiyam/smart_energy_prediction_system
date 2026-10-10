"""
prediction_service.py
Provides single-step (next-hour) and multi-step (next 24 hours)
autoregressive rolling energy forecasts.
"""

import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from backend.services.model_service import ModelService

TIER_LOW = 0.30
TIER_NORMAL = 1.65
TIER_HIGH = 3.20

def classify_level(kwh):
    if kwh < TIER_LOW:
        return "Low"
    elif kwh <= TIER_NORMAL:
        return "Normal"
    elif kwh <= TIER_HIGH:
        return "High"
    else:
        return "Unusually High"

class PredictionService:
    def __init__(self):
        self.model_svc = ModelService.get_instance()

    def predict_next_hour(self, df_24h):
        """
        df_24h: pandas DataFrame with at least 24 rows and required features.
        """
        if len(df_24h) < 24:
            raise ValueError(f"Expected at least 24 hourly rows, received {len(df_24h)}")
            
        recent_24 = df_24h.iloc[-24:][self.model_svc.feature_cols]
        last_ts = pd.to_datetime(recent_24.index[-1])
        next_ts = last_ts + pd.Timedelta(hours=1)
        
        # Scale
        scaled_seq = self.model_svc.feature_scaler.transform(recent_24)
        input_tensor = np.expand_dims(scaled_seq, axis=0) # (1, 24, D)
        
        # Inference
        pred_scaled = self.model_svc.model.predict(input_tensor, verbose=0)
        pred_wh = float(self.model_svc.target_scaler.inverse_transform(pred_scaled)[0, 0])
        pred_kwh = max(0.0, round(pred_wh / 1000.0, 3))
        
        level = classify_level(pred_kwh)
        
        return {
            "predicted_consumption": pred_kwh,
            "predicted_wh": round(pred_wh, 2),
            "unit": "kWh",
            "level": level,
            "timestamp": next_ts.isoformat(),
            "reference_timestamp": last_ts.isoformat()
        }

    def predict_next_24_hours(self, df_24h):
        """
        Autoregressively rolls the 24-hour window forward for 24 steps ahead.
        """
        if len(df_24h) < 24:
            raise ValueError(f"Expected at least 24 hourly rows, received {len(df_24h)}")
            
        working_df = df_24h.iloc[-24:].copy()
        predictions = []
        
        current_window = working_df.copy()
        
        for step in range(1, 25):
            feat_subset = current_window.iloc[-24:][self.model_svc.feature_cols]
            last_ts = pd.to_datetime(feat_subset.index[-1])
            step_ts = last_ts + pd.Timedelta(hours=1)
            
            # Scale & predict
            scaled_seq = self.model_svc.feature_scaler.transform(feat_subset)
            input_tensor = np.expand_dims(scaled_seq, axis=0)
            pred_scaled = self.model_svc.model.predict(input_tensor, verbose=0)
            pred_wh = float(self.model_svc.target_scaler.inverse_transform(pred_scaled)[0, 0])
            pred_kwh = max(0.0, round(pred_wh / 1000.0, 2))
            
            level = classify_level(pred_kwh)
            
            predictions.append({
                "step": step,
                "timestamp": step_ts.isoformat(),
                "hour_label": step_ts.strftime("%H:00"),
                "predicted_consumption": pred_kwh,
                "predicted_wh": round(pred_wh, 1),
                "unit": "kWh",
                "level": level
            })
            
            # Construct synthetic next row for roll-forward
            # Update temporal features for next step
            new_hour = step_ts.hour
            new_dow = step_ts.dayofweek
            new_is_weekend = int(new_dow >= 5)
            
            new_row = {}
            for col in self.model_svc.feature_cols:
                # Default carry over previous row
                new_row[col] = current_window.iloc[-1][col]
                
            new_row['energy_consumption'] = pred_wh
            new_row['hour'] = new_hour
            new_row['day_of_week'] = new_dow
            new_row['is_weekend'] = new_is_weekend
            new_row['sin_hour'] = np.sin(2 * np.pi * new_hour / 24.0)
            new_row['cos_hour'] = np.cos(2 * np.pi * new_hour / 24.0)
            new_row['sin_day_of_week'] = np.sin(2 * np.pi * new_dow / 7.0)
            new_row['cos_day_of_week'] = np.cos(2 * np.pi * new_dow / 7.0)
            new_row['lag_1'] = current_window.iloc[-1]['energy_consumption']
            
            # 24h lag: value from 24 steps back
            if len(current_window) >= 24:
                new_row['lag_24'] = current_window.iloc[-24]['energy_consumption']
                
            # Rolling mean 24h
            recent_energy = current_window.iloc[-23:]['energy_consumption'].tolist() + [pred_wh]
            new_row['rolling_mean_24h'] = float(np.mean(recent_energy))
            new_row['rolling_std_24h'] = float(np.std(recent_energy))
            
            # Append new row
            new_df_row = pd.DataFrame([new_row], index=[step_ts])
            current_window = pd.concat([current_window, new_df_row])
            
        return predictions
