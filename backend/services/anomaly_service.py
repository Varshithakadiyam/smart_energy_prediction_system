"""
anomaly_service.py
Statistical anomaly detection leveraging dynamic diurnal (hour-of-day)
baseline distributions to prevent false positives during legitimate peak hours.
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

class AnomalyService:
    def __init__(self):
        self.model_svc = ModelService.get_instance()
        self._hour_baselines = self._calculate_hour_baselines()

    def _calculate_hour_baselines(self):
        """Calculates historical mean and std dev for each hour of day (0-23)."""
        df = self.model_svc.sample_df
        if df is None:
            # Fallback typical values if df is unavailable
            return {h: {'mean': 1.0, 'std': 0.5} for h in range(24)}
            
        hour_stats = df.groupby('hour')['energy_consumption'].agg(['mean', 'std']).to_dict(orient='index')
        return hour_stats

    def detect_anomalies(self, observations):
        """
        observations: list of dicts or DataFrame with 'timestamp' and 'energy_consumption' (or 'actual_consumption').
        """
        if isinstance(observations, list):
            df_obs = pd.DataFrame(observations)
        else:
            df_obs = observations.copy()
            
        # Standardize column names
        if 'energy_consumption' in df_obs.columns and 'actual_consumption' not in df_obs.columns:
            df_obs['actual_consumption'] = df_obs['energy_consumption']
        elif 'actual_consumption' not in df_obs.columns and 'consumption' in df_obs.columns:
            df_obs['actual_consumption'] = df_obs['consumption']
            
        if 'actual_consumption' not in df_obs.columns:
            raise ValueError("Input data must contain an 'actual_consumption' or 'energy_consumption' column")
            
        if 'timestamp' not in df_obs.columns and df_obs.index.name == 'timestamp':
            df_obs['timestamp'] = df_obs.index
            
        results = []
        anomaly_count = 0
        
        for _, row in df_obs.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            hour = ts.hour
            raw_val = float(row['actual_consumption'])
            # Auto-detect kWh vs Wh: values under 50 are kWh, values above are Wh
            if raw_val <= 50.0:
                actual_wh = raw_val * 1000.0
            else:
                actual_wh = raw_val
            
            # Lookup diurnal baseline
            baseline = self._hour_baselines.get(hour, {'mean': 1000.0, 'std': 500.0})
            mean_wh = baseline['mean']
            std_wh = max(100.0, baseline['std']) # Prevent division by zero
            
            deviation_wh = actual_wh - mean_wh
            z_score = deviation_wh / std_wh
            
            # Convert to kWh for user-friendly reporting
            actual_kwh = round(actual_wh / 1000.0, 2)
            expected_kwh = round(mean_wh / 1000.0, 2)
            deviation_kwh = round(deviation_wh / 1000.0, 2)
            
            # Threshold: 2.5 standard deviations from the hourly expectation
            is_anomaly = abs(z_score) > 2.5
            
            if is_anomaly:
                anomaly_count += 1
                status = "⚠️ High consumption detected" if z_score > 0 else "⚠️ Unusually low consumption detected"
            else:
                status = "✓ Normal consumption"
                
            results.append({
                "timestamp": ts.isoformat(),
                "actual_consumption": actual_kwh,
                "expected_consumption": expected_kwh,
                "deviation": deviation_kwh,
                "z_score": round(float(z_score), 2),
                "is_anomaly": bool(is_anomaly),
                "anomaly_status": status
            })
            
        return {
            "total_evaluated": len(results),
            "anomalies_detected": anomaly_count,
            "has_anomalies": anomaly_count > 0,
            "summary": "⚠️ Unusual energy consumption detected" if anomaly_count > 0 else "✓ No significant anomalies detected",
            "results": results
        }
