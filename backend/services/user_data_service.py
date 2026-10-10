"""
user_data_service.py
Service for uploading, validating, cleaning, and managing user-specific
electricity consumption CSV datasets.
Preserves user privacy by using an in-memory session cache rather than
persisting files to disk.
"""

import sys
import io
import time
import uuid
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

# In-memory session storage for validated user datasets: {session_id: {"df": DataFrame, "created_at": float, "metadata": dict}}
_USER_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 7200 # 2 hours

def _cleanup_expired_sessions():
    now = time.time()
    expired = [sid for sid, data in _USER_SESSIONS.items() if now - data.get("created_at", 0) > SESSION_TTL_SECONDS]
    for sid in expired:
        _USER_SESSIONS.pop(sid, None)

class UserDataService:
    @staticmethod
    def store_session(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        _cleanup_expired_sessions()
        session_id = str(uuid.uuid4())
        _USER_SESSIONS[session_id] = {
            "df": df.copy(),
            "created_at": time.time(),
            "metadata": metadata
        }
        return session_id

    @staticmethod
    def get_session(session_id: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        _cleanup_expired_sessions()
        session_data = _USER_SESSIONS.get(session_id)
        if not session_data:
            return None
        return session_data["df"].copy(), session_data["metadata"]

    @staticmethod
    def clear_session(session_id: str) -> bool:
        if session_id in _USER_SESSIONS:
            del _USER_SESSIONS[session_id]
            return True
        return False

    @staticmethod
    def validate_and_process_csv(
        file_content: bytes,
        filename: str = "upload.csv",
        max_size_bytes: int = 10 * 1024 * 1024 # 10MB
    ) -> Dict[str, Any]:
        """
        Validates, parses, cleans, and generates personal analytics for an uploaded CSV.
        """
        # 1. File size check
        if len(file_content) > max_size_bytes:
            raise ValueError(f"File size exceeds maximum allowed limit of {max_size_bytes / (1024 * 1024):.0f}MB.")
            
        if len(file_content) == 0:
            raise ValueError("Uploaded file is empty.")

        # 2. Parse CSV
        try:
            df_raw = pd.read_csv(io.BytesIO(file_content))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file: {str(e)}")

        if df_raw.empty or len(df_raw.columns) == 0:
            raise ValueError("CSV contains no columns or rows.")

        total_rows_raw = len(df_raw)

        # 3. Identify Timestamp column
        ts_candidates = [c for c in df_raw.columns if c.lower() in [
            'timestamp', 'datetime', 'date_time', 'time', 'date', 'interval_start'
        ]]
        if not ts_candidates:
            # Fallback to column 0 if parseable
            ts_col = df_raw.columns[0]
        else:
            ts_col = ts_candidates[0]

        # 4. Identify Energy Consumption column
        energy_candidates = [c for c in df_raw.columns if c.lower() in [
            'energy_consumption_kwh', 'consumption_kwh', 'kwh', 'energy_kwh',
            'energy_consumption', 'consumption', 'global_active_power', 'power_kwh'
        ]]
        if not energy_candidates:
            # Look for any numeric column other than ts_col
            num_cols = [c for c in df_raw.columns if c != ts_col]
            if not num_cols:
                raise ValueError("Could not find an energy consumption column. Required: 'energy_consumption_kwh' or 'consumption'.")
            energy_col = num_cols[0]
        else:
            energy_col = energy_candidates[0]

        # 5. Parse and validate timestamps
        try:
            df_raw['parsed_timestamp'] = pd.to_datetime(df_raw[ts_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Invalid timestamp format in column '{ts_col}': {str(e)}")

        invalid_ts_count = int(df_raw['parsed_timestamp'].isna().sum())
        if invalid_ts_count > 0:
            df_raw = df_raw.dropna(subset=['parsed_timestamp'])
            if len(df_raw) == 0:
                raise ValueError("All timestamps in the file are invalid or unparseable.")

        # Remove timezone if present for clean uniform arithmetic
        if hasattr(df_raw['parsed_timestamp'].dt, 'tz') and df_raw['parsed_timestamp'].dt.tz is not None:
            df_raw['parsed_timestamp'] = df_raw['parsed_timestamp'].dt.tz_localize(None)

        # 6. Check duplicates
        duplicate_count = int(df_raw.duplicated(subset=['parsed_timestamp']).sum())
        if duplicate_count > 0:
            df_raw = df_raw.drop_duplicates(subset=['parsed_timestamp'], keep='first')

        # 7. Sort chronologically
        df_raw = df_raw.sort_values('parsed_timestamp').reset_index(drop=True)
        df_raw = df_raw.set_index('parsed_timestamp')

        # 8. Numeric energy validation
        energy_series = pd.to_numeric(df_raw[energy_col], errors='coerce')
        missing_values_count = int(energy_series.isna().sum())
        
        # Check negative readings
        negative_count = int((energy_series < 0).sum())
        if negative_count > 0:
            raise ValueError(f"Detected {negative_count} negative energy consumption readings. Energy consumption cannot be negative.")

        # Unit verification & scaling (Wh vs kWh)
        # If mean is > 50, it is likely Watt-hours, convert to kWh
        mean_raw = float(energy_series.dropna().mean()) if len(energy_series.dropna()) > 0 else 0.0
        is_wh = mean_raw > 50.0
        if is_wh:
            energy_series = energy_series / 1000.0

        # Physical ceiling sanity check (> 50 kWh in 1 single hour for a household is extreme)
        extreme_spikes = int((energy_series > 50.0).sum())
        if extreme_spikes > 0:
            raise ValueError(f"Detected {extreme_spikes} readings exceeding 50.0 kWh/hour. Please verify unit is kWh.")

        df_cleaned = pd.DataFrame({'energy_consumption_kwh': energy_series})

        # 9. Sampling frequency & gap analysis
        if len(df_cleaned) < 24:
            raise ValueError(f"Insufficient historical data: minimum 24 consecutive hourly observations required, received {len(df_cleaned)}.")

        # Check intervals
        time_diffs = df_cleaned.index.to_series().diff().dropna()
        median_diff_sec = time_diffs.dt.total_seconds().median()
        
        # We expect ~1 hour (3600 seconds)
        if not (1800 <= median_diff_sec <= 7200):
            # Try resampling if sub-hourly or irregular
            if median_diff_sec < 1800:
                # Sub-hourly: aggregate by hour (sum or mean)
                df_cleaned = df_cleaned.resample('1h').sum()
            else:
                raise ValueError(f"Irregular sampling frequency detected: median interval is {median_diff_sec/3600:.1f} hours. System requires 1-hour intervals.")

        # Continuous hourly reindex to detect missing time slots
        full_idx = pd.date_range(start=df_cleaned.index.min(), end=df_cleaned.index.max(), freq='1h')
        missing_intervals = len(full_idx) - len(df_cleaned)
        
        imputed_count = 0
        if missing_intervals > 0:
            pct_missing = (missing_intervals / len(full_idx)) * 100.0
            if pct_missing > 15.0:
                raise ValueError(f"Dataset has excessive temporal gaps ({missing_intervals} missing hours, {pct_missing:.1f}% of timeline). Please supply a continuous 24h+ sequence.")
                
            # Reindex with 1h frequency
            df_reindexed = df_cleaned.reindex(full_idx)
            # Check maximum contiguous gap
            is_na = df_reindexed['energy_consumption_kwh'].isna()
            max_consec_gap = (is_na != is_na.shift()).cumsum()[is_na].value_counts().max() if is_na.any() else 0
            if max_consec_gap > 3:
                raise ValueError(f"Detected contiguous gap of {max_consec_gap} hours. Maximum acceptable missing gap for linear interpolation is 3 hours.")
                
            # Time-based linear interpolation for small gaps
            imputed_count = int(df_reindexed['energy_consumption_kwh'].isna().sum())
            df_cleaned = df_reindexed.interpolate(method='time').bfill().ffill()
        else:
            # Handle any internal NaN
            if df_cleaned['energy_consumption_kwh'].isna().any():
                imputed_count = int(df_cleaned['energy_consumption_kwh'].isna().sum())
                df_cleaned = df_cleaned.interpolate(method='time').bfill().ffill()

        accepted_rows = len(df_cleaned)
        if accepted_rows < 24:
            raise ValueError(f"Dataset after cleaning has {accepted_rows} rows. Minimum 24 hourly rows required.")

        # 10. Check feature compatibility
        required_19 = [
            'energy_consumption', 'hour', 'day_of_week', 'is_weekend', 'sin_hour', 'cos_hour',
            'sin_day_of_week', 'cos_day_of_week', 'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24h',
            'rolling_std_24h', 'average_voltage', 'average_global_reactive_power',
            'average_global_intensity', 'sub_metering_1_wh', 'sub_metering_2_wh', 'sub_metering_3_wh'
        ]
        has_all_19 = all(col in df_raw.columns for col in required_19)
        compatible_model = "Multivariate GRU (19-Feature Stage 2 Model)" if has_all_19 else "Univariate GRU (2-Column Consumption Model)"

        # 11. Personal Analytics
        kwh = df_cleaned['energy_consumption_kwh']
        total_kwh = float(kwh.sum())
        avg_kwh = float(kwh.mean())
        min_kwh = float(kwh.min())
        max_kwh = float(kwh.max())
        
        # Diurnal peak hour
        hourly_means = df_cleaned.groupby(df_cleaned.index.hour)['energy_consumption_kwh'].mean()
        peak_hour_int = int(hourly_means.idxmax())
        peak_hour_str = f"{peak_hour_int:02d}:00"
        lowest_hour_int = int(hourly_means.idxmin())
        lowest_hour_str = f"{lowest_hour_int:02d}:00"
        
        days_span = max(1.0, len(df_cleaned) / 24.0)
        daily_avg = total_kwh / days_span

        analytics = {
            "total_consumption_kwh": round(total_kwh, 2),
            "average_hourly_kwh": round(avg_kwh, 3),
            "min_hourly_kwh": round(min_kwh, 3),
            "max_hourly_kwh": round(max_kwh, 3),
            "peak_historical_hour": peak_hour_str,
            "lowest_historical_hour": lowest_hour_str,
            "daily_average_kwh": round(daily_avg, 2),
            "total_days_coverage": round(days_span, 1)
        }

        # 12. Preview sample records (first 5 and last 5)
        preview_records = []
        preview_df = pd.concat([df_cleaned.head(5), df_cleaned.tail(5)]).drop_duplicates()
        for ts, row in preview_df.iterrows():
            preview_records.append({
                "timestamp": ts.isoformat(),
                "hour_label": ts.strftime("%a %H:00"),
                "energy_consumption_kwh": round(float(row['energy_consumption_kwh']), 3)
            })

        # Recent historical slice (last 24 hours) for trajectory charting
        recent_records = []
        recent_slice = df_cleaned.tail(24)
        for ts, row in recent_slice.iterrows():
            recent_records.append({
                "timestamp": ts.isoformat(),
                "hour_label": ts.strftime("%a %H:00"),
                "energy_consumption_kwh": round(float(row['energy_consumption_kwh']), 3)
            })

        # Hourly diurnal profile (0..23 averages)
        diurnal_profile = [
            {
                "hour": int(h),
                "hour_label": f"{int(h):02d}:00",
                "average_kwh": round(float(hourly_means.get(h, 0.0)), 3)
            }
            for h in range(24)
        ]

        # Chronological actual timeline (up to last 168 hours / 7 days)
        timeline_slice = df_cleaned.tail(168)
        historical_timeline = [
            {
                "timestamp": ts.isoformat(),
                "hour_label": ts.strftime("%a %H:00"),
                "energy_consumption_kwh": round(float(row['energy_consumption_kwh']), 3),
                "energy_wh": round(float(row['energy_consumption_kwh']) * 1000.0, 1)
            }
            for ts, row in timeline_slice.iterrows()
        ]

        metadata = {
            "filename": filename,
            "total_rows_raw": total_rows_raw,
            "accepted_rows": accepted_rows,
            "start_timestamp": df_cleaned.index.min().isoformat(),
            "end_timestamp": df_cleaned.index.max().isoformat(),
            "missing_intervals": missing_intervals,
            "duplicate_count": duplicate_count,
            "imputed_count": imputed_count,
            "compatible_model": compatible_model,
            "analytics": analytics,
            "diurnal_profile": diurnal_profile,
            "historical_timeline": historical_timeline
        }

        # Store in session
        session_id = UserDataService.store_session(df_cleaned, metadata)

        return {
            "session_id": session_id,
            "filename": filename,
            "is_valid": True,
            "total_rows": total_rows_raw,
            "accepted_rows": accepted_rows,
            "start_timestamp": metadata["start_timestamp"],
            "end_timestamp": metadata["end_timestamp"],
            "sampling_frequency": "1 hour (regular)",
            "duplicate_count": duplicate_count,
            "missing_count": missing_intervals,
            "imputed_count": imputed_count,
            "compatible_model": compatible_model,
            "preview_data": preview_records,
            "recent_history": recent_records,
            "diurnal_profile": diurnal_profile,
            "historical_timeline": historical_timeline,
            "personal_analytics": analytics
        }

    @staticmethod
    def detect_user_anomalies(df_user: pd.DataFrame, recent_hours: int = 24) -> Dict[str, Any]:
        """
        Calculates diurnal anomaly baseline from user's history and checks recent observations.
        """
        # Baseline from entire user dataset
        hourly_stats = df_user.groupby(df_user.index.hour)['energy_consumption_kwh'].agg(['mean', 'std']).to_dict(orient='index')
        has_sufficient_history = len(df_user) >= 168 # 7 days
        
        recent_slice = df_user.iloc[-recent_hours:]
        results = []
        anomalies_detected = 0

        for ts, row in recent_slice.iterrows():
            hour = ts.hour
            actual_kwh = float(row['energy_consumption_kwh'])
            stats = hourly_stats.get(hour, {'mean': 1.0, 'std': 0.5})
            mean_kwh = stats['mean']
            std_kwh = max(0.1, stats['std'] if not pd.isna(stats['std']) else 0.5)
            
            dev_kwh = actual_kwh - mean_kwh
            z_score = dev_kwh / std_kwh
            is_anomaly = abs(z_score) > 2.5 and has_sufficient_history
            
            if is_anomaly:
                anomalies_detected += 1
                severity = "High" if abs(z_score) > 3.5 else "Moderate"
                status = "⚠️ User-specific high surge" if z_score > 0 else "⚠️ Unusually low demand"
                direction = "above" if z_score > 0 else "below"
                explanation = (
                    f"Consumption ({actual_kwh:.2f} kWh) deviated by {abs(z_score):.1f}σ {direction} "
                    f"your personal {hour:02d}:00 historical baseline ({mean_kwh:.2f} ± {std_kwh:.2f} kWh). "
                    f"Investigation alert for concurrent heavy appliance use or uncharacteristic load (not an equipment fault)."
                )
            else:
                severity = "Normal"
                status = "✓ Normal personal consumption"
                explanation = f"Observation within normal expected personal diurnal variation for {hour:02d}:00 ({mean_kwh:.2f} ± {std_kwh:.2f} kWh)."
                
            results.append({
                "timestamp": ts.isoformat(),
                "hour_label": ts.strftime("%H:00"),
                "actual_consumption_kwh": round(actual_kwh, 3),
                "expected_kwh": round(mean_kwh, 3),
                "deviation_kwh": round(dev_kwh, 3),
                "z_score": round(float(z_score), 2),
                "is_anomaly": bool(is_anomaly),
                "status": status,
                "severity": severity,
                "explanation": explanation
            })

        return {
            "total_evaluated": len(results),
            "anomalies_detected": anomalies_detected,
            "has_sufficient_history": has_sufficient_history,
            "method": "Personal Diurnal Baseline (mean_hour ± 2.5 * std_hour)",
            "results": results
        }

    @staticmethod
    def generate_user_recommendations(predictions: List[Dict[str, Any]], user_analytics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates personalized recommendations based on uploaded history and new forecast.
        """
        recs = []
        peak_hour = user_analytics.get("peak_historical_hour", "19:00")
        avg_kwh = user_analytics.get("average_hourly_kwh", 1.0)
        
        # 1. Peak hour shift
        recs.append({
            "category": "Personalized Peak Shifting",
            "priority": "High",
            "title": f"Mitigate Demand Surges Around {peak_hour}",
            "recommendation": f"Your historical data shows peak consumption centered around {peak_hour}. Shift major loads (laundry, dishwashers, vehicle charging) outside this window.",
            "explanation": f"Customer historical load curve consistently peaks around {peak_hour} at an average of {user_analytics.get('max_hourly_kwh', 2.5):.2f} kWh, contributing to peak tariff exposure and local grid stress.",
            "trigger_pattern": f"Historical load curve peaks at {peak_hour} (daily high across {user_analytics.get('total_days_coverage', 1.0)} days)",
            "action": f"Schedule heavy discretionary appliances (washing machine, dryer, pool pump) before {peak_hour} or after 21:00.",
            "estimated_savings": "Est. 0.35–0.55 kWh per shifted appliance cycle (~15–20% reduction during peak TOU surcharge windows)."
        })
        
        # 2. Forecast peak check
        if predictions:
            max_pred = max(predictions, key=lambda x: x['predicted_consumption'])
            pred_kwh = max_pred['predicted_consumption']
            pred_time = max_pred.get('hour_label', 'Peak')
            if pred_kwh > avg_kwh * 1.5:
                recs.append({
                    "category": "Forecast Surge Alert",
                    "priority": "High",
                    "title": f"Upcoming Demand Spike at {pred_time}",
                    "recommendation": f"Forecast anticipates a surge of ~{pred_kwh:.2f} kWh at {pred_time}. Stagger appliance usage to reduce peak grid stress.",
                    "explanation": f"Neural forecast projects demand to reach {pred_kwh:.2f} kWh at {pred_time}, exceeding your personal average ({avg_kwh:.2f} kWh) by {(pred_kwh/avg_kwh - 1)*100:.0f}%.",
                    "trigger_pattern": f"Day-ahead neural prediction reaches {pred_kwh:.2f} kWh at {pred_time} (> 1.5x personal baseline)",
                    "action": f"Avoid simultaneous operation of oven, electric heater, and laundry equipment at {pred_time}.",
                    "estimated_savings": f"Staggering loads avoids a concentrated {pred_kwh:.2f} kWh spike and reduces demand-capacity threshold charges."
                })
                
        # 3. Base load efficiency
        min_kwh = user_analytics.get("min_hourly_kwh", 0.3)
        recs.append({
            "category": "Standby Optimization",
            "priority": "Medium",
            "title": "Overnight Phantom Load Review",
            "recommendation": "Review continuous background loads (entertainment units, chargers, sub-circuits) during early morning hours to trim baseline demand.",
            "explanation": f"Continuous minimum baseline consumption is measured at {min_kwh:.2f} kWh/h during low-activity night hours ({user_analytics.get('lowest_historical_hour', '03:00')}).",
            "trigger_pattern": f"Overnight baseline draw of {min_kwh:.2f} kWh/h measured at {user_analytics.get('lowest_historical_hour', '03:00')}",
            "action": "Use smart power strips or timers for entertainment centers, idle PC setups, and continuous standby electronics.",
            "estimated_savings": f"Trimming 50–100W of phantom standby saves ~1.2–2.4 kWh/day (~{round(min_kwh*0.3*24, 2)} kWh daily baseline reduction)."
        })
        
        return recs
