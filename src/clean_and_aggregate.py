"""
clean_and_aggregate.py
Handles missing values, datetime conversion, time-based interpolation,
and minute-to-hour aggregation for the Smart Energy Consumption Prediction project.
"""

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inspect_data import find_raw_dataset_path

NUMERICAL_COLS = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3'
]

def load_and_clean_minute_data(csv_path=None):
    """
    Loads raw CSV, constructs timestamp, converts '?' to NaN,
    and performs time-based interpolation.
    """
    if csv_path is None:
        csv_path = find_raw_dataset_path()
        
    print(f"Reading raw data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 1. Datetime creation
    print("Parsing datetime and building timestamp index...")
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.set_index('timestamp').sort_index()
    
    # Check duplicates
    if df.index.duplicated().any():
        print("Warning: Duplicate timestamps detected. Dropping duplicates keeping first...")
        df = df[~df.index.duplicated(keep='first')]
        
    # 2. Convert '?' to NaN and ensure float64
    print("Converting numeric fields and replacing missing indicators ('?')...")
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
        
    missing_before = df[NUMERICAL_COLS].isna().sum().to_dict()
    print(f"Missing values before interpolation: {missing_before}")
    
    # 3. Time-based interpolation
    print("Performing time-based linear interpolation...")
    # Time-based interpolation handles evenly or irregularly spaced time series accurately
    df[NUMERICAL_COLS] = df[NUMERICAL_COLS].interpolate(method='time')
    
    # Boundary fallback in case edge values were NaN
    df[NUMERICAL_COLS] = df[NUMERICAL_COLS].bfill().ffill()
    
    missing_after = df[NUMERICAL_COLS].isna().sum().to_dict()
    print(f"Missing values after interpolation: {missing_after}")
    
    return df, missing_before, missing_after

def aggregate_to_hourly(df_minute):
    """
    Aggregates minute-level data to hourly-level data.
    
    Units & Mathematical Formulation:
    - Global_active_power: minute-averaged active power in kW.
    - Hourly average active power: mean of 60 minute readings in kW.
    - Energy consumption (Wh): Active energy consumed over the hour.
      Each minute kW reading corresponds to: kW * 1000 W/kW * (1/60) h = (kW * 1000 / 60) Wh.
      Hourly energy consumption (Wh) = sum_m=1^60 (Global_active_power_m * 1000 / 60)
                                     = average_global_active_power * 1000 Wh.
    - Energy consumption (kWh): average_global_active_power * 1 hour.
    - Sub-meterings: Sub_metering_1, 2, 3 in raw data are active energy in Wh per minute.
      Summing over 60 minutes gives sub-metering active energy in Wh for the hour.
    """
    print("Aggregating minute data to hourly resolution...")
    
    # Count minutes per hour to filter incomplete boundary hours
    minute_counts = df_minute['Global_active_power'].resample('1h').count()
    full_hours = minute_counts[minute_counts == 60].index
    
    # Hourly means
    hourly_mean = df_minute[NUMERICAL_COLS].resample('1h').mean()
    
    # Hourly sums for sub-meterings (Wh per hour)
    sub_sums = df_minute[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].resample('1h').sum()
    
    # Filter to complete hours
    hourly_mean = hourly_mean.loc[full_hours]
    sub_sums = sub_sums.loc[full_hours]
    
    hourly_df = pd.DataFrame(index=full_hours)
    
    # Core target and power features
    hourly_df['average_global_active_power'] = hourly_mean['Global_active_power'] # in kW
    hourly_df['energy_consumption'] = hourly_mean['Global_active_power'] * 1000.0  # in Watt-hours (Wh)
    hourly_df['energy_consumption_kwh'] = hourly_mean['Global_active_power']      # in Kilowatt-hours (kWh)
    
    # Electrical network parameters
    hourly_df['average_global_reactive_power'] = hourly_mean['Global_reactive_power'] # in kVAR
    hourly_df['average_voltage'] = hourly_mean['Voltage']                             # in Volts
    hourly_df['average_global_intensity'] = hourly_mean['Global_intensity']           # in Amperes
    
    # Sub-metering averages (rate in Wh/min)
    hourly_df['average_sub_metering_1'] = hourly_mean['Sub_metering_1']
    hourly_df['average_sub_metering_2'] = hourly_mean['Sub_metering_2']
    hourly_df['average_sub_metering_3'] = hourly_mean['Sub_metering_3']
    
    # Sub-metering total energy per hour (in Wh)
    hourly_df['sub_metering_1_wh'] = sub_sums['Sub_metering_1']
    hourly_df['sub_metering_2_wh'] = sub_sums['Sub_metering_2']
    hourly_df['sub_metering_3_wh'] = sub_sums['Sub_metering_3']
    
    # Unmetered remainder energy in Wh: Total Wh - (Sub1 + Sub2 + Sub3)
    sub_total = hourly_df['sub_metering_1_wh'] + hourly_df['sub_metering_2_wh'] + hourly_df['sub_metering_3_wh']
    hourly_df['sub_metering_remainder_wh'] = np.maximum(0.0, hourly_df['energy_consumption'] - sub_total)
    
    hourly_df.index.name = 'timestamp'
    print(f"Hourly aggregation complete: {hourly_df.shape[0]} full hourly rows, {hourly_df.shape[1]} columns.")
    return hourly_df

if __name__ == "__main__":
    df_clean, m_before, m_after = load_and_clean_minute_data()
    hourly_df = aggregate_to_hourly(df_clean)
    print("Hourly head:")
    print(hourly_df.head(3))
