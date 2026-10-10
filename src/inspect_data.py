"""
inspect_data.py
Thorough dataset inspection for the Smart Energy Consumption Prediction project.
Reports dimensions, types, missing values, duplicates, continuity, and statistical distributions.
"""

import os
import pandas as pd
import numpy as np

def find_raw_dataset_path(base_dir="."):
    candidate_names = [
        "household_power_consumption_part_1(1).csv",
        "household_power_consumption_part_1.csv"
    ]
    for name in candidate_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find raw dataset among {candidate_names} in {base_dir}")

def inspect_dataset(csv_path=None):
    if csv_path is None:
        csv_path = find_raw_dataset_path()
    
    print(f"Loading raw dataset from: {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    report = {}
    report['raw_path'] = csv_path
    report['rows'] = len(df)
    report['cols'] = len(df.columns)
    report['columns'] = list(df.columns)
    report['initial_dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Missing values: '?' and NaN
    qmark_counts = {col: int((df[col] == '?').sum()) if df[col].dtype == object else 0 for col in df.columns}
    nan_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    total_missing_before = {col: qmark_counts[col] + nan_counts[col] for col in df.columns}
    
    report['qmark_counts'] = qmark_counts
    report['nan_counts'] = nan_counts
    report['total_missing_before'] = total_missing_before
    
    # Duplicates
    report['duplicate_rows'] = int(df.duplicated().sum())
    
    # Datetime parse
    ts = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    report['invalid_timestamps'] = int(ts.isna().sum())
    report['duplicate_timestamps'] = int(ts.duplicated().sum())
    report['min_timestamp'] = str(ts.min())
    report['max_timestamp'] = str(ts.max())
    report['is_chronological'] = bool(ts.is_monotonic_increasing)
    
    # Continuity
    expected_minutes = int((ts.max() - ts.min()).total_seconds() / 60) + 1
    actual_minutes = len(ts)
    missing_minute_slots = expected_minutes - actual_minutes
    report['expected_minutes'] = expected_minutes
    report['actual_minutes'] = actual_minutes
    report['missing_minute_slots'] = missing_minute_slots
    report['is_continuous'] = (missing_minute_slots == 0)
    
    # Convert numerical columns for range and outlier analysis
    num_cols = [c for c in df.columns if c not in ['Date', 'Time']]
    num_stats = {}
    df_clean_num = df.copy()
    for c in num_cols:
        df_clean_num[c] = pd.to_numeric(df_clean_num[c].replace('?', np.nan), errors='coerce')
        desc = df_clean_num[c].describe()
        q1 = desc['25%']
        q3 = desc['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = int(((df_clean_num[c] < lower_bound) | (df_clean_num[c] > upper_bound)).sum())
        negatives_count = int((df_clean_num[c] < 0).sum())
        
        num_stats[c] = {
            'count_valid': int(desc['count']),
            'mean': float(desc['mean']),
            'std': float(desc['std']),
            'min': float(desc['min']),
            'max': float(desc['max']),
            'q25': float(q1),
            'median': float(desc['50%']),
            'q75': float(q3),
            'iqr': float(iqr),
            'outliers_count_iqr': outliers_count,
            'negative_count': negatives_count
        }
    report['numerical_stats'] = num_stats
    
    # Unique values summary
    unique_counts = {c: int(df[c].nunique()) for c in df.columns}
    report['unique_counts'] = unique_counts
    
    return report

def print_inspection_report(report):
    print("=" * 80)
    print("           RAW DATASET INSPECTION REPORT")
    print("=" * 80)
    print(f"File Path: {report['raw_path']}")
    print(f"Total Rows: {report['rows']:,}")
    print(f"Total Columns: {report['cols']}")
    print(f"Columns: {', '.join(report['columns'])}")
    print(f"Duplicate Rows: {report['duplicate_rows']}")
    print("\n--- Datetime Analysis ---")
    print(f"Date Range: {report['min_timestamp']} to {report['max_timestamp']}")
    print(f"Chronologically Ordered: {report['is_chronological']}")
    print(f"Duplicate Timestamps: {report['duplicate_timestamps']}")
    print(f"Expected Continuous Minutes: {report['expected_minutes']:,}")
    print(f"Actual Minute Observations: {report['actual_minutes']:,}")
    print(f"Missing Minute Time Gaps: {report['missing_minute_slots']} (Continuous: {report['is_continuous']})")
    
    print("\n--- Missing Values Before Cleaning ---")
    for col in report['columns']:
        q = report['qmark_counts'][col]
        n = report['nan_counts'][col]
        tot = report['total_missing_before'][col]
        pct = (tot / report['rows']) * 100
        print(f"  {col:<25}: {tot:>6} missing ({pct:.3f}%) [? count: {q}, NaN count: {n}]")
        
    print("\n--- Numerical Range & Distribution Checks ---")
    header = f"{'Column':<23} {'Min':>8} {'Median':>8} {'Mean':>8} {'Max':>8} {'Negatives':>9} {'IQR Outliers':>12}"
    print(header)
    print("-" * len(header))
    for col, st in report['numerical_stats'].items():
        print(f"{col:<23} {st['min']:>8.3f} {st['median']:>8.3f} {st['mean']:>8.3f} {st['max']:>8.3f} {st['negative_count']:>9} {st['outliers_count_iqr']:>12}")
    print("=" * 80)

if __name__ == "__main__":
    rep = inspect_dataset()
    print_inspection_report(rep)
