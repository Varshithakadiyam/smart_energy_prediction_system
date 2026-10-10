"""
eda_stage2.py
Generates the 10 EDA visualizations for reports/figures/
focusing on energy_consumption (Wh and kWh equivalents),
revealing diurnal/weekly seasonality, macro trends, rolling stats,
and representative 7-day appliance load dynamics.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def setup_style():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#eaeaea'
    plt.rcParams['grid.linestyle'] = '--'

def generate_eda_figures(df_path="data/processed/hourly_energy_consumption.csv", output_dir="reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    setup_style()
    print(f"Loading processed hourly dataset from {df_path}...")
    df = pd.read_csv(df_path, index_col=0, parse_dates=True)
    
    # Target column is energy_consumption (in Wh)
    target = df['energy_consumption'] # in Wh
    target_kwh = df['energy_consumption_kwh'] # in kWh
    
    # -------------------------------------------------------------------
    # 1. Complete energy consumption time series
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(df.index, target_kwh, color='#1f77b4', alpha=0.55, linewidth=0.7, label='Hourly Energy Consumption (kWh)')
    trend = target_kwh.rolling(window=24*7, center=True).mean()
    ax.plot(df.index, trend, color='#d62728', linewidth=2.0, label='7-Day Centered Moving Average')
    ax.set_title('Complete Hourly Energy Consumption Time Series (Dec 2006 – Nov 2007)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    p1 = os.path.join(output_dir, '01_complete_energy_consumption_time_series.png')
    plt.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")

    # -------------------------------------------------------------------
    # 2. Daily average energy consumption
    # -------------------------------------------------------------------
    daily = target_kwh.resample('D').mean()
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(daily.index, daily.values, color='#2ca02c', linewidth=1.5, marker='.', markersize=3, label='Daily Average (kWh)')
    ax.axhline(daily.mean(), color='#e377c2', linestyle='--', linewidth=1.8, label=f'Overall Mean ({daily.mean():.2f} kWh)')
    ax.fill_between(daily.index, daily.mean() - daily.std(), daily.mean() + daily.std(), color='#2ca02c', alpha=0.15, label=r'Mean $\pm$ 1 Std Dev')
    ax.set_title('Daily Average Energy Consumption Trend', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Daily Mean Energy (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p2 = os.path.join(output_dir, '02_daily_average_energy_consumption.png')
    plt.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------------------------
    # 3. Weekly average energy consumption
    # -------------------------------------------------------------------
    weekly = target_kwh.resample('W').mean()
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.bar(weekly.index, weekly.values, width=5, color='#ff7f0e', edgecolor='#d96504', alpha=0.85, label='Weekly Average (kWh)')
    ax.plot(weekly.index, weekly.values, color='#b34700', marker='o', linewidth=1.5)
    ax.set_title('Weekly Average Energy Consumption', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Week', fontsize=11)
    ax.set_ylabel('Weekly Mean Energy (kWh)', fontsize=11)
    plt.tight_layout()
    p3 = os.path.join(output_dir, '03_weekly_average_energy_consumption.png')
    plt.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")

    # -------------------------------------------------------------------
    # 4. Monthly average energy consumption
    # -------------------------------------------------------------------
    monthly = target_kwh.resample('ME').mean()
    month_labels = [d.strftime('%b %Y') for d in monthly.index]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    bars = ax.bar(range(len(monthly)), monthly.values, color='#9467bd', edgecolor='#6d4493', width=0.6, alpha=0.85)
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(month_labels, rotation=30, ha='right', fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f} kWh',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Monthly Average Hourly Energy Consumption (kWh)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Mean Energy (kWh)', fontsize=11)
    plt.tight_layout()
    p4 = os.path.join(output_dir, '04_monthly_average_energy_consumption.png')
    plt.savefig(p4)
    plt.close()
    print(f"Saved: {p4}")

    # -------------------------------------------------------------------
    # 5. Energy consumption by hour of day (Diurnal Profile)
    # -------------------------------------------------------------------
    hour_stats = df.groupby('hour')['energy_consumption_kwh'].agg(['mean', 'median', 'std'])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    hrs = hour_stats.index
    ax.plot(hrs, hour_stats['mean'], color='#1f77b4', marker='o', linewidth=2.5, label='Mean Consumption')
    ax.plot(hrs, hour_stats['median'], color='#2ca02c', linestyle='--', marker='s', linewidth=1.8, label='Median Consumption')
    ax.fill_between(hrs, hour_stats['mean'] - 0.5 * hour_stats['std'], hour_stats['mean'] + 0.5 * hour_stats['std'],
                    color='#1f77b4', alpha=0.15, label=r'Mean $\pm$ 0.5 Std Dev')
    ax.axvspan(18, 21, color='orange', alpha=0.2, label='Peak Period (18:00 – 21:00)')
    ax.axvspan(1, 5, color='gray', alpha=0.15, label='Low Period (01:00 – 05:00)')
    ax.set_xticks(range(24))
    ax.set_title('Hourly Diurnal Energy Consumption Profile', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Hour of Day (0–23)', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    p5 = os.path.join(output_dir, '05_energy_consumption_by_hour_of_day.png')
    plt.savefig(p5)
    plt.close()
    print(f"Saved: {p5}")

    # -------------------------------------------------------------------
    # 6. Energy consumption by day of week
    # -------------------------------------------------------------------
    dow_stats = df.groupby('day_of_week')['energy_consumption_kwh'].agg(['mean', 'std'])
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(day_names, dow_stats['mean'], yerr=dow_stats['std']*0.15, capsize=4,
                  color=['#3b528b']*5 + ['#5ec962']*2, edgecolor='#333333', alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f} kWh',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Average Energy Consumption by Day of Week (Weekday vs. Weekend)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Mean Energy Consumption (kWh)', fontsize=11)
    plt.tight_layout()
    p6 = os.path.join(output_dir, '06_energy_consumption_by_day_of_week.png')
    plt.savefig(p6)
    plt.close()
    print(f"Saved: {p6}")

    # -------------------------------------------------------------------
    # 7. Hour x Day-of-week heatmap
    # -------------------------------------------------------------------
    heatmap_matrix = df.pivot_table(index='day_of_week', columns='hour', values='energy_consumption_kwh', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    im = ax.imshow(heatmap_matrix.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Mean Energy (kWh)', fontsize=11)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(day_names)
    ax.set_title('Load Heatmap: Hour of Day × Day of Week', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Hour of Day (0–23)', fontsize=11)
    ax.set_ylabel('Day of Week', fontsize=11)
    plt.tight_layout()
    p7 = os.path.join(output_dir, '07_hour_x_day_of_week_heatmap.png')
    plt.savefig(p7)
    plt.close()
    print(f"Saved: {p7}")

    # -------------------------------------------------------------------
    # 8. Distribution of energy consumption
    # -------------------------------------------------------------------
    vals = target_kwh.values
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.hist(vals, bins=50, density=True, color='#440154', edgecolor='white', alpha=0.75, label='Empirical Distribution')
    kde = gaussian_kde(vals)
    xg = np.linspace(0, vals.max(), 300)
    ax.plot(xg, kde(xg), color='#fde725', linewidth=2.5, label='KDE Density')
    mean_val, med_val, q95 = np.mean(vals), np.median(vals), np.percentile(vals, 95)
    ax.axvline(mean_val, color='#e377c2', linestyle='--', linewidth=1.8, label=f'Mean: {mean_val:.2f} kWh')
    ax.axvline(med_val, color='#17becf', linestyle=':', linewidth=2, label=f'Median: {med_val:.2f} kWh')
    ax.axvline(q95, color='#d62728', linestyle='-.', linewidth=1.5, label=f'95th Pct: {q95:.2f} kWh')
    ax.set_title('Probability Distribution of Hourly Energy Consumption (kWh)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Energy Consumption (kWh)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p8 = os.path.join(output_dir, '08_distribution_of_energy_consumption.png')
    plt.savefig(p8)
    plt.close()
    print(f"Saved: {p8}")

    # -------------------------------------------------------------------
    # 9. Rolling mean and rolling standard deviation
    # -------------------------------------------------------------------
    rm_24 = target_kwh.rolling(window=24).mean()
    rstd_24 = target_kwh.rolling(window=24).std()
    rm_168 = target_kwh.rolling(window=168).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=150)
    ax1.plot(df.index, target_kwh, color='#cccccc', alpha=0.4, label='Hourly Actual')
    ax1.plot(df.index, rm_24, color='#1f77b4', linewidth=1.2, label='24-Hour Rolling Mean')
    ax1.plot(df.index, rm_168, color='#d62728', linewidth=1.8, label='168-Hour (7-Day) Rolling Mean')
    ax1.set_title('Rolling Mean Smoothing Trends (24-Hour & 7-Day)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy (kWh)', fontsize=10)
    ax1.legend(loc='upper right')
    
    ax2.plot(df.index, rstd_24, color='#2ca02c', linewidth=1.2, label='24-Hour Rolling Std Dev (Volatility)')
    ax2.set_title('24-Hour Rolling Volatility (Standard Deviation)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timestamp', fontsize=11)
    ax2.set_ylabel('Std Dev (kWh)', fontsize=10)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    p9 = os.path.join(output_dir, '09_rolling_mean_and_rolling_std.png')
    plt.savefig(p9)
    plt.close()
    print(f"Saved: {p9}")

    # -------------------------------------------------------------------
    # 10. Actual consumption for a representative 7-day period
    # -------------------------------------------------------------------
    # Select a clear representative week in winter/spring: e.g., 2007-01-15 to 2007-01-22
    week_sample = target_kwh.loc['2007-01-15 00:00:00':'2007-01-21 23:00:00']
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(week_sample.index, week_sample.values, color='#1f77b4', marker='o', markersize=4, linewidth=1.8, label='Hourly Consumption (kWh)')
    for day in pd.date_range('2007-01-15', '2007-01-21'):
        peak_time = pd.Timestamp(day).replace(hour=19)
        if peak_time in week_sample.index:
            ax.scatter([peak_time], [week_sample.loc[peak_time]], color='#d62728', s=60, zorder=5)
    ax.scatter([], [], color='#d62728', s=60, label='Evening Peak (19:00)')
    ax.set_title('Representative 7-Day Load Dynamics (Jan 15 – Jan 21, 2007)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Timestamp (Hourly Resolution)', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p10 = os.path.join(output_dir, '10_actual_consumption_representative_7_day.png')
    plt.savefig(p10)
    plt.close()
    print(f"Saved: {p10}")
    
    print("All 10 Stage 2 EDA visualizations successfully created in reports/figures/!")

if __name__ == "__main__":
    generate_eda_figures()
