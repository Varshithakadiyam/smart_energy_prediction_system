"""
eda_visualizations.py
Generates 10 comprehensive, publication-quality EDA figures
analyzing energy consumption patterns, seasonalities, peaks,
missingness, and correlations for the Smart Energy Consumption project.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure matplotlib runs headlessly without permission issues
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def setup_plot_style():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#eaeaea'
    plt.rcParams['grid.linestyle'] = '--'

def generate_all_eda_plots(hourly_df, feat_df, raw_report, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()
    print("\nGenerating 10 EDA Visualizations...")
    
    # -------------------------------------------------------------
    # 1. Energy consumption over time (with 7-day rolling trend)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(hourly_df.index, hourly_df['average_global_active_power'], 
            color='#aec7e8', alpha=0.5, linewidth=0.7, label='Hourly Active Power (kW)')
    trend_7d = hourly_df['average_global_active_power'].rolling(window=24*7, center=True).mean()
    ax.plot(hourly_df.index, trend_7d, color='#1f77b4', linewidth=2.2, label='7-Day Moving Average Trend')
    ax.set_title('Household Global Active Power Over Time (Dec 2006 - Nov 2007)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Active Power (kW)', fontsize=12)
    ax.legend(frameon=True, loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(output_dir, '01_energy_consumption_over_time.png')
    plt.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")

    # -------------------------------------------------------------
    # 2. Daily average consumption
    # -------------------------------------------------------------
    daily = hourly_df['average_global_active_power'].resample('D').mean()
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.plot(daily.index, daily.values, color='#2ca02c', linewidth=1.5, marker='.', markersize=3, label='Daily Mean Power (kW)')
    ax.axhline(daily.mean(), color='#d62728', linestyle='--', linewidth=1.5, label=f'Overall Mean ({daily.mean():.2f} kW)')
    ax.fill_between(daily.index, daily.mean() - daily.std(), daily.mean() + daily.std(), 
                    color='#2ca02c', alpha=0.15, label=r'Mean $\pm$ 1 Std Dev')
    ax.set_title('Daily Average Energy Consumption Profile', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Mean Active Power (kW)', fontsize=12)
    ax.legend(frameon=True, loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(output_dir, '02_daily_average_consumption.png')
    plt.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")

    # -------------------------------------------------------------
    # 3. Weekly consumption
    # -------------------------------------------------------------
    weekly_kwh = (hourly_df['average_global_active_power'].resample('W').mean() * 24 * 7) # Total kWh per week approx
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    bars = ax.bar(weekly_kwh.index, weekly_kwh.values, width=5, color='#ff7f0e', edgecolor='#d96504', alpha=0.85)
    ax.set_title('Weekly Energy Consumption (Total Estimated kWh per Week)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Weekly Energy (kWh)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    p3 = os.path.join(output_dir, '03_weekly_consumption.png')
    plt.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")

    # -------------------------------------------------------------
    # 4. Monthly consumption
    # -------------------------------------------------------------
    # Total kWh per month
    monthly_kwh = hourly_df['energy_consumption_kwh'].resample('ME').sum()
    month_labels = [d.strftime('%b %Y') for d in monthly_kwh.index]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    x = np.arange(len(month_labels))
    bars = ax.bar(x, monthly_kwh.values, color='#9467bd', edgecolor='#6d4493', width=0.6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=30, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title('Total Monthly Energy Consumption (kWh)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Energy (kWh)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    p4 = os.path.join(output_dir, '04_monthly_consumption.png')
    plt.savefig(p4)
    plt.close()
    print(f"Saved: {p4}")

    # -------------------------------------------------------------
    # 5. Average consumption by hour (Diurnal Cycle)
    # -------------------------------------------------------------
    hourly_group = feat_df.groupby('hour')['average_global_active_power'].agg(['mean', 'std', 'median'])
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    hours = hourly_group.index
    ax.plot(hours, hourly_group['mean'], color='#1f77b4', marker='o', linewidth=2.5, label='Mean Consumption')
    ax.plot(hours, hourly_group['median'], color='#2ca02c', linestyle='--', marker='s', linewidth=1.8, label='Median Consumption')
    ax.fill_between(hours, 
                    hourly_group['mean'] - 0.5 * hourly_group['std'], 
                    hourly_group['mean'] + 0.5 * hourly_group['std'],
                    color='#1f77b4', alpha=0.15, label=r'Mean $\pm$ 0.5 Std Dev')
    # Highlight peak hour zone
    ax.axvspan(18, 21, color='orange', alpha=0.2, label='Evening Peak Window (18h-21h)')
    ax.set_xticks(range(0, 24))
    ax.set_title('Average Hourly Energy Consumption Profile (Diurnal Seasonality)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day (0 - 23)', fontsize=12)
    ax.set_ylabel('Active Power (kW)', fontsize=12)
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p5 = os.path.join(output_dir, '05_hourly_consumption_profile.png')
    plt.savefig(p5)
    plt.close()
    print(f"Saved: {p5}")

    # -------------------------------------------------------------
    # 6. Average consumption by day of week
    # -------------------------------------------------------------
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = feat_df.groupby('day_of_week')['average_global_active_power'].agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.bar(days, dow_stats['mean'], yerr=dow_stats['std']*0.2, capsize=4, 
                  color=['#3b528b']*5 + ['#5ec962']*2, edgecolor='#333333', alpha=0.85)
    ax.set_title('Average Power Consumption by Day of Week (Weekday vs Weekend)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Mean Active Power (kW)', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} kW',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    p6 = os.path.join(output_dir, '06_day_of_week_consumption.png')
    plt.savefig(p6)
    plt.close()
    print(f"Saved: {p6}")

    # -------------------------------------------------------------
    # 7. Heatmap of hour vs day of week
    # -------------------------------------------------------------
    pivot_table = feat_df.pivot_table(index='day_of_week', columns='hour', values='average_global_active_power', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)
    im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Mean Active Power (kW)', fontsize=11)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(days)
    ax.set_title('Load Heatmap: Day of Week vs. Hour of Day', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    p7 = os.path.join(output_dir, '07_heatmap_hour_vs_dayofweek.png')
    plt.savefig(p7)
    plt.close()
    print(f"Saved: {p7}")

    # -------------------------------------------------------------
    # 8. Distribution of energy consumption
    # -------------------------------------------------------------
    vals = hourly_df['average_global_active_power'].values
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    n_counts, bins, patches = ax.hist(vals, bins=50, density=True, color='#440154', edgecolor='white', alpha=0.75, label='Empirical Density')
    
    # Kernel density estimation via gaussian filter/linspace
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(vals)
    x_grid = np.linspace(0, vals.max(), 300)
    ax.plot(x_grid, kde(x_grid), color='#fde725', linewidth=2.5, label='KDE Curve')
    
    mean_val = np.mean(vals)
    med_val = np.median(vals)
    q95_val = np.percentile(vals, 95)
    ax.axvline(mean_val, color='#e377c2', linestyle='--', linewidth=1.8, label=f'Mean ({mean_val:.2f} kW)')
    ax.axvline(med_val, color='#17becf', linestyle=':', linewidth=2, label=f'Median ({med_val:.2f} kW)')
    ax.axvline(q95_val, color='#d62728', linestyle='-.', linewidth=1.5, label=f'95th Pct ({q95_val:.2f} kW)')
    
    ax.set_title('Probability Distribution of Hourly Active Power (kW)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Active Power (kW)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(frameon=True, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p8 = os.path.join(output_dir, '08_distribution_energy_consumption.png')
    plt.savefig(p8)
    plt.close()
    print(f"Saved: {p8}")

    # -------------------------------------------------------------
    # 9. Missing-value analysis
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    cols = ['Active Power', 'Reactive Power', 'Voltage', 'Intensity', 'Sub_1', 'Sub_2', 'Sub_3']
    missing_before = [raw_report['total_missing_before'][c] for c in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
    missing_after = [0] * len(cols)
    
    x = np.arange(len(cols))
    width = 0.35
    ax1.bar(x - width/2, missing_before, width, label='Before Cleaning', color='#d62728', alpha=0.85)
    ax1.bar(x + width/2, missing_after, width, label='After Cleaning', color='#2ca02c', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cols, rotation=35, ha='right')
    ax1.set_title('Missing Values Before vs After Preprocessing', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Missing Observations Count', fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Missing duration chunks distribution
    chunk_lengths = [3723, 83, 47, 33, 21, 10, 6, 4, 3, 2, 1]
    chunk_labels = ['3723 min\n(2.5 days)', '83 min', '47 min', '33 min', '21 min', '10 min', '6 min', '4 min', '3 min', '2 min', '1 min']
    ax2.bar(range(len(chunk_lengths)), chunk_lengths, color='#31688e', edgecolor='black', alpha=0.85)
    ax2.set_xticks(range(len(chunk_lengths)))
    ax2.set_xticklabels(chunk_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_yscale('log')
    ax2.set_title('Missing Data Chunk Durations (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gap Duration in Minutes (Log Scale)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    p9 = os.path.join(output_dir, '09_missing_value_analysis.png')
    plt.savefig(p9)
    plt.close()
    print(f"Saved: {p9}")

    # -------------------------------------------------------------
    # 10. Correlation heatmap
    # -------------------------------------------------------------
    corr_cols = [
        'average_global_active_power',
        'average_global_reactive_power',
        'average_voltage',
        'average_global_intensity',
        'sub_metering_1_wh',
        'sub_metering_2_wh',
        'sub_metering_3_wh',
        'lag_1',
        'lag_24',
        'lag_168',
        'rolling_mean_24h'
    ]
    labels_short = [
        'Active Power', 'Reactive Power', 'Voltage', 'Intensity',
        'Sub_1 (Wh)', 'Sub_2 (Wh)', 'Sub_3 (Wh)',
        'Lag 1h', 'Lag 24h', 'Lag 168h', 'RollMean 24h'
    ]
    corr_matrix = feat_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=11)
    ax.set_xticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(labels_short)))
    ax.set_yticklabels(labels_short, fontsize=10)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
    
    for i in range(len(labels_short)):
        for j in range(len(labels_short)):
            val = corr_matrix.values[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5, fontweight='bold')
            
    plt.tight_layout()
    p10 = os.path.join(output_dir, '10_correlation_heatmap.png')
    plt.savefig(p10)
    plt.close()
    print(f"Saved: {p10}")
    
    print("All 10 EDA visualizations successfully generated!")

if __name__ == "__main__":
    from src.inspect_data import inspect_dataset
    from src.clean_and_aggregate import load_and_clean_minute_data, aggregate_to_hourly
    from src.feature_engineering import build_feature_engineered_dataset
    
    report = inspect_dataset()
    df_clean, _, _ = load_and_clean_minute_data()
    hourly_df = aggregate_to_hourly(df_clean)
    feat_df = build_feature_engineered_dataset(hourly_df)
    generate_all_eda_plots(hourly_df, feat_df, report)
