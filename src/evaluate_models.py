import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent if 'src' in str(Path(__file__).resolve().parent) else Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import src.compat
"""
evaluate_models.py
Evaluates LSTM, GRU, and 3 statistical baselines strictly on the unseen test set.
Converts predictions to original units (Wh and kWh), computes MAE, MSE, RMSE, MAPE, R2,
saves data/processed/test_predictions.csv, generates 5 prediction visual analysis plots,
and persists models/best_energy_model.keras.
"""

import sys
import os
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['KERAS_HOME'] = os.path.abspath('.keras')
# sys.modules['pyarrow'] = None
# sys.modules['pyarrow._generated_version'] = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import joblib
from tensorflow import keras

from src.baselines import evaluate_baselines, calculate_metrics

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#eaeaea'
    plt.rcParams['grid.linestyle'] = '--'

def evaluate_all(data, output_fig_dir="reports/figures", output_csv_path="data/processed/test_predictions.csv"):
    setup_style()
    os.makedirs(output_fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    X_test = data['X_test']
    y_test_scaled = data['y_test']
    raw_y_test = data['raw_y_test'].flatten()
    ts_test = pd.to_datetime(data['ts_test'])
    target_scaler = data['scaler_bundle']['target_scaler']
    feat_scaler = data['scaler_bundle']['feature_scaler']
    
    # 1. Evaluate Baselines
    N, L, D = X_test.shape
    unscaled_test_seq = np.zeros_like(X_test)
    for i in range(N):
        unscaled_test_seq[i] = feat_scaler.inverse_transform(X_test[i])
    base_results = evaluate_baselines(unscaled_test_seq, raw_y_test, target_feature_idx=0)
    
    # 2. Evaluate LSTM
    print("\nLoading LSTM model for evaluation...")
    lstm_model = keras.models.load_model("models/lstm_energy_model.keras")
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_pred_wh = target_scaler.inverse_transform(lstm_pred_scaled).flatten()
    lstm_metrics = calculate_metrics(raw_y_test, lstm_pred_wh)
    
    # 3. Evaluate GRU
    print("Loading GRU model for evaluation...")
    gru_model = keras.models.load_model("models/gru_energy_model.keras")
    gru_pred_scaled = gru_model.predict(X_test, verbose=0)
    gru_pred_wh = target_scaler.inverse_transform(gru_pred_scaled).flatten()
    gru_metrics = calculate_metrics(raw_y_test, gru_pred_wh)
    
    # 4. Compile Comparison
    comparison = {
        'Previous Hour': base_results['Previous Hour']['metrics'],
        'Previous Day': base_results['Previous Day']['metrics'],
        'Moving Average': base_results['Moving Average']['metrics'],
        'LSTM': lstm_metrics,
        'GRU': gru_metrics
    }
    
    print("\n" + "=" * 90)
    print(f"{'Model':<18} {'MAE (Wh)':>10} {'RMSE (Wh)':>11} {'MAPE (%)':>10} {'R2':>8} {'MAE (kWh)':>11} {'RMSE (kWh)':>12}")
    print("-" * 90)
    for m_name, m_val in comparison.items():
        mae_kwh = m_val['MAE'] / 1000.0
        rmse_kwh = m_val['RMSE'] / 1000.0
        print(f"{m_name:<18} {m_val['MAE']:>10.2f} {m_val['RMSE']:>11.2f} {m_val['MAPE']:>9.2f}% {m_val['R2']:>8.4f} {mae_kwh:>11.3f} {rmse_kwh:>12.3f}")
    print("=" * 90)
    
    # Determine best model
    # Priority: lowest test RMSE and highest R2
    best_name = "LSTM" if lstm_metrics['RMSE'] <= gru_metrics['RMSE'] else "GRU"
    best_pred_wh = lstm_pred_wh if best_name == "LSTM" else gru_pred_wh
    best_model_path = "models/lstm_energy_model.keras" if best_name == "LSTM" else "models/gru_energy_model.keras"
    shutil.copy(best_model_path, "models/best_energy_model.keras")
    print(f"\nBest Model selected: {best_name} (saved as models/best_energy_model.keras)")
    
    # 5. Save test_predictions.csv
    pred_df = pd.DataFrame({
        'timestamp': ts_test,
        'actual_consumption': raw_y_test,
        'predicted_consumption': best_pred_wh,
        'error': raw_y_test - best_pred_wh
    })
    pred_df['actual_consumption_kwh'] = pred_df['actual_consumption'] / 1000.0
    pred_df['predicted_consumption_kwh'] = pred_df['predicted_consumption'] / 1000.0
    pred_df['error_kwh'] = pred_df['error'] / 1000.0
    pred_df.to_csv(output_csv_path, index=False)
    print(f"Saved test predictions table to: {output_csv_path}")
    
    # 6. Generate 5 Prediction Analysis Figures
    print("\nGenerating 5 Prediction Visualizations in reports/figures/...")
    
    # Figure 13: Actual vs predicted for complete test period
    fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
    ax.plot(ts_test, raw_y_test / 1000.0, color='#1f77b4', alpha=0.7, linewidth=0.9, label='Actual Consumption (kWh)')
    ax.plot(ts_test, best_pred_wh / 1000.0, color='#d62728', alpha=0.85, linewidth=1.1, linestyle='--', label=f'{best_name} Forecast (kWh)')
    ax.set_title(f'Actual vs. Predicted Energy Consumption Across Full Test Period ({best_name})', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p13 = os.path.join(output_fig_dir, "13_actual_vs_predicted_full_test.png")
    plt.savefig(p13)
    plt.close()
    print(f"Saved: {p13}")
    
    # Figure 14: Actual vs predicted for 24 hours
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    zoom_24_idx = slice(100, 124) # 24 consecutive hours
    ax.plot(ts_test[zoom_24_idx], raw_y_test[zoom_24_idx] / 1000.0, color='#1f77b4', marker='o', linewidth=2.0, label='Actual (kWh)')
    ax.plot(ts_test[zoom_24_idx], best_pred_wh[zoom_24_idx] / 1000.0, color='#d62728', marker='s', linestyle='--', linewidth=2.0, label=f'{best_name} Predicted (kWh)')
    ax.set_title(f'Actual vs. Predicted Consumption Zoom (24-Hour Day-Ahead Window)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    p14 = os.path.join(output_fig_dir, "14_actual_vs_predicted_24h.png")
    plt.savefig(p14)
    plt.close()
    print(f"Saved: {p14}")
    
    # Figure 15: Actual vs predicted for 7 days
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    zoom_7d_idx = slice(100, 100 + 168) # 7 days = 168 hours
    ax.plot(ts_test[zoom_7d_idx], raw_y_test[zoom_7d_idx] / 1000.0, color='#1f77b4', linewidth=1.5, label='Actual (kWh)')
    ax.plot(ts_test[zoom_7d_idx], best_pred_wh[zoom_7d_idx] / 1000.0, color='#d62728', linestyle='--', linewidth=1.5, label=f'{best_name} Predicted (kWh)')
    ax.set_title(f'Actual vs. Predicted Consumption Zoom (Representative 7-Day Window)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Timestamp', fontsize=11)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p15 = os.path.join(output_fig_dir, "15_actual_vs_predicted_7d.png")
    plt.savefig(p15)
    plt.close()
    print(f"Saved: {p15}")
    
    # Figure 16: Prediction error distribution
    errors_wh = raw_y_test - best_pred_wh
    errors_kwh = errors_wh / 1000.0
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.hist(errors_kwh, bins=50, density=True, color='#2ca02c', edgecolor='white', alpha=0.75, label='Error Density')
    kde = gaussian_kde(errors_kwh)
    xg = np.linspace(errors_kwh.min(), errors_kwh.max(), 300)
    ax.plot(xg, kde(xg), color='#117733', linewidth=2.2, label='KDE Curve')
    mean_err = np.mean(errors_kwh)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2, label='Zero Error')
    ax.axvline(mean_err, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean Error ({mean_err:+.3f} kWh)')
    ax.set_title(f'Prediction Error Distribution ({best_name}) [Actual - Predicted]', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Error (kWh)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    plt.tight_layout()
    p16 = os.path.join(output_fig_dir, "16_prediction_error_distribution.png")
    plt.savefig(p16)
    plt.close()
    print(f"Saved: {p16}")
    
    # Figure 17: Scatter plot of actual vs predicted values
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.scatter(raw_y_test / 1000.0, best_pred_wh / 1000.0, alpha=0.45, color='#31688e', edgecolor='none', s=25, label='Predictions')
    max_bound = max(raw_y_test.max(), best_pred_wh.max()) / 1000.0
    ax.plot([0, max_bound], [0, max_bound], color='#d62728', linestyle='--', linewidth=1.8, label='Perfect Prediction (y = x)')
    ax.set_title(f'Actual vs. Predicted Parity Plot ({best_name})', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Actual Energy Consumption (kWh)', fontsize=11)
    ax.set_ylabel('Predicted Energy Consumption (kWh)', fontsize=11)
    r2_val = comparison[best_name]['R2']
    rmse_val = comparison[best_name]['RMSE'] / 1000.0
    ax.text(0.05, 0.90, f'$R^2$ = {r2_val:.3f}\nRMSE = {rmse_val:.3f} kWh', transform=ax.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    ax.legend(frameon=True, loc='lower right')
    ax.set_xlim(0, max_bound * 1.05)
    ax.set_ylim(0, max_bound * 1.05)
    plt.tight_layout()
    p17 = os.path.join(output_fig_dir, "17_actual_vs_predicted_scatter.png")
    plt.savefig(p17)
    plt.close()
    print(f"Saved: {p17}")
    
    return {
        'comparison': comparison,
        'best_model': best_name,
        'pred_df': pred_df
    }

if __name__ == "__main__":
    from src.sequence_dataset import prepare_stage2_data
    data = prepare_stage2_data()
    results = evaluate_all(data)
