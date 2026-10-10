import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""
baselines.py
Implements and evaluates three simple statistical forecasting baselines:
1. Baseline 1 (Previous Hour Persistence): y_hat(t+1) = y(t)
2. Baseline 2 (Previous Day Persistence):  y_hat(t+1) = y(t - 23) (same hour yesterday)
3. Baseline 3 (24h Moving Average):       y_hat(t+1) = mean(y(t-23)...y(t))
All metrics evaluated on unscaled test data.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred, eps=1e-5):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Safe MAPE avoiding division by near-zero values
    denom = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2)
    }

def evaluate_baselines(unscaled_test_sequences, unscaled_y_test, target_feature_idx=0):
    """
    unscaled_test_sequences: shape (N, 24, num_features) in physical units (Wh)
    unscaled_y_test: shape (N, 1) in physical units (Wh)
    """
    # Extract the target series across the 24 lookback steps
    # seq[:, 23] is time t (previous hour)
    # seq[:, 0] is time t - 23 (24 hours earlier, same hour yesterday)
    seq_target = unscaled_test_sequences[:, :, target_feature_idx]
    y_true = unscaled_y_test.flatten()
    
    # Baseline 1: Previous hour
    pred_prev_hour = seq_target[:, -1]
    m_prev_hour = calculate_metrics(y_true, pred_prev_hour)
    
    # Baseline 2: Previous day (same hour yesterday)
    pred_prev_day = seq_target[:, 0]
    m_prev_day = calculate_metrics(y_true, pred_prev_day)
    
    # Baseline 3: 24h Moving Average
    pred_moving_avg = np.mean(seq_target, axis=1)
    m_moving_avg = calculate_metrics(y_true, pred_moving_avg)
    
    results = {
        'Previous Hour': {
            'metrics': m_prev_hour,
            'predictions': pred_prev_hour
        },
        'Previous Day': {
            'metrics': m_prev_day,
            'predictions': pred_prev_day
        },
        'Moving Average': {
            'metrics': m_moving_avg,
            'predictions': pred_moving_avg
        }
    }
    
    return results

if __name__ == "__main__":
    from src.sequence_dataset import prepare_stage2_data
    data = prepare_stage2_data()
    
    # Build unscaled test sequences
    feat_scaler = data['scaler_bundle']['feature_scaler']
    # Inverse transform each test sequence back to physical units
    N, L, D = data['X_test'].shape
    unscaled_test_seq = np.zeros_like(data['X_test'])
    for i in range(N):
        unscaled_test_seq[i] = feat_scaler.inverse_transform(data['X_test'][i])
        
    baselines = evaluate_baselines(unscaled_test_seq, data['raw_y_test'], target_feature_idx=0)
    print("\n=== BASELINE EVALUATION (Target: Wh) ===")
    for name, res in baselines.items():
        m = res['metrics']
        print(f"{name:<16}: MAE={m['MAE']:>7.2f} Wh | RMSE={m['RMSE']:>7.2f} Wh | MAPE={m['MAPE']:>5.2f}% | R2={m['R2']:>6.3f}")
