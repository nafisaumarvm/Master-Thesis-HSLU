# Temporal Validation Module

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def rolling_window_validation(
    exposure_log: pd.DataFrame,
    date_col: str = 'timestamp',
    train_months: int = 1,
    test_months: int = 1,
    metric_func: callable = None
) -> pd.DataFrame:
    # Rolling window validation: Train on month N, test on month N+1

    if date_col not in exposure_log.columns:
        return pd.DataFrame()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(exposure_log[date_col]):
        exposure_log[date_col] = pd.to_datetime(exposure_log[date_col])
    
    # Get date range
    min_date = exposure_log[date_col].min()
    max_date = exposure_log[date_col].max()
    
    results = []
    current_date = min_date
    
    while current_date < max_date:
        # Training window
        train_end = current_date + pd.DateOffset(months=train_months)
        train_data = exposure_log[
            (exposure_log[date_col] >= current_date) &
            (exposure_log[date_col] < train_end)
        ]
        
        # Test window
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        test_data = exposure_log[
            (exposure_log[date_col] >= test_start) &
            (exposure_log[date_col] < test_end)
        ]
        
        if len(train_data) > 0 and len(test_data) > 0:
            # Compute metrics
            if metric_func is not None:
                train_metric = metric_func(train_data)
                test_metric = metric_func(test_data)
            else:
                # Default: scan rate
                train_metric = train_data['click'].mean() if 'click' in train_data.columns else 0.0
                test_metric = test_data['click'].mean() if 'click' in test_data.columns else 0.0
            
            results.append({
                'train_start': current_date,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_metric': train_metric,
                'test_metric': test_metric,
                'performance_degradation': train_metric - test_metric if train_metric > 0 else 0.0,
                'n_train': len(train_data),
                'n_test': len(test_data)
            })
        
        # Move to next window
        current_date = test_start
    
    return pd.DataFrame(results)


def analyze_concept_drift(
    rolling_results: pd.DataFrame,
    metric_col: str = 'test_metric'
) -> Dict[str, float]:
    # Analyze concept drift from rolling window results

    if len(rolling_results) == 0 or metric_col not in rolling_results.columns:
        return {}
    
    metrics = rolling_results[metric_col].values
    
    # Linear trend (slope)
    x = np.arange(len(metrics))
    if len(metrics) > 1:
        slope, intercept = np.polyfit(x, metrics, 1)
    else:
        slope, intercept = 0.0, metrics[0] if len(metrics) > 0 else 0.0
    
    # Performance degradation rate (per month)
    degradation_rate = slope
    
    # Total degradation
    if len(metrics) > 1:
        total_degradation = metrics[-1] - metrics[0]
        degradation_pct = (total_degradation / (metrics[0] + 1e-10)) * 100 if metrics[0] > 0 else 0.0
    else:
        total_degradation = 0.0
        degradation_pct = 0.0
    
    # Variance (stability)
    variance = np.var(metrics)
    std_dev = np.std(metrics)
    
    # Coefficient of variation
    cv = std_dev / (np.mean(metrics) + 1e-10)
    
    return {
        'degradation_rate_per_month': degradation_rate,
        'total_degradation': total_degradation,
        'degradation_pct': degradation_pct,
        'variance': variance,
        'std_dev': std_dev,
        'coefficient_of_variation': cv,
        'initial_performance': metrics[0] if len(metrics) > 0 else 0.0,
        'final_performance': metrics[-1] if len(metrics) > 0 else 0.0,
        'n_windows': len(metrics)
    }


def time_since_training_analysis(
    exposure_log: pd.DataFrame,
    training_date: datetime,
    date_col: str = 'timestamp',
    metric_func: callable = None,
    time_bins: List[int] = None
) -> pd.DataFrame:
    # Analyze performance vs. time since training

    if date_col not in exposure_log.columns:
        return pd.DataFrame()
    
    if time_bins is None:
        time_bins = [0, 30, 60, 90, 180, 365]
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(exposure_log[date_col]):
        exposure_log[date_col] = pd.to_datetime(exposure_log[date_col])
    
    # Compute days since training
    exposure_log['days_since_training'] = (exposure_log[date_col] - training_date).dt.days
    
    results = []
    
    for i, bin_end in enumerate(time_bins):
        bin_start = time_bins[i-1] if i > 0 else 0
        
        bin_data = exposure_log[
            (exposure_log['days_since_training'] >= bin_start) &
            (exposure_log['days_since_training'] < bin_end)
        ]
        
        if len(bin_data) > 0:
            if metric_func is not None:
                metric_value = metric_func(bin_data)
            else:
                metric_value = bin_data['click'].mean() if 'click' in bin_data.columns else 0.0
            
            results.append({
                'days_since_training_start': bin_start,
                'days_since_training_end': bin_end,
                'days_since_training_mid': (bin_start + bin_end) / 2,
                'metric_value': metric_value,
                'n_samples': len(bin_data)
            })
    
    return pd.DataFrame(results)


def plot_temporal_degradation(
    temporal_results: pd.DataFrame,
    x_col: str = 'days_since_training_mid',
    y_col: str = 'metric_value',
    save_path: Optional[str] = None
) -> None:
    # Plot performance degradation over time

    try:
        import matplotlib.pyplot as plt
        
        if x_col not in temporal_results.columns or y_col not in temporal_results.columns:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(temporal_results[x_col], temporal_results[y_col], marker='o', linewidth=2)
        plt.xlabel('Days Since Training')
        plt.ylabel('Performance Metric')
        plt.title('Model Performance Degradation Over Time')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    except ImportError:
        print("Matplotlib not available for plotting")

