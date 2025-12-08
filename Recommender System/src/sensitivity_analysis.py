# Sensitivity Analysis Module

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def sensitivity_analysis_poisson_lambda(
    base_lambda: float = 1.8,
    lambda_range: List[float] = None,
    simulation_func: callable = None,
    **simulation_kwargs
) -> pd.DataFrame:
    # Sensitivity analysis for Poisson lambda (TV session frequency)

    if lambda_range is None:
        lambda_range = [1.0, 1.5, 2.0, 2.5]
    
    if simulation_func is None:
        # Return empty DataFrame if no simulation function provided
        return pd.DataFrame()
    
    results = []
    
    for lambda_val in lambda_range:
        # Run simulation with this lambda value
        sim_kwargs = simulation_kwargs.copy()
        sim_kwargs['poisson_lambda'] = lambda_val
        
        metrics = simulation_func(**sim_kwargs)
        
        results.append({
            'lambda': lambda_val,
            'reach': metrics.get('reach', 0.0),
            'frequency': metrics.get('frequency', 0.0),
            'scan_rate': metrics.get('scan_rate', 0.0),
            'awareness_uplift': metrics.get('awareness_uplift', 0.0),
            'grp': metrics.get('grp', 0.0),
            'revenue_per_guest': metrics.get('revenue_per_guest', 0.0)
        })
    
    return pd.DataFrame(results)


def sensitivity_analysis_awareness_params(
    base_alpha: float = 0.3,
    base_delta: float = 0.1,
    alpha_range: List[float] = None,
    delta_range: List[float] = None,
    simulation_func: callable = None,
    **simulation_kwargs
) -> pd.DataFrame:
    # Sensitivity analysis for awareness parameters (alpha, delta)  
    if alpha_range is None:
        alpha_range = [0.2, 0.3, 0.4, 0.5]
    
    if delta_range is None:
        delta_range = [0.05, 0.1, 0.15, 0.2]
    
    if simulation_func is None:
        return pd.DataFrame()
    
    results = []
    
    for alpha in alpha_range:
        for delta in delta_range:
            # Run simulation with these parameters
            sim_kwargs = simulation_kwargs.copy()
            sim_kwargs['alpha'] = alpha
            sim_kwargs['delta'] = delta
            
            metrics = simulation_func(**sim_kwargs)
            
            results.append({
                'alpha': alpha,
                'delta': delta,
                'reach': metrics.get('reach', 0.0),
                'frequency': metrics.get('frequency', 0.0),
                'scan_rate': metrics.get('scan_rate', 0.0),
                'awareness_uplift': metrics.get('awareness_uplift', 0.0),
                'grp': metrics.get('grp', 0.0)
            })
    
    return pd.DataFrame(results)


def compute_robustness_metrics(
    sensitivity_results: pd.DataFrame,
    base_metric_value: float,
    metric_name: str = 'scan_rate',
    tolerance_pct: float = 10.0
) -> Dict[str, float]:
    # Compute robustness metrics from sensitivity analysis
    if len(sensitivity_results) == 0 or metric_name not in sensitivity_results.columns:
        return {}
    
    metric_values = sensitivity_results[metric_name].values
    
    # Coefficient of variation
    cv = np.std(metric_values) / (np.mean(metric_values) + 1e-10)
    
    # Min/max deviation from base
    min_deviation = (np.min(metric_values) - base_metric_value) / (base_metric_value + 1e-10) * 100
    max_deviation = (np.max(metric_values) - base_metric_value) / (base_metric_value + 1e-10) * 100
    if len(sensitivity_results) == 0 or metric_name not in sensitivity_results.columns:
        return {}
    
    metric_values = sensitivity_results[metric_name].values
    
    # Coefficient of variation
    cv = np.std(metric_values) / (np.mean(metric_values) + 1e-10)
    
    # Min/max deviation from base
    min_deviation = (np.min(metric_values) - base_metric_value) / (base_metric_value + 1e-10) * 100
    max_deviation = (np.max(metric_values) - base_metric_value) / (base_metric_value + 1e-10) * 100
    
    # Percentage of results within tolerance
    within_tolerance = np.abs(metric_values - base_metric_value) / (base_metric_value + 1e-10) * 100 <= tolerance_pct
    robust_pct = within_tolerance.mean() * 100
    
    # Range (max - min)
    metric_range = np.max(metric_values) - np.min(metric_values)
    
    return {
        'coefficient_of_variation': cv,
        'min_deviation_pct': min_deviation,
        'max_deviation_pct': max_deviation,
        'robust_pct': robust_pct,
        'range': metric_range,
        'min_value': np.min(metric_values),
        'max_value': np.max(metric_values),
        'mean_value': np.mean(metric_values),
        'base_value': base_metric_value
    }


def plot_sensitivity_heatmap(
    sensitivity_results: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
    save_path: Optional[str] = None
) -> None:
    # Create heatmap visualization of sensitivity analysis
    if x_param not in sensitivity_results.columns or y_param not in sensitivity_results.columns:
        return
    
    if metric not in sensitivity_results.columns:
        return
    
    # Pivot for heatmap
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if x_param not in sensitivity_results.columns or y_param not in sensitivity_results.columns:
            return
        
        if metric not in sensitivity_results.columns:
            return
        
        # Pivot for heatmap
        pivot_data = sensitivity_results.pivot_table(
            values=metric,
            index=y_param,
            columns=x_param,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Sensitivity Analysis: {metric} vs {x_param} and {y_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")


def identify_critical_parameters(
    sensitivity_results: pd.DataFrame,
    metric: str = 'scan_rate',
    threshold_pct: float = 15.0
) -> List[str]:
    # Identify parameters that have critical impact on results
    if len(sensitivity_results) == 0 or metric not in sensitivity_results.columns:
        return []
    
    critical_params = []
    
    # Get parameter columns (exclude metric columns)
    param_cols = [col for col in sensitivity_results.columns 
                  if col not in ['reach', 'frequency', 'scan_rate', 'awareness_uplift', 'grp', 'revenue_per_guest']]
    
    if len(sensitivity_results) == 0 or metric not in sensitivity_results.columns:
        return []
    
    critical_params = []
    
    # Get parameter columns (exclude metric columns)
    param_cols = [col for col in sensitivity_results.columns 
                  if col not in ['reach', 'frequency', 'scan_rate', 'awareness_uplift', 'grp', 'revenue_per_guest']]
    
    for param in param_cols:
        if param not in sensitivity_results.columns:
            continue
        
        # Group by parameter value and compute metric range
        grouped = sensitivity_results.groupby(param)[metric]
        metric_range = grouped.max() - grouped.min()
        metric_mean = grouped.mean().mean()
        
        # Check if range exceeds threshold
        if metric_mean > 0:
            range_pct = (metric_range.max() / metric_mean) * 100
            if range_pct > threshold_pct:
                critical_params.append(param)
    
    return critical_params

