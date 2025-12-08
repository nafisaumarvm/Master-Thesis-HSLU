# External Validation Module

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Ebbinghaus forgetting curve

def ebbinghaus_forgetting_curve(
    t: np.ndarray,
    initial_strength: float = 1.0,
    decay_rate: float = 0.1
) -> np.ndarray:
    # Ebbinghaus forgetting curve: R = S * exp(-λt)

    return initial_strength * np.exp(-decay_rate * t)


def fit_ebbinghaus_curve(
    time_since_exposure: np.ndarray,
    awareness_values: np.ndarray
) -> Dict[str, float]:
    # Fit Ebbinghaus forgetting curve to awareness decay data

    if len(time_since_exposure) == 0 or len(awareness_values) == 0:
        return {}
    
    try:
        # Fit exponential decay: y = a * exp(-b * x)
        popt, pcov = curve_fit(
            lambda t, a, b: a * np.exp(-b * t),
            time_since_exposure,
            awareness_values,
            p0=[1.0, 0.1],
            bounds=([0, 0], [2, 1])
        )
        
        initial_strength, decay_rate = popt
        
        # Compute R²
        predicted = ebbinghaus_forgetting_curve(time_since_exposure, initial_strength, decay_rate)
        ss_res = np.sum((awareness_values - predicted) ** 2)
        ss_tot = np.sum((awareness_values - np.mean(awareness_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'initial_strength': initial_strength,
            'decay_rate': decay_rate,
            'r_squared': r_squared,
            'half_life_days': np.log(2) / decay_rate if decay_rate > 0 else np.inf
        }
    except Exception as e:
        return {'error': str(e)}


def compare_to_ebbinghaus(
    simulation_awareness: pd.DataFrame,
    time_col: str = 'days_since_exposure',
    awareness_col: str = 'awareness'
) -> Dict[str, Any]:
    # Compare simulation awareness decay to Ebbinghaus curve

    if time_col not in simulation_awareness.columns or awareness_col not in simulation_awareness.columns:
        return {}
    
    time_values = simulation_awareness[time_col].values
    awareness_values = simulation_awareness[awareness_col].values
    
    # Fit Ebbinghaus curve
    fitted_params = fit_ebbinghaus_curve(time_values, awareness_values)
    
    if 'error' in fitted_params:
        return {'error': fitted_params['error']}
    
    # Expected decay rate from literature: 0.05-0.15 per day (Ebbinghaus found ~0.1)
    literature_decay_range = (0.05, 0.15)
    our_decay_rate = fitted_params['decay_rate']
    
    # Check if our decay rate is within literature range
    within_literature = (
        literature_decay_range[0] <= our_decay_rate <= literature_decay_range[1]
    )
    
    return {
        'fitted_parameters': fitted_params,
        'decay_rate': our_decay_rate,
        'literature_range': literature_decay_range,
        'within_literature_range': within_literature,
        'half_life_days': fitted_params.get('half_life_days', np.inf),
        'r_squared': fitted_params.get('r_squared', 0.0)
    }


# Marketing literature validation

def validate_against_marketing_literature(
    simulation_results: Dict[str, float]
) -> Dict[str, Any]:
    # Validate simulation results against marketing literature benchmarks

    validation = {}
    
    # Brand recall benchmarks
    if 'awareness_after_1_exposure' in simulation_results:
        our_value = simulation_results['awareness_after_1_exposure']
        literature_range = (0.10, 0.20)
        validation['awareness_1_exposure'] = {
            'our_value': our_value,
            'literature_range': literature_range,
            'within_range': literature_range[0] <= our_value <= literature_range[1],
            'deviation_pct': ((our_value - np.mean(literature_range)) / np.mean(literature_range)) * 100
        }
    
    if 'awareness_after_3_exposures' in simulation_results:
        our_value = simulation_results['awareness_after_3_exposures']
        literature_range = (0.30, 0.50)
        validation['awareness_3_exposures'] = {
            'our_value': our_value,
            'literature_range': literature_range,
            'within_range': literature_range[0] <= our_value <= literature_range[1],
            'deviation_pct': ((our_value - np.mean(literature_range)) / np.mean(literature_range)) * 100
        }
    
    # Frequency benchmarks
    if 'optimal_frequency' in simulation_results:
        our_value = simulation_results['optimal_frequency']
        literature_range = (3.0, 7.0)
        validation['optimal_frequency'] = {
            'our_value': our_value,
            'literature_range': literature_range,
            'within_range': literature_range[0] <= our_value <= literature_range[1],
            'deviation_pct': ((our_value - np.mean(literature_range)) / np.mean(literature_range)) * 100
        }
    
    # Decay rate benchmarks
    if 'decay_rate' in simulation_results:
        our_value = simulation_results['decay_rate']
        literature_range = (0.05, 0.15)
        validation['decay_rate'] = {
            'our_value': our_value,
            'literature_range': literature_range,
            'within_range': literature_range[0] <= our_value <= literature_range[1],
            'deviation_pct': ((our_value - np.mean(literature_range)) / np.mean(literature_range)) * 100
        }
    
    return validation


# Sensitivity analysis for awareness validation

def sensitivity_analysis_awareness_validation(
    base_alpha: float = 0.3,
    base_delta: float = 0.1,
    alpha_range: List[float] = None,
    delta_range: List[float] = None
) -> pd.DataFrame:
    # Sensitivity analysis
    if alpha_range is None:
        alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    if delta_range is None:
        delta_range = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    results = []
    
    # Simulate awareness dynamics with different parameters
    for alpha in alpha_range:
        for delta in delta_range:
            time_points = np.arange(0, 30, 1)
            awareness = np.zeros(len(time_points))
            awareness[0] = 0.05  # Initial awareness
            
            for t in range(1, len(time_points)):
                # Growth (if exposed)
                if t % 3 == 0:  # Exposure every 3 days
                    awareness[t] = awareness[t-1] + alpha * (1 - awareness[t-1])
                else:
                    # Decay
                    awareness[t] = awareness[t-1] * (1 - delta)
            
            # Fit Ebbinghaus curve and get R²
            fitted = fit_ebbinghaus_curve(time_points, awareness)
            r_squared = fitted.get('r_squared', 0.0)
            
            results.append({
                'alpha': alpha,
                'delta': delta,
                'r_squared': r_squared,
                'is_base_params': (alpha == base_alpha and delta == base_delta)
            })
    
    return pd.DataFrame(results)


# Comprehensive external validation

def perform_external_validation(
    simulation_awareness_data: pd.DataFrame,
    simulation_metrics: Dict[str, float],
    time_col: str = 'days_since_exposure',
    awareness_col: str = 'awareness'
) -> Dict[str, Any]:
    # Perform comprehensive external validation

    results = {}
    
    # Ebbinghaus curve comparison
    ebbinghaus_comparison = compare_to_ebbinghaus(
        simulation_awareness_data,
        time_col=time_col,
        awareness_col=awareness_col
    )
    results['ebbinghaus_comparison'] = ebbinghaus_comparison
    
    # Marketing literature validation
    literature_validation = validate_against_marketing_literature(simulation_metrics)
    results['literature_validation'] = literature_validation
    
    # Sensitivity analysis
    sensitivity_results = sensitivity_analysis_awareness_validation()
    results['sensitivity_analysis'] = {
        'r_squared_range': (
            sensitivity_results['r_squared'].min(),
            sensitivity_results['r_squared'].max()
        ),
        'base_r_squared': sensitivity_results[
            sensitivity_results['is_base_params']
        ]['r_squared'].values[0] if len(sensitivity_results[sensitivity_results['is_base_params']]) > 0 else 0.0,
        'r_squared_with_wrong_params': sensitivity_results[
            ~sensitivity_results['is_base_params']
        ]['r_squared'].mean()
    }
    
    # Limitations statement
    results['limitations_statement'] = generate_validation_limitations_statement()
    
    return results

