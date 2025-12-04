"""
Statistical Rigor - Final Polish for Thesis

Implements:
A. Confidence intervals (bootstrap)
B. BH-corrected significance tests (FDR control)
C. Hyperparameter search/optimization
D. Two-stage model selection
E. Explainability (permutation importance)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from scipy.stats import bootstrap
from statsmodels.stats.multitest import multipletests


# ============================================================================
# A. CONFIDENCE INTERVALS (Bootstrap)
# ============================================================================

def compute_bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for any statistic.
    
    Uses scipy.stats.bootstrap for robust CI estimation.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    statistic : callable
        Statistic function (default: np.mean)
    confidence_level : float
        Confidence level (default: 0.95 for 95% CI)
    n_resamples : int
        Number of bootstrap samples
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Point estimate, lower bound, upper bound, standard error
    """
    # Point estimate
    point_estimate = statistic(data)
    
    # Bootstrap CI
    rng = np.random.default_rng(random_state)
    result = bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng,
        method='percentile'
    )
    
    # Standard error (from bootstrap distribution)
    bootstrap_samples = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(statistic(sample))
    
    std_error = np.std(bootstrap_samples)
    
    return {
        'point_estimate': point_estimate,
        'lower_bound': result.confidence_interval.low,
        'upper_bound': result.confidence_interval.high,
        'std_error': std_error,
        'confidence_level': confidence_level
    }


def compute_ctr_ci(
    clicks: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute CTR with confidence interval.
    
    Parameters
    ----------
    clicks : np.ndarray
        Binary array (0/1) of clicks
    confidence_level : float
        Confidence level
        
    Returns
    -------
    dict
        CTR with CI
    """
    ctr = clicks.mean()
    n = len(clicks)
    
    # Wilson score interval (better for proportions)
    z = stats.norm.ppf((1 + confidence_level) / 2)
    denominator = 1 + z**2 / n
    center = (ctr + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(ctr * (1 - ctr) / n + z**2 / (4*n**2)) / denominator
    
    return {
        'ctr': ctr,
        'lower_bound': max(0, center - margin),
        'upper_bound': min(1, center + margin),
        'confidence_level': confidence_level,
        'n': n
    }


def add_cis_to_results(
    exposure_log: pd.DataFrame
) -> pd.DataFrame:
    """
    Add confidence intervals to all key metrics.
    
    Parameters
    ----------
    exposure_log : pd.DataFrame
        Exposure log
        
    Returns
    -------
    pd.DataFrame
        Summary with CIs
    """
    results = []
    
    # Overall CTR
    ctr_ci = compute_ctr_ci(exposure_log['click'].values)
    results.append({
        'metric': 'CTR (Overall)',
        'value': ctr_ci['ctr'],
        'ci_lower': ctr_ci['lower_bound'],
        'ci_upper': ctr_ci['upper_bound'],
        'ci_width': ctr_ci['upper_bound'] - ctr_ci['lower_bound']
    })
    
    # Revenue per exposure
    rev_ci = compute_bootstrap_ci(exposure_log['revenue'].values)
    results.append({
        'metric': 'Revenue per Exposure',
        'value': rev_ci['point_estimate'],
        'ci_lower': rev_ci['lower_bound'],
        'ci_upper': rev_ci['upper_bound'],
        'ci_width': rev_ci['upper_bound'] - rev_ci['lower_bound']
    })
    
    # Awareness uplift
    if 'awareness_after' in exposure_log.columns and 'awareness_before' in exposure_log.columns:
        awareness_uplift = (exposure_log['awareness_after'] - exposure_log['awareness_before']).values
        aware_ci = compute_bootstrap_ci(awareness_uplift)
        results.append({
            'metric': 'Awareness Uplift',
            'value': aware_ci['point_estimate'],
            'ci_lower': aware_ci['lower_bound'],
            'ci_upper': aware_ci['upper_bound'],
            'ci_width': aware_ci['upper_bound'] - aware_ci['lower_bound']
        })
    
    # Base utility
    if 'base_utility' in exposure_log.columns:
        util_ci = compute_bootstrap_ci(exposure_log['base_utility'].values)
        results.append({
            'metric': 'Base Utility',
            'value': util_ci['point_estimate'],
            'ci_lower': util_ci['lower_bound'],
            'ci_upper': util_ci['upper_bound'],
            'ci_width': util_ci['upper_bound'] - util_ci['lower_bound']
        })
    
    return pd.DataFrame(results)


# ============================================================================
# B. BENJAMINI-HOCHBERG FDR CORRECTION
# ============================================================================

def compute_segment_category_significance(
    exposure_log: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test significance of segment-category effects with FDR correction.
    
    Uses Benjamini-Hochberg procedure to control false discovery rate.
    
    Parameters
    ----------
    exposure_log : pd.DataFrame
        Exposure log
    alpha : float
        Family-wise error rate
        
    Returns
    -------
    pd.DataFrame
        Test results with corrected p-values
    """
    results = []
    
    # For each segment-category pair, test if CTR differs from baseline
    baseline_ctr = exposure_log['click'].mean()
    
    for segment in exposure_log['segment'].unique():
        for category in exposure_log['advertiser_type'].unique():
            mask = (exposure_log['segment'] == segment) & (exposure_log['advertiser_type'] == category)
            
            if mask.sum() < 10:  # Skip small samples
                continue
            
            group_data = exposure_log[mask]
            group_ctr = group_data['click'].mean()
            n = len(group_data)
            
            # Proportion test
            successes = group_data['click'].sum()
            baseline_successes = baseline_ctr * n
            
            # Z-test for proportions
            p_pooled = (successes + baseline_successes) / (n + n)
            se = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)
            
            if se > 0:
                z = (group_ctr - baseline_ctr) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                p_value = 1.0
            
            results.append({
                'segment': segment,
                'category': category,
                'ctr': group_ctr,
                'baseline_ctr': baseline_ctr,
                'difference': group_ctr - baseline_ctr,
                'n': n,
                'p_value': p_value
            })
    
    results_df = pd.DataFrame(results)
    
    # BH correction
    if len(results_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            results_df['p_value'].values,
            alpha=alpha,
            method='fdr_bh'
        )
        
        results_df['p_value_corrected'] = pvals_corrected
        results_df['significant'] = reject
        results_df['fdr_controlled'] = True
    
    return results_df


def report_fdr_summary(
    significance_results: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    Generate FDR correction summary report.
    
    Parameters
    ----------
    significance_results : pd.DataFrame
        Results from compute_segment_category_significance
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Summary statistics
    """
    n_tests = len(significance_results)
    n_significant_uncorrected = (significance_results['p_value'] < alpha).sum()
    n_significant_corrected = significance_results['significant'].sum()
    
    expected_false_positives = n_tests * alpha
    observed_false_positives_estimate = n_significant_corrected * alpha
    
    return {
        'n_tests': n_tests,
        'n_significant_uncorrected': int(n_significant_uncorrected),
        'n_significant_corrected': int(n_significant_corrected),
        'false_positive_rate_uncorrected': n_significant_uncorrected / n_tests if n_tests > 0 else 0,
        'false_discovery_rate_controlled': alpha,
        'expected_false_discoveries': observed_false_positives_estimate,
        'reduction_pct': (n_significant_uncorrected - n_significant_corrected) / n_significant_uncorrected * 100 if n_significant_uncorrected > 0 else 0
    }


# ============================================================================
# C. HYPERPARAMETER SEARCH
# ============================================================================

def grid_search_awareness_params(
    exposure_log: pd.DataFrame,
    alpha_range: List[float] = None,
    beta_range: List[float] = None,
    gamma_range: List[float] = None,
    metric: str = 'ctr'
) -> pd.DataFrame:
    """
    Grid search for optimal awareness parameters.
    
    Parameters
    ----------
    exposure_log : pd.DataFrame
        Exposure log (validation set)
    alpha_range : list
        Awareness growth rates to test
    beta_range : list
        Awareness effect strengths to test
    gamma_range : list
        Position bias decay rates to test
    metric : str
        Optimization metric ('ctr', 'revenue', 'calibration')
        
    Returns
    -------
    pd.DataFrame
        Grid search results
    """
    if alpha_range is None:
        alpha_range = [0.2, 0.3, 0.4]
    
    if beta_range is None:
        beta_range = [0.4, 0.5, 0.6]
    
    if gamma_range is None:
        gamma_range = [0.6, 0.7, 0.8]
    
    results = []
    
    for alpha in alpha_range:
        for beta in beta_range:
            for gamma in gamma_range:
                # Simulate awareness effect with these params
                simulated_awareness = exposure_log['awareness_before'].values.copy()
                
                # Update awareness (simplified)
                exposed_mask = exposure_log['click'].values == 1
                simulated_awareness[exposed_mask] += alpha * (1 - simulated_awareness[exposed_mask])
                
                # Compute click probability
                base_util = exposure_log['base_utility'].values if 'base_utility' in exposure_log.columns else np.random.random(len(exposure_log))
                pos_effect = gamma ** (exposure_log['position'].values - 1) if 'position' in exposure_log.columns else 1.0
                
                click_prob = 1 / (1 + np.exp(-(base_util + beta * simulated_awareness + np.log(pos_effect + 1e-10))))
                
                # Evaluate
                actual_clicks = exposure_log['click'].values
                
                if metric == 'ctr':
                    score = -np.abs(click_prob.mean() - actual_clicks.mean())
                elif metric == 'calibration':
                    score = -np.mean((click_prob - actual_clicks)**2)  # Negative Brier
                elif metric == 'revenue':
                    predicted_revenue = click_prob * exposure_log['revenue'].values if 'revenue' in exposure_log.columns else click_prob
                    actual_revenue = actual_clicks * exposure_log['revenue'].values if 'revenue' in exposure_log.columns else actual_clicks
                    score = -np.abs(predicted_revenue.mean() - actual_revenue.mean())
                else:
                    score = 0
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'score': score,
                    'metric': metric
                })
    
    results_df = pd.DataFrame(results)
    results_df['rank'] = results_df['score'].rank(ascending=False)
    
    return results_df.sort_values('score', ascending=False)


def report_best_hyperparams(
    grid_results: pd.DataFrame
) -> Dict:
    """
    Report best hyperparameters from grid search.
    
    Parameters
    ----------
    grid_results : pd.DataFrame
        Results from grid_search_awareness_params
        
    Returns
    -------
    dict
        Best parameters and performance
    """
    best = grid_results.iloc[0]
    
    return {
        'alpha_optimal': best['alpha'],
        'beta_optimal': best['beta'],
        'gamma_optimal': best['gamma'],
        'score': best['score'],
        'metric_optimized': best['metric'],
        'n_configurations_tested': len(grid_results)
    }


# ============================================================================
# D. TWO-STAGE MODEL SELECTION
# ============================================================================

def two_stage_model_selection(
    models_performance: pd.DataFrame,
    auc_threshold: float = 0.70,
    ece_threshold: float = 0.10
) -> pd.DataFrame:
    """
    Two-stage model selection: AUC for ranking, then ECE for calibration.
    
    Stage 1: Filter by AUC (ranking quality)
    Stage 2: Select best by ECE (calibration quality)
    
    Parameters
    ----------
    models_performance : pd.DataFrame
        DataFrame with columns: model, auc, ece
    auc_threshold : float
        Minimum acceptable AUC
    ece_threshold : float
        Maximum acceptable ECE
        
    Returns
    -------
    pd.DataFrame
        Selected models with selection criteria
    """
    # Stage 1: Filter by AUC
    stage1_pass = models_performance[models_performance['auc'] >= auc_threshold].copy()
    stage1_pass['stage1_pass'] = True
    
    # Stage 2: Select by ECE (among those passing stage 1)
    if len(stage1_pass) > 0:
        stage1_pass['stage2_rank'] = stage1_pass['ece'].rank()
        best_model = stage1_pass.nsmallest(1, 'ece')
    else:
        best_model = models_performance.nlargest(1, 'auc')
    
    # Add selection info
    models_performance['stage1_pass'] = models_performance['auc'] >= auc_threshold
    models_performance['stage2_pass'] = (
        models_performance['stage1_pass'] &
        (models_performance['ece'] <= ece_threshold)
    )
    models_performance['selected'] = False
    if len(best_model) > 0:
        models_performance.loc[best_model.index, 'selected'] = True
    
    return models_performance


# ============================================================================
# E. EXPLAINABILITY (Permutation Importance)
# ============================================================================

def compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    model: object,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for interpretability.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model : object
        Fitted model with predict method
    feature_names : list
        Feature names
    n_repeats : int
        Number of permutation repeats
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Feature importances
    """
    from sklearn.metrics import log_loss
    
    # Baseline score
    if hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X)[:, 1] if len(model.predict_proba(X).shape) > 1 else model.predict_proba(X)
    else:
        y_pred = model.predict(X)
    
    baseline_score = log_loss(y, y_pred)
    
    importances = []
    rng = np.random.RandomState(random_state)
    
    for i, feature in enumerate(feature_names):
        scores = []
        
        for _ in range(n_repeats):
            # Permute feature
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            
            # Score
            if hasattr(model, 'predict_proba'):
                y_pred_perm = model.predict_proba(X_permuted)[:, 1] if len(model.predict_proba(X_permuted).shape) > 1 else model.predict_proba(X_permuted)
            else:
                y_pred_perm = model.predict(X_permuted)
            
            score = log_loss(y, y_pred_perm)
            scores.append(score - baseline_score)  # Increase in loss
        
        importances.append({
            'feature': feature,
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores),
            'rank': 0  # Will be filled later
        })
    
    importance_df = pd.DataFrame(importances)
    importance_df['rank'] = importance_df['importance_mean'].rank(ascending=False)
    
    return importance_df.sort_values('importance_mean', ascending=False)


# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================

def generate_statistical_rigor_report(
    exposure_log: pd.DataFrame,
    models_performance: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Generate comprehensive statistical rigor report.
    
    Parameters
    ----------
    exposure_log : pd.DataFrame
        Exposure log
    models_performance : pd.DataFrame, optional
        Model performance metrics
        
    Returns
    -------
    dict
        Complete statistical report
    """
    report = {}
    
    # A. Confidence intervals
    report['confidence_intervals'] = add_cis_to_results(exposure_log)
    
    # B. FDR-corrected significance
    sig_results = compute_segment_category_significance(exposure_log)
    report['significance_tests'] = sig_results
    report['fdr_summary'] = report_fdr_summary(sig_results)
    
    # C. Hyperparameter search
    grid_results = grid_search_awareness_params(exposure_log)
    report['hyperparameter_search'] = grid_results
    report['best_hyperparams'] = report_best_hyperparams(grid_results)
    
    # D. Two-stage model selection (if provided)
    if models_performance is not None:
        report['model_selection'] = two_stage_model_selection(models_performance)
    
    return report


# Example usage
if __name__ == '__main__':
    print("STATISTICAL RIGOR - DEMONSTRATION")
    print("="*70)
    
    # Simulate data
    np.random.seed(42)
    n = 1000
    
    data = {
        'click': np.random.binomial(1, 0.08, n),
        'revenue': np.random.lognormal(3, 1, n),
        'awareness_before': np.random.beta(2, 5, n),
        'awareness_after': np.random.beta(3, 4, n),
        'base_utility': np.random.normal(0.5, 0.2, n),
        'position': np.random.randint(1, 6, n),
        'segment': np.random.choice(['luxury', 'business', 'family'], n),
        'advertiser_type': np.random.choice(['restaurant', 'tour', 'spa'], n)
    }
    
    exposure_log = pd.DataFrame(data)
    exposure_log['revenue'] = exposure_log['click'] * exposure_log['revenue']
    
    # A. Confidence intervals
    print("\n1. CONFIDENCE INTERVALS:")
    cis = add_cis_to_results(exposure_log)
    print(cis.to_string(index=False))
    
    # B. FDR correction
    print("\n2. FDR-CORRECTED SIGNIFICANCE:")
    sig_results = compute_segment_category_significance(exposure_log, alpha=0.05)
    print(f"\nTests performed: {len(sig_results)}")
    print(f"Significant (uncorrected): {(sig_results['p_value'] < 0.05).sum()}")
    print(f"Significant (FDR-corrected): {sig_results['significant'].sum()}")
    
    # C. Hyperparameter search
    print("\n3. HYPERPARAMETER SEARCH:")
    grid = grid_search_awareness_params(exposure_log)
    print(f"\nConfigurations tested: {len(grid)}")
    print(f"\nTop 3 configurations:")
    print(grid[['alpha', 'beta', 'gamma', 'score']].head(3).to_string(index=False))
    
    print("\n" + "="*70)





