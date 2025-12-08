# Enhanced Evaluation Module


import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Enhanced Evaluation Metrics

def compute_precision_at_k(
    recommendations: pd.DataFrame,
    k: int = 2,
    relevance_col: str = 'click'
) -> float:
    # Compute Precision@K: % of top-K recommendations that are relevant.    

    if len(recommendations) == 0:
        return 0.0
    
    # Group by guest and rank recommendations
    guest_recs = recommendations.groupby('guest_id').apply(
        lambda x: x.nlargest(k, 'predicted_score') if 'predicted_score' in x.columns 
        else x.head(k)
    ).reset_index(drop=True)
    
    if len(guest_recs) == 0:
        return 0.0
    
    # Count relevant items in top-K
    relevant_count = (guest_recs[relevance_col] == 1).sum()
    total_count = len(guest_recs)
    
    return relevant_count / total_count if total_count > 0 else 0.0


def compute_hit_rate(
    recommendations: pd.DataFrame,
    k: int = 2,
    relevance_col: str = 'click'
) -> float:
    # Compute Hit Rate: % of guests who scanned at least one recommendation.

    if len(recommendations) == 0:
        return 0.0
    
    # Get top-K per guest
    guest_recs = recommendations.groupby('guest_id').apply(
        lambda x: x.nlargest(k, 'predicted_score') if 'predicted_score' in x.columns 
        else x.head(k)
    ).reset_index(drop=True)
    
    # Count guests with at least one relevant item
    guests_with_hit = guest_recs.groupby('guest_id')[relevance_col].max()
    hit_rate = (guests_with_hit == 1).mean()
    
    return hit_rate


def compute_mrr(
    recommendations: pd.DataFrame,
    relevance_col: str = 'click'
) -> float:
    # Compute Mean Reciprocal Rank (MRR): Average of 1/rank for first relevant item.
    if len(recommendations) == 0:
        return 0.0
    
    # Rank recommendations per guest
    if 'predicted_score' in recommendations.columns:
        recommendations = recommendations.sort_values(
            ['guest_id', 'predicted_score'], 
            ascending=[True, False]
        )
    else:
        recommendations = recommendations.sort_values('guest_id')
    
    # Compute reciprocal rank for each guest
    reciprocal_ranks = []
    for guest_id, guest_recs in recommendations.groupby('guest_id'):
        guest_recs = guest_recs.reset_index(drop=True)
        relevant_indices = guest_recs[guest_recs[relevance_col] == 1].index
        
        if len(relevant_indices) > 0:
            first_relevant_rank = relevant_indices[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_diversity_metrics(
    recommendations: pd.DataFrame,
    category_col: str = 'category'
) -> Dict[str, float]:
    # Compute diversity metrics: Simpson's diversity index and Gini coefficient.
    if len(recommendations) == 0 or category_col not in recommendations.columns:
        return {'simpson_diversity': 0.0, 'gini_coefficient': 0.0, 'entropy': 0.0}
    
    # Category distribution
    category_counts = recommendations[category_col].value_counts()
    proportions = category_counts / category_counts.sum()
    
    # Simpson's diversity index (1 - sum(p_i^2))
    simpson_diversity = 1 - (proportions ** 2).sum()
    
    # Gini coefficient
    sorted_proportions = np.sort(proportions.values)
    n = len(sorted_proportions)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_proportions)) / (n * np.sum(sorted_proportions)) - (n + 1) / n
    
    # Entropy
    entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
    
    return {
        'simpson_diversity': simpson_diversity,
        'gini_coefficient': gini,
        'entropy': entropy,
        'num_categories': len(category_counts)
    }

# Bootstrap Confidence Intervals

def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    # Compute bootstrap confidence interval for a statistic.

    if len(data) == 0:
        return (0.0, 0.0, 0.0)
    
    rng = np.random.default_rng(seed)
    n = len(data)
    
    # Compute statistic on original data
    original_stat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(data, size=n, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return (original_stat, lower_bound, upper_bound)


def compute_metric_with_ci(
    recommendations: pd.DataFrame,
    metric_func: callable,
    metric_name: str,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
    **metric_kwargs
) -> Dict[str, float]:
    # Compute a metric with bootstrap confidence interval.

    # Compute metric on full data
    metric_value = metric_func(recommendations, **metric_kwargs)
    
    # Bootstrap by guest (resample guests, not individual recommendations)
    if 'guest_id' in recommendations.columns:
        unique_guests = recommendations['guest_id'].unique()
        rng = np.random.default_rng(seed)
        
        bootstrap_metrics = []
        for _ in range(n_bootstrap):
            # Resample guests with replacement
            sampled_guests = rng.choice(unique_guests, size=len(unique_guests), replace=True)
            bootstrap_recs = recommendations[recommendations['guest_id'].isin(sampled_guests)]
            
            if len(bootstrap_recs) > 0:
                bootstrap_metric = metric_func(bootstrap_recs, **metric_kwargs)
                bootstrap_metrics.append(bootstrap_metric)
        
        if bootstrap_metrics:
            alpha = 1 - confidence
            lower_bound = np.percentile(bootstrap_metrics, (alpha / 2) * 100)
            upper_bound = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
        else:
            lower_bound = upper_bound = metric_value
    else:
        # Fallback: bootstrap individual recommendations
        metric_values = []
        for _ in range(min(n_bootstrap, 1000)):  # Limit for performance
            sample = recommendations.sample(frac=1.0, replace=True, random_state=seed)
            if len(sample) > 0:
                metric_values.append(metric_func(sample, **metric_kwargs))
        
        if metric_values:
            alpha = 1 - confidence
            lower_bound = np.percentile(metric_values, (alpha / 2) * 100)
            upper_bound = np.percentile(metric_values, (1 - alpha / 2) * 100)
        else:
            lower_bound = upper_bound = metric_value
    
    return {
        f'{metric_name}': metric_value,
        f'{metric_name}_ci_lower': lower_bound,
        f'{metric_name}_ci_upper': upper_bound,
        f'{metric_name}_ci_width': upper_bound - lower_bound
    }

# Segment-Level Analysis

def compute_segment_performance(
    recommendations: pd.DataFrame,
    segment_col: str = 'segment_name',
    metrics: List[str] = ['precision@2', 'hit_rate', 'mrr', 'scan_rate']
) -> pd.DataFrame:
    # Compute performance metrics for each segment.
    if segment_col not in recommendations.columns:
        return pd.DataFrame()
    
    segment_results = []
    
    for segment_name, segment_recs in recommendations.groupby(segment_col):
        segment_metrics = {'segment': segment_name, 'n_guests': segment_recs['guest_id'].nunique()}
        
        if 'precision@2' in metrics:
            segment_metrics['precision@2'] = compute_precision_at_k(segment_recs, k=2)
        
        if 'hit_rate' in metrics:
            segment_metrics['hit_rate'] = compute_hit_rate(segment_recs, k=2)
        
        if 'mrr' in metrics:
            segment_metrics['mrr'] = compute_mrr(segment_recs)
        
        if 'scan_rate' in metrics:
            if 'click' in segment_recs.columns:
                segment_metrics['scan_rate'] = segment_recs['click'].mean()
        
        if 'awareness_uplift' in metrics:
            if 'awareness' in segment_recs.columns:
                initial_awareness = 0.05
                final_awareness = segment_recs['awareness'].mean()
                segment_metrics['awareness_uplift'] = final_awareness - initial_awareness
        
        segment_results.append(segment_metrics)
    
    return pd.DataFrame(segment_results)


def compute_segment_fairness(
    recommendations: pd.DataFrame,
    segment_col: str = 'segment_name',
    outcome_col: str = 'click'
) -> Dict[str, float]:
    # Compute fairness metrics across segments.
    if segment_col not in recommendations.columns or outcome_col not in recommendations.columns:
        return {}
    
    segment_performance = recommendations.groupby(segment_col).agg({
        outcome_col: ['mean', 'count']
    })
    segment_performance.columns = ['mean_outcome', 'count']
    
    # Compute fairness metrics
    mean_outcomes = segment_performance['mean_outcome'].values
    
    # Max-min ratio (ratio of best to worst performing segment)
    max_min_ratio = np.max(mean_outcomes) / (np.min(mean_outcomes) + 1e-10)
    
    # Coefficient of variation (std / mean)
    cv = np.std(mean_outcomes) / (np.mean(mean_outcomes) + 1e-10)
    
    # Gini coefficient of segment outcomes
    sorted_outcomes = np.sort(mean_outcomes)
    n = len(sorted_outcomes)
    if n > 1:
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_outcomes)) / (n * np.sum(sorted_outcomes)) - (n + 1) / n
    else:
        gini = 0.0
    
    return {
        'max_min_ratio': max_min_ratio,
        'coefficient_of_variation': cv,
        'gini_coefficient': gini,
        'best_segment': segment_performance['mean_outcome'].idxmax(),
        'worst_segment': segment_performance['mean_outcome'].idxmin(),
        'best_outcome': np.max(mean_outcomes),
        'worst_outcome': np.min(mean_outcomes)
    }


def compute_segment_error_analysis(
    recommendations: pd.DataFrame,
    segment_col: str = 'segment_name',
    prediction_col: str = 'predicted_score',
    outcome_col: str = 'click'
) -> pd.DataFrame:
    # Analyze prediction errors by segment.
    if segment_col not in recommendations.columns:
        return pd.DataFrame()
    
    if prediction_col not in recommendations.columns or outcome_col not in recommendations.columns:
        return pd.DataFrame()
    
    segment_errors = []
    
    for segment_name, segment_recs in recommendations.groupby(segment_col):
        predictions = segment_recs[prediction_col].values
        outcomes = segment_recs[outcome_col].values
        
        # Mean squared error
        mse = np.mean((predictions - outcomes) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(predictions - outcomes))
        
        # Calibration error (if predictions are probabilities)
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            # Expected Calibration Error (ECE)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = outcomes[in_bin].mean()
                    avg_confidence_in_bin = predictions[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            ece = np.nan
        
        segment_errors.append({
            'segment': segment_name,
            'mse': mse,
            'mae': mae,
            'ece': ece,
            'n_samples': len(segment_recs)
        })
    
    return pd.DataFrame(segment_errors)


# Comprehensive Evaluation Function

def evaluate_comprehensive(
    recommendations: pd.DataFrame,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    # Comprehensive evaluation with all enhanced metrics and CIs
    results = {}
    
    # Basic metrics with CIs
    if 'click' in recommendations.columns:
        scan_rate = recommendations['click'].mean()
        scan_rate_ci = bootstrap_ci(
            recommendations['click'].values,
            np.mean,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed
        )
        results['scan_rate'] = scan_rate_ci[0]
        results['scan_rate_ci_lower'] = scan_rate_ci[1]
        results['scan_rate_ci_upper'] = scan_rate_ci[2]
    
    # Enhanced metrics
    if 'predicted_score' in recommendations.columns:
        results.update(compute_metric_with_ci(
            recommendations, compute_precision_at_k, 'precision@2',
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed, k=2
        ))
        
        results.update(compute_metric_with_ci(
            recommendations, compute_hit_rate, 'hit_rate',
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed, k=2
        ))
        
        results.update(compute_metric_with_ci(
            recommendations, compute_mrr, 'mrr',
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed
        ))
    
    # Diversity metrics
    if 'category' in recommendations.columns:
        diversity = compute_diversity_metrics(recommendations)
        results.update(diversity)
    
    # Segment-level analysis
    if 'segment_name' in recommendations.columns:
        segment_perf = compute_segment_performance(recommendations)
        results['segment_performance'] = segment_perf
        
        fairness = compute_segment_fairness(recommendations)
        results['fairness_metrics'] = fairness
        
        if 'predicted_score' in recommendations.columns:
            error_analysis = compute_segment_error_analysis(recommendations)
            results['segment_error_analysis'] = error_analysis
    
    return results

