# Advanced Evaluation Metrics & Methods

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Advanced Off-Policy Estimators

def self_normalized_ips(
    exposure_log: pd.DataFrame,
    target_policy_probs: np.ndarray,
    logging_policy_probs: np.ndarray,
    rewards: np.ndarray,
    clip_weights: float = 10.0
) -> Dict[str, float]:
    # Self-Normalized Inverse Propensity Scoring (SNIPS)
    # Reduces variance compared to standard IPS, especially for skewed propensities

    # Importance weights
    weights = target_policy_probs / (logging_policy_probs + 1e-10)
    
    # Clip weights
    weights_clipped = np.clip(weights, 0, clip_weights)
    
    # Self-normalized estimate
    numerator = np.sum(weights_clipped * rewards)
    denominator = np.sum(weights_clipped)
    
    snips_estimate = numerator / (denominator + 1e-10)
    
    # Effective sample size
    ess = (np.sum(weights_clipped)**2) / np.sum(weights_clipped**2)
    
    # Variance (Hesterberg 1995)
    normalized_weights = weights_clipped / np.sum(weights_clipped)
    variance = np.sum(normalized_weights**2 * (rewards - snips_estimate)**2)
    
    return {
        'snips_estimate': snips_estimate,
        'variance': variance,
        'std_error': np.sqrt(variance),
        'effective_sample_size': ess,
        'mean_weight': weights_clipped.mean(),
        'max_weight': weights_clipped.max(),
        'clipped_fraction': (weights > clip_weights).mean()
    }


def weighted_ips(
    rewards: np.ndarray,
    weights: np.ndarray,
    clip_weights: float = 10.0
) -> Dict[str, float]:
    # Weighted IPS with variance reduction

    weights_clipped = np.clip(weights, 0, clip_weights)
    
    ips_estimate = np.mean(weights_clipped * rewards)
    variance = np.var(weights_clipped * rewards)
    
    return {
        'ips_estimate': ips_estimate,
        'variance': variance,
        'std_error': np.sqrt(variance / len(rewards)),
        'confidence_interval_95': (
            ips_estimate - 1.96 * np.sqrt(variance / len(rewards)),
            ips_estimate + 1.96 * np.sqrt(variance / len(rewards))
        )
    }


# Calibration Metrics

def compute_brier_score(
    predicted_probs: np.ndarray,
    observed_outcomes: np.ndarray
) -> float:
    # Brier Score - measures calibration quality.
    return np.mean((predicted_probs - observed_outcomes)**2)


def compute_ece(
    predicted_probs: np.ndarray,
    observed_outcomes: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    # Expected Calibration Error (ECE).
    # Measures the expected difference between predicted probability and observed frequency across bins
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_size = mask.sum()
        bin_probs = predicted_probs[mask]
        bin_outcomes = observed_outcomes[mask]
        
        # Average confidence in bin
        avg_confidence = bin_probs.mean()
        
        # Average accuracy in bin
        avg_accuracy = bin_outcomes.mean()
        
        # Contribution to ECE
        ece += (bin_size / len(predicted_probs)) * abs(avg_accuracy - avg_confidence)
        
        bin_stats.append({
            'bin': i,
            'bin_lower': bins[i],
            'bin_upper': bins[i+1],
            'n_samples': int(bin_size),
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'calibration_error': abs(avg_accuracy - avg_confidence)
        })
    
    return {
        'ece': ece,
        'n_bins': n_bins,
        'bin_statistics': pd.DataFrame(bin_stats)
    }


def compute_reliability_diagram_data(
    predicted_probs: np.ndarray,
    observed_outcomes: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    # Generate data for reliability diagram.

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted_probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    reliability_data = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_probs = predicted_probs[mask]
        bin_outcomes = observed_outcomes[mask]
        
        reliability_data.append({
            'bin_center': (bins[i] + bins[i+1]) / 2,
            'predicted_prob': bin_probs.mean(),
            'observed_freq': bin_outcomes.mean(),
            'n_samples': int(mask.sum()),
            'std_error': np.sqrt(bin_outcomes.mean() * (1 - bin_outcomes.mean()) / mask.sum())
        })
    
    return pd.DataFrame(reliability_data)


# Diversity & Fairness Metrics

def compute_diversity_metrics(
    exposure_log: pd.DataFrame,
    category_column: str = 'advertiser_type'
) -> Dict[str, float]:
    # Compute diversity metrics for shown ads

    # Category distribution
    category_counts = exposure_log[category_column].value_counts()
    category_probs = category_counts / len(exposure_log)
    
    # Entropy
    entropy = -np.sum(category_probs * np.log2(category_probs + 1e-10))
    max_entropy = np.log2(len(category_probs))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Gini coefficient
    sorted_counts = np.sort(category_counts.values)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    
    # Coverage
    n_categories_total = exposure_log[category_column].nunique()
    coverage = len(category_probs) / n_categories_total if n_categories_total > 0 else 0
    
    # Jain's fairness index
    sum_x = category_counts.sum()
    sum_x_squared = (category_counts**2).sum()
    jains_index = (sum_x**2) / (len(category_counts) * sum_x_squared) if sum_x_squared > 0 else 0
    
    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini,
        'coverage': coverage,
        'jains_fairness_index': jains_index,
        'n_categories_shown': len(category_probs),
        'category_distribution': category_probs.to_dict()
    }


def compute_advertiser_fairness(
    exposure_log: pd.DataFrame,
    ad_id_column: str = 'ad_id'
) -> Dict[str, float]:
    # Compute fairness metrics for advertiser exposure

    ad_exposures = exposure_log[ad_id_column].value_counts()
    
    # Share of voice
    total_exposures = len(exposure_log)
    share_of_voice = ad_exposures / total_exposures
    
    # Concentration metrics
    hhi = (share_of_voice**2).sum()  # Herfindahl-Hirschman Index
    
    # Top-k concentration
    top_5_share = share_of_voice.head(5).sum()
    top_10_share = share_of_voice.head(10).sum()
    
    # Gini for advertisers
    sorted_exposures = np.sort(ad_exposures.values)
    n = len(sorted_exposures)
    cumsum = np.cumsum(sorted_exposures)
    gini = (2 * np.sum((np.arange(n) + 1) * sorted_exposures)) / (n * cumsum[-1]) - (n + 1) / n
    
    return {
        'herfindahl_index': hhi,
        'gini_coefficient': gini,
        'top_5_concentration': top_5_share,
        'top_10_concentration': top_10_share,
        'n_advertisers_shown': len(ad_exposures),
        'mean_exposures_per_ad': ad_exposures.mean(),
        'std_exposures_per_ad': ad_exposures.std()
    }


# Long-Term Value Metrics

def compute_ltv_proxy(
    exposure_log: pd.DataFrame,
    gamma: float = 0.95,
    awareness_column: str = 'awareness_after'
) -> Dict[str, float]:
    # Compute Long-Term Value (LTV) proxy

    # Group by guest
    guest_groups = exposure_log.groupby('guest_id')
    
    ltv_values = []
    
    for guest_id, group in guest_groups:
        # Sort by session chronologically (assuming session_id is chronological)
        group_sorted = group.sort_values('session_id')
        
        # Compute discounted cumulative reward
        discounts = gamma ** np.arange(len(group_sorted))
        discounted_rewards = discounts * group_sorted['revenue'].values
        
        # LTV = sum of discounted rewards
        ltv = discounted_rewards.sum()
        
        # Future value from awareness
        final_awareness = group_sorted[awareness_column].iloc[-1]
        future_value = final_awareness * 10  # Proxy: awareness worth $10
        
        total_ltv = ltv + gamma**len(group_sorted) * future_value
        
        ltv_values.append({
            'guest_id': guest_id,
            'immediate_value': group_sorted['revenue'].sum(),
            'discounted_value': ltv,
            'awareness_value': future_value,
            'total_ltv': total_ltv
        })
    
    ltv_df = pd.DataFrame(ltv_values)
    
    return {
        'mean_immediate_value': ltv_df['immediate_value'].mean(),
        'mean_discounted_value': ltv_df['discounted_value'].mean(),
        'mean_awareness_value': ltv_df['awareness_value'].mean(),
        'mean_total_ltv': ltv_df['total_ltv'].mean(),
        'ltv_by_guest': ltv_df
    }


# Distribution Shift Analysis

def compute_distribution_shift(
    dist1: np.ndarray,
    dist2: np.ndarray,
    bins: int = 50
) -> Dict[str, float]:
    # Measure distribution shift using multiple metrics.

    # Create histograms
    range_min = min(dist1.min(), dist2.min())
    range_max = max(dist1.max(), dist2.max())
    
    hist1, bin_edges = np.histogram(dist1, bins=bins, range=(range_min, range_max), density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=(range_min, range_max), density=True)
    
    # Normalize to probabilities
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # KL divergence
    kl_div = np.sum(hist1 * np.log(hist1 / hist2))
    
    # Jensen-Shannon divergence
    js_div = jensenshannon(hist1, hist2)
    
    # Wasserstein distance (Earth Mover's Distance)
    wasserstein = stats.wasserstein_distance(dist1, dist2)
    
    # Mean shift
    mean_shift = dist2.mean() - dist1.mean()
    
    # Variance ratio
    variance_ratio = dist2.var() / (dist1.var() + 1e-10)
    
    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'wasserstein_distance': wasserstein,
        'mean_shift': mean_shift,
        'variance_ratio': variance_ratio
    }


def analyze_temporal_drift(
    exposure_log: pd.DataFrame,
    metric_column: str = 'base_utility',
    time_column: str = 'day_of_stay'
) -> pd.DataFrame:
    # Analyze how a metric drifts over time.

    drift_results = []
    
    time_periods = sorted(exposure_log[time_column].unique())
    
    # Compare consecutive periods
    for i in range(len(time_periods) - 1):
        period1 = time_periods[i]
        period2 = time_periods[i + 1]
        
        dist1 = exposure_log[exposure_log[time_column] == period1][metric_column].values
        dist2 = exposure_log[exposure_log[time_column] == period2][metric_column].values
        
        if len(dist1) > 0 and len(dist2) > 0:
            shift_metrics = compute_distribution_shift(dist1, dist2)
            
            drift_results.append({
                'period_from': period1,
                'period_to': period2,
                **shift_metrics
            })
    
    return pd.DataFrame(drift_results)


# Comprehensive Evaluation Report

def generate_comprehensive_report(
    exposure_log: pd.DataFrame,
    target_policy_probs: Optional[np.ndarray] = None,
    logging_policy_probs: Optional[np.ndarray] = None
) -> Dict:
    # Generate comprehensive evaluation report with all advanced metrics.
    
    report = {}
    
    # 1. Calibration metrics
    if 'click_prob' in exposure_log.columns and 'click' in exposure_log.columns:
        report['calibration'] = {
            'brier_score': compute_brier_score(
                exposure_log['click_prob'].values,
                exposure_log['click'].values
            ),
            'ece': compute_ece(
                exposure_log['click_prob'].values,
                exposure_log['click'].values
            )
        }
    
    # 2. Diversity metrics
    if 'advertiser_type' in exposure_log.columns:
        report['diversity'] = compute_diversity_metrics(exposure_log)
    
    # 3. Fairness metrics
    if 'ad_id' in exposure_log.columns:
        report['fairness'] = compute_advertiser_fairness(exposure_log)
    
    # 4. LTV metrics
    if 'awareness_after' in exposure_log.columns and 'guest_id' in exposure_log.columns:
        report['ltv'] = compute_ltv_proxy(exposure_log)
    
    # 5. Distribution shift
    if 'base_utility' in exposure_log.columns and 'day_of_stay' in exposure_log.columns:
        report['temporal_drift'] = analyze_temporal_drift(exposure_log)
    
    # 6. Off-policy evaluation
    if target_policy_probs is not None and logging_policy_probs is not None:
        rewards = exposure_log['click'].values if 'click' in exposure_log.columns else np.zeros(len(exposure_log))
        report['off_policy'] = self_normalized_ips(
            exposure_log,
            target_policy_probs,
            logging_policy_probs,
            rewards
        )
    
    return report


# Example usage
if __name__ == '__main__':
    # Simulate some data
    n = 1000
    predicted_probs = np.random.beta(2, 5, n)
    observed_outcomes = (np.random.random(n) < predicted_probs).astype(int)
    
    # 1. Calibration
    brier = compute_brier_score(predicted_probs, observed_outcomes)
    ece_result = compute_ece(predicted_probs, observed_outcomes)
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE: {ece_result['ece']:.4f}")
    
    # 2. SNIPS
    target_probs = np.random.beta(3, 3, n)
    logging_probs = np.random.beta(2, 5, n)
    rewards = observed_outcomes * np.random.lognormal(3, 1, n) 
    
    snips_result = self_normalized_ips(
        None, target_probs, logging_probs, rewards
    )
    print(f"SNIPS Estimate: {snips_result['snips_estimate']:.4f}")
    print(f"Effective Sample Size: {snips_result['effective_sample_size']:.1f}")
    
    # 3. Diversity
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], n, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    df = pd.DataFrame({'advertiser_type': categories})
    diversity = compute_diversity_metrics(df)
    print(f"Normalized Entropy: {diversity['normalized_entropy']:.4f}")
    print(f"Gini Coefficient: {diversity['gini_coefficient']:.4f}")
    print(f"Jain's Index: {diversity['jains_fairness_index']:.4f}")





