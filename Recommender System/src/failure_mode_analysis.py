# Failure Mode Analysis Module

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Cold start analysis

def analyze_cold_start_guests(
    recommendations: pd.DataFrame,
    training_guests: set,
    guest_id_col: str = 'guest_id',
    segment_col: str = 'segment_name'
) -> Dict[str, Any]:
    # Analyze performance on cold-start guests (not in training data)

    if guest_id_col not in recommendations.columns:
        return {}
    
    # Identify cold-start guests
    recommendations['is_cold_start'] = ~recommendations[guest_id_col].isin(training_guests)
    
    # Compare performance
    cold_start_data = recommendations[recommendations['is_cold_start']]
    warm_start_data = recommendations[~recommendations['is_cold_start']]
    
    results = {
        'n_cold_start_guests': cold_start_data[guest_id_col].nunique() if len(cold_start_data) > 0 else 0,
        'n_warm_start_guests': warm_start_data[guest_id_col].nunique() if len(warm_start_data) > 0 else 0,
        'cold_start_pct': (len(cold_start_data) / len(recommendations) * 100) if len(recommendations) > 0 else 0.0
    }
    
    # Performance comparison
    if 'click' in recommendations.columns:
        if len(cold_start_data) > 0:
            results['cold_start_scan_rate'] = cold_start_data['click'].mean()
        else:
            results['cold_start_scan_rate'] = 0.0
        
        if len(warm_start_data) > 0:
            results['warm_start_scan_rate'] = warm_start_data['click'].mean()
        else:
            results['warm_start_scan_rate'] = 0.0
        
        if results['warm_start_scan_rate'] > 0:
            results['cold_start_degradation_pct'] = (
                (results['warm_start_scan_rate'] - results['cold_start_scan_rate']) /
                results['warm_start_scan_rate'] * 100
            )
        else:
            results['cold_start_degradation_pct'] = 0.0
    
    # Segment-level cold start
    if segment_col in recommendations.columns:
        segment_cold_start = {}
        for segment in recommendations[segment_col].unique():
            segment_data = recommendations[recommendations[segment_col] == segment]
            segment_cold = segment_data[segment_data['is_cold_start']]
            
            if len(segment_cold) > 0 and 'click' in segment_cold.columns:
                segment_cold_start[segment] = {
                    'n_guests': segment_cold[guest_id_col].nunique(),
                    'scan_rate': segment_cold['click'].mean()
                }
        
        results['segment_cold_start'] = segment_cold_start
    
    return results


def analyze_cold_start_segments(
    recommendations: pd.DataFrame,
    training_segments: set,
    segment_col: str = 'segment_name'
) -> Dict[str, Any]:
    # Analyze performance on new segments not in training data
    if segment_col not in recommendations.columns:
        return {}
    
    # Identify cold-start segments
    recommendations['is_cold_start_segment'] = ~recommendations[segment_col].isin(training_segments)
    
    cold_start_segments = recommendations[recommendations['is_cold_start_segment']]
    warm_start_segments = recommendations[~recommendations['is_cold_start_segment']]
    
    results = {
        'n_cold_start_segments': cold_start_segments[segment_col].nunique() if len(cold_start_segments) > 0 else 0,
        'cold_start_segment_names': list(cold_start_segments[segment_col].unique()) if len(cold_start_segments) > 0 else []
    }
    
    if 'click' in recommendations.columns:
        if len(cold_start_segments) > 0:
            results['cold_start_segment_scan_rate'] = cold_start_segments['click'].mean()
        else:
            results['cold_start_segment_scan_rate'] = 0.0
        
        if len(warm_start_segments) > 0:
            results['warm_start_segment_scan_rate'] = warm_start_segments['click'].mean()
        else:
            results['warm_start_segment_scan_rate'] = 0.0
    
    return results


# Adversarial scenario analysis

def analyze_adversarial_gaming(
    recommendations: pd.DataFrame,
    popularity_col: str = 'base_utility',
    scan_col: str = 'click'
) -> Dict[str, Any]:
    # Analyze vulnerability to adversarial gaming (artificially boosting popularity)
    if popularity_col not in recommendations.columns:
        return {}
    
    # Identify potential gaming (high popularity but low actual engagement)
    recommendations['popularity_rank'] = recommendations.groupby('guest_id')[popularity_col].rank(ascending=False)
    
    # Top-ranked ads that don't perform well
    top_ranked = recommendations[recommendations['popularity_rank'] <= 2]
    
    if len(top_ranked) > 0 and scan_col in top_ranked.columns:
        top_ranked_scan_rate = top_ranked[scan_col].mean()
        overall_scan_rate = recommendations[scan_col].mean() if scan_col in recommendations.columns else 0.0
        
        # If top-ranked perform worse, might indicate gaming
        gaming_risk = (overall_scan_rate - top_ranked_scan_rate) / (overall_scan_rate + 1e-10) * 100
        
        results = {
            'top_ranked_scan_rate': top_ranked_scan_rate,
            'overall_scan_rate': overall_scan_rate,
            'gaming_risk_pct': max(0, gaming_risk),
            'n_top_ranked': len(top_ranked)
        }
    else:
        results = {}
    
    # Advertiser-level analysis (potential for gaming)
    if 'ad_id' in recommendations.columns and scan_col in recommendations.columns:
        ad_performance = recommendations.groupby('ad_id').agg({
            popularity_col: 'mean',
            scan_col: 'mean',
            'guest_id': 'count'
        }).reset_index()
        ad_performance.columns = ['ad_id', 'avg_popularity', 'scan_rate', 'n_impressions']
        
        # Correlation between popularity and actual performance
        if len(ad_performance) > 1:
            correlation = np.corrcoef(ad_performance['avg_popularity'], ad_performance['scan_rate'])[0, 1]
            results['popularity_performance_correlation'] = correlation
            
            # Low correlation might indicate gaming
            if correlation < 0.3:
                results['gaming_risk'] = 'HIGH'
            elif correlation < 0.5:
                results['gaming_risk'] = 'MEDIUM'
            else:
                results['gaming_risk'] = 'LOW'
    
    return results


# Privacy leakage analysis

def analyze_privacy_leakage(
    recommendations: pd.DataFrame,
    sensitive_categories: List[str] = None,
    guest_id_col: str = 'guest_id',
    category_col: str = 'category'
) -> Dict[str, Any]:
    # Analyze potential privacy leakage from ad sequences
    if sensitive_categories is None:
        sensitive_categories = ['Wellness', 'Spa']  # Could reveal health conditions
    
    if category_col not in recommendations.columns:
        return {}
    
    results = {}
    
    # Check if sensitive categories appear in recommendations
    sensitive_exposures = recommendations[recommendations[category_col].isin(sensitive_categories)]
    
    if len(sensitive_exposures) > 0:
        results['n_sensitive_exposures'] = len(sensitive_exposures)
        results['sensitive_exposure_pct'] = len(sensitive_exposures) / len(recommendations) * 100
        
        # Guests exposed to sensitive categories
        guests_exposed = sensitive_exposures[guest_id_col].nunique()
        total_guests = recommendations[guest_id_col].nunique()
        results['guests_exposed_to_sensitive_pct'] = (guests_exposed / total_guests * 100) if total_guests > 0 else 0.0
        
        # Pattern analysis: consecutive sensitive ads might reveal more
        guest_sequences = recommendations.groupby(guest_id_col)[category_col].apply(list)
        consecutive_sensitive = 0
        for seq in guest_sequences:
            for i in range(len(seq) - 1):
                if seq[i] in sensitive_categories and seq[i+1] in sensitive_categories:
                    consecutive_sensitive += 1
                    break
        
        results['guests_with_consecutive_sensitive'] = consecutive_sensitive
        results['consecutive_sensitive_pct'] = (consecutive_sensitive / total_guests * 100) if total_guests > 0 else 0.0
    
    # Diversity analysis: low diversity might reveal preferences too clearly
    guest_diversity = recommendations.groupby(guest_id_col)[category_col].apply(
        lambda x: x.nunique()
    )
    results['avg_category_diversity_per_guest'] = guest_diversity.mean()
    results['low_diversity_guests_pct'] = (guest_diversity < 2).mean() * 100  # Only 1-2 categories
    
    return results


def compute_privacy_risk_score(
    privacy_analysis: Dict[str, Any]
) -> float:
    # Compute overall privacy risk score (0-1, higher = more risk)
    risk_score = 0.0
    
    # Sensitive exposure risk
    if 'sensitive_exposure_pct' in privacy_analysis:
        risk_score += min(privacy_analysis['sensitive_exposure_pct'] / 100, 0.4)
    
    # Consecutive sensitive risk
    if 'consecutive_sensitive_pct' in privacy_analysis:
        risk_score += min(privacy_analysis['consecutive_sensitive_pct'] / 100, 0.3)
    
    # Low diversity risk (reveals preferences too clearly)
    if 'low_diversity_guests_pct' in privacy_analysis:
        risk_score += min(privacy_analysis['low_diversity_guests_pct'] / 100, 0.3)
    
    return min(risk_score, 1.0)


# Comprehensive failure mode analysis

def analyze_all_failure_modes(
    recommendations: pd.DataFrame,
    training_guests: set,
    training_segments: set,
    guest_id_col: str = 'guest_id',
    segment_col: str = 'segment_name'
) -> Dict[str, Any]:
    # Comprehensive failure mode analysis

    results = {}
    
    # Cold start
    results['cold_start_guests'] = analyze_cold_start_guests(
        recommendations, training_guests, guest_id_col, segment_col
    )
    results['cold_start_segments'] = analyze_cold_start_segments(
        recommendations, training_segments, segment_col
    )
    
    # Adversarial gaming
    results['adversarial_gaming'] = analyze_adversarial_gaming(recommendations)
    
    # Privacy leakage
    results['privacy_leakage'] = analyze_privacy_leakage(
        recommendations, guest_id_col=guest_id_col
    )
    results['privacy_risk_score'] = compute_privacy_risk_score(results['privacy_leakage'])
    
    return results

