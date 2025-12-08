# Counterfactual Baseline Policies


import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import beta
from scipy.special import expit as sigmoid
import warnings
warnings.filterwarnings('ignore')


# Content-based filtering baseline

def content_based_policy(
    guest_id: str,
    guest_row: pd.Series,
    candidate_ads_df: pd.DataFrame,
    k: int = 3,
    **kwargs
) -> List[str]:
    # Content-based filtering: Match ads to guest demographics without awareness dynamics   

    if len(candidate_ads_df) == 0:
        return []
    
    scores = []
    
    # Segment-category preferences (from segment analysis)
    segment_category_prefs = {
        'luxury_leisure': {'Experiences': 0.9, 'Wellness': 0.8, 'Shopping': 0.7, 'Restaurants': 0.8},
        'budget_family': {'Experiences': 0.7, 'Shopping': 0.6, 'Restaurants': 0.8, 'Nightlife': 0.3},
        'business_traveler': {'Restaurants': 0.8, 'Nightlife': 0.6, 'Experiences': 0.5, 'Shopping': 0.4},
        'solo_backpacker': {'Experiences': 0.9, 'Nightlife': 0.7, 'Restaurants': 0.6, 'Shopping': 0.5},
        'couple_romantic': {'Wellness': 0.9, 'Restaurants': 0.9, 'Experiences': 0.7, 'Nightlife': 0.6},
        'group_friends': {'Nightlife': 0.9, 'Experiences': 0.8, 'Restaurants': 0.7, 'Shopping': 0.6},
        'extended_stay': {'Shopping': 0.7, 'Restaurants': 0.8, 'Experiences': 0.6, 'Wellness': 0.5},
        'budget_solo': {'Experiences': 0.8, 'Restaurants': 0.7, 'Shopping': 0.5, 'Nightlife': 0.4}
    }
    
    segment_name = guest_row.get('segment_name', 'luxury_leisure')
    category_prefs = segment_category_prefs.get(segment_name, {})
    
    # Guest price level
    guest_price = guest_row.get('price_per_night', 100)
    if guest_price < 80:
        guest_price_tier = 'budget'
    elif guest_price < 150:
        guest_price_tier = 'mid'
    else:
        guest_price_tier = 'premium'
    
    for _, ad in candidate_ads_df.iterrows():
        score = 0.0
        
        # Category preference match
        ad_category = ad.get('category', 'Experiences')
        category_score = category_prefs.get(ad_category, 0.5)
        score += category_score * 0.4
        
        # Price level match
        ad_price_tier = ad.get('price_level', 'mid')
        if ad_price_tier == guest_price_tier:
            score += 0.3
        elif abs(ord(ad_price_tier[0]) - ord(guest_price_tier[0])) <= 1:
            score += 0.15  # Adjacent tiers
        
        # Distance preference (closer = better, but not too close)
        distance = ad.get('distance_km', 5.0)
        if distance <= 2.0:
            score += 0.2  # Very close
        elif distance <= 5.0:
            score += 0.15  # Close
        elif distance <= 8.0:
            score += 0.1  # Moderate
        
        # Base utility (if available)
        if 'base_utility' in ad:
            score += ad['base_utility'] * 0.1
        
        scores.append((ad['ad_id'], score))
    
    # Sort by score and select top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [ad_id for ad_id, _ in scores[:k]]
    
    return selected

# Thompson Sampling Bandit

class ThompsonSamplingBandit:
    
    def __init__(self, seed: Optional[int] = None):
        self.alpha = {}  # Success counts per ad
        self.beta = {}   # Failure counts per ad
        self.rng = np.random.default_rng(seed)
    
    def select_ads(
        self,
        candidate_ads_df: pd.DataFrame,
        k: int = 3
    ) -> List[str]:
        # Select ads using Thompson Sampling    

        if len(candidate_ads_df) == 0:
            return []
        
        sampled_probs = []
        
        for _, ad in candidate_ads_df.iterrows():
            ad_id = ad['ad_id']
            
            # Get Beta parameters (default: uniform prior)
            alpha = self.alpha.get(ad_id, 1.0)
            beta_param = self.beta.get(ad_id, 1.0)
            
            # Sample from Beta distribution
            sampled_prob = self.rng.beta(alpha, beta_param)
            sampled_probs.append((ad_id, sampled_prob))
        
        # Select top-k by sampled probability
        sampled_probs.sort(key=lambda x: x[1], reverse=True)
        selected = [ad_id for ad_id, _ in sampled_probs[:k]]
        
        return selected
    
    def update(self, ad_id: str, clicked: bool):
        # Update Beta parameters based on observed outcome

        if ad_id not in self.alpha:
            self.alpha[ad_id] = 1.0
            self.beta[ad_id] = 1.0
        
        if clicked:
            self.alpha[ad_id] += 1.0
        else:
            self.beta[ad_id] += 1.0


class UCBBandit:
    # Upper Confidence Bound (UCB) for contextual bandit ad selection.
    
    def __init__(self, confidence_level: float = 2.0, seed: Optional[int] = None):
        self.confidence_level = confidence_level
        self.clicks = {}  # Click counts per ad
        self.impressions = {}  # Impression counts per ad
        self.rng = np.random.default_rng(seed)
    
    def select_ads(
        self,
        candidate_ads_df: pd.DataFrame,
        k: int = 3
    ) -> List[str]:
        # Select ads using UCB

        if len(candidate_ads_df) == 0:
            return []
        
        total_impressions = sum(self.impressions.values()) if self.impressions else 1
        
        ucb_scores = []
        
        for _, ad in candidate_ads_df.iterrows():
            ad_id = ad['ad_id']
            
            # Get empirical mean
            if ad_id in self.impressions and self.impressions[ad_id] > 0:
                empirical_mean = self.clicks.get(ad_id, 0) / self.impressions[ad_id]
            else:
                empirical_mean = 0.5  # Optimistic initialization
            
            # Compute UCB
            n_ad = self.impressions.get(ad_id, 1)
            confidence_bound = self.confidence_level * np.sqrt(np.log(total_impressions + 1) / (n_ad + 1))
            ucb = empirical_mean + confidence_bound
            
            ucb_scores.append((ad_id, ucb))
        
        # Select top-k by UCB
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [ad_id for ad_id, _ in ucb_scores[:k]]
        
        return selected
    
    def update(self, ad_id: str, clicked: bool):
        # Update statistics based on observed outcome

        if ad_id not in self.impressions:
            self.impressions[ad_id] = 0
            self.clicks[ad_id] = 0
        
        self.impressions[ad_id] += 1
        if clicked:
            self.clicks[ad_id] += 1


def thompson_sampling_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    bandit: Optional[ThompsonSamplingBandit] = None,
    k: int = 3,
    **kwargs
) -> Tuple[List[str], ThompsonSamplingBandit]:
    # Thompson Sampling policy wrapper
    
    if bandit is None:
        bandit = ThompsonSamplingBandit(seed=kwargs.get('seed', None))
    
    selected = bandit.select_ads(candidate_ads_df, k=k)
    
    return selected, bandit


def ucb_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    bandit: Optional[UCBBandit] = None,
    k: int = 3,
    **kwargs
) -> Tuple[List[str], UCBBandit]:
    # UCB policy wrapper

    if bandit is None:
        bandit = UCBBandit(confidence_level=kwargs.get('confidence_level', 2.0), 
                          seed=kwargs.get('seed', None))
    
    selected = bandit.select_ads(candidate_ads_df, k=k)
    
    return selected, bandit

# Random temporal policy

def random_temporal_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    current_time: Optional[str] = None,
    k: int = 3,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    # Random temporal policy: Show ads at random times (not optimized for room-entry)

    if len(candidate_ads_df) == 0:
        return []
    
    rng = np.random.default_rng(seed)
    
    # Random selection (no timing optimization)
    n_sample = min(k, len(candidate_ads_df))
    selected_indices = rng.choice(len(candidate_ads_df), size=n_sample, replace=False)
    selected = candidate_ads_df.iloc[selected_indices]
    
    return selected['ad_id'].tolist()


# Baseline comparison function

def compare_baselines(
    recommendations: pd.DataFrame,
    baseline_name: str,
    proposed_name: str = 'proposed'
) -> Dict[str, float]:
    # Compare a baseline policy against proposed system

    if baseline_name not in recommendations.columns or proposed_name not in recommendations.columns:
        return {}
    
    baseline_scores = recommendations[baseline_name].values
    proposed_scores = recommendations[proposed_name].values
    
    # Improvement percentage
    if baseline_scores.mean() > 0:
        improvement_pct = ((proposed_scores.mean() - baseline_scores.mean()) / baseline_scores.mean()) * 100
    else:
        improvement_pct = 0.0
    
    # Statistical significance (t-test)
    from scipy.stats import ttest_rel
    if len(baseline_scores) == len(proposed_scores):
        t_stat, p_value = ttest_rel(proposed_scores, baseline_scores)
    else:
        t_stat, p_value = 0.0, 1.0
    
    return {
        'baseline_mean': baseline_scores.mean(),
        'proposed_mean': proposed_scores.mean(),
        'improvement_pct': improvement_pct,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

