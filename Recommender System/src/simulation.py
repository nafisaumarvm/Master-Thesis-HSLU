# Simulation framework with exposure/awareness dynamics

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy.special import expit as sigmoid

from .utils import set_random_seed, clip_probability, parse_tags_string
from .exposure_log import filter_candidate_ads


class AwarenessSimulator:
    # Simulate ad recommendations with exposure/awareness dynamics
    
    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.5,
        lambda_intrusion: float = 0.1,
        f_max: int = 5,
        seed: int = 42
    ):
        # Initialize awareness simulator

        self.alpha = alpha
        self.gamma = gamma
        self.lambda_intrusion = lambda_intrusion
        self.f_max = f_max
        self.rng = set_random_seed(seed)
        
        # State tracking
        self.awareness = {}  # (guest_id, ad_id) -> awareness level
        self.frequency = {}  # (guest_id, ad_id) -> exposure count
        
    def get_awareness(self, guest_id: str, ad_id: str) -> float:
        # Get current awareness level for guest-ad pair
        key = (guest_id, ad_id)
        return self.awareness.get(key, 0.05)  # Small initial awareness
    
    def get_frequency(self, guest_id: str, ad_id: str) -> int:
        # Get exposure frequency for guest-ad pair
        key = (guest_id, ad_id)
        return self.frequency.get(key, 0)
    
    def update_awareness(self, guest_id: str, ad_id: str):
        # Update awareness after showing an ad

        key = (guest_id, ad_id)
        current_awareness = self.get_awareness(guest_id, ad_id)
        
        # Update awareness
        new_awareness = current_awareness + self.alpha * (1 - current_awareness)
        self.awareness[key] = min(new_awareness, 1.0)
        
        # Update frequency
        self.frequency[key] = self.frequency.get(key, 0) + 1
    
    def compute_click_probability(
        self,
        guest_id: str,
        ad_id: str,
        base_click_prob: float
    ) -> float:
        # Compute click probability with awareness boost
        p_click = base_click_prob * (1 + self.gamma * awareness)
        
        return clip_probability(p_click)
        awareness = self.get_awareness(guest_id, ad_id)
        
        # Click boost from awareness
        p_click = base_click_prob * (1 + self.gamma * awareness)
        
        return clip_probability(p_click)
    
    def compute_intrusion_cost(
        self,
        guest_id: str,
        ad_id: str,
        guest_context: pd.Series,
        ad_context: pd.Series
    ) -> float:
        # Compute intrusion cost for showing an ad
        awareness = self.get_awareness(guest_id, ad_id)
        freq = self.get_frequency(guest_id, ad_id)
        
        # Frequency penalty
        freq_penalty = max(0, freq - self.f_max)
        
        # Category mismatch penalty
        mismatch_penalty = 0.0
        
        # Business travelers shown nightlife/bars repeatedly
        if guest_context.get('is_business', False):
            if ad_context['advertiser_type'] in ['nightlife', 'bar']:
                mismatch_penalty = 0.5
        
        # Families shown non-family-friendly places
        if guest_context.get('is_family', False):
            ad_tags = set(parse_tags_string(ad_context.get('category_tags', '')))
            if 'family_friendly' not in ad_tags:
                if ad_context['advertiser_type'] in ['nightlife', 'bar']:
                    mismatch_penalty = 0.7
        
        # Total intrusion cost
        intrusion = self.lambda_intrusion * (freq_penalty + mismatch_penalty)
        
        return intrusion
    
    def reset(self):
        # Reset awareness and frequency state
        self.awareness = {}
        self.frequency = {}


def run_simulation(
    policy,
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    guest_ad_prefs_df: pd.DataFrame,
    n_sessions_per_stay: int = 4,
    alpha: float = 0.3,
    gamma: float = 0.5,
    lambda_intrusion: float = 0.1,
    f_max: int = 5,
    k_ads_per_session: int = 3,
    max_distance_km: float = 8.0,
    seed: int = 42,
    policy_name: str = "policy"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Run simulation with a given policy
    
    return (simulation_log, metrics_dict)
    rng = set_random_seed(seed)
    
    # Initialize awareness simulator
    awareness_sim = AwarenessSimulator(
        alpha=alpha,
        gamma=gamma,
        lambda_intrusion=lambda_intrusion,
        f_max=f_max,
        seed=seed
    )
    
    # Time of day options
    time_of_day_options = ['morning', 'afternoon', 'evening', 'late_night']
    
    # Simulation log
    sim_records = []
    
    # Metrics accumulators
    total_clicks = 0
    total_revenue = 0.0
    total_intrusion = 0.0
    total_impressions = 0
    
    session_counter = 0
    
    for idx, guest in guests_df.iterrows():
        guest_id = guest['guest_id']
        stay_id = guest['stay_id']
        
        # Get nights and ensure it's a valid integer
        nights = guest.get('nights', guest.get('stay_nights', 2))
        if pd.isna(nights) or nights <= 0:
            nights = 2  # Default to 2 nights if missing or invalid
        nights = int(nights)
        
        source = guest.get('source', 'unknown')
        
        guest_clicks = 0
        guest_revenue = 0.0
        guest_intrusion = 0.0
        guest_impressions = 0
        
        for session_idx in range(n_sessions_per_stay):
            session_id = f"SIM_SESSION_{session_counter:08d}"
            session_counter += 1
            
            # Sample context
            time_of_day = rng.choice(time_of_day_options)
            day_of_stay = min(int(rng.integers(1, nights + 2)), nights)
            
            # Filter candidate ads
            candidate_ads = filter_candidate_ads(ads_df, time_of_day, max_distance_km)
            
            if len(candidate_ads) == 0:
                continue
            
            # Policy selects ads
            try:
                selected_ad_ids = policy.select_ads(
                    guest, candidate_ads, k=k_ads_per_session
                )
            except Exception as e:
                # Fallback to random if policy fails
                n_sample = min(k_ads_per_session, len(candidate_ads))
                selected_indices = rng.choice(len(candidate_ads), size=n_sample, replace=False)
                selected_ad_ids = candidate_ads.iloc[selected_indices]['ad_id'].tolist()
            
            if len(selected_ad_ids) == 0:
                continue
            
            # Show ads and simulate clicks
            for position, ad_id in enumerate(selected_ad_ids, start=1):
                # Get base click probability
                pref_match = guest_ad_prefs_df[
                    (guest_ad_prefs_df['guest_id'] == guest_id) &
                    (guest_ad_prefs_df['ad_id'] == ad_id)
                ]
                
                if len(pref_match) == 0:
                    base_click_prob = 0.05
                else:
                    base_click_prob = pref_match.iloc[0]['base_click_prob']
                
                # Compute click probability with awareness
                click_prob = awareness_sim.compute_click_probability(
                    guest_id, ad_id, base_click_prob
                )
                
                # Sample click
                click = rng.binomial(1, click_prob)
                
                # Get ad info
                ad_row = ads_df[ads_df['ad_id'] == ad_id].iloc[0]
                
                # Compute revenue
                revenue = click * ad_row['revenue_per_conversion']
                
                # Compute intrusion cost
                intrusion_cost = awareness_sim.compute_intrusion_cost(
                    guest_id, ad_id, guest, ad_row
                )
                
                # Update awareness
                awareness_sim.update_awareness(guest_id, ad_id)
                
                # Update policy if it has update method
                if hasattr(policy, 'update'):
                    policy.update(ad_id, float(click))
                
                # Record
                sim_records.append({
                    'guest_id': guest_id,
                    'stay_id': stay_id,
                    'session_id': session_id,
                    'ad_id': ad_id,
                    'position': position,
                    'base_click_prob': base_click_prob,
                    'awareness': awareness_sim.get_awareness(guest_id, ad_id),
                    'frequency': awareness_sim.get_frequency(guest_id, ad_id),
                    'click_prob': click_prob,
                    'click': click,
                    'revenue': revenue,
                    'intrusion_cost': intrusion_cost,
                    'time_of_day': time_of_day,
                    'day_of_stay': day_of_stay,
                    'source': source,
                    'policy': policy_name
                })
                
                # Accumulate
                total_clicks += click
                total_revenue += revenue
                total_intrusion += intrusion_cost
                total_impressions += 1
                
                guest_clicks += click
                guest_revenue += revenue
                guest_intrusion += intrusion_cost
                guest_impressions += 1
    
    # Create simulation log
    sim_log = pd.DataFrame(sim_records)
    
    # Compute metrics
    metrics = {
        'policy': policy_name,
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_revenue': total_revenue,
        'total_intrusion': total_intrusion,
        'ctr': total_clicks / total_impressions if total_impressions > 0 else 0,
        'rpm': (total_revenue / total_impressions) * 1000 if total_impressions > 0 else 0,
        'revenue_per_stay': total_revenue / len(guests_df) if len(guests_df) > 0 else 0,
        'avg_intrusion': total_intrusion / total_impressions if total_impressions > 0 else 0,
        'guest_experience_index': -total_intrusion / len(guests_df) if len(guests_df) > 0 else 0
    }
    
    return sim_log, metrics


def compare_policies(
    policies: Dict[str, Any],
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    guest_ad_prefs_df: pd.DataFrame,
    n_sessions_per_stay: int = 4,
    alpha: float = 0.3,
    gamma: float = 0.5,
    seed: int = 42
) -> pd.DataFrame:
    # Compare multiple policies via simulation
    return pd.DataFrame(all_metrics)
    all_metrics = []
    
    for policy_name, policy in policies.items():
        print(f"Running simulation for: {policy_name}")
        
        _, metrics = run_simulation(
            policy=policy,
            guests_df=guests_df,
            ads_df=ads_df,
            guest_ad_prefs_df=guest_ad_prefs_df,
            n_sessions_per_stay=n_sessions_per_stay,
            alpha=alpha,
            gamma=gamma,
            seed=seed,
            policy_name=policy_name
        )
        
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

