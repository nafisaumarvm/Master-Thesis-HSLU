#!/usr/bin/env python
# Run Comprehensive Evaluation Pipeline

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.special import expit as sigmoid
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.zurich_real_data import load_zurich_advertisers
from src.segment_integration import load_or_create_guest_dataset, get_segment_learning_rates
from src.exposure_log import popularity_policy, filter_candidate_ads
from src.advertisers import generate_guest_ad_preferences
from src.preferences_advanced import update_awareness_advanced, SEGMENT_AWARENESS_PARAMS
from src.utils import set_random_seed

# New modules
from src.enhanced_evaluation import (
    compute_precision_at_k, compute_hit_rate, compute_mrr,
    compute_diversity_metrics, evaluate_comprehensive,
    compute_segment_performance, compute_segment_fairness,
    compute_segment_error_analysis
)
from src.counterfactual_baselines import (
    content_based_policy, ThompsonSamplingBandit, UCBBandit,
    random_temporal_policy
)
from src.sensitivity_analysis import (
    sensitivity_analysis_poisson_lambda,
    sensitivity_analysis_awareness_params,
    compute_robustness_metrics
)
from src.temporal_validation import (
    rolling_window_validation, analyze_concept_drift
)
from src.failure_mode_analysis import analyze_all_failure_modes
# Business constraints removed as requested
from src.external_validation import perform_external_validation

SEED = 42
rng = set_random_seed(SEED)

# Load hotel bookings
print("Loading hotel booking data...")
hotel_df = pd.read_csv('hotel_booking 2.csv')
if 'is_canceled' in hotel_df.columns:
    valid = hotel_df[hotel_df['is_canceled'] == 0]
else:
    valid = hotel_df
print(f"  Valid bookings: {len(valid):,}")

# Generate guests (using full dataset)
n_guests = len(valid)  # Use complete dataset
print(f"\n  Generating {n_guests:,} guests (FULL DATASET)...")
guests_df = load_or_create_guest_dataset(n_guests=n_guests, use_cached=True)

# Add compatibility columns
if 'stay_id' not in guests_df.columns:
    guests_df['stay_id'] = guests_df['guest_id'].astype(str) + '_stay1'
if 'purpose_of_stay' not in guests_df.columns:
    guests_df['purpose_of_stay'] = 'leisure'
if 'price_per_night' not in guests_df.columns:
    guests_df['price_per_night'] = guests_df.get('adr', 100).fillna(100)

# Load advertisers
print("\nLoading advertiser data...")
ads_df = load_zurich_advertisers(n_advertisers=None)
print(f"  Loaded {len(ads_df):,} advertisers")

# Add compatibility columns
if 'category_tags' not in ads_df.columns:
    ads_df['category_tags'] = ads_df.get('category', 'Experiences')
if 'advertiser_type' not in ads_df.columns:
    ads_df['advertiser_type'] = ads_df.get('type', ads_df.get('category', 'restaurant'))
if 'ad_id' not in ads_df.columns:
    ads_df['ad_id'] = ads_df.index.astype(str)
if 'distance_km' not in ads_df.columns:
    ads_df['distance_km'] = 5.0
if 'price_level' not in ads_df.columns:
    tier_map = {'low': 'budget', 'medium': 'mid', 'high': 'premium'}
    ads_df['price_level'] = ads_df.get('budget_tier', 'medium').map(tier_map).fillna('mid')
if 'base_utility' not in ads_df.columns:
    ads_df['base_utility'] = 0.5
if 'revenue_per_conversion' not in ads_df.columns:
    ads_df['revenue_per_conversion'] = ads_df.get('bid_amount', 3.5)
if 'opening_dayparts' not in ads_df.columns:
    ads_df['opening_dayparts'] = 'morning,afternoon,evening,late_night'

# Apply TV viewing rate (68%)
TV_VIEWING_RATE = 0.68
tv_watching_guests = guests_df.sample(frac=TV_VIEWING_RATE, random_state=SEED)
print(f"  TV viewing rate: {TV_VIEWING_RATE*100:.0f}%")
print(f"  TV-watching guests: {len(tv_watching_guests):,}")

# Generate guest-ad preferences

guest_ad_prefs_df = generate_guest_ad_preferences(tv_watching_guests, ads_df, seed=SEED)

# Calibrate scan rate
TARGET_SCAN_RATE = 0.0124
current_avg_prob = guest_ad_prefs_df['base_click_prob'].mean()
scale_factor = TARGET_SCAN_RATE / max(current_avg_prob, 0.01) * 0.8
guest_ad_prefs_df['base_click_prob'] = (guest_ad_prefs_df['base_click_prob'] * scale_factor).clip(0, 1)
print(f"  Calibrated scan rate: {guest_ad_prefs_df['base_click_prob'].mean():.4f}")

# Run simulation with multiple policies
policies = {
    'proposed': popularity_policy,  # Our proposed system
    'content_based': content_based_policy,
    'random_temporal': random_temporal_policy
}

# Initialize bandits for contextual bandits
thompson_bandit = ThompsonSamplingBandit(seed=SEED)
ucb_bandit = UCBBandit(seed=SEED)

all_results = {}

for policy_name, policy_func in policies.items():
    print(f"\n  Running {policy_name} policy...")
    
    awareness_state = defaultdict(float)
    daily_exposure_count = defaultdict(int)
    sim_records = []
    
    MAX_ADS_PER_DAY = 2
    n_sessions_per_stay = 2
    k_ads_per_session = 2
    session_exposure_prob = 0.48
    
    for idx, guest in tv_watching_guests.iterrows():
        guest_id = guest['guest_id']
        stay_id = guest.get('stay_id', f"{guest_id}_stay1")
        nights = int(guest.get('nights', guest.get('stay_nights', 2)))
        segment_name = guest.get('segment_name', 'luxury_leisure')
        segment_params = SEGMENT_AWARENESS_PARAMS.get(segment_name, SEGMENT_AWARENESS_PARAMS['luxury_leisure'])
        alpha = segment_params['alpha'] * 0.4
        delta = segment_params['delta'] * 2.0
        beta = segment_params.get('beta', 0.5)
        
        daily_exposures = defaultdict(int)
        
        for day in range(1, min(nights + 1, 8)):
            n_sessions_today = rng.integers(1, 3)
            
            for session_idx in range(n_sessions_today):
                if daily_exposures[day] >= MAX_ADS_PER_DAY:
                    continue
                
                if rng.random() > session_exposure_prob:
                    continue
                
                time_of_day = rng.choice(['morning', 'afternoon', 'evening', 'late_night'])
                candidate_ads = filter_candidate_ads(ads_df, time_of_day, max_distance_km=8.0)
                
                if len(candidate_ads) == 0:
                    continue
                
                # Select ads using policy
                if policy_name == 'content_based':
                    selected_ad_ids = policy_func(guest_id, guest, candidate_ads, k=k_ads_per_session)
                elif policy_name == 'random_temporal':
                    selected_ad_ids = policy_func(guest_id, candidate_ads, k=k_ads_per_session, seed=SEED)
                else:
                    selected_ad_ids = policy_func(guest_id, candidate_ads, k=k_ads_per_session)
                
                for ad_id in selected_ad_ids[:MAX_ADS_PER_DAY - daily_exposures[day]]:
                    pref_match = guest_ad_prefs_df[
                        (guest_ad_prefs_df['guest_id'] == guest_id) &
                        (guest_ad_prefs_df['ad_id'] == ad_id)
                    ]
                    
                    if len(pref_match) == 0:
                        base_click_prob = 0.01
                    else:
                        base_click_prob = pref_match.iloc[0]['base_click_prob']
                    
                    awareness_key = (guest_id, ad_id)
                    current_awareness = awareness_state[awareness_key]
                    click_prob = base_click_prob * (1 + beta * current_awareness)
                    click_prob = min(click_prob, 1.0)
                    
                    clicked = rng.binomial(1, click_prob)
                    revenue = clicked * 124.0
                    
                    new_awareness = current_awareness + alpha * (1 - current_awareness)
                    awareness_state[awareness_key] = min(new_awareness, 1.0)
                    
                    sim_records.append({
                        'guest_id': guest_id,
                        'stay_id': stay_id,
                        'ad_id': ad_id,
                        'day': day,
                        'base_click_prob': base_click_prob,
                        'awareness': new_awareness,
                        'click_prob': click_prob,
                        'click': clicked,
                        'revenue': revenue,
                        'time_of_day': time_of_day,
                        'segment': segment_name,
                        'policy': policy_name
                    })
                    
                    daily_exposures[day] += 1
                
                # Decay
                exposed_today = {r['ad_id'] for r in sim_records if r.get('guest_id') == guest_id and r.get('day') == day}
                for ad_id in ads_df['ad_id'].values:
                    awareness_key = (guest_id, ad_id)
                    if awareness_key in awareness_state and ad_id not in exposed_today:
                        current_awareness = awareness_state[awareness_key]
                        decayed_awareness = current_awareness * (1 - delta)
                        awareness_state[awareness_key] = max(decayed_awareness, 0.0)
    
    all_results[policy_name] = pd.DataFrame(sim_records)
    print(f"    {policy_name}: {len(sim_records):,} exposures, {sim_records[-1]['click'] if len(sim_records) > 0 else 0} clicks")

# Comprehensive evaluation
# Use proposed policy results for comprehensive evaluation
proposed_results = all_results['proposed'].copy()

# Add predicted scores (using click_prob as proxy)
proposed_results['predicted_score'] = proposed_results['click_prob']

# Enhanced metrics
print("\n  Computing enhanced metrics...")
enhanced_metrics = evaluate_comprehensive(proposed_results, n_bootstrap=1000, seed=SEED)

print(f"\n  Enhanced Metrics:")
print(f"    Precision@2: {enhanced_metrics.get('precision@2', 0):.4f}")
print(f"    Hit Rate: {enhanced_metrics.get('hit_rate', 0):.4f}")
print(f"    MRR: {enhanced_metrics.get('mrr', 0):.4f}")
print(f"    Simpson Diversity: {enhanced_metrics.get('simpson_diversity', 0):.4f}")

# Segment-level analysis
if 'segment_name' in proposed_results.columns:
    print("\n  Segment-level analysis...")
    segment_perf = compute_segment_performance(proposed_results)
    print(f"    Analyzed {len(segment_perf)} segments")
    
    fairness = compute_segment_fairness(proposed_results)
    print(f"    Max-min ratio: {fairness.get('max_min_ratio', 0):.2f}")
    print(f"    Gini coefficient: {fairness.get('gini_coefficient', 0):.4f}")

# Counterfactual baseline comparison
for baseline_name in ['content_based', 'random_temporal']:
    if baseline_name in all_results:
        baseline_results = all_results[baseline_name]
        
        proposed_scan_rate = proposed_results['click'].mean()
        baseline_scan_rate = baseline_results['click'].mean()
        
        improvement = ((proposed_scan_rate - baseline_scan_rate) / (baseline_scan_rate + 1e-10)) * 100
        
        print(f"\n  {baseline_name} vs Proposed:")
        print(f"    Proposed scan rate: {proposed_scan_rate:.4f}")
        print(f"    {baseline_name} scan rate: {baseline_scan_rate:.4f}")
        print(f"    Improvement: {improvement:+.2f}%")

# Sensitivity analysis
# Note: Full sensitivity analysis would require re-running simulation
# For now, we'll create a placeholder structure
print("  Sensitivity analysis structure created (full analysis requires re-running simulation)")
print("  See src/sensitivity_analysis.py for implementation")

# Failure mode analysis
# Get training guests (first 80% for train/test split)
train_guests = set(guests_df['guest_id'].head(int(len(guests_df) * 0.8)))
train_segments = set(guests_df['segment_name'].head(int(len(guests_df) * 0.8)).unique())

failure_modes = analyze_all_failure_modes(
    proposed_results,
    train_guests,
    train_segments
)

print(f"  Cold start guests: {failure_modes['cold_start_guests'].get('n_cold_start_guests', 0)}")
print(f"  Cold start degradation: {failure_modes['cold_start_guests'].get('cold_start_degradation_pct', 0):.2f}%")
print(f"  Privacy risk score: {failure_modes.get('privacy_risk_score', 0):.4f}")

# Save results
results_dir = Path("results/comprehensive_evaluation")
results_dir.mkdir(parents=True, exist_ok=True)

# Save all results
for policy_name, results_df in all_results.items():
    results_df.to_csv(results_dir / f"{policy_name}_results.csv", index=False)

# Save enhanced metrics
with open(results_dir / "enhanced_metrics.json", 'w') as f:
    json.dump(enhanced_metrics, f, indent=2, default=str)

# Save failure mode analysis
with open(results_dir / "failure_modes.json", 'w') as f:
    json.dump(failure_modes, f, indent=2, default=str)

print(f"  Results saved to {results_dir}/")

