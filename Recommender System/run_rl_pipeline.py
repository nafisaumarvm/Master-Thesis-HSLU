"""
Run Complete RL Training Pipeline

This script implements the 5-step RL training process:
1. Train 4 baseline models (Logistic, XGBoost, Random, Popularity)
2. Select strongest model (XGBoost) as base recommender
3. Run Phase 2 simulation with awareness dynamics
4. Train ε-greedy RL policy
5. Compare base recommender vs RL policy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl_policy_training import run_complete_rl_pipeline
from src.enhanced_data_loader import load_hotel_booking_large
from src.zurich_real_data import load_zurich_advertisers
from src.segment_integration import DataDrivenSegmentMapper, load_or_create_guest_dataset
from src.exposure_log import generate_exposure_log

print("="*80)
print("RL TRAINING PIPELINE FOR IN-ROOM TV ADVERTISING")
print("="*80)

# Load data
print("\n📊 Loading data...")
# Load guests with data-driven segments
guests_df = load_or_create_guest_dataset(n_guests=1000, use_cached=False)

# Add missing columns for compatibility
if 'stay_id' not in guests_df.columns:
    guests_df['stay_id'] = guests_df['guest_id'].astype(str) + '_stay1'
if 'guest_id' not in guests_df.columns and 'guest_id' in guests_df.index:
    guests_df = guests_df.reset_index()
if 'purpose_of_stay' not in guests_df.columns:
    guests_df['purpose_of_stay'] = 'leisure'  # Default
if 'price_per_night' not in guests_df.columns:
    guests_df['price_per_night'] = guests_df.get('adr', 100).fillna(100)

ads_df = load_zurich_advertisers(n_advertisers=200)  # Use 200 for speed

# Add compatibility columns for real data to match expected format
if 'category_tags' not in ads_df.columns:
    ads_df['category_tags'] = ads_df.get('category', 'Experiences')
if 'advertiser_type' not in ads_df.columns:
    ads_df['advertiser_type'] = ads_df.get('type', ads_df.get('category', 'restaurant'))
if 'ad_id' not in ads_df.columns:
    ads_df['ad_id'] = (ads_df.index.astype(str) if 'advertiser_idx' not in ads_df.columns 
                      else ads_df['advertiser_idx'].astype(str))
if 'distance_km' not in ads_df.columns:
    ads_df['distance_km'] = 5.0  # Default, computed dynamically per guest
if 'price_level' not in ads_df.columns:
    # Map budget_tier to price_level
    tier_map = {'low': 'budget', 'medium': 'mid', 'high': 'premium'}
    ads_df['price_level'] = ads_df.get('budget_tier', 'medium').map(tier_map).fillna('mid')
if 'base_utility' not in ads_df.columns:
    ads_df['base_utility'] = 0.5  # Default base utility
if 'revenue_per_conversion' not in ads_df.columns:
    ads_df['revenue_per_conversion'] = ads_df.get('bid_amount', 3.5)  # Use bid as proxy
if 'opening_dayparts' not in ads_df.columns:
    # Map opening_hours to dayparts format (default: all dayparts open)
    # Format should be comma-separated string like "morning,afternoon,evening,late_night"
    ads_df['opening_dayparts'] = 'morning,afternoon,evening,late_night'  # Default: all open

# Create synthetic exposure log for training baselines
print("\n📝 Creating synthetic exposure log for baseline training...")
# Generate guest-ad preferences first
from src.advertisers import generate_guest_ad_preferences
guest_ad_prefs_df = generate_guest_ad_preferences(guests_df, ads_df, seed=42)

exposure_log = generate_exposure_log(
    guests_df=guests_df,
    ads_df=ads_df,
    guest_ad_prefs_df=guest_ad_prefs_df,
    n_sessions_per_stay=5,
    seed=42
)

print(f"   Created {len(exposure_log):,} exposure records")
# Check which column name is used for outcomes
outcome_col = 'click' if 'click' in exposure_log.columns else ('outcome' if 'outcome' in exposure_log.columns else None)
if outcome_col:
    print(f"   Scan rate: {exposure_log[outcome_col].mean():.2%}")
else:
    print(f"   Columns: {list(exposure_log.columns)}")

# Run complete pipeline
results = run_complete_rl_pipeline(
    exposure_log=exposure_log,
    guests_df=guests_df,
    ads_df=ads_df,
    n_simulation_days=7,
    seed=42
)

# Print summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

eval_results = results['evaluation_results']

print("\n📊 Baseline Model Performance:")
for name, metrics in results['baseline_results']['results'].items():
    print(f"   {name:20s}: AUC = {metrics['auc']:.4f}")

print(f"\n✅ Best Model: {results['baseline_results']['best_model']}")

print("\n📈 RL Policy vs Base Recommender:")
print(f"   Scan Volume:")
print(f"      Base: {eval_results['scan_volume']['base']}")
print(f"      RL:   {eval_results['scan_volume']['rl']}")
print(f"      Improvement: {eval_results['scan_volume']['improvement']:+.2f}%")

print(f"\n   Exposure Spread (Diversity):")
print(f"      Base: {eval_results['exposure_spread']['base']:.3f}")
print(f"      RL:   {eval_results['exposure_spread']['rl']:.3f}")
print(f"      Improvement: {eval_results['exposure_spread']['improvement']:+.2f}%")

print(f"\n   Guest Experience Penalty:")
print(f"      Base: {eval_results['guest_experience']['base']:.2f}%")
print(f"      RL:   {eval_results['guest_experience']['rl']:.2f}%")
print(f"      Improvement: {eval_results['guest_experience']['improvement']:+.2f}%")

print(f"\n   Awareness Estimation Error:")
print(f"      Base MSE: {eval_results['awareness_error']['base_mse']:.6f}")
print(f"      RL MSE:   {eval_results['awareness_error']['rl_mse']:.6f}")
print(f"      Improvement: {eval_results['awareness_error']['improvement']:+.2f}%")

print("\n✅ RL Pipeline Complete!")

