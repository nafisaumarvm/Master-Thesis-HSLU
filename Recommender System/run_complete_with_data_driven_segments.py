# Run Complete System with Data-Driven Guest Segments

import pandas as pd
import numpy as np
from pathlib import Path

# Import data-driven segment integration
from src.segment_integration import (
    DataDrivenSegmentMapper,
    create_preference_matrix_for_recommender,
    load_or_create_guest_dataset,
    get_segment_learning_rates
)

# Import core recommender components
from src.zurich_real_data import load_zurich_advertisers
from src.preferences_advanced import (
    update_awareness_advanced,
    compute_multi_objective_reward
)

# Load data-driven guest segments
mapper = DataDrivenSegmentMapper()
segment_names = mapper.get_segment_names()
preference_matrix = mapper.get_preference_matrix()

print(f"Loaded {len(segment_names)} data-driven segments:")
for i, name in enumerate(segment_names):
    profile = mapper.get_segment_profile(i)
    print(f"   {i}. {name:30s} ({profile['proportion']*100:4.1f}% of guests)")

# Load real Swiss advertisers
advertisers = load_zurich_advertisers(n_advertisers=None)  # All 801
print(f"Loaded {len(advertisers)} real Swiss establishments")

# Generate guest dataset with realistic segment distribution
n_guests = 1000
guests = load_or_create_guest_dataset(n_guests=n_guests, use_cached=False)

print(f"Generated {len(guests):,} guests with realistic segment distribution")
print("\nSegment Distribution:")
for name, count in guests['segment_name'].value_counts().items():
    pct = count / len(guests) * 100
# Simulate ad recommendations with awareness dynamics
# Initialize awareness (segment × advertiser)
n_segments = len(segment_names)
n_ads = len(advertisers)
awareness = np.zeros((n_segments, n_ads))

# Get segment-specific learning rates
learning_rates = get_segment_learning_rates()

# Simulate 7-day stay
n_days = 7
n_impressions_per_day = 2  # Max ads per day (guest experience constraint)

results = []

print(f"Simulating {n_days} days of ad exposure...")

for guest_idx in range(min(100, n_guests)):  # Simulate subset for speed
    guest = guests.iloc[guest_idx]
    segment_id = int(guest['segment_id'])
    los = int(guest['length_of_stay'])
    
    # Guest's awareness vector
    guest_awareness = awareness[segment_id].copy()
    
    # Simulate each day
    for day in range(min(los, n_days)):
        # Select top ads based on utility (preference + awareness)
        base_utility = preference_matrix.iloc[segment_id].values
        
        # Match categories to advertisers
        ad_utilities = []
        for i, ad in advertisers.iterrows():
            cat = ad['category']
            cat_idx = list(preference_matrix.columns).index(cat)
            utility = base_utility[cat_idx] * (1 + 0.5 * guest_awareness[i])
            ad_utilities.append(utility)
        
        ad_utilities = np.array(ad_utilities)
        
        # Select top K ads
        top_k = np.argsort(ad_utilities)[::-1][:n_impressions_per_day]
        
        # Record impressions
        for ad_idx in top_k:
            ad = advertisers.iloc[ad_idx]
            
            # Update awareness
            alpha = learning_rates[segment_id]['alpha']
            delta = learning_rates[segment_id]['delta']
            
            old_awareness = guest_awareness[ad_idx]
            guest_awareness[ad_idx] = old_awareness + alpha * (1 - old_awareness)  # Growth formula
            
            # Simulate engagement (QR scan)
            scan_prob = guest_awareness[ad_idx] * base_utility[list(preference_matrix.columns).index(ad['category'])]
            scanned = np.random.random() < scan_prob
            
            results.append({
                'guest_id': guest['guest_id'],
                'segment_id': segment_id,
                'segment_name': guest['segment_name'],
                'day': day + 1,
                'advertiser_id': ad_idx,
                'advertiser_name': ad['name'],
                'category': ad['category'],
                'awareness': guest_awareness[ad_idx],
                'utility': ad_utilities[ad_idx],
                'scanned': scanned
            })
        
        # Decay awareness for non-exposed ads
        for ad_idx in range(n_ads):
            if ad_idx not in top_k:
                delta = learning_rates[segment_id]['delta']
                guest_awareness[ad_idx] = guest_awareness[ad_idx] * (1 - delta)  # Decay formula

results_df = pd.DataFrame(results)

print(f"Simulated {len(results_df):,} ad impressions")

# Analyze results by segment
for segment_id in range(n_segments):
    segment_results = results_df[results_df['segment_id'] == segment_id]
    
    if len(segment_results) == 0:
        continue
    
    segment_name = segment_names[segment_id]
    
    print(f"\n{segment_name}")
    
    # Metrics
    n_impressions = len(segment_results)
    n_scans = segment_results['scanned'].sum()
    scan_rate = n_scans / n_impressions * 100 if n_impressions > 0 else 0
    avg_awareness = segment_results['awareness'].mean()
    
    print(f"   Impressions: {n_impressions:,}")
    print(f"   QR Scans:    {n_scans:,} ({scan_rate:.1f}%)")
    print(f"   Avg Awareness: {avg_awareness:.3f}")
    
    # Top categories shown
    top_categories = segment_results['category'].value_counts().head(3)
    print(f"   Top Categories:")
    for cat, count in top_categories.items():
        pct = count / n_impressions * 100
        print(f"      {cat:20s}: {count:3d} ({pct:4.1f}%)")

# Overall metrics
print("\nOverall system metrics")

print(f"\nData Sources:")
print(f"   Guest Segments:    8 data-driven clusters from 74,486 real bookings")
print(f"   Advertisers:       801 real Swiss establishments (100%)")
print(f"   Segment Method:    Hierarchical clustering + k-means")
print(f"   Feature Count:     61 engineered features")

print(f"\nSimulation Results:")
print(f"   Total Impressions: {len(results_df):,}")
print(f"   Total QR Scans:    {results_df['scanned'].sum():,}")
print(f"   Overall Scan Rate: {results_df['scanned'].mean()*100:.2f}%")
print(f"   Avg Awareness:     {results_df['awareness'].mean():.3f}")

print(f"\nPerformance by Segment:")
segment_performance = results_df.groupby('segment_name').agg({
    'scanned': ['sum', 'mean'],
    'awareness': 'mean'
}).round(3)
segment_performance.columns = ['Total Scans', 'Scan Rate', 'Avg Awareness']
print(segment_performance.sort_values('Scan Rate', ascending=False))

print(f"\nTop Performing Advertisers:")
top_ads = results_df.groupby('advertiser_name').agg({
    'scanned': 'sum',
    'awareness': 'mean'
}).sort_values('scanned', ascending=False).head(10)
top_ads.columns = ['Total Scans', 'Avg Awareness']
print(top_ads)

# Save results
results_df.to_csv("results/simulation_results_data_driven.csv", index=False)
segment_performance.to_csv("results/segment_performance.csv")

