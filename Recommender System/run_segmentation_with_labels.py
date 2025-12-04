"""
Run complete guest segmentation pipeline with business label assignment.
"""

from src.guest_segmentation import (
    run_guest_segmentation_pipeline,
    SegmentAdAffinityMapper
)
import pandas as pd

print("="*80)
print("COMPLETE GUEST SEGMENTATION + LABELING")
print("="*80)

# Run segmentation
data_path = "/Users/nafisaumar/Documents/Master Thesis/Recommender System NEW/hotel_booking 2.csv"

model, features, results = run_guest_segmentation_pipeline(
    booking_data_path=data_path,
    n_clusters=8,
    sample_size=10000
)

# Assign business labels based on cluster characteristics
print("\n" + "="*80)
print("ASSIGNING BUSINESS LABELS")
print("="*80)

# Analyze each cluster to assign meaningful labels
profiles = model.cluster_profiles

labels = []
for i in range(len(profiles)):
    prof = profiles.loc[i]
    
    # Cluster 0: Solo travelers, short stay, budget
    if prof['pct_solo'] > 90:
        labels.append("Budget Solo Travelers")
    
    # Cluster 4: Families, luxury, medium stay
    elif prof['pct_family'] > 40 and prof['pct_luxury'] > 90:
        labels.append("Luxury Families")
    
    # Cluster 7: Very long stay, couples
    elif prof['los_mean'] > 10:
        labels.append("Extended Stay Guests")
    
    # Cluster 1: Medium stay, couples, early booking
    elif prof['los_mean'] > 5 and prof['lead_time_mean'] > 100:
        labels.append("Planned Leisure Couples")
    
    # Cluster 3: Medium stay, couples, high ADR
    elif prof['pct_couple'] > 80 and prof['pct_luxury'] > 50:
        labels.append("Premium Couples")
    
    # Cluster 2: Long lead time, couples
    elif prof['lead_time_mean'] > 200:
        labels.append("Early Planners")
    
    # Cluster 5: Very short stay, mixed party
    elif prof['los_mean'] < 2 and prof['lead_time_mean'] < 40:
        labels.append("Last-Minute City Breakers")
    
    # Cluster 6: Short stay, couples, domestic
    elif prof['pct_couple'] > 90 and prof['pct_domestic'] > 40:
        labels.append("Domestic Weekend Couples")
    
    else:
        labels.append(f"Cluster {i}")

label_map = model.assign_business_labels(labels)

# Generate segment-category affinities
print("\n" + "="*80)
print("GENERATING SEGMENT-CATEGORY AFFINITIES")
print("="*80)

mapper = SegmentAdAffinityMapper(model.cluster_profiles)
affinity_matrix = mapper.generate_expert_affinities()

# Save everything
print("\n💾 Saving results...")

# Save cluster profiles with labels
profiles_with_labels = model.cluster_profiles.copy()
results.to_csv("results/guest_clusters.csv", index=False)
profiles_with_labels.to_csv("results/cluster_profiles.csv", index=False)
affinity_matrix.to_csv("results/segment_category_affinities.csv")

# Create a summary report
print("\n" + "="*80)
print("SEGMENT SUMMARY REPORT")
print("="*80)

for i in range(len(profiles)):
    prof = profiles_with_labels.loc[i]
    label = prof['business_label']
    
    print(f"\n🏷️  {label}")
    print(f"   {'='*70}")
    print(f"   Size: {prof['size']:,} guests ({prof['proportion']*100:.1f}% of total)")
    print(f"   Length of Stay: {prof['los_mean']:.1f} nights (median: {prof['los_median']:.1f})")
    print(f"   Party Composition:")
    print(f"      - Families: {prof['pct_family']:.0f}%")
    print(f"      - Couples: {prof['pct_couple']:.0f}%")
    print(f"      - Solo: {prof['pct_solo']:.0f}%")
    if not pd.isna(prof['adr_mean']):
        print(f"   Spend: €{prof['adr_mean']:.0f} ADR (Luxury: {prof['pct_luxury']:.0f}%)")
    if not pd.isna(prof['lead_time_mean']):
        print(f"   Booking: {prof['lead_time_mean']:.0f} days lead time")
    if not pd.isna(prof['pct_domestic']):
        print(f"   Origin: {prof['pct_domestic']:.0f}% domestic, {prof['pct_long_haul']:.0f}% long-haul")
    
    print(f"\n   Ad Category Affinities:")
    for cat in affinity_matrix.columns:
        score = affinity_matrix.loc[i, cat]
        bar = '█' * int(score * 20)
        print(f"      {cat:15s}: {score:.2f} {bar}")

print("\n" + "="*80)
print("✅ SEGMENTATION COMPLETE - FILES SAVED:")
print("   - results/guest_clusters.csv")
print("   - results/cluster_profiles.csv")
print("   - results/segment_category_affinities.csv")
print("="*80)




