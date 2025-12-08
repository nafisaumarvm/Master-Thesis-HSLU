# Integration module: Connect data-driven guest segments to recommender system.

# Replaces hard-coded segments with data-driven clusters from guest_segmentation.py.


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Load saved segmentation results
RESULTS_DIR = Path(__file__).parent.parent / "results"


class DataDrivenSegmentMapper:
    
    def __init__(self):
        # Initialize with saved segmentation results
        self.cluster_profiles = None
        self.affinity_matrix = None
        self.segment_names = None
        self._load_segmentation_results()
    
    def _load_segmentation_results(self):
        # Load saved cluster profiles and affinities
        profiles_path = RESULTS_DIR / "cluster_profiles.csv"
        affinity_path = RESULTS_DIR / "segment_category_affinities.csv"
        
        if not profiles_path.exists():
            raise FileNotFoundError(
                f"Cluster profiles not found at {profiles_path}. "
                "Run run_segmentation_with_labels.py first!"
            )
        
        if not affinity_path.exists():
            raise FileNotFoundError(
                f"Affinity matrix not found at {affinity_path}. "
                "Run run_segmentation_with_labels.py first!"
            )
        
        self.cluster_profiles = pd.read_csv(profiles_path)
        self.affinity_matrix = pd.read_csv(affinity_path, index_col=0)
        
        # Extract segment names
        if 'business_label' in self.cluster_profiles.columns:
            self.segment_names = self.cluster_profiles['business_label'].tolist()
        else:
            self.segment_names = [f"Segment {i}" for i in range(len(self.cluster_profiles))]
    
    def get_segment_names(self) -> list:
        # Get list of data-driven segment names
        return self.segment_names
    
    def get_preference_matrix(self) -> pd.DataFrame:
        # Get segment-category preference matrix for recommender
        return self.affinity_matrix
    
    def get_segment_profile(self, segment_id: int) -> Dict:
        # Get detailed profile for a segment
        if segment_id >= len(self.cluster_profiles):
            raise ValueError(f"Invalid segment_id {segment_id}. Valid range: 0-{len(self.cluster_profiles)-1}")
        
        return self.cluster_profiles.loc[segment_id].to_dict()
    
    def map_old_to_new_segments(self) -> Dict[str, int]:
        # Map old hard-coded segments to new data-driven segments
        # Analyze cluster characteristics to find best matches
        mapping = {}
        
        for i, profile in self.cluster_profiles.iterrows():
            name = profile['business_label']
            
            # Luxury Leisure → Premium Couples or Luxury Families
            if 'Premium' in name or ('Luxury' in name and 'Family' in name):
                if 'Luxury Leisure' not in mapping:  # First match
                    mapping['Luxury Leisure'] = i
            
            # Family Group → Luxury Families
            if 'Family' in name or 'Families' in name:
                if 'Family Group' not in mapping:
                    mapping['Family Group'] = i
            
            # Bargain Hunter → Budget Solo or Last-Minute
            if 'Budget' in name or ('Solo' in name and profile['adr_mean'] < 80):
                if 'Bargain Hunter' not in mapping:
                    mapping['Bargain Hunter'] = i
            
            # Business Traveler → Domestic Weekend or short-stay solo
            if profile['los_mean'] <= 2 and (profile['pct_solo'] > 50 or 'Weekend' in name):
                if 'Business Traveler' not in mapping:
                    mapping['Business Traveler'] = i
        
        # Fill any missing with reasonable defaults
        if 'Luxury Leisure' not in mapping:
            mapping['Luxury Leisure'] = 3  # Premium Couples
        if 'Family Group' not in mapping:
            mapping['Family Group'] = 4  # Luxury Families
        if 'Bargain Hunter' not in mapping:
            mapping['Bargain Hunter'] = 0  # Budget Solo
        if 'Business Traveler' not in mapping:
            mapping['Business Traveler'] = 6  # Domestic Weekend Couples
        
        return mapping
    
    def generate_guest_dataset_with_segments(
        self, 
        n_guests: int = 1000
    ) -> pd.DataFrame:
        # Generate guest dataset with realistic segment distribution
        # Use actual segment proportions from clustering
        proportions = self.cluster_profiles['proportion'].values
        
        # Sample segment IDs according to real distribution
        segment_ids = np.random.choice(
            len(proportions),
            size=n_guests,
            p=proportions,
            replace=True
        )
        
        guests = pd.DataFrame({
            'guest_id': range(n_guests),
            'segment_id': segment_ids,
            'segment_name': [self.segment_names[sid] for sid in segment_ids]
        })
        
        # Add segment characteristics (sampled from cluster distributions)
        for i, row in guests.iterrows():
            sid = row['segment_id']
            profile = self.cluster_profiles.loc[sid]
            
            # Sample length of stay from cluster distribution
            los_mean = profile['los_mean']
            los_std = max(1, los_mean * 0.3)  # Assume CV of 0.3
            guests.loc[i, 'length_of_stay'] = max(1, int(np.random.normal(los_mean, los_std)))
            
            # Sample party size
            party_mean = profile['party_size_mean']
            guests.loc[i, 'party_size'] = max(1, int(np.random.normal(party_mean, 0.5)))
            
            # Sample ADR (if available)
            if not np.isnan(profile['adr_mean']):
                adr_mean = profile['adr_mean']
                adr_std = max(10, adr_mean * 0.2)
                guests.loc[i, 'adr'] = max(20, np.random.normal(adr_mean, adr_std))
            
            # Assign categorical features based on cluster probabilities
            guests.loc[i, 'is_family'] = np.random.random() < (profile['pct_family'] / 100)
            guests.loc[i, 'is_couple'] = np.random.random() < (profile['pct_couple'] / 100)
            guests.loc[i, 'is_solo'] = np.random.random() < (profile['pct_solo'] / 100)
        
        return guests


def create_preference_matrix_for_recommender() -> Tuple[pd.DataFrame, list]:
    # Create preference matrix in format expected by existing recommender system
    mapper = DataDrivenSegmentMapper()
    
    # Get data-driven preference matrix
    pref_matrix = mapper.get_preference_matrix()
    segment_names = mapper.get_segment_names()
    
    return pref_matrix, segment_names


def load_or_create_guest_dataset(
    n_guests: int = 1000,
    use_cached: bool = True
) -> pd.DataFrame:
    # Load or create guest dataset with data-driven segments
    cache_path = RESULTS_DIR / f"guest_dataset_{n_guests}.csv"
    
    if use_cached and cache_path.exists():
        print(f"Loading cached guest dataset from {cache_path}")
        return pd.read_csv(cache_path)
    
    print(f"Generating {n_guests:,} guests with data-driven segments...")
    mapper = DataDrivenSegmentMapper()
    guests = mapper.generate_guest_dataset_with_segments(n_guests)
    
    # Cache for future use
    guests.to_csv(cache_path, index=False)
    print(f"Saved to {cache_path}")
    
    return guests


def get_segment_learning_rates() -> Dict[int, Dict[str, float]]:
    # Get segment-specific awareness learning rates
    mapper = DataDrivenSegmentMapper()
    
    learning_rates = {}
    
    for i, profile in mapper.cluster_profiles.iterrows():
        # Alpha (awareness growth rate): Higher for luxury/leisure, lower for budget/business
        if profile['pct_luxury'] > 50:
            alpha = 0.40  # High responsiveness
        elif profile['pct_luxury'] > 30:
            alpha = 0.30  # Medium responsiveness
        else:
            alpha = 0.20  # Lower responsiveness
        
        # Delta (decay rate): Higher for short stays, lower for long stays
        if profile['los_mean'] <= 2:
            delta = 0.15  # Fast decay (short stay)
        elif profile['los_mean'] <= 5:
            delta = 0.10  # Medium decay
        else:
            delta = 0.05  # Slow decay (long stay)
        
        learning_rates[i] = {'alpha': alpha, 'delta': delta}
    
    return learning_rates


if __name__ == "__main__":
    
    # Test 1: Load segmentation results
    print("\nLoading segmentation results...")
    mapper = DataDrivenSegmentMapper()
    print(f"Loaded {len(mapper.segment_names)} segments")
    
    # Test 2: Get preference matrix
    print("\nLoading preference matrix...")
    pref_matrix = mapper.get_preference_matrix()
    print(f"Matrix shape: {pref_matrix.shape}")
    print("\nPreview:")
    print(pref_matrix.head())
    
    # Test 3: Map old to new segments
    print("\nMapping old segments to new clusters...")
    old_to_new = mapper.map_old_to_new_segments()
    for old_name, new_id in old_to_new.items():
        new_name = mapper.segment_names[new_id]
        print(f"   {old_name:20s} → Cluster {new_id}: {new_name}")
    
    # Test 4: Generate guest dataset
    print("\nGenerating guest dataset...")
    guests = mapper.generate_guest_dataset_with_segments(n_guests=1000)
    print(f"Generated {len(guests):,} guests")
    print("\nSegment distribution:")
    print(guests['segment_name'].value_counts())
    
    # Test 5: Get learning rates
    print("\nGetting segment-specific learning rates...")
    learning_rates = get_segment_learning_rates()
    for seg_id, rates in learning_rates.items():
        seg_name = mapper.segment_names[seg_id]
        print(f"   {seg_name:30s}: α={rates['alpha']:.2f}, δ={rates['delta']:.2f}")





