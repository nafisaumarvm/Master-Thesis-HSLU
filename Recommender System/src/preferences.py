# Guest-Advertiser Preference Matrix & Utility Model

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from .utils import set_random_seed


# Segment-Category Affinity Matrix 
# Higher values = stronger preference
SEGMENT_CATEGORY_AFFINITIES = {
    'luxury_leisure': {
        'restaurant': 0.8,
        'spa': 0.9,
        'attraction': 0.7,
        'tour': 0.6,
        'cafe': 0.5,
        'bar': 0.4,
        'museum': 0.6,
        'nightlife': 0.3,
        'transport': 0.5,
        'experience': 0.8
    },
    'cultural_tourist': {
        'restaurant': 0.7,
        'spa': 0.4,
        'attraction': 0.9,
        'tour': 0.8,
        'cafe': 0.7,
        'bar': 0.3,
        'museum': 0.9,
        'nightlife': 0.2,
        'transport': 0.6,
        'experience': 0.7
    },
    'business_traveler': {
        'restaurant': 0.6,
        'spa': 0.5,
        'attraction': 0.2,
        'tour': 0.1,
        'cafe': 0.8,
        'bar': 0.4,
        'museum': 0.2,
        'nightlife': 0.3,
        'transport': 0.9,
        'experience': 0.2
    },
    'weekend_explorer': {
        'restaurant': 0.7,
        'spa': 0.5,
        'attraction': 0.8,
        'tour': 0.7,
        'cafe': 0.6,
        'bar': 0.7,
        'museum': 0.6,
        'nightlife': 0.8,
        'transport': 0.5,
        'experience': 0.7
    },
    'budget_family': {
        'restaurant': 0.6,
        'spa': 0.2,
        'attraction': 0.8,
        'tour': 0.6,
        'cafe': 0.7,
        'bar': 0.1,
        'museum': 0.7,
        'nightlife': 0.1,
        'transport': 0.7,
        'experience': 0.6
    },
    'adventure_seeker': {
        'restaurant': 0.5,
        'spa': 0.3,
        'attraction': 0.9,
        'tour': 0.9,
        'cafe': 0.4,
        'bar': 0.6,
        'museum': 0.4,
        'nightlife': 0.7,
        'transport': 0.7,
        'experience': 0.9
    },
    'extended_stay': {
        'restaurant': 0.7,
        'spa': 0.6,
        'attraction': 0.5,
        'tour': 0.4,
        'cafe': 0.8,
        'bar': 0.5,
        'museum': 0.5,
        'nightlife': 0.4,
        'transport': 0.6,
        'experience': 0.5
    },
    'bargain_hunter': {
        'restaurant': 0.4,
        'spa': 0.2,
        'attraction': 0.6,
        'tour': 0.5,
        'cafe': 0.6,
        'bar': 0.4,
        'museum': 0.6,
        'nightlife': 0.3,
        'transport': 0.7,
        'experience': 0.4
    }
}


def generate_preference_matrix(
    segments: List[str],
    categories: List[str],
    noise_scale: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    # Generate segment-category preference matrix U[s, c]

    rng = np.random.RandomState(seed)
    
    matrix = []
    for segment in segments:
        row = []
        base_affinities = SEGMENT_CATEGORY_AFFINITIES.get(
            segment,
            {cat: 0.5 for cat in categories}  # Default: neutral
        )
        
        for category in categories:
            # Base affinity
            base = base_affinities.get(category, 0.5)
            
            # Add individual variation
            noise = rng.normal(0, noise_scale)
            utility = np.clip(base + noise, 0.0, 1.0)
            
            row.append(utility)
        
        matrix.append(row)
    
    df = pd.DataFrame(matrix, index=segments, columns=categories)
    return df


def compute_base_utility(
    guest_segment: str,
    advertiser_category: str,
    guest_context: Optional[Dict] = None,
    advertiser_attrs: Optional[Dict] = None,
    preference_matrix: Optional[pd.DataFrame] = None
) -> float:
    # Compute base utility U0 for guest-advertiser pair

    # 1. Segment-category affinity (core preference)
    if preference_matrix is not None and guest_segment in preference_matrix.index:
        if advertiser_category in preference_matrix.columns:
            base_affinity = preference_matrix.loc[guest_segment, advertiser_category]
        else:
            base_affinity = 0.5
    else:
        affinities = SEGMENT_CATEGORY_AFFINITIES.get(guest_segment, {})
        base_affinity = affinities.get(advertiser_category, 0.5)
    
    # Convert to logit scale
    utility = np.log(base_affinity / (1 - base_affinity + 1e-10))
    
    # 2. Context modifiers
    if guest_context:
        # Party size match
        if 'party_size' in guest_context and advertiser_attrs:
            if advertiser_attrs.get('family_friendly', False) and guest_context['party_size'] >= 3:
                utility += 0.5
        
        # Purpose match
        if 'trip_purpose' in guest_context:
            purpose = guest_context['trip_purpose']
            if purpose == 'business' and advertiser_category == 'cafe':
                utility += 0.3
            elif purpose == 'leisure' and advertiser_category in ['attraction', 'tour', 'experience']:
                utility += 0.4
            elif purpose == 'visiting_friends' and advertiser_category in ['restaurant', 'bar']:
                utility += 0.3
        
        # Time of day match
        if 'time_of_day' in guest_context and advertiser_attrs:
            time = guest_context['time_of_day']
            if time == 'morning' and advertiser_category == 'cafe':
                utility += 0.4
            elif time == 'evening' and advertiser_category in ['restaurant', 'bar', 'nightlife']:
                utility += 0.3
            elif time == 'afternoon' and advertiser_category in ['museum', 'attraction']:
                utility += 0.2
        
        # Weather match
        if 'weather' in guest_context:
            weather = guest_context['weather']
            if weather == 'rainy' and advertiser_category in ['museum', 'spa', 'cafe']:
                utility += 0.4
            elif weather == 'sunny' and advertiser_category in ['tour', 'attraction', 'experience']:
                utility += 0.4
    
    # 3. Advertiser attribute match
    if advertiser_attrs:
        # Price match
        if 'price_level' in advertiser_attrs:
            price = advertiser_attrs['price_level']
            if guest_segment == 'luxury_leisure' and price == 'high':
                utility += 0.5
            elif guest_segment == 'bargain_hunter' and price == 'low':
                utility += 0.5
            elif guest_segment == 'budget_family' and price == 'low':
                utility += 0.4
        
        # Distance penalty (farther = less utility)
        if 'distance_km' in advertiser_attrs:
            distance = advertiser_attrs['distance_km']
            utility -= 0.05 * distance  # Decay with distance
    
    return utility


def compute_choice_probability(
    base_utility: float,
    awareness: float,
    position: int,
    awareness_weight: float = 0.5,
    position_weights: Optional[Dict[int, float]] = None
) -> float:
    # Compute choice probability

    if position_weights is None:
        position_weights = {1: 1.0, 2: 0.7, 3: 0.5}
    
    # Position bias on log scale
    pos_weight = position_weights.get(position, 0.5)
    pos_bias = np.log(pos_weight + 1e-10)
    
    # Awareness effect
    awareness_effect = awareness_weight * awareness
    
    # Total utility
    total_utility = base_utility + awareness_effect + pos_bias
    
    # Sigmoid
    prob = 1.0 / (1.0 + np.exp(-total_utility))
    
    return prob


def update_awareness(
    current_awareness: float,
    was_exposed: bool,
    alpha: float = 0.3
) -> float:
    # Update awareness

    if was_exposed:
        return current_awareness + alpha * (1 - current_awareness)
    else:
        return current_awareness


def generate_guest_advertiser_utilities(
    guests_df: pd.DataFrame,
    advertisers_df: pd.DataFrame,
    preference_matrix: Optional[pd.DataFrame] = None,
    sample_per_guest: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    # Generate utility matrix for all guest-advertiser pairs
    rng = np.random.RandomState(seed)
    
    # Generate preference matrix if not provided
    if preference_matrix is None:
        segments = guests_df['segment'].unique().tolist()
        categories = advertisers_df['advertiser_type'].unique().tolist()
        preference_matrix = generate_preference_matrix(segments, categories, seed=seed)
    
    utilities = []
    
    for _, guest in guests_df.iterrows():
        guest_id = guest['guest_id']
        segment = guest['segment']
        
        # Sample advertisers (or use all if small)
        if len(advertisers_df) > sample_per_guest:
            sampled_ads = advertisers_df.sample(sample_per_guest, random_state=rng)
        else:
            sampled_ads = advertisers_df
        
        # Guest context
        context = {
            'party_size': guest.get('total_guests', 2),
            'trip_purpose': guest.get('purpose_of_stay', 'leisure'),
            'weather': guest.get('weather', 'sunny')
        }
        
        for _, ad in sampled_ads.iterrows():
            ad_id = ad['ad_id']
            category = ad['advertiser_type']
            
            # Advertiser attributes
            attrs = {
                'price_level': ad.get('price_level', 'medium'),
                'distance_km': ad.get('distance_km', 5.0),
                'family_friendly': 'family_friendly' in str(ad.get('category_tags', ''))
            }
            
            # Compute base utility
            utility = compute_base_utility(
                segment, category, context, attrs, preference_matrix
            )
            
            # Base click probability (no awareness, position 1)
            base_prob = compute_choice_probability(
                utility, awareness=0.0, position=1,
                awareness_weight=0.0  # Pure utility
            )
            
            utilities.append({
                'guest_id': guest_id,
                'ad_id': ad_id,
                'segment': segment,
                'category': category,
                'base_utility': utility,
                'base_click_prob': base_prob
            })
    
    return pd.DataFrame(utilities)


def get_preference_summary(preference_matrix: pd.DataFrame) -> pd.DataFrame:
    # Summarize preference matrix for analysis
    summary = []
    for segment in preference_matrix.index:
        row = preference_matrix.loc[segment]
        summary.append({
            'segment': segment,
            'mean_affinity': row.mean(),
            'std_affinity': row.std(),
            'top_category': row.idxmax(),
            'top_affinity': row.max(),
            'bottom_category': row.idxmin(),
            'bottom_affinity': row.min()
        })
    return pd.DataFrame(summary)


# Example usage
if __name__ == '__main__':
    # Generate preference matrix
    segments = ['luxury_leisure', 'cultural_tourist', 'business_traveler']
    categories = ['restaurant', 'museum', 'cafe', 'spa', 'tour']
    
    pref_matrix = generate_preference_matrix(segments, categories)
    print("Preference Matrix:")
    print(pref_matrix)
    
    print("\nSummary:")
    print(get_preference_summary(pref_matrix))
    
    # Test utility computation
    utility = compute_base_utility(
        'luxury_leisure',
        'spa',
        guest_context={'party_size': 2, 'trip_purpose': 'leisure', 'weather': 'rainy'},
        advertiser_attrs={'price_level': 'high', 'distance_km': 2.0}
    )
    print(f"\nBase utility (luxury guest, spa, rainy day): {utility:.3f}")
    
    # Test choice probability with awareness
    for awareness in [0.0, 0.3, 0.6, 0.9]:
        prob = compute_choice_probability(utility, awareness, position=1)
        print(f"Awareness {awareness:.1f} → Click prob: {prob:.3f}")





