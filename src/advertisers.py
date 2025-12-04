"""
Synthetic local advertiser catalogue and guest-ad preference generation (in case real advertiser data isn't loaded)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.special import expit as sigmoid

from .utils import set_random_seed, tags_to_string, parse_tags_string


# Advertiser type definitions
ADVERTISER_TYPES = [
    'restaurant', 'bar', 'cafe', 'museum', 'gallery', 
    'spa', 'tour', 'attraction', 'nightlife', 'transport'
]

# Category tags pool
CATEGORY_TAGS = [
    'foodie', 'family_friendly', 'nightlife', 'culture', 'outdoor',
    'budget', 'luxury', 'romantic', 'adventure', 'wellness',
    'shopping', 'local', 'touristy', 'quiet', 'trendy'
]

# Dayparts
DAYPARTS = ['morning', 'afternoon', 'evening', 'late_night']

# Price levels
PRICE_LEVELS = ['low', 'medium', 'high']


def generate_advertisers(
    n_ads: int = 150,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic local advertiser catalogue.
    
    Args:
        n_ads: Number of advertisers to generate
        seed: Random seed
        
    Returns:
        Dataframe with advertiser catalogue:
        - ad_id, advertiser_name, advertiser_type, category_tags,
        - distance_km, opening_dayparts, price_level, base_utility,
        - revenue_per_conversion
    """
    rng = set_random_seed(seed)
    
    advertisers = []
    
    for i in range(n_ads):
        ad_id = f"AD_{i:04d}"
        
        # Random advertiser type
        adv_type = rng.choice(ADVERTISER_TYPES)
        
        # Generate name based on type
        advertiser_name = _generate_advertiser_name(adv_type, i, rng)
        
        # Sample category tags (2-4 tags)
        n_tags = rng.integers(2, 5)
        category_tags = rng.choice(CATEGORY_TAGS, size=n_tags, replace=False).tolist()
        
        # Add type-specific tags
        type_tags = _get_type_specific_tags(adv_type)
        category_tags = list(set(category_tags + type_tags))
        
        # Distance from hotel/center (0.1 to 15 km)
        distance_km = rng.uniform(0.1, 15.0)
        
        # Opening dayparts (at least 1, at most all 4)
        n_dayparts = rng.integers(1, 5)
        opening_dayparts = rng.choice(DAYPARTS, size=n_dayparts, replace=False).tolist()
        
        # Add type-specific dayparts
        opening_dayparts = _adjust_dayparts_for_type(adv_type, opening_dayparts)
        opening_dayparts = list(set(opening_dayparts))
        
        # Price level
        price_level = rng.choice(PRICE_LEVELS)
        
        # Base utility (sampled from N(0,1))
        base_utility = rng.normal(0, 1)
        
        # Revenue per conversion (depends on type and price level)
        revenue_per_conversion = _compute_revenue(adv_type, price_level, rng)
        
        advertisers.append({
            'ad_id': ad_id,
            'advertiser_name': advertiser_name,
            'advertiser_type': adv_type,
            'category_tags': tags_to_string(category_tags),
            'distance_km': distance_km,
            'opening_dayparts': tags_to_string(opening_dayparts),
            'price_level': price_level,
            'base_utility': base_utility,
            'revenue_per_conversion': revenue_per_conversion
        })
    
    return pd.DataFrame(advertisers)


def _generate_advertiser_name(adv_type: str, index: int, rng: np.random.Generator) -> str:
    """Generate fake advertiser name based on type."""
    
    prefixes = {
        'restaurant': ['The', 'Le', 'Il', 'Chez', 'Casa'],
        'bar': ['The', 'Bar', 'Pub', 'Lounge'],
        'cafe': ['Cafe', 'Coffee', 'Espresso'],
        'museum': ['Museum of', 'House of', 'Gallery of'],
        'gallery': ['Gallery', 'Art Space', 'Atelier'],
        'spa': ['Spa', 'Wellness', 'Zen'],
        'tour': ['Tours', 'Excursions', 'Adventures'],
        'attraction': ['The', 'Adventure', 'Experience'],
        'nightlife': ['Club', 'Night', 'Dance'],
        'transport': ['Transport', 'Shuttle', 'Travel']
    }
    
    suffixes = {
        'restaurant': ['Bistro', 'Kitchen', 'Table', 'House', 'Grill'],
        'bar': ['Bar', 'Pub', 'Tavern', 'Lounge', 'Club'],
        'cafe': ['Cafe', 'Coffee', 'Roastery', 'House', 'Bar'],
        'museum': ['Art', 'History', 'Culture', 'Science', 'Heritage'],
        'gallery': ['Modern', 'Contemporary', 'Fine Art', 'Studio'],
        'spa': ['Retreat', 'Center', 'Sanctuary', 'Oasis'],
        'tour': ['City', 'Wine', 'Food', 'Walking', 'Bike'],
        'attraction': ['Park', 'Tower', 'Gardens', 'Castle', 'Palace'],
        'nightlife': ['Lounge', 'Club', 'Bar', 'Scene'],
        'transport': ['Service', 'Shuttle', 'Express', 'Link']
    }
    
    prefix = rng.choice(prefixes.get(adv_type, ['The']))
    suffix = rng.choice(suffixes.get(adv_type, ['Place']))
    
    return f"{prefix} {suffix} {index % 100}"


def _get_type_specific_tags(adv_type: str) -> List[str]:
    """Get tags that are natural for a given advertiser type."""
    
    type_tag_map = {
        'restaurant': ['foodie'],
        'bar': ['nightlife'],
        'cafe': ['local'],
        'museum': ['culture'],
        'gallery': ['culture'],
        'spa': ['wellness', 'quiet'],
        'tour': ['adventure', 'touristy'],
        'attraction': ['touristy'],
        'nightlife': ['nightlife', 'trendy'],
        'transport': ['local']
    }
    
    return type_tag_map.get(adv_type, [])


def _adjust_dayparts_for_type(adv_type: str, dayparts: List[str]) -> List[str]:
    """Adjust dayparts to be realistic for advertiser type."""
    
    # Museums/galleries typically not open late night
    if adv_type in ['museum', 'gallery']:
        dayparts = [d for d in dayparts if d != 'late_night']
        if not dayparts:
            dayparts = ['morning', 'afternoon']
    
    # Nightlife primarily evening/late_night
    if adv_type in ['nightlife', 'bar']:
        if 'evening' not in dayparts:
            dayparts.append('evening')
    
    # Cafes typically morning/afternoon
    if adv_type == 'cafe':
        if 'morning' not in dayparts:
            dayparts.append('morning')
    
    return dayparts if dayparts else ['afternoon']


def _compute_revenue(adv_type: str, price_level: str, rng: np.random.Generator) -> float:
    """Compute revenue per conversion based on type and price level."""
    
    base_revenue = {
        'restaurant': 50,
        'bar': 30,
        'cafe': 15,
        'museum': 25,
        'gallery': 20,
        'spa': 80,
        'tour': 60,
        'attraction': 40,
        'nightlife': 35,
        'transport': 25
    }
    
    price_multiplier = {
        'low': 0.6,
        'medium': 1.0,
        'high': 1.8
    }
    
    base = base_revenue.get(adv_type, 30)
    multiplier = price_multiplier.get(price_level, 1.0)
    
    # Add some noise
    revenue = base * multiplier * rng.uniform(0.8, 1.2)
    
    return round(revenue, 2)


def compute_guest_ad_intrinsic_score(
    guest_row: pd.Series,
    ad_row: pd.Series
) -> float:
    """
    Compute intrinsic compatibility score between guest and ad.
    
    Args:
        guest_row: Row from unified guest dataframe
        ad_row: Row from advertiser catalogue
        
    Returns:
        Intrinsic score (higher = better match)
    """
    score = 0.0
    
    # Parse tags
    ad_tags = set(parse_tags_string(ad_row['category_tags']))
    
    # 1. Purpose-type matching
    purpose = guest_row.get('purpose_of_stay', 'leisure')
    adv_type = ad_row['advertiser_type']
    
    if purpose == 'business':
        # Business travelers prefer practical, quiet places
        if adv_type in ['cafe', 'restaurant', 'transport']:
            score += 0.5
        if 'quiet' in ad_tags or 'local' in ad_tags:
            score += 0.3
        if adv_type in ['nightlife', 'bar']:
            score -= 0.3
    
    elif purpose == 'leisure':
        # Leisure travelers like attractions, tours
        if adv_type in ['attraction', 'tour', 'museum', 'gallery']:
            score += 0.5
        if adv_type in ['restaurant', 'cafe', 'spa']:
            score += 0.3
    
    elif purpose == 'visiting_friends':
        # Social activities
        if adv_type in ['restaurant', 'bar', 'cafe']:
            score += 0.4
    
    elif purpose == 'events':
        # Event-goers like nightlife, dining
        if adv_type in ['restaurant', 'bar', 'nightlife']:
            score += 0.5
    
    # 2. Family compatibility
    is_family = guest_row.get('is_family', False)
    if is_family:
        if 'family_friendly' in ad_tags:
            score += 0.6
        if adv_type in ['attraction', 'museum', 'cafe']:
            score += 0.3
        if adv_type in ['nightlife', 'bar']:
            score -= 0.5
    else:
        # Singles/couples might prefer trendy, nightlife
        if 'trendy' in ad_tags or 'romantic' in ad_tags:
            score += 0.2
    
    # 3. Distance penalty (exponential decay)
    distance = ad_row['distance_km']
    distance_penalty = np.exp(-distance / 5.0) - 0.5
    score += distance_penalty
    
    # 4. Price compatibility
    price_per_night = guest_row.get('price_per_night', 100)
    price_level = ad_row['price_level']
    
    # Guests paying more might prefer higher-end ads
    if price_per_night < 80:  # Budget guest
        if price_level == 'low':
            score += 0.3
        elif price_level == 'high':
            score -= 0.3
        if 'budget' in ad_tags:
            score += 0.2
    elif price_per_night > 200:  # Luxury guest
        if price_level == 'high':
            score += 0.3
        elif price_level == 'low':
            score -= 0.2
        if 'luxury' in ad_tags:
            score += 0.3
    else:  # Mid-range
        if price_level == 'medium':
            score += 0.2
    
    # 5. Weekend effect
    is_weekend = guest_row.get('is_weekend_stay', False)
    if is_weekend:
        if adv_type in ['nightlife', 'bar', 'restaurant']:
            score += 0.2
    
    # 6. Base utility from ad
    score += ad_row['base_utility'] * 0.3
    
    return score


def generate_guest_ad_preferences(
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    n_samples_per_guest: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate guest-ad preference matrix.
    
    For each guest, sample candidate ads and compute intrinsic scores.
    
    Args:
        guests_df: Unified guest dataframe
        ads_df: Advertiser catalogue
        n_samples_per_guest: Number of candidate ads per guest
        seed: Random seed
        
    Returns:
        Dataframe with columns: guest_id, ad_id, intrinsic_score, base_click_prob
    """
    rng = set_random_seed(seed)
    
    preferences = []
    
    for idx, guest in guests_df.iterrows():
        guest_id = guest['guest_id']
        
        # Sample candidate ads (or use all if fewer than n_samples)
        if len(ads_df) <= n_samples_per_guest:
            candidate_ads = ads_df
        else:
            candidate_indices = rng.choice(len(ads_df), size=n_samples_per_guest, replace=False)
            candidate_ads = ads_df.iloc[candidate_indices]
        
        for _, ad in candidate_ads.iterrows():
            # Compute intrinsic score
            intrinsic_score = compute_guest_ad_intrinsic_score(guest, ad)
            
            # Convert to base click probability via sigmoid
            base_click_prob = sigmoid(intrinsic_score)
            
            preferences.append({
                'guest_id': guest_id,
                'ad_id': ad['ad_id'],
                'intrinsic_score': intrinsic_score,
                'base_click_prob': base_click_prob
            })
    
    return pd.DataFrame(preferences)





