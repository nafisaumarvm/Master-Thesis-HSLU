# Advanced guest segmentation for better targeting.

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import set_random_seed


def create_guest_segments(
    guests_df: pd.DataFrame,
    method: str = 'rules',
    n_clusters: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    # Create meaningful guest segments

    df = guests_df.copy()
    
    if method == 'rules':
        df['segment'] = df.apply(_assign_rule_based_segment, axis=1)
    elif method == 'clustering':
        df = _assign_cluster_based_segment(df, n_clusters, seed)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Add segment descriptions
    df['segment_description'] = df['segment'].map(SEGMENT_DESCRIPTIONS)
    
    return df


def _assign_rule_based_segment(row: pd.Series) -> str:
    # Assign segment based on business rules

    price = row.get('price_per_night', 100)
    nights = row.get('nights', 2)
    is_family = row.get('is_family', False)
    purpose = row.get('purpose_of_stay', 'leisure')
    is_weekend = row.get('is_weekend_stay', False)
    
    # Business traveler
    if purpose == 'business' and nights <= 3:
        return 'business_traveler'
    
    # Extended stay
    if nights >= 7:
        if price > 150:
            return 'luxury_leisure'
        else:
            return 'extended_stay'
    
    # Luxury leisure
    if price > 200 and purpose == 'leisure':
        return 'luxury_leisure'
    
    # Budget family
    if is_family and price < 120:
        return 'budget_family'
    
    # Weekend explorer
    if is_weekend and nights <= 2 and purpose == 'leisure':
        return 'weekend_explorer'
    
    # Bargain hunter
    if price < 70:
        return 'bargain_hunter'
    
    # Cultural tourist (default for mid-range leisure)
    if purpose in ['leisure', 'visiting_friends'] and 100 <= price <= 180:
        return 'cultural_tourist'
    
    # Default: cultural tourist
    return 'cultural_tourist'


def _assign_cluster_based_segment(
    df: pd.DataFrame,
    n_clusters: int,
    seed: int
) -> pd.DataFrame:
    # Assign segments using K-means clustering
    # Select features for clustering
    features = []
    feature_names = []
    
    # Numeric features
    for col in ['nights', 'total_guests', 'price_per_night']:
        if col in df.columns:
            features.append(df[col].fillna(df[col].median()).values)
            feature_names.append(col)
    
    # Binary features
    for col in ['is_family', 'is_business', 'is_weekend_stay']:
        if col in df.columns:
            features.append(df[col].astype(int).values)
            feature_names.append(col)
    
    # Create feature matrix
    X = np.column_stack(features)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Assign cluster labels
    df['segment'] = [f"cluster_{i}" for i in clusters]
    
    return df


# Segment descriptions
SEGMENT_DESCRIPTIONS = {
    'luxury_leisure': 'High-end leisure travelers, premium experiences',
    'budget_family': 'Cost-conscious families, family-friendly activities',
    'business_traveler': 'Business travelers, efficiency-focused',
    'weekend_explorer': 'Weekend getaway seekers, local experiences',
    'extended_stay': 'Long-stay guests, local immersion',
    'bargain_hunter': 'Budget-conscious travelers, deals-focused',
    'cultural_tourist': 'Culture and dining enthusiasts',
    'adventure_seeker': 'Active outdoor adventurers'
}


# Segment-specific ad preferences
SEGMENT_AD_PREFERENCES = {
    'luxury_leisure': {
        'preferred_types': ['spa', 'restaurant', 'tour', 'attraction'],
        'preferred_tags': ['luxury', 'romantic', 'wellness', 'foodie'],
        'price_preference': 'high',
        'distance_tolerance': 15.0,  # km
        'boost_factor': 1.3
    },
    'budget_family': {
        'preferred_types': ['attraction', 'museum', 'cafe', 'tour'],
        'preferred_tags': ['family_friendly', 'budget', 'outdoor'],
        'price_preference': 'low',
        'distance_tolerance': 8.0,
        'boost_factor': 1.2
    },
    'business_traveler': {
        'preferred_types': ['cafe', 'restaurant', 'transport'],
        'preferred_tags': ['local', 'quiet', 'foodie'],
        'price_preference': 'medium',
        'distance_tolerance': 5.0,
        'boost_factor': 1.1
    },
    'weekend_explorer': {
        'preferred_types': ['restaurant', 'bar', 'nightlife', 'attraction'],
        'preferred_tags': ['trendy', 'local', 'nightlife'],
        'price_preference': 'medium',
        'distance_tolerance': 10.0,
        'boost_factor': 1.2
    },
    'extended_stay': {
        'preferred_types': ['cafe', 'restaurant', 'museum', 'gallery'],
        'preferred_tags': ['local', 'culture', 'quiet'],
        'price_preference': 'medium',
        'distance_tolerance': 12.0,
        'boost_factor': 1.15
    },
    'bargain_hunter': {
        'preferred_types': ['cafe', 'museum', 'attraction'],
        'preferred_tags': ['budget', 'local'],
        'price_preference': 'low',
        'distance_tolerance': 6.0,
        'boost_factor': 1.0
    },
    'cultural_tourist': {
        'preferred_types': ['museum', 'gallery', 'restaurant', 'tour'],
        'preferred_tags': ['culture', 'foodie', 'touristy'],
        'price_preference': 'medium',
        'distance_tolerance': 10.0,
        'boost_factor': 1.2
    },
    'adventure_seeker': {
        'preferred_types': ['tour', 'attraction', 'transport'],
        'preferred_tags': ['adventure', 'outdoor', 'active'],
        'price_preference': 'medium',
        'distance_tolerance': 20.0,
        'boost_factor': 1.25
    }
}


def apply_segment_boost(
    ad_scores: Dict[str, float],
    guest_segment: str,
    ads_df: pd.DataFrame
) -> Dict[str, float]:
    # Apply segment-specific boost to ad scores
    from .utils import parse_tags_string
    
    if guest_segment not in SEGMENT_AD_PREFERENCES:
        return ad_scores
    
    prefs = SEGMENT_AD_PREFERENCES[guest_segment]
    boosted_scores = ad_scores.copy()
    
    for ad_id, score in ad_scores.items():
        ad_info = ads_df[ads_df['ad_id'] == ad_id]
        
        if len(ad_info) == 0:
            continue
        
        ad_info = ad_info.iloc[0]
        boost = 1.0
        
        # Type match
        if ad_info['advertiser_type'] in prefs['preferred_types']:
            boost *= prefs['boost_factor']
        
        # Tag match
        ad_tags = parse_tags_string(ad_info['category_tags'])
        matching_tags = sum(1 for tag in ad_tags if tag in prefs['preferred_tags'])
        if matching_tags > 0:
            boost *= (1.0 + 0.1 * matching_tags)
        
        # Price match
        ad_price = ad_info['price_level']
        if (prefs['price_preference'] == 'low' and ad_price == 'low') or \
           (prefs['price_preference'] == 'high' and ad_price == 'high') or \
           (prefs['price_preference'] == 'medium' and ad_price == 'medium'):
            boost *= 1.15
        
        boosted_scores[ad_id] = score * boost
    
    return boosted_scores


def get_segment_summary(guests_df: pd.DataFrame) -> pd.DataFrame:
    # Get summary statistics per segment
    if 'segment' not in guests_df.columns:
        guests_df = create_guest_segments(guests_df)
    
    # Build aggregation dict dynamically based on available columns
    agg_dict = {'guest_id': 'count'}
    
    # Try different column names for nights
    if 'nights' in guests_df.columns:
        agg_dict['nights'] = 'mean'
    elif 'stay_nights' in guests_df.columns:
        agg_dict['stay_nights'] = 'mean'
    
    if 'price_per_night' in guests_df.columns:
        agg_dict['price_per_night'] = 'mean'
    
    if 'total_guests' in guests_df.columns:
        agg_dict['total_guests'] = 'mean'
    
    if 'is_family' in guests_df.columns:
        agg_dict['is_family'] = 'mean'
    
    if 'is_business' in guests_df.columns:
        agg_dict['is_business'] = 'mean'
    
    summary = guests_df.groupby('segment').agg(agg_dict).round(2)
    
    # Build column names based on what we aggregated
    new_cols = ['n_guests']
    if 'nights' in agg_dict or 'stay_nights' in agg_dict:
        new_cols.append('avg_nights')
    if 'price_per_night' in agg_dict:
        new_cols.append('avg_price')
    if 'total_guests' in agg_dict:
        new_cols.append('avg_party_size')
    if 'is_family' in agg_dict:
        new_cols.append('pct_family')
    if 'is_business' in agg_dict:
        new_cols.append('pct_business')
    
    summary.columns = new_cols
    
    summary = summary.reset_index()
    
    # Add descriptions
    summary['description'] = summary['segment'].map(SEGMENT_DESCRIPTIONS)
    
    return summary

