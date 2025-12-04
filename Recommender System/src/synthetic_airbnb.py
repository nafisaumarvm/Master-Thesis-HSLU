"""
Synthetic Airbnb-style dataset generation from personal export schema.

PRIVACY NOTE: This module treats the user's personal Airbnb export purely as a 
schema/distribution template. All generated records are synthetic, anonymized, 
and non-identifiable.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

from .utils import (
    set_random_seed,
    compute_empirical_distribution,
    sample_from_distribution,
    jitter_dates,
    generate_fake_ids
)


def infer_airbnb_schema(airbnb_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Infer canonical schema from raw Airbnb export.
    
    Maps various column names to a standard schema and applies
    heuristics to infer missing fields.
    
    Args:
        airbnb_raw: Raw Airbnb export dataframe
        
    Returns:
        Dataframe with canonical schema:
        - guest_id, stay_id, city, arrival_date, nights, adults, children,
        - purpose_of_stay, price_per_night, total_price, country, device_type
    """
    df = airbnb_raw.copy()
    
    # Normalize column names for matching
    col_map = {col: col.lower().replace(' ', '_') for col in df.columns}
    df.rename(columns=col_map, inplace=True)
    
    result = pd.DataFrame()
    
    # Map guest_id
    guest_id_candidates = ['guest_id', 'user_id', 'id', 'confirmation_code']
    for candidate in guest_id_candidates:
        if candidate in df.columns:
            result['guest_id'] = df[candidate].astype(str)
            break
    if 'guest_id' not in result.columns:
        result['guest_id'] = [f"GUEST_{i:08d}" for i in range(len(df))]
    
    # Map stay_id/booking_id
    stay_id_candidates = ['booking_id', 'reservation_id', 'stay_id', 'confirmation_code']
    for candidate in stay_id_candidates:
        if candidate in df.columns:
            result['stay_id'] = df[candidate].astype(str)
            break
    if 'stay_id' not in result.columns:
        result['stay_id'] = [f"STAY_{i:08d}" for i in range(len(df))]
    
    # Map city/location
    city_candidates = ['city', 'listing_city', 'location', 'destination']
    for candidate in city_candidates:
        if candidate in df.columns:
            result['city'] = df[candidate].fillna('Unknown')
            break
    if 'city' not in result.columns:
        result['city'] = 'Unknown'
    
    # Map arrival_date
    date_candidates = ['start_date', 'check_in', 'arrival_date', 'stay_start_date']
    for candidate in date_candidates:
        if candidate in df.columns:
            result['arrival_date'] = pd.to_datetime(df[candidate], errors='coerce')
            break
    if 'arrival_date' not in result.columns:
        result['arrival_date'] = pd.to_datetime('2020-01-01')
    
    # Map nights
    nights_candidates = ['nights', 'nights_stayed', 'length_of_stay', 'duration']
    for candidate in nights_candidates:
        if candidate in df.columns:
            result['nights'] = pd.to_numeric(df[candidate], errors='coerce').fillna(1).astype(int)
            break
    
    # If not found, try to compute from start/end dates
    if 'nights' not in result.columns:
        end_date_candidates = ['end_date', 'check_out', 'stay_end_date', 'checkout_date']
        for candidate in end_date_candidates:
            if candidate in df.columns:
                end_dates = pd.to_datetime(df[candidate], errors='coerce')
                result['nights'] = (end_dates - result['arrival_date']).dt.days
                result['nights'] = result['nights'].fillna(1).clip(lower=1).astype(int)
                break
    
    if 'nights' not in result.columns:
        result['nights'] = 1
    
    # Map adults
    adults_candidates = ['adults', 'number_of_adults', 'num_adults', 'guest_count']
    for candidate in adults_candidates:
        if candidate in df.columns:
            result['adults'] = pd.to_numeric(df[candidate], errors='coerce').fillna(1).astype(int)
            break
    if 'adults' not in result.columns:
        result['adults'] = 1
    
    # Map children
    children_candidates = ['children', 'number_of_children', 'num_children', 'kids']
    for candidate in children_candidates:
        if candidate in df.columns:
            result['children'] = pd.to_numeric(df[candidate], errors='coerce').fillna(0).astype(int)
            break
    if 'children' not in result.columns:
        result['children'] = 0
    
    # Map infants (fold into children)
    infant_candidates = ['infants', 'babies', 'number_of_infants']
    for candidate in infant_candidates:
        if candidate in df.columns:
            infants = pd.to_numeric(df[candidate], errors='coerce').fillna(0).astype(int)
            result['children'] = result['children'] + infants
            break
    
    # Infer purpose_of_stay
    result['purpose_of_stay'] = _infer_purpose_of_stay(df)
    
    # Map prices
    price_candidates = ['total_price', 'price', 'total_cost', 'amount_paid']
    for candidate in price_candidates:
        if candidate in df.columns:
            result['total_price'] = pd.to_numeric(df[candidate], errors='coerce').fillna(0)
            break
    if 'total_price' not in result.columns:
        result['total_price'] = 0
    
    # Compute price_per_night
    result['price_per_night'] = result['total_price'] / result['nights']
    result['price_per_night'] = result['price_per_night'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Map country
    country_candidates = ['country', 'country_of_origin', 'guest_country', 'nationality']
    for candidate in country_candidates:
        if candidate in df.columns:
            result['country'] = df[candidate].fillna('Unknown')
            break
    if 'country' not in result.columns:
        result['country'] = 'Unknown'
    
    # Map device_type
    device_candidates = ['device_type', 'device', 'platform', 'user_agent']
    if any(c in df.columns for c in device_candidates):
        for candidate in device_candidates:
            if candidate in df.columns:
                result['device_type'] = _infer_device_type(df[candidate])
                break
    else:
        result['device_type'] = 'unknown'
    
    return result


def _infer_purpose_of_stay(df: pd.DataFrame) -> pd.Series:
    """
    Infer purpose of stay using keyword heuristics.
    
    Args:
        df: Raw dataframe (with lowercase column names)
        
    Returns:
        Series with purpose labels
    """
    purpose = pd.Series(['leisure'] * len(df), index=df.index)
    
    # Try to find trip reason column
    reason_candidates = ['reason_for_trip', 'trip_purpose', 'purpose', 'trip_reason']
    
    for candidate in reason_candidates:
        if candidate in df.columns:
            reasons = df[candidate].fillna('').astype(str).str.lower()
            
            # Business keywords
            purpose[reasons.str.contains('business|work|conference|meeting', regex=True)] = 'business'
            
            # Visiting friends/family
            purpose[reasons.str.contains('visit|family|friend', regex=True)] = 'visiting_friends'
            
            # Events
            purpose[reasons.str.contains('event|wedding|concert|festival', regex=True)] = 'events'
            
            return purpose
    
    # Try to infer from review text or notes
    text_candidates = ['review_text', 'notes', 'comments', 'description']
    for candidate in text_candidates:
        if candidate in df.columns:
            text = df[candidate].fillna('').astype(str).str.lower()
            
            purpose[text.str.contains('business|work|conference', regex=True)] = 'business'
            purpose[text.str.contains('family|friend|visit', regex=True)] = 'visiting_friends'
            purpose[text.str.contains('wedding|concert|event', regex=True)] = 'events'
            
            break
    
    return purpose


def _infer_device_type(device_series: pd.Series) -> pd.Series:
    """
    Infer device type from user agent or device field.
    
    Args:
        device_series: Series containing device/user agent info
        
    Returns:
        Series with device types: desktop, mobile, tablet, unknown
    """
    device = device_series.fillna('').astype(str).str.lower()
    
    result = pd.Series(['unknown'] * len(device), index=device.index)
    
    result[device.str.contains('mobile|iphone|android', regex=True)] = 'mobile'
    result[device.str.contains('tablet|ipad', regex=True)] = 'tablet'
    result[device.str.contains('desktop|windows|mac|linux', regex=True)] = 'desktop'
    
    return result


def generate_synthetic_airbnb(
    airbnb_raw: pd.DataFrame,
    n_samples: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic Airbnb stays from empirical distributions.
    
    PRIVACY: This function samples from distributions only; no real
    identifiers or exact data are copied.
    
    Args:
        airbnb_raw: Canonical Airbnb schema (output of infer_airbnb_schema)
        n_samples: Number of synthetic stays to generate
        seed: Random seed
        
    Returns:
        Synthetic Airbnb dataset with same schema plus 'source' column
    """
    rng = set_random_seed(seed)
    
    # Compute empirical distributions
    nights_dist = compute_empirical_distribution(airbnb_raw['nights'], normalize=True)
    adults_dist = compute_empirical_distribution(airbnb_raw['adults'], normalize=True)
    children_dist = compute_empirical_distribution(airbnb_raw['children'], normalize=True)
    city_dist = compute_empirical_distribution(airbnb_raw['city'], normalize=True)
    country_dist = compute_empirical_distribution(airbnb_raw['country'], normalize=True)
    purpose_dist = compute_empirical_distribution(airbnb_raw['purpose_of_stay'], normalize=True)
    device_dist = compute_empirical_distribution(airbnb_raw['device_type'], normalize=True)
    
    # Compute price statistics by nights bracket
    price_stats = airbnb_raw.groupby('nights')['price_per_night'].agg(['mean', 'std']).to_dict()
    
    # Generate synthetic records
    synthetic_data = []
    
    for i in range(n_samples):
        # Sample nights
        nights = sample_from_distribution(
            list(nights_dist.keys()),
            list(nights_dist.values()),
            size=1,
            rng=rng
        )
        
        # Sample adults and children
        adults = sample_from_distribution(
            list(adults_dist.keys()),
            list(adults_dist.values()),
            size=1,
            rng=rng
        )
        
        children = sample_from_distribution(
            list(children_dist.keys()),
            list(children_dist.values()),
            size=1,
            rng=rng
        )
        
        # Sample city and country
        city = sample_from_distribution(
            list(city_dist.keys()),
            list(city_dist.values()),
            size=1,
            rng=rng
        )
        
        country = sample_from_distribution(
            list(country_dist.keys()),
            list(country_dist.values()),
            size=1,
            rng=rng
        )
        
        # Sample purpose
        purpose = sample_from_distribution(
            list(purpose_dist.keys()),
            list(purpose_dist.values()),
            size=1,
            rng=rng
        )
        
        # Sample device type
        device = sample_from_distribution(
            list(device_dist.keys()),
            list(device_dist.values()),
            size=1,
            rng=rng
        )
        
        # Sample price (with noise)
        if nights in price_stats['mean']:
            mean_price = price_stats['mean'][nights]
            std_price = price_stats['std'][nights] if nights in price_stats['std'] else mean_price * 0.3
        else:
            mean_price = airbnb_raw['price_per_night'].median()
            std_price = mean_price * 0.3
        
        price_per_night = max(10, rng.normal(mean_price, std_price))
        total_price = price_per_night * nights
        
        # Generate synthetic date (random within past 2 years)
        days_ago = int(rng.integers(0, 730))
        arrival_date = datetime.now() - timedelta(days=days_ago)
        
        # Create synthetic IDs
        guest_id = f"SYNTH_GUEST_{i:08d}"
        stay_id = f"SYNTH_STAY_{i:08d}"
        
        synthetic_data.append({
            'guest_id': guest_id,
            'stay_id': stay_id,
            'city': city,
            'arrival_date': arrival_date,
            'nights': nights,
            'adults': adults,
            'children': children,
            'purpose_of_stay': purpose,
            'price_per_night': price_per_night,
            'total_price': total_price,
            'country': country,
            'device_type': device,
            'source': 'synthetic_airbnb'
        })
    
    return pd.DataFrame(synthetic_data)


def harmonize_guests(
    hotel_df: pd.DataFrame,
    airbnb_synth_df: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Map hotel and synthetic Airbnb guests to unified schema.
    
    Args:
        hotel_df: Hotel guests dataframe
        airbnb_synth_df: Synthetic Airbnb dataframe
        seed: Random seed
        
    Returns:
        Unified guest_sessions dataframe
    """
    rng = set_random_seed(seed)
    
    # Process hotel guests
    hotel_unified = pd.DataFrame()
    hotel_unified['guest_id'] = hotel_df['guest_id']
    hotel_unified['stay_id'] = hotel_df['stay_id']
    hotel_unified['source'] = 'hotel'
    hotel_unified['arrival_date'] = hotel_df['arrival_date']
    hotel_unified['nights'] = hotel_df['stay_nights']
    hotel_unified['total_guests'] = hotel_df['total_guests']
    hotel_unified['adults'] = hotel_df['adults']
    hotel_unified['children'] = hotel_df['children'] + hotel_df.get('babies', 0)
    hotel_unified['country'] = hotel_df['country']
    hotel_unified['city_or_hotel_location'] = hotel_df.get('hotel', 'Unknown Hotel')
    hotel_unified['purpose_of_stay'] = hotel_df['stay_purpose']
    hotel_unified['price_per_night'] = hotel_df['price_per_night']
    
    # Booking channel
    if 'distribution_channel' in hotel_df.columns:
        hotel_unified['booking_channel'] = hotel_df['distribution_channel']
    else:
        hotel_unified['booking_channel'] = 'OTA'
    
    hotel_unified['is_repeated_guest'] = hotel_df.get('is_repeated_guest', False)
    
    # Segment features
    hotel_unified['is_family'] = hotel_df.get('is_family', False)
    hotel_unified['is_business'] = hotel_df['stay_purpose'] == 'business'
    hotel_unified['is_weekend_stay'] = hotel_df.get('is_weekend_stay', False)
    
    # Process Airbnb guests
    airbnb_unified = pd.DataFrame()
    airbnb_unified['guest_id'] = airbnb_synth_df['guest_id']
    airbnb_unified['stay_id'] = airbnb_synth_df['stay_id']
    airbnb_unified['source'] = 'synthetic_airbnb'
    airbnb_unified['arrival_date'] = airbnb_synth_df['arrival_date']
    airbnb_unified['nights'] = airbnb_synth_df['nights']
    airbnb_unified['adults'] = airbnb_synth_df['adults']
    airbnb_unified['children'] = airbnb_synth_df['children']
    airbnb_unified['total_guests'] = airbnb_unified['adults'] + airbnb_unified['children']
    airbnb_unified['country'] = airbnb_synth_df['country']
    airbnb_unified['city_or_hotel_location'] = airbnb_synth_df['city']
    airbnb_unified['purpose_of_stay'] = airbnb_synth_df['purpose_of_stay']
    airbnb_unified['price_per_night'] = airbnb_synth_df['price_per_night']
    airbnb_unified['booking_channel'] = 'Airbnb'
    
    # Synthetic repeated guest flag
    airbnb_unified['is_repeated_guest'] = rng.choice([False, False, False, True], size=len(airbnb_unified))
    
    # Segment features
    airbnb_unified['is_family'] = airbnb_unified['children'] > 0
    airbnb_unified['is_business'] = airbnb_unified['purpose_of_stay'] == 'business'
    airbnb_unified['is_weekend_stay'] = airbnb_unified['arrival_date'].dt.dayofweek >= 5
    
    # Combine
    unified = pd.concat([hotel_unified, airbnb_unified], ignore_index=True)
    
    return unified

