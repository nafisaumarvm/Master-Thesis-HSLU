"""
Real-world dataset integration for hotel bookings and ad click logs.

This module loads and processes:
1. Hotel Booking 2 dataset (for hotel attributes)
2. Criteo Click Logs (for advertiser/ad dataset)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import warnings

from .utils import set_random_seed


def load_hotel_booking_dataset(
    path: Optional[str] = None,
    sample_frac: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load Hotel Booking Demand dataset.
    
    This is a real-world hotel booking dataset with rich attributes.
    
    Args:
        path: Path to hotel_bookings.csv (if None, will look in data/raw/)
        sample_frac: Fraction of data to sample (1.0 = all data)
        seed: Random seed
        
    Returns:
        Processed hotel bookings dataframe
    """
    if path is None:
        # Try common locations
        possible_paths = [
            '/Users/nafisaumar/Documents/Master Thesis/Recommender System NEW/hotel_booking 2.csv',
            '/Users/nafisaumar/Documents/Master Thesis/Recommender System NEW/hotel_guests_dataset.csv',
            'hotel_booking 2.csv',
            'hotel_guests_dataset.csv',
            'data/raw/hotel_bookings.csv',
            'data/raw/hotel_booking.csv',
            '../data/raw/hotel_bookings.csv',
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                path = p
                break
        
        if path is None:
            raise FileNotFoundError(
                "Hotel booking dataset not found. Please download from:\n"
                "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand\n"
                "and place in data/raw/hotel_bookings.csv"
            )
    
    print(f"Loading hotel booking dataset from: {path}")
    df = pd.read_csv(path)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    
    print(f"Loaded {len(df)} hotel booking records")
    
    # Process and clean
    df = _process_hotel_booking_dataset(df, seed)
    
    return df


def _process_hotel_booking_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Process the Hotel Booking Demand dataset.
    
    Args:
        df: Raw hotel booking dataframe
        seed: Random seed
        
    Returns:
        Processed dataframe with standardized columns
    """
    processed = pd.DataFrame()
    
    # Generate unique IDs
    processed['guest_id'] = [f"REAL_GUEST_{i:08d}" for i in range(len(df))]
    processed['stay_id'] = [f"REAL_STAY_{i:08d}" for i in range(len(df))]
    processed['source'] = 'hotel_booking_real'
    
    # Arrival date
    if 'arrival_date_year' in df.columns:
        # Construct full date
        year = df['arrival_date_year'].astype(str)
        month_map = {
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'May': '05', 'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
        month = df['arrival_date_month'].map(month_map)
        day = df['arrival_date_day_of_month'].astype(str).str.zfill(2)
        
        processed['arrival_date'] = pd.to_datetime(
            year + '-' + month + '-' + day,
            errors='coerce'
        )
    
    # Stay nights
    processed['stay_nights'] = (
        df.get('stays_in_weekend_nights', 0) + 
        df.get('stays_in_week_nights', 0)
    )
    
    # Guests
    processed['adults'] = df.get('adults', 2).fillna(2).astype(int)
    processed['children'] = df.get('children', 0).fillna(0).astype(int)
    processed['babies'] = df.get('babies', 0).fillna(0).astype(int)
    processed['total_guests'] = processed['adults'] + processed['children'] + processed['babies']
    processed['total_guests'] = processed['total_guests'].clip(lower=1)
    
    # Location info
    processed['country'] = df.get('country', 'Unknown').fillna('Unknown')
    processed['hotel'] = df.get('hotel', 'Unknown Hotel').fillna('Unknown Hotel')
    processed['city_or_hotel_location'] = processed['hotel']
    
    # Booking details
    processed['market_segment'] = df.get('market_segment', 'Online TA').fillna('Online TA')
    processed['distribution_channel'] = df.get('distribution_channel', 'TA/TO').fillna('TA/TO')
    processed['customer_type'] = df.get('customer_type', 'Transient').fillna('Transient')
    processed['booking_channel'] = processed['distribution_channel']
    
    # Pricing
    processed['adr'] = df.get('adr', 100).fillna(100).clip(lower=0)
    processed['price_per_night'] = processed['adr']
    
    # Lead time
    processed['lead_time'] = df.get('lead_time', 0).fillna(0)
    
    # Repeated guest
    processed['is_repeated_guest'] = df.get('is_repeated_guest', 0).fillna(0).astype(bool)
    
    # Meal type
    processed['meal'] = df.get('meal', 'BB').fillna('BB')
    
    # Room info
    processed['reserved_room_type'] = df.get('reserved_room_type', 'A').fillna('A')
    processed['assigned_room_type'] = df.get('assigned_room_type', 'A').fillna('A')
    
    # Derived features
    processed['stay_type'] = pd.cut(
        processed['stay_nights'],
        bins=[0, 2, 6, 999],
        labels=['short', 'medium', 'long']
    )
    
    # Infer purpose
    processed['stay_purpose'] = 'leisure'
    business_mask = (
        (processed['market_segment'] == 'Corporate') |
        (processed['customer_type'].isin(['Contract', 'Group'])) |
        ((processed['total_guests'] == 1) & (processed['stay_nights'] <= 3))
    )
    processed.loc[business_mask, 'stay_purpose'] = 'business'
    
    # Additional flags
    processed['is_family'] = (processed['children'] > 0) | (processed['babies'] > 0)
    processed['is_business'] = processed['stay_purpose'] == 'business'
    
    # Weekend stay (from arrival date)
    processed['is_weekend_stay'] = processed['arrival_date'].dt.dayofweek >= 5
    
    # Nights attribute for compatibility
    processed['nights'] = processed['stay_nights']
    
    # Purpose of stay (standardized)
    processed['purpose_of_stay'] = processed['stay_purpose']
    
    return processed


def download_criteo_dataset(
    output_dir: str = 'data/raw/criteo',
    sample_size: int = 100000
) -> str:
    """
    Download and prepare Criteo Click Logs dataset.
    
    Note: This dataset is VERY large (>40GB). This function will:
    1. Download a sample
    2. Process it for ad recommendation
    
    Args:
        output_dir: Directory to save the dataset
        sample_size: Number of records to sample
        
    Returns:
        Path to processed file
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install datasets library: pip install datasets\n"
            "This is needed to download Criteo dataset from Hugging Face"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading Criteo Click Logs dataset (this may take a while)...")
    print(f"Sampling {sample_size} records...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(
        "criteo/CriteoClickLogs",
        split=f"train[:{sample_size}]",
        trust_remote_code=True
    )
    
    # Convert to pandas
    df = dataset.to_pandas()
    
    output_path = os.path.join(output_dir, 'criteo_sample.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved Criteo sample to: {output_path}")
    
    return output_path


def process_criteo_for_advertisers(
    criteo_path: str,
    n_advertisers: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Process Criteo Click Logs to create advertiser catalogue.
    
    The Criteo dataset has click/conversion data that we can use to
    create realistic advertiser profiles with:
    - Click-through rates
    - Conversion rates
    - Category information
    
    Args:
        criteo_path: Path to Criteo CSV
        n_advertisers: Number of advertisers to generate
        seed: Random seed
        
    Returns:
        Advertiser catalogue dataframe
    """
    print(f"Processing Criteo data for advertiser generation...")
    
    df = pd.read_csv(criteo_path)
    rng = set_random_seed(seed)
    
    # Criteo has features like:
    # - Label (0/1 for click)
    # - I1-I13 (integer features)
    # - C1-C26 (categorical features)
    
    # Group by categorical features to create advertiser profiles
    # Use the most common categories as advertiser types
    
    advertisers = []
    
    for i in range(n_advertisers):
        ad_id = f"CRITEO_AD_{i:04d}"
        
        # Sample features from the Criteo data
        if len(df) > 0:
            sample_idx = rng.integers(0, len(df))
            sample = df.iloc[sample_idx]
            
            # Estimate CTR from similar ads in dataset
            # (In production, you'd do proper clustering)
            base_ctr = df['Label'].mean() if 'Label' in df.columns else 0.05
            
            # Add noise
            base_ctr = max(0.001, base_ctr + rng.normal(0, 0.02))
        else:
            base_ctr = 0.05
        
        # Map to our advertiser schema
        advertiser_types = [
            'restaurant', 'bar', 'cafe', 'museum', 'gallery',
            'spa', 'tour', 'attraction', 'nightlife', 'transport'
        ]
        
        adv_type = rng.choice(advertiser_types)
        
        # Generate name
        advertiser_name = f"Criteo Advertiser {i:04d}"
        
        # Category tags (inferred from type)
        category_tags = _get_tags_for_type(adv_type, rng)
        
        # Distance (realistic)
        distance_km = rng.lognormal(1.0, 0.8)  # Log-normal for realistic distances
        distance_km = np.clip(distance_km, 0.1, 20.0)
        
        # Opening dayparts
        n_dayparts = rng.integers(2, 5)
        dayparts = rng.choice(
            ['morning', 'afternoon', 'evening', 'late_night'],
            size=n_dayparts,
            replace=False
        ).tolist()
        
        # Price level
        price_level = rng.choice(['low', 'medium', 'high'])
        
        # Base utility (based on CTR)
        base_utility = np.log(base_ctr / 0.05)  # Log-odds style
        
        # Revenue
        revenue_per_conversion = _compute_revenue_for_type(adv_type, price_level, rng)
        
        from .utils import tags_to_string
        
        advertisers.append({
            'ad_id': ad_id,
            'advertiser_name': advertiser_name,
            'advertiser_type': adv_type,
            'category_tags': tags_to_string(category_tags),
            'distance_km': distance_km,
            'opening_dayparts': tags_to_string(dayparts),
            'price_level': price_level,
            'base_utility': base_utility,
            'revenue_per_conversion': revenue_per_conversion,
            'estimated_ctr': base_ctr,
            'source': 'criteo'
        })
    
    ads_df = pd.DataFrame(advertisers)
    print(f"✓ Generated {len(ads_df)} advertisers from Criteo data")
    
    return ads_df


def _get_tags_for_type(adv_type: str, rng: np.random.Generator) -> list:
    """Get relevant tags for advertiser type."""
    
    type_tag_map = {
        'restaurant': ['foodie', 'local', 'touristy'],
        'bar': ['nightlife', 'trendy', 'local'],
        'cafe': ['local', 'quiet', 'budget'],
        'museum': ['culture', 'family_friendly', 'touristy'],
        'gallery': ['culture', 'quiet'],
        'spa': ['wellness', 'luxury', 'quiet'],
        'tour': ['adventure', 'touristy', 'outdoor'],
        'attraction': ['family_friendly', 'touristy', 'outdoor'],
        'nightlife': ['nightlife', 'trendy'],
        'transport': ['local']
    }
    
    base_tags = type_tag_map.get(adv_type, ['local'])
    
    # Add 1-2 random tags
    all_tags = [
        'foodie', 'family_friendly', 'nightlife', 'culture', 'outdoor',
        'budget', 'luxury', 'romantic', 'adventure', 'wellness',
        'shopping', 'local', 'touristy', 'quiet', 'trendy'
    ]
    
    additional = rng.choice(
        [t for t in all_tags if t not in base_tags],
        size=min(2, len([t for t in all_tags if t not in base_tags])),
        replace=False
    ).tolist()
    
    return base_tags + additional


def _compute_revenue_for_type(
    adv_type: str,
    price_level: str,
    rng: np.random.Generator
) -> float:
    """Compute revenue per conversion."""
    
    base_revenue = {
        'restaurant': 50, 'bar': 30, 'cafe': 15, 'museum': 25,
        'gallery': 20, 'spa': 80, 'tour': 60, 'attraction': 40,
        'nightlife': 35, 'transport': 25
    }
    
    price_multiplier = {'low': 0.6, 'medium': 1.0, 'high': 1.8}
    
    base = base_revenue.get(adv_type, 30)
    multiplier = price_multiplier.get(price_level, 1.0)
    
    revenue = base * multiplier * rng.uniform(0.8, 1.2)
    
    return round(revenue, 2)


def create_integrated_dataset(
    use_hotel_booking: bool = True,
    use_criteo: bool = False,
    hotel_path: Optional[str] = None,
    criteo_path: Optional[str] = None,
    sample_hotel: float = 0.1,
    n_advertisers: int = 300,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create integrated dataset from real-world sources.
    
    Args:
        use_hotel_booking: Use Hotel Booking Demand dataset
        use_criteo: Use Criteo dataset for advertisers
        hotel_path: Path to hotel booking CSV
        criteo_path: Path to Criteo CSV
        sample_hotel: Fraction of hotel data to use
        n_advertisers: Number of advertisers to generate
        seed: Random seed
        
    Returns:
        (guests_df, ads_df) tuple
    """
    print("="*70)
    print("CREATING INTEGRATED DATASET FROM REAL-WORLD SOURCES")
    print("="*70)
    
    # Load hotel data
    if use_hotel_booking:
        try:
            guests_df = load_hotel_booking_dataset(
                path=hotel_path,
                sample_frac=sample_hotel,
                seed=seed
            )
            print(f"✓ Loaded {len(guests_df)} real hotel bookings")
        except FileNotFoundError as e:
            print(f"⚠ {e}")
            print("⚠ Falling back to synthetic hotel data")
            from .data_loading import load_hotel_guests
            guests_df = load_hotel_guests('data/raw/demo_hotel.csv', seed=seed)
    else:
        from .data_loading import load_hotel_guests
        guests_df = load_hotel_guests('data/raw/demo_hotel.csv', seed=seed)
    
    # Load advertiser data
    if use_criteo:
        if criteo_path is None or not os.path.exists(criteo_path):
            print("\n⚠ Criteo dataset not found. Downloading sample...")
            try:
                criteo_path = download_criteo_dataset(sample_size=50000)
            except Exception as e:
                print(f"⚠ Failed to download Criteo: {e}")
                print("⚠ Falling back to synthetic advertisers")
                use_criteo = False
        
        if use_criteo:
            ads_df = process_criteo_for_advertisers(
                criteo_path,
                n_advertisers=n_advertisers,
                seed=seed
            )
    
    if not use_criteo:
        from .advertisers import generate_advertisers
        ads_df = generate_advertisers(n_ads=n_advertisers, seed=seed)
        print(f"✓ Generated {len(ads_df)} synthetic advertisers")
    
    print("\n" + "="*70)
    print("DATASET INTEGRATION COMPLETE")
    print("="*70)
    print(f"Guests: {len(guests_df)} records")
    print(f"Advertisers: {len(ads_df)} records")
    
    return guests_df, ads_df

