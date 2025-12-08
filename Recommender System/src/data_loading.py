# Data loading and preprocessing for hotel guests dataset

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings


def load_hotel_guests(
    path: str,
    drop_missing: bool = True,
    seed: int = 42
) -> pd.DataFrame:
    # Load and clean the Kaggle Hotel Guests Dataset

    # Load raw data
    df = pd.read_csv(path)
    
    # Store original column names for flexible mapping
    original_cols = df.columns.tolist()
    
    # Common column name variations (normalize to lowercase for matching)
    col_mapping = {}
    
    # Flexible column detection
    for col in original_cols:
        col_lower = col.lower()
        
        # Arrival date variations
        if 'arrival' in col_lower and 'date' in col_lower:
            col_mapping['arrival_date'] = col
        # Length of stay
        elif 'length' in col_lower or 'nights' in col_lower or 'stays_in' in col_lower:
            if 'night' in col_lower or 'length' in col_lower:
                col_mapping['stay_nights'] = col
        # Adults
        elif col_lower in ['adults', 'adult']:
            col_mapping['adults'] = col
        # Children
        elif col_lower in ['children', 'child']:
            col_mapping['children'] = col
        # Babies
        elif col_lower in ['babies', 'baby', 'infants']:
            col_mapping['babies'] = col
        # Market segment
        elif 'market' in col_lower and 'segment' in col_lower:
            col_mapping['market_segment'] = col
        # Customer type
        elif 'customer' in col_lower and 'type' in col_lower:
            col_mapping['customer_type'] = col
        # Hotel
        elif col_lower in ['hotel', 'hotel_name']:
            col_mapping['hotel'] = col
        # Booking changes
        elif 'booking' in col_lower and 'changes' in col_lower:
            col_mapping['booking_changes'] = col
        # ADR (Average Daily Rate)
        elif col_lower == 'adr':
            col_mapping['adr'] = col
        # Country
        elif col_lower in ['country', 'country_of_origin']:
            col_mapping['country'] = col
        # Reserved room type
        elif 'reserved' in col_lower and 'room' in col_lower:
            col_mapping['reserved_room_type'] = col
        # Assigned room type
        elif 'assigned' in col_lower and 'room' in col_lower:
            col_mapping['assigned_room_type'] = col
        # Lead time
        elif 'lead' in col_lower and 'time' in col_lower:
            col_mapping['lead_time'] = col
        # Is repeated guest
        elif 'repeat' in col_lower or 'is_repeated' in col_lower:
            col_mapping['is_repeated_guest'] = col
        # Meal
        elif col_lower in ['meal', 'meal_type']:
            col_mapping['meal'] = col
        # Distribution channel
        elif 'distribution' in col_lower and 'channel' in col_lower:
            col_mapping['distribution_channel'] = col
    
    # Parse arrival date if present
    if 'arrival_date' in col_mapping:
        arrival_col = col_mapping['arrival_date']
        # Try to parse as date
        try:
            df['arrival_date'] = pd.to_datetime(df[arrival_col], errors='coerce')
        except:
            # If single column doesn't work, try combining year/month/day columns
            year_col = [c for c in original_cols if 'year' in c.lower()]
            month_col = [c for c in original_cols if 'month' in c.lower()]
            day_col = [c for c in original_cols if 'day' in c.lower() or 'week' in c.lower()]
            
            if year_col and month_col and day_col:
                # Construct date from components
                df['arrival_date'] = pd.to_datetime(
                    df[year_col[0]].astype(str) + '-' + 
                    df[month_col[0]].astype(str) + '-' + 
                    df[day_col[0]].astype(str),
                    errors='coerce'
                )
            else:
                # Create synthetic dates
                df['arrival_date'] = pd.date_range('2019-01-01', periods=len(df), freq='H')
    else:
        # Create synthetic arrival dates
        df['arrival_date'] = pd.date_range('2019-01-01', periods=len(df), freq='H')
    
    # Extract numeric columns with validation
    rng = np.random.default_rng(seed)
    
    # Stay nights
    if 'stay_nights' in col_mapping:
        stay_col = col_mapping['stay_nights']
        df['stay_nights'] = pd.to_numeric(df[stay_col], errors='coerce')
    else:
        # Check for separate weekend/week nights columns
        weekend_col = [c for c in original_cols if 'weekend' in c.lower() and 'night' in c.lower()]
        week_col = [c for c in original_cols if 'week' in c.lower() and 'night' in c.lower() and 'weekend' not in c.lower()]
        
        if weekend_col and week_col:
            df['stay_nights'] = (
                pd.to_numeric(df[weekend_col[0]], errors='coerce').fillna(0) + 
                pd.to_numeric(df[week_col[0]], errors='coerce').fillna(0)
            )
        else:
            # Sample from reasonable distribution
            df['stay_nights'] = rng.integers(1, 15, size=len(df))
    
    # Adults
    if 'adults' in col_mapping:
        df['adults'] = pd.to_numeric(df[col_mapping['adults']], errors='coerce').fillna(1).astype(int)
    else:
        df['adults'] = rng.integers(1, 4, size=len(df))
    
    # Children
    if 'children' in col_mapping:
        df['children'] = pd.to_numeric(df[col_mapping['children']], errors='coerce').fillna(0).astype(int)
    else:
        df['children'] = rng.choice([0, 0, 0, 1, 2], size=len(df))
    
    # Babies
    if 'babies' in col_mapping:
        df['babies'] = pd.to_numeric(df[col_mapping['babies']], errors='coerce').fillna(0).astype(int)
    else:
        df['babies'] = rng.choice([0, 0, 0, 0, 1], size=len(df))
    
    # Total guests
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['total_guests'] = df['total_guests'].clip(lower=1)
    
    # Market segment and customer type
    if 'market_segment' in col_mapping:
        df['market_segment'] = df[col_mapping['market_segment']].fillna('Unknown')
    else:
        df['market_segment'] = rng.choice(
            ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups'],
            size=len(df)
        )
    
    if 'customer_type' in col_mapping:
        df['customer_type'] = df[col_mapping['customer_type']].fillna('Transient')
    else:
        df['customer_type'] = rng.choice(
            ['Transient', 'Contract', 'Transient-Party', 'Group'],
            size=len(df)
        )
    
    # Country
    if 'country' in col_mapping:
        df['country'] = df[col_mapping['country']].fillna('Unknown')
    else:
        df['country'] = rng.choice(
            ['USA', 'GBR', 'FRA', 'DEU', 'CHE', 'ITA', 'ESP'],
            size=len(df)
        )
    
    # Hotel
    if 'hotel' in col_mapping:
        df['hotel'] = df[col_mapping['hotel']].fillna('Hotel A')
    else:
        df['hotel'] = rng.choice(['Resort Hotel', 'City Hotel'], size=len(df))
    
    # ADR (Average Daily Rate)
    if 'adr' in col_mapping:
        df['adr'] = pd.to_numeric(df[col_mapping['adr']], errors='coerce')
        df['adr'] = df['adr'].clip(lower=0)
        df['adr'] = df['adr'].fillna(df['adr'].median())
    else:
        df['adr'] = rng.normal(100, 50, size=len(df)).clip(min=20)
    
    # Is repeated guest
    if 'is_repeated_guest' in col_mapping:
        df['is_repeated_guest'] = df[col_mapping['is_repeated_guest']].fillna(0).astype(bool)
    else:
        df['is_repeated_guest'] = rng.choice([False, False, False, True], size=len(df))
    
    # Lead time
    if 'lead_time' in col_mapping:
        df['lead_time'] = pd.to_numeric(df[col_mapping['lead_time']], errors='coerce').fillna(0)
    else:
        df['lead_time'] = rng.integers(0, 365, size=len(df))
    
    # Meal
    if 'meal' in col_mapping:
        df['meal'] = df[col_mapping['meal']].fillna('BB')
    else:
        df['meal'] = rng.choice(['BB', 'HB', 'FB', 'SC'], size=len(df))
    
    # Distribution channel
    if 'distribution_channel' in col_mapping:
        df['distribution_channel'] = df[col_mapping['distribution_channel']].fillna('TA/TO')
    else:
        df['distribution_channel'] = rng.choice(['TA/TO', 'Direct', 'Corporate', 'GDS'], size=len(df))
    
    # Drop rows with missing critical values if requested
    if drop_missing:
        initial_len = len(df)
        df = df[df['stay_nights'] > 0]
        df = df[df['total_guests'] > 0]
        df = df[df['arrival_date'].notna()]
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing critical values")
    
    # Engineered features
    
    # Stay type (short/medium/long)
    df['stay_type'] = pd.cut(
        df['stay_nights'],
        bins=[0, 2, 6, np.inf],
        labels=['short', 'medium', 'long']
    )
    
    # Stay purpose (business vs leisure)
    df['stay_purpose'] = 'leisure'
    
    # Business heuristics
    business_mask = (
        (df['market_segment'] == 'Corporate') |
        (df['customer_type'].isin(['Contract', 'Group'])) |
        ((df['total_guests'] == 1) & (df['stay_nights'] <= 3)) |
        (df['lead_time'] < 7)
    )
    df.loc[business_mask, 'stay_purpose'] = 'business'
    
    # Additional segment features
    df['is_family'] = (df['children'] > 0) | (df['babies'] > 0)
    df['is_business'] = (df['stay_purpose'] == 'business')
    df['is_weekend_stay'] = df['arrival_date'].dt.dayofweek >= 5
    df['is_long_lead'] = df['lead_time'] > 90
    df['price_per_night'] = df['adr']
    
    # Add 'nights' as alias for 'stay_nights' (for compatibility)
    df['nights'] = df['stay_nights']
    
    # Add 'purpose_of_stay' as alias for 'stay_purpose' (for compatibility)
    df['purpose_of_stay'] = df['stay_purpose']
    
    # Create a guest_id and stay_id
    df['guest_id'] = [f"HOTEL_{i:08d}" for i in range(len(df))]
    df['stay_id'] = [f"STAY_{i:08d}" for i in range(len(df))]
    df['source'] = 'hotel'
    
    # Select and order final columns
    final_columns = [
        'guest_id', 'stay_id', 'source', 'arrival_date', 'stay_nights', 'nights',
        'total_guests', 'adults', 'children', 'babies',
        'country', 'hotel', 'market_segment', 'customer_type',
        'distribution_channel', 'is_repeated_guest',
        'price_per_night', 'adr', 'stay_type', 'stay_purpose', 'purpose_of_stay',
        'is_family', 'is_business', 'is_weekend_stay', 'lead_time', 'meal'
    ]
    
    # Only include columns that exist
    final_columns = [c for c in final_columns if c in df.columns]
    
    return df[final_columns].reset_index(drop=True)


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    # Compute summary statistics for a guest dataset
    stats = {
        'n_guests': len(df),
        'n_unique_countries': df['country'].nunique() if 'country' in df.columns else 0,
        'avg_stay_nights': df['stay_nights'].mean() if 'stay_nights' in df.columns else 0,
        'avg_party_size': df['total_guests'].mean() if 'total_guests' in df.columns else 0,
        'pct_business': (df['stay_purpose'] == 'business').mean() * 100 if 'stay_purpose' in df.columns else 0,
        'pct_family': df['is_family'].mean() * 100 if 'is_family' in df.columns else 0,
        'avg_price_per_night': df['price_per_night'].mean() if 'price_per_night' in df.columns else 0,
    }
    return stats

