"""
Enhanced Data Loader - Combines All Three Datasets

Integrates:
1. hotel_guests_dataset.csv (4K guests) - baseline
2. hotel_booking 2.csv (119K bookings) - main training data  
3. Dataset_Ads.csv (10K ads) - CTR calibration

Provides 30× more training data for improved model performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os


def load_hotel_booking_large(
    filepath: str = 'hotel_booking 2.csv',
    sample_frac: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load and preprocess large hotel booking dataset.
    
    Features extracted:
    - guest_id (synthesized from booking)
    - stay_nights (weekend + week nights)
    - adults, children (party composition)
    - is_business (from customer_type)
    - country, market_segment, distribution_channel
    - lead_time, is_repeated_guest
    - price (ADR)
    
    Parameters
    ----------
    filepath : str
        Path to hotel_booking 2.csv
    sample_frac : float
        Fraction to sample (1.0 = all data, 0.1 = 10%)
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Processed guest data (119K rows)
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Sample if requested
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
    
    print(f"  Loaded {len(df):,} bookings")
    
    # Process features
    processed = pd.DataFrame()
    
    # Guest ID (use name or create unique ID)
    if 'name' in df.columns:
        # Fill missing names with guest IDs based on index
        guest_ids = df['name'].copy()
        missing_mask = guest_ids.isna()
        guest_ids.loc[missing_mask] = 'Guest_' + df.index[missing_mask].astype(str)
        processed['guest_id'] = guest_ids
    else:
        processed['guest_id'] = 'Guest_' + df.index.astype(str)
    
    # Stay duration
    processed['stay_nights'] = (
        df['stays_in_weekend_nights'].fillna(0) + 
        df['stays_in_week_nights'].fillna(0)
    ).astype(int)
    
    # Party composition
    processed['adults'] = df['adults'].fillna(1).astype(int)
    processed['children'] = df['children'].fillna(0).fillna(0).astype(int)  # handles NaN
    processed['total_guests'] = processed['adults'] + processed['children']
    
    # Stay type
    processed['is_weekend_stay'] = df['stays_in_weekend_nights'] > 0
    processed['is_family'] = processed['children'] > 0
    
    # Business classification
    if 'customer_type' in df.columns:
        processed['is_business'] = df['customer_type'].str.contains('Corporate', case=False, na=False)
    elif 'market_segment' in df.columns:
        processed['is_business'] = df['market_segment'].str.contains('Corporate', case=False, na=False)
    else:
        processed['is_business'] = False
    
    processed['stay_purpose'] = processed['is_business'].map({True: 'business', False: 'leisure'})
    
    # Booking details
    if 'lead_time' in df.columns:
        processed['lead_time'] = df['lead_time'].fillna(0).astype(int)
    
    if 'is_repeated_guest' in df.columns:
        processed['is_repeated_guest'] = df['is_repeated_guest'].fillna(0).astype(bool)
    
    # Categorical features
    if 'country' in df.columns:
        processed['country'] = df['country'].fillna('UNKNOWN')
    
    if 'market_segment' in df.columns:
        processed['market_segment'] = df['market_segment'].fillna('Direct')
    
    if 'distribution_channel' in df.columns:
        processed['booking_channel'] = df['distribution_channel'].fillna('Direct')
    
    # Price
    if 'adr' in df.columns:
        processed['price_per_night'] = df['adr'].fillna(df['adr'].median())
    
    # Hotel type
    if 'hotel' in df.columns:
        processed['hotel_type'] = df['hotel']
    
    # Arrival date (for temporal features)
    if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns:
        processed['arrival_year'] = df['arrival_date_year']
        processed['arrival_month'] = df['arrival_date_month']
    
    # Cancellation (for filtering)
    if 'is_canceled' in df.columns:
        processed['is_canceled'] = df['is_canceled'].fillna(0).astype(bool)
        # Filter out cancellations for guest behavior
        processed = processed[~processed['is_canceled']].copy()
        print(f"  After removing cancellations: {len(processed):,} bookings")
    
    # Clean data
    processed = processed[processed['stay_nights'] > 0].copy()  # Valid stays only
    processed = processed[processed['stay_nights'] <= 30].copy()  # Remove outliers
    
    print(f"  Final dataset: {len(processed):,} valid guest stays")
    
    return processed


def load_ads_dataset(
    filepath: str = 'Dataset_Ads.csv',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load online ads dataset for CTR calibration.
    
    Features:
    - Age, Gender, Income, Location
    - Ad Type, Ad Topic, Ad Placement
    - Clicks, CTR, Conversion Rate
    
    Parameters
    ----------
    filepath : str
        Path to Dataset_Ads.csv
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Ads data (10K rows)
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df):,} ad records")
    
    # Clean CTR and conversion rate
    if 'CTR' in df.columns:
        df['CTR'] = df['CTR'].clip(0, 1)
    
    if 'Conversion Rate' in df.columns:
        df['Conversion Rate'] = df['Conversion Rate'].clip(0, 1)
    
    return df


def extract_ctr_calibration(ads_df: pd.DataFrame) -> Dict:
    """
    Extract CTR statistics from ads dataset for calibration.
    
    Returns position bias estimates, CTR distributions, etc.
    
    Parameters
    ----------
    ads_df : pd.DataFrame
        Ads dataset
        
    Returns
    -------
    dict
        Calibration parameters
    """
    calibration = {}
    
    # Overall CTR statistics
    calibration['mean_ctr'] = ads_df['CTR'].mean()
    calibration['median_ctr'] = ads_df['CTR'].median()
    calibration['std_ctr'] = ads_df['CTR'].std()
    
    # CTR by ad type
    if 'Ad Type' in ads_df.columns:
        calibration['ctr_by_type'] = ads_df.groupby('Ad Type')['CTR'].mean().to_dict()
    
    # CTR by topic (can map to advertiser categories)
    if 'Ad Topic' in ads_df.columns:
        calibration['ctr_by_topic'] = ads_df.groupby('Ad Topic')['CTR'].mean().to_dict()
        
        # Map ad topics to advertiser categories
        topic_mapping = {
            'Travel': ['tour', 'attraction', 'transport'],
            'Food': ['restaurant', 'cafe'],
            'Health': ['spa', 'wellness'],
            'Entertainment': ['nightlife', 'event'],
            'Shopping': ['shop']
        }
        calibration['topic_category_mapping'] = topic_mapping
    
    # CTR by placement (proxy for position)
    if 'Ad Placement' in ads_df.columns:
        calibration['ctr_by_placement'] = ads_df.groupby('Ad Placement')['CTR'].mean().to_dict()
    
    # Demographic patterns
    if 'Age' in ads_df.columns:
        # Age bins
        ads_df['age_bin'] = pd.cut(ads_df['Age'], bins=[0, 30, 50, 70, 100], 
                                    labels=['young', 'middle', 'senior', 'elderly'])
        calibration['ctr_by_age'] = ads_df.groupby('age_bin')['CTR'].mean().to_dict()
    
    if 'Gender' in ads_df.columns:
        calibration['ctr_by_gender'] = ads_df.groupby('Gender')['CTR'].mean().to_dict()
    
    if 'Location' in ads_df.columns:
        calibration['ctr_by_location'] = ads_df.groupby('Location')['CTR'].mean().to_dict()
    
    # Conversion rates
    if 'Conversion Rate' in ads_df.columns:
        calibration['mean_conversion_rate'] = ads_df['Conversion Rate'].mean()
        calibration['ctr_to_conversion_ratio'] = calibration['mean_conversion_rate'] / calibration['mean_ctr']
    
    print(f"\nCTR Calibration Extracted:")
    print(f"  Mean CTR: {calibration['mean_ctr']:.4f}")
    print(f"  CTR by topic: {calibration.get('ctr_by_topic', {})}")
    
    return calibration


def combine_guest_datasets(
    guests_small: pd.DataFrame,
    guests_large: pd.DataFrame,
    weight_large: float = 0.9
) -> pd.DataFrame:
    """
    Combine small and large guest datasets with sampling.
    
    Strategy:
    - Use 90% from large dataset (119K → ~107K)
    - Use 10% from small dataset (4K → ~4K)
    - Total: ~111K guests
    
    Parameters
    ----------
    guests_small : pd.DataFrame
        Small dataset (4K)
    guests_large : pd.DataFrame
        Large dataset (119K)
    weight_large : float
        Fraction from large dataset
        
    Returns
    -------
    pd.DataFrame
        Combined dataset
    """
    # Sample from large
    n_large = int(len(guests_large) * weight_large)
    large_sampled = guests_large.sample(n=min(n_large, len(guests_large)), random_state=42)
    
    # Use all of small (it's already small)
    small_used = guests_small.copy()
    
    # Ensure consistent columns
    common_cols = list(set(large_sampled.columns) & set(small_used.columns))
    
    # Combine
    combined = pd.concat([
        large_sampled[common_cols],
        small_used[common_cols]
    ], ignore_index=True)
    
    print(f"\nCombined Guest Data:")
    print(f"  From large dataset: {len(large_sampled):,}")
    print(f"  From small dataset: {len(small_used):,}")
    print(f"  Total: {len(combined):,} guests")
    
    return combined


def load_all_datasets(
    use_large_hotel_data: bool = True,
    use_ads_calibration: bool = True,
    hotel_sample_frac: float = 1.0
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load all available datasets.
    
    Parameters
    ----------
    use_large_hotel_data : bool
        Whether to include hotel_booking 2.csv
    use_ads_calibration : bool
        Whether to load Dataset_Ads.csv for calibration
    hotel_sample_frac : float
        Fraction of large dataset to use (1.0 = all 119K)
        
    Returns
    -------
    tuple
        (guests_df, ads_df, calibration_params)
    """
    from .data_loading import load_hotel_guests
    
    print("="*70)
    print("ENHANCED DATA LOADING - ALL DATASETS")
    print("="*70)
    
    # 1. Load small guest dataset (baseline)
    print("\n1. Loading baseline guest dataset...")
    guests_small = load_hotel_guests('hotel_guests_dataset.csv')
    
    # 2. Load large hotel booking dataset
    guests_df = guests_small
    if use_large_hotel_data and os.path.exists('hotel_booking 2.csv'):
        print("\n2. Loading large hotel booking dataset...")
        guests_large = load_hotel_booking_large('hotel_booking 2.csv', sample_frac=hotel_sample_frac)
        
        # Combine datasets
        guests_df = combine_guest_datasets(guests_small, guests_large)
    else:
        print("\n2. Skipping large dataset (not found or disabled)")
    
    # 3. Load ads dataset for calibration
    ads_df = None
    calibration_params = {}
    
    if use_ads_calibration and os.path.exists('Dataset_Ads.csv'):
        print("\n3. Loading ads dataset for CTR calibration...")
        ads_df = load_ads_dataset('Dataset_Ads.csv')
        calibration_params = extract_ctr_calibration(ads_df)
    else:
        print("\n3. Skipping ads dataset (not found or disabled)")
    
    print("\n" + "="*70)
    print("DATA LOADING COMPLETE")
    print("="*70)
    print(f"Final guest dataset: {len(guests_df):,} stays")
    if ads_df is not None:
        print(f"Ads dataset: {len(ads_df):,} records")
    print("="*70)
    
    return guests_df, ads_df, calibration_params


# Example usage
if __name__ == '__main__':
    # Load all datasets
    guests, ads, calibration = load_all_datasets(
        use_large_hotel_data=True,
        use_ads_calibration=True,
        hotel_sample_frac=1.0  # Use all 119K bookings
    )
    
    print("\nGuest Data Summary:")
    print(guests.describe())
    
    print("\nGuest Data Columns:")
    print(guests.columns.tolist())
    
    if ads is not None:
        print("\nAds Data Summary:")
        print(ads.describe())
    
    print("\nCTR Calibration Parameters:")
    for key, value in calibration.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")




