# CTR Logs Loader

import pandas as pd
import numpy as np
import gzip
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings

from .utils import set_random_seed, tags_to_string


def load_ctr_gz_file(
    gz_path: str,
    n_rows: Optional[int] = None,
    chunksize: int = 100000
) -> pd.DataFrame:
    # Load a .gz compressed CTR log file

    print(f"Loading {os.path.basename(gz_path)}...")
    
    try:
        # Try reading with pandas directly (handles .gz automatically)
        if n_rows:
            df = pd.read_csv(gz_path, compression='gzip', sep='\t', nrows=n_rows, header=None)
        else:
            # Read in chunks for large files
            chunks = []
            for chunk in pd.read_csv(gz_path, compression='gzip', sep='\t', 
                                    chunksize=chunksize, header=None):
                chunks.append(chunk)
                if len(chunks) * chunksize >= (n_rows or float('inf')):
                    break
            df = pd.concat(chunks, ignore_index=True)
            if n_rows:
                df = df.head(n_rows)
        
        print(f"Loaded {len(df)} records")
        return df
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return pd.DataFrame()


def infer_ctr_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Infer schema for CTR log files (Criteo-style format)

    # Standard Criteo format
    if df.shape[1] == 40:  # 1 label + 13 int + 26 cat
        col_names = ['Label'] + \
                    [f'I{i}' for i in range(1, 14)] + \
                    [f'C{i}' for i in range(1, 27)]
        df.columns = col_names
    elif df.shape[1] == 14:  # Label + 13 features
        col_names = ['Label'] + [f'F{i}' for i in range(1, 14)]
        df.columns = col_names
    else:
        # Generic naming
        df.columns = [f'Col_{i}' for i in range(df.shape[1])]
    
    return df


def load_multiple_ctr_logs(
    log_paths: List[str],
    sample_per_file: int = 50000,
    seed: int = 42
) -> pd.DataFrame:
    # Load multiple CTR log files and combine

    print(f"\nLoading {len(log_paths)} CTR log files...")
    
    all_data = []
    
    for path in log_paths:
        if not os.path.exists(path):
            print(f"⚠ File not found: {path}")
            continue
        
        df = load_ctr_gz_file(path, n_rows=sample_per_file)
        if len(df) > 0:
            df = infer_ctr_schema(df)
            df['source_file'] = os.path.basename(path)
            all_data.append(df)
    
    if not all_data:
        print("No data loaded from CTR logs")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n Combined: {len(combined)} total records")
    print(f"Click rate: {combined['Label'].mean():.4f}")
    
    return combined


def create_advertisers_from_ctr_logs(
    ctr_logs: pd.DataFrame,
    n_advertisers: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    # Create advertiser catalogue from CTR logs

    print(f"\nCreating {n_advertisers} advertisers from CTR logs...")
    
    rng = set_random_seed(seed)
    
    # Advertiser types
    advertiser_types = [
        'restaurant', 'bar', 'cafe', 'museum', 'gallery',
        'spa', 'tour', 'attraction', 'nightlife', 'transport'
    ]
    
    # Category tags
    all_tags = [
        'foodie', 'family_friendly', 'nightlife', 'culture', 'outdoor',
        'budget', 'luxury', 'romantic', 'adventure', 'wellness',
        'shopping', 'local', 'touristy', 'quiet', 'trendy'
    ]
    
    advertisers = []
    
    # Global CTR statistics
    global_ctr = ctr_logs['Label'].mean() if 'Label' in ctr_logs.columns else 0.05
    
    # Group by categorical features to create clusters
    if 'C1' in ctr_logs.columns:
        # Use first categorical feature for clustering
        category_groups = ctr_logs.groupby('C1')['Label'].agg(['mean', 'count'])
        category_groups = category_groups[category_groups['count'] >= 10]  # Min samples
    else:
        category_groups = None
    
    for i in range(n_advertisers):
        ad_id = f"CTR_AD_{i:04d}"
        
        # Sample CTR from actual data
        if category_groups is not None and len(category_groups) > 0:
            # Sample from a category cluster
            if rng.random() < 0.7:  # 70% use actual cluster CTRs
                cluster = category_groups.sample(1, random_state=seed+i)
                base_ctr = cluster['mean'].values[0]
                # Add noise
                base_ctr = max(0.001, base_ctr + rng.normal(0, 0.01))
            else:
                # Sample from global distribution
                base_ctr = global_ctr + rng.normal(0, 0.02)
        else:
            # Sample from overall distribution
            base_ctr = global_ctr + rng.normal(0, 0.02)
        
        base_ctr = np.clip(base_ctr, 0.001, 0.5)
        
        # Advertiser type
        adv_type = rng.choice(advertiser_types)
        
        # Name
        advertiser_name = f"CTR Advertiser {i:04d}"
        
        # Category tags (2-4 tags)
        n_tags = rng.integers(2, 5)
        category_tags = rng.choice(all_tags, size=n_tags, replace=False).tolist()
        
        # Add type-specific tags
        type_tags = _get_type_tags(adv_type)
        category_tags = list(set(category_tags + type_tags))
        
        # Distance (log-normal distribution for realism)
        distance_km = np.clip(rng.lognormal(1.0, 0.8), 0.1, 20.0)
        
        # Opening dayparts
        n_dayparts = rng.integers(2, 5)
        dayparts = rng.choice(
            ['morning', 'afternoon', 'evening', 'late_night'],
            size=n_dayparts,
            replace=False
        ).tolist()
        
        # Price level (correlate with CTR - higher CTR often means better targeting)
        if base_ctr > global_ctr * 1.5:
            price_level = rng.choice(['medium', 'high'], p=[0.3, 0.7])
        elif base_ctr < global_ctr * 0.5:
            price_level = rng.choice(['low', 'medium'], p=[0.7, 0.3])
        else:
            price_level = 'medium'
        
        # Base utility (log-odds of CTR)
        base_utility = np.log(base_ctr / (1 - base_ctr))
        
        # Revenue (based on type and price level)
        revenue_per_conversion = _compute_revenue(adv_type, price_level, rng)
        
        # Expected revenue (CTR × Revenue)
        expected_revenue = base_ctr * revenue_per_conversion
        
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
            'expected_revenue': expected_revenue,
            'source': 'ctr_logs'
        })
    
    ads_df = pd.DataFrame(advertisers)
    
    print(f"Created {len(ads_df)} advertisers")
    print(f"CTR range: {ads_df['estimated_ctr'].min():.4f} - {ads_df['estimated_ctr'].max():.4f}")
    print(f"Mean CTR: {ads_df['estimated_ctr'].mean():.4f}")
    print(f"Expected revenue range: ${ads_df['expected_revenue'].min():.2f} - ${ads_df['expected_revenue'].max():.2f}")
    
    return ads_df


def _get_type_tags(adv_type: str) -> List[str]:
    # Get tags for advertiser type
    type_tag_map = {
        'restaurant': ['foodie', 'local'],
        'bar': ['nightlife', 'trendy'],
        'cafe': ['local', 'quiet'],
        'museum': ['culture', 'family_friendly'],
        'gallery': ['culture', 'quiet'],
        'spa': ['wellness', 'luxury'],
        'tour': ['adventure', 'touristy'],
        'attraction': ['family_friendly', 'touristy'],
        'nightlife': ['nightlife', 'trendy'],
        'transport': ['local']
    }
    return type_tag_map.get(adv_type, ['local'])


def _compute_revenue(adv_type: str, price_level: str, rng: np.random.Generator) -> float:
    # Compute revenue per conversion
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


def analyze_ctr_logs(ctr_logs: pd.DataFrame) -> Dict:
    # Analyze CTR logs to understand patterns

    if len(ctr_logs) == 0:
        return {}
    
    analysis = {
        'total_impressions': len(ctr_logs),
        'total_clicks': 0,
        'overall_ctr': 0.0,
        'ctr_by_day': {},
        'feature_stats': {}
    }
    
    if 'Label' in ctr_logs.columns:
        analysis['total_clicks'] = ctr_logs['Label'].sum()
        analysis['overall_ctr'] = ctr_logs['Label'].mean()
    
    # CTR by source file (day)
    if 'source_file' in ctr_logs.columns:
        for source in ctr_logs['source_file'].unique():
            subset = ctr_logs[ctr_logs['source_file'] == source]
            if 'Label' in subset.columns:
                analysis['ctr_by_day'][source] = subset['Label'].mean()
    
    # Feature statistics
    for col in ctr_logs.columns:
        if col.startswith('I'):  # Integer features
            analysis['feature_stats'][col] = {
                'mean': ctr_logs[col].mean(),
                'std': ctr_logs[col].std(),
                'missing_pct': ctr_logs[col].isna().mean()
            }
    
    return analysis


def load_your_ctr_logs(
    base_dir: str = "/Users/nafisaumar/Documents/Master Thesis/Recommender System NEW",
    sample_per_file: int = 50000,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Your CTR log files
    log_files = [
        os.path.join(base_dir, 'day_0.gz'),
        os.path.join(base_dir, 'day_1.gz'),
        os.path.join(base_dir, 'day_2.gz'),
        os.path.join(base_dir, 'day_3.gz'),
    ]
    
    # Check which files exist
    existing_files = [f for f in log_files if os.path.exists(f)]
    
    print(f"\nFound {len(existing_files)} / {len(log_files)} CTR log files:")
    for f in log_files:
        status = "✓" if os.path.exists(f) else "✗"
        print(f"  {status} {os.path.basename(f)}")
    
    if not existing_files:
        print("\n No CTR log files found")
        return pd.DataFrame(), pd.DataFrame()
    
    # Load all files
    ctr_logs = load_multiple_ctr_logs(existing_files, sample_per_file, seed)
    
    if len(ctr_logs) == 0:
        print("\n No data loaded from CTR logs")
        return pd.DataFrame(), pd.DataFrame()
    
    
    analysis = analyze_ctr_logs(ctr_logs)
    
    print(f"\nOverall Statistics:")
    print(f"Total impressions: {analysis['total_impressions']:,}")
    print(f"Total clicks: {analysis['total_clicks']:,}")
    print(f"Overall CTR: {analysis['overall_ctr']:.4f} ({analysis['overall_ctr']*100:.2f}%)")
    
    if analysis['ctr_by_day']:
        print(f"\nCTR by Day:")
        for day, ctr in sorted(analysis['ctr_by_day'].items()):
            print(f"  {day:15s}: {ctr:.4f} ({ctr*100:.2f}%)")
    
    
    ads_df = create_advertisers_from_ctr_logs(ctr_logs, n_advertisers=500, seed=seed)
    
    return ctr_logs, ads_df





