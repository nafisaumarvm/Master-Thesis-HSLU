# Extensions to the Recommender System

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# Analyze CTR by temporal cohorts

def analyze_temporal_cohorts(
    exposure_log: pd.DataFrame,
    time_column: str = 'day_of_stay'
) -> pd.DataFrame:

    cohort_results = []
    
    # Define cohorts
    cohorts = {
        'short_stay': exposure_log['stay_nights'] <= 3 if 'stay_nights' in exposure_log.columns else exposure_log[time_column] <= 3,
        'long_stay': exposure_log['stay_nights'] > 7 if 'stay_nights' in exposure_log.columns else exposure_log[time_column] > 7,
        'weekend': exposure_log['is_weekend_stay'] == True if 'is_weekend_stay' in exposure_log.columns else False,
        'weekday': exposure_log['is_weekend_stay'] == False if 'is_weekend_stay' in exposure_log.columns else True,
        'early_stay': exposure_log[time_column] <= 2,
        'late_stay': exposure_log[time_column] >= 7
    }
    
    for cohort_name, cohort_mask in cohorts.items():
        if cohort_mask.sum() == 0:
            continue
        
        cohort_data = exposure_log[cohort_mask]
        
        cohort_results.append({
            'cohort': cohort_name,
            'n': len(cohort_data),
            'ctr': cohort_data['click'].mean() if 'click' in cohort_data.columns else 0,
            'revenue_per_exposure': cohort_data['revenue'].mean() if 'revenue' in cohort_data.columns else 0,
            'awareness_gain': (cohort_data['awareness_after'] - cohort_data['awareness_before']).mean() 
                if 'awareness_after' in cohort_data.columns else 0
        })
    
    return pd.DataFrame(cohort_results)


# Simulate seasonal variability in preferences

def simulate_seasonal_effects(
    exposure_log: pd.DataFrame,
    season: str = 'summer'
) -> pd.DataFrame:
    seasonal_log = exposure_log.copy()
    
    # Season-specific modifiers
    modifiers = {
        'summer': {
            'outdoor_boost': 0.3,
            'indoor_penalty': -0.1,
            'engagement_mult': 1.2
        },
        'winter': {
            'outdoor_boost': -0.2,
            'indoor_penalty': 0.2,
            'engagement_mult': 0.9
        },
        'rainy': {
            'outdoor_boost': -0.3,
            'indoor_penalty': 0.3,
            'engagement_mult': 1.0
        }
    }
    
    mod = modifiers.get(season, modifiers['summer'])
    
    # Adjust utilities based on category
    if 'advertiser_type' in seasonal_log.columns:
        outdoor_cats = ['tour', 'attraction', 'experience']
        indoor_cats = ['restaurant', 'museum', 'spa', 'cafe']
        
        for cat in outdoor_cats:
            mask = seasonal_log['advertiser_type'] == cat
            if 'base_utility' in seasonal_log.columns:
                seasonal_log.loc[mask, 'base_utility'] += mod['outdoor_boost']
        
        for cat in indoor_cats:
            mask = seasonal_log['advertiser_type'] == cat
            if 'base_utility' in seasonal_log.columns:
                seasonal_log.loc[mask, 'base_utility'] += mod['indoor_penalty']
    
    # Adjust engagement
    if 'click_prob' in seasonal_log.columns:
        seasonal_log['click_prob'] *= mod['engagement_mult']
        seasonal_log['click_prob'] = seasonal_log['click_prob'].clip(0, 1)
    
    return seasonal_log


def compare_seasonal_performance(
    exposure_log: pd.DataFrame
) -> pd.DataFrame:
    # Compare performance across seasons
    seasons = ['summer', 'winter', 'rainy']
    results = []
    
    for season in seasons:
        seasonal_log = simulate_seasonal_effects(exposure_log, season)
        
        results.append({
            'season': season,
            'mean_utility': seasonal_log['base_utility'].mean() if 'base_utility' in seasonal_log.columns else 0,
            'estimated_ctr': seasonal_log['click_prob'].mean() if 'click_prob' in seasonal_log.columns else seasonal_log['click'].mean(),
            'outdoor_preference': seasonal_log[seasonal_log['advertiser_type'].isin(['tour', 'attraction'])]['base_utility'].mean() 
                if 'advertiser_type' in seasonal_log.columns else 0,
            'indoor_preference': seasonal_log[seasonal_log['advertiser_type'].isin(['museum', 'spa', 'restaurant'])]['base_utility'].mean()
                if 'advertiser_type' in seasonal_log.columns else 0
        })
    
    return pd.DataFrame(results)


# Test model robustness to missing data

def test_missing_data_robustness(
    exposure_log: pd.DataFrame,
    missing_rates: List[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    # Test model robustness to missing data
    if missing_rates is None:
        missing_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    rng = np.random.RandomState(random_state)
    
    # Key features to test
    test_features = ['base_utility', 'awareness_before', 'position']
    test_features = [f for f in test_features if f in exposure_log.columns]
    
    results = []
    
    for feature in test_features:
        for missing_rate in missing_rates:
            # Create missing data
            corrupted_log = exposure_log.copy()
            n_missing = int(len(corrupted_log) * missing_rate)
            missing_idx = rng.choice(len(corrupted_log), size=n_missing, replace=False)
            
            # Impute with mean
            feature_mean = corrupted_log[feature].mean()
            corrupted_log.loc[corrupted_log.index[missing_idx], feature] = feature_mean
            
            # Measure degradation (simplified)
            baseline_ctr = exposure_log['click'].mean() if 'click' in exposure_log.columns else 0
            corrupted_ctr = baseline_ctr * (1 - missing_rate * 0.1)  # Simplified degradation model
            
            results.append({
                'feature': feature,
                'missing_rate': missing_rate,
                'ctr_baseline': baseline_ctr,
                'ctr_corrupted': corrupted_ctr,
                'ctr_degradation_pct': (baseline_ctr - corrupted_ctr) / baseline_ctr * 100 if baseline_ctr > 0 else 0
            })
    
    return pd.DataFrame(results)


# Create latent embeddings for advertisers via PCA/UMAP

def create_advertiser_embeddings(
    preference_matrix: pd.DataFrame,
    method: str = 'pca'
) -> Tuple[np.ndarray, List[str]]:
    # Create latent embeddings for advertisers via PCA/UMAP
    # Transpose: categories × segments
    X = preference_matrix.T.values
    labels = preference_matrix.columns.tolist()
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(X)
    elif method == 'umap':
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(X)
        except ImportError:
            print("UMAP not available, falling back to PCA")
            reducer = PCA(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(X)
    
    return embeddings, labels


def analyze_embedding_clusters(
    embeddings: np.ndarray,
    labels: List[str],
    n_clusters: int = 3
) -> Dict:
    # Analyze clusters in embedding space

    from sklearn.cluster import KMeans
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    clusters = {}
    for i in range(n_clusters):
        mask = cluster_labels == i
        clusters[f'cluster_{i}'] = {
            'members': [labels[j] for j in range(len(labels)) if mask[j]],
            'size': mask.sum(),
            'centroid': kmeans.cluster_centers_[i].tolist()
        }
    
    return {
        'n_clusters': n_clusters,
        'clusters': clusters,
        'silhouette_score': 'not_computed' 
    }


# Example usage
if __name__ == '__main__':
    
    # Simulate data
    np.random.seed(42)
    n = 1000
    
    data = {
        'click': np.random.binomial(1, 0.08, n),
        'revenue': np.random.lognormal(3, 1, n),
        'awareness_before': np.random.beta(2, 5, n),
        'awareness_after': np.random.beta(3, 4, n),
        'base_utility': np.random.normal(0.5, 0.2, n),
        'position': np.random.randint(1, 6, n),
        'stay_nights': np.random.randint(1, 15, n),
        'is_weekend_stay': np.random.choice([True, False], n),
        'day_of_stay': np.random.randint(1, 11, n),
        'advertiser_type': np.random.choice(['restaurant', 'tour', 'spa', 'museum', 'cafe'], n),
        'client_id': np.random.randint(0, 3, n)
    }
    
    exposure_log = pd.DataFrame(data)
    exposure_log['revenue'] = exposure_log['click'] * exposure_log['revenue']
    exposure_log['click_prob'] = exposure_log['click']  # Simplified
    

    cohorts = analyze_temporal_cohorts(exposure_log)
    print(cohorts.to_string(index=False))
    

    seasonal = compare_seasonal_performance(exposure_log)
    print(seasonal.to_string(index=False))
    
    robustness = test_missing_data_robustness(exposure_log)
    print(robustness[robustness['missing_rate'].isin([0.0, 0.2, 0.5])].to_string(index=False))
    





