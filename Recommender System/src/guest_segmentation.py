# Guest Segmentation Module

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Workaround for threadpoolctl
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Patch threadpoolctl to handle macOS issue
try:
    import threadpoolctl
    original_get_config = getattr(threadpoolctl, '_get_config', None)
    if original_get_config is None:
        # Monkey patch to avoid AttributeError
        def safe_get_config():
            try:
                import subprocess
                result = subprocess.run(['/usr/bin/otool', '-L'], 
                                      capture_output=True, timeout=1)
                return result.stdout.decode('utf-8', errors='ignore') if result.returncode == 0 else ''
            except:
                return ''
        pass
except ImportError:
    pass

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

try:
    from gower import gower_matrix
    GOWER_AVAILABLE = True
except ImportError:
    GOWER_AVAILABLE = False
    warnings.warn("gower package not available. Install with: pip install gower")


class GuestFeatureEngineer:
    # Engineers guest-level features from raw booking data
    def __init__(self, booking_data: pd.DataFrame):
        # Initialize guest feature engineer
        self.df = booking_data.copy()
        self._preprocess_raw_data()
    
    def _preprocess_raw_data(self):
        # Clean and prepare raw booking data
        print("Preprocessing raw booking data...")
        
        # Remove cancellations
        if 'is_canceled' in self.df.columns:
            n_before = len(self.df)
            self.df = self.df[self.df['is_canceled'] == 0].copy()
            print(f"Removed {n_before - len(self.df):,} cancellations")
        
        # Remove anomalous stays (0 nights or >365 nights)
        if 'stays_in_weekend_nights' in self.df.columns and 'stays_in_week_nights' in self.df.columns:
            self.df['total_nights'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
            n_before = len(self.df)
            self.df = self.df[(self.df['total_nights'] > 0) & (self.df['total_nights'] <= 365)].copy()
            print(f"Removed {n_before - len(self.df):,} anomalous stays")
        
        # Create date columns if needed
        if 'arrival_date_year' in self.df.columns:
            self._create_date_columns()
        
        print(f"Clean dataset: {len(self.df):,} valid bookings")
    
    def _create_date_columns(self):
        # Create proper date columns from year/month/day
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        self.df['arrival_month_num'] = self.df['arrival_date_month'].map(month_map)
        self.df['arrival_date'] = pd.to_datetime(
            self.df['arrival_date_year'].astype(str) + '-' +
            self.df['arrival_month_num'].astype(str) + '-' +
            self.df['arrival_date_day_of_month'].astype(str),
            errors='coerce'
        )
    
    def engineer_features(self) -> pd.DataFrame:
        # Engineer comprehensive guest-level features
        print("Engineering guest features...")
        
        features = pd.DataFrame(index=self.df.index)
        
        # 1. STAY CHARACTERISTICS
        features = self._add_stay_features(features)
        
        # 2. PARTY COMPOSITION
        features = self._add_party_features(features)
        
        # 3. BOOKING BEHAVIOR
        features = self._add_booking_features(features)
        
        # 4. GEOGRAPHY
        features = self._add_geography_features(features)
        
        # 5. REVENUE PROXIES
        features = self._add_revenue_features(features)
        
        # 6. TEMPORAL PATTERNS
        features = self._add_temporal_features(features)
        
        print(f"Engineered {len(features.columns)} features for {len(features):,} stays")
        
        return features
    
    def _add_stay_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add length-of-stay and stay pattern features
        # Total nights
        if 'total_nights' in self.df.columns:
            features['los_total'] = self.df['total_nights']
        else:
            features['los_total'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        
        # LOS categories
        features['los_single_night'] = (features['los_total'] == 1).astype(int)
        features['los_short'] = ((features['los_total'] >= 2) & (features['los_total'] <= 3)).astype(int)
        features['los_medium'] = ((features['los_total'] >= 4) & (features['los_total'] <= 7)).astype(int)
        features['los_long'] = (features['los_total'] > 7).astype(int)
        
        # Weekend proportion
        features['weekend_prop'] = self.df['stays_in_weekend_nights'] / (features['los_total'] + 0.01)
        
        # Weekend vs weekday dominant
        features['weekend_dominant'] = (features['weekend_prop'] > 0.5).astype(int)
        
        return features
    
    def _add_party_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add party composition features
        features['adults'] = self.df['adults'].fillna(2)
        features['children'] = self.df['children'].fillna(0)
        features['babies'] = self.df['babies'].fillna(0)
        
        # Total party size
        features['party_size'] = features['adults'] + features['children'] + features['babies']
        
        # Party type flags
        features['is_family'] = (features['children'] + features['babies'] > 0).astype(int)
        features['is_couple'] = ((features['adults'] == 2) & (features['children'] + features['babies'] == 0)).astype(int)
        features['is_solo'] = ((features['adults'] == 1) & (features['children'] + features['babies'] == 0)).astype(int)
        features['is_group'] = (features['adults'] > 2).astype(int)
        
        return features
    
    def _add_booking_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add booking channel and behavior features
        # Lead time (days between booking and arrival)
        if 'lead_time' in self.df.columns:
            features['lead_time'] = self.df['lead_time'].fillna(self.df['lead_time'].median())
            
            # Lead time categories
            features['lead_last_minute'] = (features['lead_time'] <= 7).astype(int)
            features['lead_normal'] = ((features['lead_time'] > 7) & (features['lead_time'] <= 60)).astype(int)
            features['lead_early_bird'] = (features['lead_time'] > 60).astype(int)
        
        # Market segment (one-hot encode main categories)
        if 'market_segment' in self.df.columns:
            for segment in ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups']:
                features[f'market_{segment.lower().replace(" ", "_").replace("/", "_")}'] = \
                    (self.df['market_segment'] == segment).astype(int)
        
        # Distribution channel
        if 'distribution_channel' in self.df.columns:
            for channel in ['Direct', 'Corporate', 'TA/TO']:
                features[f'channel_{channel.lower().replace("/", "_")}'] = \
                    (self.df['distribution_channel'] == channel).astype(int)
        
        # Repeat guest
        if 'is_repeated_guest' in self.df.columns:
            features['is_repeat_guest'] = self.df['is_repeated_guest'].fillna(0)
        
        # Previous cancellations (proxy for booking reliability)
        if 'previous_cancellations' in self.df.columns:
            features['prev_cancellations'] = self.df['previous_cancellations'].fillna(0)
            features['has_cancelled_before'] = (features['prev_cancellations'] > 0).astype(int)
        
        return features
    
    def _add_geography_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add country and geographic features
        if 'country' not in self.df.columns:
            return features
        
        # Group countries into regions
        domestic = ['PRT']
        western_europe = ['GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'BEL', 'NLD', 'CHE', 'AUT']
        eastern_europe = ['POL', 'RUS', 'CZE', 'ROU', 'HUN']
        north_america = ['USA', 'CAN']
        asia = ['CHN', 'JPN', 'KOR', 'IND', 'SGP', 'HKG', 'TWN']
        south_america = ['BRA', 'ARG', 'CHL', 'COL']
        africa_middle_east = ['ZAF', 'EGY', 'MAR', 'ARE', 'ISR', 'SAU']
        
        features['origin_domestic'] = self.df['country'].isin(domestic).astype(int)
        features['origin_western_europe'] = self.df['country'].isin(western_europe).astype(int)
        features['origin_eastern_europe'] = self.df['country'].isin(eastern_europe).astype(int)
        features['origin_north_america'] = self.df['country'].isin(north_america).astype(int)
        features['origin_asia'] = self.df['country'].isin(asia).astype(int)
        features['origin_south_america'] = self.df['country'].isin(south_america).astype(int)
        features['origin_africa_middle_east'] = self.df['country'].isin(africa_middle_east).astype(int)
        features['origin_other'] = (~self.df['country'].isin(
            domestic + western_europe + eastern_europe + north_america + 
            asia + south_america + africa_middle_east
        )).astype(int)
        
        # Long-haul vs short-haul
        features['is_long_haul'] = (
            features['origin_north_america'] + 
            features['origin_asia'] + 
            features['origin_south_america']
        ).clip(0, 1)
        
        return features
    
    def _add_revenue_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add revenue and spend-potential proxies
        # ADR (Average Daily Rate) if available
        if 'adr' in self.df.columns:
            features['adr'] = self.df['adr'].fillna(self.df['adr'].median())
            
            # ADR categories (proxy for budget vs luxury)
            adr_33 = self.df['adr'].quantile(0.33)
            adr_67 = self.df['adr'].quantile(0.67)
            features['adr_budget'] = (features['adr'] <= adr_33).astype(int)
            features['adr_mid'] = ((features['adr'] > adr_33) & (features['adr'] <= adr_67)).astype(int)
            features['adr_luxury'] = (features['adr'] > adr_67).astype(int)
        
        # Deposit type (proxy for booking security/trust)
        if 'deposit_type' in self.df.columns:
            features['deposit_no_deposit'] = (self.df['deposit_type'] == 'No Deposit').astype(int)
            features['deposit_required'] = (self.df['deposit_type'] != 'No Deposit').astype(int)
        
        # Customer type
        if 'customer_type' in self.df.columns:
            features['customer_transient'] = (self.df['customer_type'] == 'Transient').astype(int)
            features['customer_contract'] = (self.df['customer_type'] == 'Contract').astype(int)
            features['customer_group'] = (self.df['customer_type'] == 'Group').astype(int)
        
        # Meal plan (proxy for spend behavior)
        if 'meal' in self.df.columns:
            features['meal_bb'] = (self.df['meal'] == 'BB').astype(int)  # Bed & Breakfast
            features['meal_hb'] = (self.df['meal'] == 'HB').astype(int)  # Half Board
            features['meal_fb'] = (self.df['meal'] == 'FB').astype(int)  # Full Board
            features['meal_none'] = (self.df['meal'].isin(['SC', 'Undefined'])).astype(int)
        
        # Special requests (proxy for service expectations)
        if 'total_of_special_requests' in self.df.columns:
            features['special_requests'] = self.df['total_of_special_requests'].fillna(0)
            features['has_special_requests'] = (features['special_requests'] > 0).astype(int)
        
        return features
    
    def _add_temporal_features(self, features: pd.DataFrame) -> pd.DataFrame:
        # Add seasonality and temporal pattern features
        # Season
        if 'arrival_month_num' in self.df.columns:
            month = self.df['arrival_month_num']
            features['season_winter'] = month.isin([12, 1, 2]).astype(int)
            features['season_spring'] = month.isin([3, 4, 5]).astype(int)
            features['season_summer'] = month.isin([6, 7, 8]).astype(int)
            features['season_fall'] = month.isin([9, 10, 11]).astype(int)
            
            # Peak season (summer + holidays)
            features['is_peak_season'] = month.isin([7, 8, 12]).astype(int)
        
        # Hotel type
        if 'hotel' in self.df.columns:
            features['hotel_resort'] = (self.df['hotel'] == 'Resort Hotel').astype(int)
            features['hotel_city'] = (self.df['hotel'] == 'City Hotel').astype(int)
        
        return features


class GuestSegmentationModel:
    # Performs data-driven guest segmentation using hierarchical clustering + k-means
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        # Initialize guest segmentation model

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers
        self.kmeans = None
        self.cluster_profiles = None
        self.cluster_labels = None
        self.feature_names = None
    
    def fit(self, features: pd.DataFrame, sample_size: int = 10000) -> 'GuestSegmentationModel':
        # Fit segmentation model using hierarchical clustering + k-means

        print(f"Fitting Guest Segmentation Model (k={self.n_clusters})...")
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Remove any remaining NaN values
        features_clean = features.fillna(features.median())
        
        # Scale features
        print("Scaling features...") 
        X_scaled = self.scaler.fit_transform(features_clean)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features.columns, index=features.index)
        
        # Step 1: Hierarchical clustering on sample to explore structure
        print(f"Running hierarchical clustering on {sample_size:,} sample...")
        if len(features_clean) > sample_size:
            sample_idx = features_clean.sample(n=sample_size, random_state=self.random_state).index
            X_sample = X_scaled_df.loc[sample_idx]
        else:
            X_sample = X_scaled_df
        
        # Hierarchical clustering (Ward's method)
        hierarchical = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward',
            compute_distances=True
        )
        sample_labels = hierarchical.fit_predict(X_sample)
        
        # Step 2: K-means on full dataset, initialized from hierarchical solution
        print(f"Running k-means on full dataset ({len(features_clean):,} stays)...")
        
        # Get cluster centers from hierarchical solution
        initial_centers = np.array([
            X_sample[sample_labels == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])
        
        # K-means with hierarchical initialization
        try:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=initial_centers,
                n_init=1,
                max_iter=300,
                random_state=self.random_state
            )
            cluster_labels = self.kmeans.fit_predict(X_scaled)
        except (AttributeError, RuntimeError, Exception) as e:
            error_str = str(e)
            if "'NoneType' object has no attribute 'split'" in error_str or "threadpoolctl" in error_str.lower():
                # Fallback: Use hierarchical clustering on full dataset
                print("K-means failed due to threading issue, using hierarchical clustering on full dataset...")
                hierarchical_full = AgglomerativeClustering(
                    n_clusters=self.n_clusters,
                    linkage='ward',
                    compute_distances=False
                )
                cluster_labels = hierarchical_full.fit_predict(X_scaled)
                # Create a dummy kmeans object for compatibility
                from sklearn.base import BaseEstimator, ClusterMixin
                class DummyKMeans(BaseEstimator, ClusterMixin):
                    def __init__(self, centers, labels):
                        self.cluster_centers_ = centers
                        self.labels_ = labels
                    def predict(self, X):
                        # Simple nearest centroid prediction
                        from scipy.spatial.distance import cdist
                        distances = cdist(X, self.cluster_centers_)
                        return distances.argmin(axis=1)
                # Compute cluster centers from hierarchical results
                cluster_centers = np.array([
                    X_scaled[cluster_labels == i].mean(axis=0)
                    for i in range(self.n_clusters)
                ])
                self.kmeans = DummyKMeans(cluster_centers, cluster_labels)
            else:
                raise
        
        # Calculate quality metrics
        silhouette = silhouette_score(X_scaled, cluster_labels, sample_size=min(10000, len(X_scaled)))
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Calinski-Harabasz: {calinski:.1f}")
        
        # Store cluster labels
        self.cluster_labels = cluster_labels
        
        # Generate cluster profiles
        self._generate_cluster_profiles(features_clean, cluster_labels)
        
        return self
    
    def _generate_cluster_profiles(self, features: pd.DataFrame, labels: np.ndarray):
        # Generate interpretable profiles for each cluster
        
        profiles = []
        
        for i in range(self.n_clusters):
            mask = (labels == i)
            cluster_data = features[mask]
            
            profile = {
                'cluster_id': i,
                'size': mask.sum(),
                'proportion': mask.sum() / len(features),
                
                # Stay characteristics
                'los_mean': cluster_data['los_total'].mean(),
                'los_median': cluster_data['los_total'].median(),
                'weekend_prop_mean': cluster_data['weekend_prop'].mean(),
                
                # Party composition
                'party_size_mean': cluster_data['party_size'].mean(),
                'pct_family': cluster_data['is_family'].mean() * 100,
                'pct_couple': cluster_data['is_couple'].mean() * 100,
                'pct_solo': cluster_data['is_solo'].mean() * 100,
                'pct_group': cluster_data['is_group'].mean() * 100,
                
                # Booking behavior
                'lead_time_mean': cluster_data['lead_time'].mean() if 'lead_time' in cluster_data.columns else np.nan,
                'pct_repeat': cluster_data['is_repeat_guest'].mean() * 100 if 'is_repeat_guest' in cluster_data.columns else np.nan,
                
                # Geography
                'pct_domestic': cluster_data['origin_domestic'].mean() * 100 if 'origin_domestic' in cluster_data.columns else np.nan,
                'pct_long_haul': cluster_data['is_long_haul'].mean() * 100 if 'is_long_haul' in cluster_data.columns else np.nan,
                
                # Revenue proxy
                'adr_mean': cluster_data['adr'].mean() if 'adr' in cluster_data.columns else np.nan,
                'pct_luxury': cluster_data['adr_luxury'].mean() * 100 if 'adr_luxury' in cluster_data.columns else np.nan,
                
                # Temporal
                'pct_peak_season': cluster_data['is_peak_season'].mean() * 100 if 'is_peak_season' in cluster_data.columns else np.nan,
                'pct_resort': cluster_data['hotel_resort'].mean() * 100 if 'hotel_resort' in cluster_data.columns else np.nan,
            }
            
            profiles.append(profile)
        
        self.cluster_profiles = pd.DataFrame(profiles)
        
        # Display profiles
        for i, row in self.cluster_profiles.iterrows():
            print(f"Cluster {i} (n={row['size']:,}, {row['proportion']*100:.1f}%):")
            print(f"LOS: {row['los_mean']:.1f} nights (median: {row['los_median']:.1f})")
            print(f"Party: {row['party_size_mean']:.1f} people (Family: {row['pct_family']:.0f}%, Couple: {row['pct_couple']:.0f}%, Solo: {row['pct_solo']:.0f}%)")
            if not np.isnan(row['adr_mean']):
                print(f"   ADR: €{row['adr_mean']:.0f} (Luxury: {row['pct_luxury']:.0f}%)")
            if not np.isnan(row['lead_time_mean']):
                print(f"   Lead time: {row['lead_time_mean']:.0f} days")
            if not np.isnan(row['pct_domestic']):
                print(f"   Origin: Domestic {row['pct_domestic']:.0f}%, Long-haul {row['pct_long_haul']:.0f}%")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        # Predict cluster labels for new data
        if self.kmeans is None:
            raise ValueError("Model not fitted yet!")
        
        features_clean = features.fillna(features.median())
        X_scaled = self.scaler.transform(features_clean)
        return self.kmeans.predict(X_scaled)
    
    def assign_business_labels(self, labels: List[str]) -> Dict[int, str]:
        # Assign human-readable business labels to clusters
        if len(labels) != self.n_clusters:
            raise ValueError(f"Need exactly {self.n_clusters} labels, got {len(labels)}")
        
        label_map = {i: label for i, label in enumerate(labels)}
        self.cluster_profiles['business_label'] = self.cluster_profiles['cluster_id'].map(label_map)
        
        print("Business labels assigned:")
        for i, label in label_map.items():
            size = self.cluster_profiles.loc[i, 'size']
            pct = self.cluster_profiles.loc[i, 'proportion'] * 100
            print(f"   {i}: {label} (n={size:,}, {pct:.1f}%)")
        
        return label_map


class SegmentAdAffinityMapper:
    # Maps guest segments to advertiser category affinities
    
    def __init__(self, cluster_profiles: pd.DataFrame):
        # Initialize segment ad affinity mapper
        self.profiles = cluster_profiles
        self.affinity_matrix = None
        self.categories = [
            'Experiences',
            'Restaurants',
            'Shopping',
            'Wellness',
            'Nightlife',
            'Accommodation'
        ]
    
    def generate_expert_affinities(self) -> pd.DataFrame:
        # Generate expert-based segment-category affinity matrix
        print("Generating segment-category affinities...")
        
        n_clusters = len(self.profiles)
        affinities = pd.DataFrame(
            index=range(n_clusters),
            columns=self.categories
        )
        
        # For each cluster, assign affinities based on profile characteristics
        for i in range(n_clusters):
            profile = self.profiles.loc[i]
            
            # Base affinity scores (will be normalized)
            scores = {}
            
            # Experiences: Favored by long-stay leisure, families, tourists
            scores['Experiences'] = (
                0.3 * (profile['los_mean'] > 3) +  # Long stay
                0.2 * (profile['pct_family'] > 50) +  # Families
                0.2 * (profile['pct_long_haul'] > 30 if not np.isnan(profile['pct_long_haul']) else 0) +  # Tourists
                0.3 * (profile['pct_luxury'] > 50 if not np.isnan(profile['pct_luxury']) else 0.3)  # High spenders
            )
            
            # Restaurants: Universal, but stronger for leisure and long-stay
            scores['Restaurants'] = (
                0.5 +  # Base interest
                0.2 * (profile['los_mean'] > 2) +  # Multi-night stay
                0.2 * (profile['pct_couple'] > 50) +  # Couples
                0.1 * (profile['weekend_prop_mean'] > 0.3)  # Leisure timing
            )
            
            # Shopping: Leisure travelers, especially tourists and families
            scores['Shopping'] = (
                0.2 +  # Base interest
                0.3 * (profile['pct_long_haul'] > 30 if not np.isnan(profile['pct_long_haul']) else 0) +  # Tourists
                0.2 * (profile['pct_family'] > 40) +  # Families
                0.3 * (profile['pct_luxury'] > 50 if not np.isnan(profile['pct_luxury']) else 0.2)  # High spenders
            )
            
            # Wellness: Luxury leisure, long-stay, couples
            scores['Wellness'] = (
                0.1 +  # Base interest
                0.4 * (profile['pct_luxury'] > 60 if not np.isnan(profile['pct_luxury']) else 0) +  # Luxury guests
                0.2 * (profile['los_mean'] > 5) +  # Long stay
                0.2 * (profile['pct_couple'] > 50) +  # Couples
                0.1 * (profile['pct_resort'] > 60 if not np.isnan(profile['pct_resort']) else 0)  # Resort guests
            )
            
            # Nightlife: Young adults, couples, weekend guests
            scores['Nightlife'] = (
                0.2 +  # Base interest
                0.3 * (profile['pct_couple'] > 40) +  # Couples
                0.2 * (profile['weekend_prop_mean'] > 0.5) +  # Weekend stays
                0.2 * (1 - profile['pct_family'] / 100) +  # Not families
                0.1 * (profile['los_mean'] <= 3)  # Short stays
            )
            
            # Accommodation: Mainly for friends/family recommendations
            scores['Accommodation'] = (
                0.1 +  # Low base (already booked)
                0.3 * (profile['pct_repeat'] > 50 if not np.isnan(profile['pct_repeat']) else 0) +  # Repeat guests
                0.2 * (profile['pct_long_haul'] > 40 if not np.isnan(profile['pct_long_haul']) else 0)  # Tourists
            )
            
            # Normalize to [0.1, 1.0] range
            for cat in self.categories:
                affinities.loc[i, cat] = min(1.0, max(0.1, scores[cat]))
        
        self.affinity_matrix = affinities.astype(float)
        
        # Display
        print("Segment-Category Affinity Matrix:")
        print(self.affinity_matrix.round(2))
        
        return self.affinity_matrix
    
    def get_preference_matrix(self) -> pd.DataFrame:
        # Get preference matrix in format expected by recommender system
        if self.affinity_matrix is None:
            self.generate_expert_affinities()
        
        return self.affinity_matrix


def run_guest_segmentation_pipeline(
    booking_data_path: str,
    n_clusters: int = 8,
    sample_size: int = 10000,
    random_state: int = 42
) -> Tuple[GuestSegmentationModel, pd.DataFrame, pd.DataFrame]:
    # Complete pipeline: load data → engineer features → cluster → profile
    
    # 1. Load booking data
    print(f"Loading booking data from {booking_data_path}...")
    df = pd.read_csv(booking_data_path, low_memory=False)
    print(f"Loaded {len(df):,} bookings")
    
    # 2. Engineer features
    engineer = GuestFeatureEngineer(df)
    features = engineer.engineer_features()
    
    # 3. Fit segmentation model
    model = GuestSegmentationModel(n_clusters=n_clusters, random_state=random_state)
    model.fit(features, sample_size=sample_size)
    
    # 4. Create results DataFrame
    results = pd.DataFrame({
        'cluster_id': model.cluster_labels
    }, index=features.index)
        
    return model, features, results


if __name__ == "__main__":
    # Demo: Run segmentation on hotel booking data
    data_path = "/Users/nafisaumar/Documents/Master Thesis/Recommender System NEW/hotel_booking 2.csv"
    
    model, features, results = run_guest_segmentation_pipeline(
        booking_data_path=data_path,
        n_clusters=8,
        sample_size=10000
    )
    
    results.to_csv("results/guest_clusters.csv", index=False)
    model.cluster_profiles.to_csv("results/cluster_profiles.csv", index=False)
    
