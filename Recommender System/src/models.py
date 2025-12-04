"""
CTR prediction models and ranking algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import warnings

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .utils import parse_tags_string


class FeatureBuilder:
    """Build training features from exposure log and metadata."""
    
    def __init__(self, categorical_encoding: str = 'label'):
        """
        Args:
            categorical_encoding: 'label' or 'onehot'
        """
        self.categorical_encoding = categorical_encoding
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def build_training_frame(
        self,
        exposure_log: pd.DataFrame,
        guests_df: pd.DataFrame,
        ads_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build training features from exposure log.
        
        Args:
            exposure_log: Exposure log with clicks
            guests_df: Guest metadata
            ads_df: Advertiser metadata
            
        Returns:
            (X, y): Features and labels
        """
        # Merge guest features (only include columns that exist)
        guest_cols = ['guest_id']
        optional_guest_cols = [
            'purpose_of_stay', 'is_family', 'is_business',
            'is_weekend_stay', 'total_guests', 'nights', 'price_per_night',
            'country', 'source'
        ]
        for col in optional_guest_cols:
            if col in guests_df.columns:
                guest_cols.append(col)
        
        df = exposure_log.merge(
            guests_df[guest_cols],
            on='guest_id',
            how='left'
        )
        
        # Merge ad features
        df = df.merge(
            ads_df[[
                'ad_id', 'advertiser_type', 'category_tags', 'distance_km',
                'price_level', 'base_utility', 'revenue_per_conversion'
            ]],
            on='ad_id',
            how='left'
        )
        
        # Extract features
        features = pd.DataFrame()
        
        # Numeric features (with defaults if missing)
        features['position'] = df['position'].fillna(1) if 'position' in df.columns else 1
        features['day_of_stay'] = df['day_of_stay'].fillna(1) if 'day_of_stay' in df.columns else 1
        features['total_guests'] = df['total_guests'].fillna(2) if 'total_guests' in df.columns else 2
        features['nights'] = df['nights'].fillna(2) if 'nights' in df.columns else 2
        features['price_per_night'] = df['price_per_night'].fillna(100) if 'price_per_night' in df.columns else 100
        features['distance_km'] = df['distance_km'].fillna(5) if 'distance_km' in df.columns else 5
        features['base_utility'] = df['base_utility'].fillna(0) if 'base_utility' in df.columns else 0
        features['revenue_per_conversion'] = df['revenue_per_conversion'].fillna(30) if 'revenue_per_conversion' in df.columns else 30
        
        # Binary features (with defaults if missing)
        if 'is_family' in df.columns:
            features['is_family'] = df['is_family'].fillna(0).astype(int)
        else:
            features['is_family'] = 0
        
        if 'is_business' in df.columns:
            features['is_business'] = df['is_business'].fillna(0).astype(int)
        else:
            features['is_business'] = 0
        
        if 'is_weekend_stay' in df.columns:
            features['is_weekend_stay'] = df['is_weekend_stay'].fillna(0).astype(int)
        else:
            features['is_weekend_stay'] = 0
        
        # Categorical features (only include if columns exist)
        cat_features = {}
        cat_feature_defaults = {
            'purpose_of_stay': 'unknown',
            'advertiser_type': 'unknown',
            'price_level': 'medium',
            'time_of_day': 'afternoon',
            'country': 'unknown',
            'source': 'unknown'
        }
        
        for col, default in cat_feature_defaults.items():
            if col in df.columns:
                cat_features[col] = df[col].fillna(default)
            else:
                cat_features[col] = pd.Series([default] * len(df), index=df.index)
        
        if self.categorical_encoding == 'label':
            for col_name, col_data in cat_features.items():
                if col_name not in self.label_encoders:
                    le = LabelEncoder()
                    features[col_name] = le.fit_transform(col_data)
                    self.label_encoders[col_name] = le
                else:
                    le = self.label_encoders[col_name]
                    # Handle unseen categories
                    features[col_name] = col_data.map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        else:
            # One-hot encoding
            for col_name, col_data in cat_features.items():
                dummies = pd.get_dummies(col_data, prefix=col_name, drop_first=True)
                features = pd.concat([features, dummies], axis=1)
        
        # Tag-based features (binary indicators for common tags)
        common_tags = [
            'foodie', 'family_friendly', 'nightlife', 'culture',
            'outdoor', 'budget', 'luxury', 'wellness'
        ]
        
        if 'category_tags' in df.columns:
            for tag in common_tags:
                features[f'tag_{tag}'] = df['category_tags'].apply(
                    lambda x: 1 if tag in parse_tags_string(x) else 0
                )
        else:
            for tag in common_tags:
                features[f'tag_{tag}'] = 0
        
        # Interaction features
        try:
            features['price_match'] = (
                (features['price_per_night'] < 100) & (features['price_level'] == 0)
            ).astype(int)
        except:
            features['price_match'] = 0
        
        try:
            features['family_friendly_match'] = (
                features['is_family'] * features['tag_family_friendly']
            )
        except:
            features['family_friendly_match'] = 0
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Clean NaN values - fill with appropriate defaults
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                # Numeric columns: fill with 0 or median
                if col in ['distance_km', 'base_utility', 'revenue_per_conversion']:
                    features[col] = features[col].fillna(features[col].median())
                else:
                    features[col] = features[col].fillna(0)
            else:
                # Categorical/object columns
                features[col] = features[col].fillna(0)
        
        # Final safety check - replace any remaining NaN/inf
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        # Labels
        y = df['click']
        
        return features, y
    
    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features (e.g., scaling)."""
        # For now, no scaling to keep interpretability
        # Can add if needed for neural models
        return features


class PopularityRanker:
    """Rank ads by global CTR (popularity baseline)."""
    
    def __init__(self):
        self.ad_ctr = {}
        self.global_ctr = 0.0
        
    def fit(self, exposure_log: pd.DataFrame):
        """
        Fit popularity model.
        
        Args:
            exposure_log: Historical exposure log with clicks
        """
        # Compute CTR per ad
        ad_stats = exposure_log.groupby('ad_id').agg({
            'click': ['sum', 'count']
        })
        
        ad_stats.columns = ['clicks', 'impressions']
        ad_stats['ctr'] = ad_stats['clicks'] / ad_stats['impressions']
        
        self.ad_ctr = ad_stats['ctr'].to_dict()
        self.global_ctr = exposure_log['click'].mean()
    
    def predict_proba(self, ad_ids: List[str]) -> np.ndarray:
        """
        Predict click probabilities.
        
        Args:
            ad_ids: List of ad IDs
            
        Returns:
            Array of click probabilities
        """
        probs = [self.ad_ctr.get(ad_id, self.global_ctr) for ad_id in ad_ids]
        return np.array(probs)
    
    def rank(self, ad_ids: List[str], k: int = 3) -> List[str]:
        """
        Rank ads by predicted CTR.
        
        Args:
            ad_ids: Candidate ad IDs
            k: Number to return
            
        Returns:
            Top-k ad IDs
        """
        probs = self.predict_proba(ad_ids)
        top_k_indices = np.argsort(probs)[::-1][:k]
        return [ad_ids[i] for i in top_k_indices]


class LogisticRegressionRanker:
    """Logistic regression CTR model."""
    
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: Arguments for sklearn LogisticRegression
        """
        self.model = LogisticRegression(max_iter=1000, **kwargs)
        self.feature_builder = FeatureBuilder(categorical_encoding='label')
        
    def fit(
        self,
        exposure_log: pd.DataFrame,
        guests_df: pd.DataFrame,
        ads_df: pd.DataFrame
    ):
        """
        Fit logistic regression model.
        
        Args:
            exposure_log: Exposure log with clicks
            guests_df: Guest metadata
            ads_df: Advertiser metadata
        """
        X, y = self.feature_builder.build_training_frame(
            exposure_log, guests_df, ads_df
        )
        
        self.model.fit(X, y)
        
    def predict_proba(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        time_of_day: str = 'afternoon',
        day_of_stay: int = 1
    ) -> np.ndarray:
        """
        Predict click probabilities for candidate ads.
        
        Args:
            guest_context: Guest information (single row)
            candidate_ads: Candidate ads dataframe
            time_of_day: Time of day
            day_of_stay: Day of stay
            
        Returns:
            Click probabilities
        """
        # Build feature matrix
        features = self._build_prediction_features(
            guest_context, candidate_ads, time_of_day, day_of_stay
        )
        
        probs = self.model.predict_proba(features)[:, 1]
        return probs
    
    def rank(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        k: int = 3,
        time_of_day: str = 'afternoon',
        day_of_stay: int = 1
    ) -> List[str]:
        """Rank candidate ads for a guest."""
        probs = self.predict_proba(
            guest_context, candidate_ads, time_of_day, day_of_stay
        )
        
        top_k_indices = np.argsort(probs)[::-1][:k]
        return candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
    
    def _build_prediction_features(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        time_of_day: str,
        day_of_stay: int
    ) -> pd.DataFrame:
        """Build features for prediction."""
        n = len(candidate_ads)
        
        features = pd.DataFrame()
        
        # Replicate guest features
        features['position'] = 1  # Default position
        features['day_of_stay'] = day_of_stay
        features['total_guests'] = guest_context['total_guests']
        features['nights'] = guest_context['nights']
        features['price_per_night'] = guest_context['price_per_night']
        
        # Ad features
        features['distance_km'] = candidate_ads['distance_km'].values
        features['base_utility'] = candidate_ads['base_utility'].values
        features['revenue_per_conversion'] = candidate_ads['revenue_per_conversion'].values
        
        # Binary
        features['is_family'] = int(guest_context.get('is_family', False))
        features['is_business'] = int(guest_context.get('is_business', False))
        features['is_weekend_stay'] = int(guest_context.get('is_weekend_stay', False))
        
        # Categorical (encode)
        for col_name in ['purpose_of_stay', 'country', 'source']:
            val = guest_context.get(col_name, 'unknown')
            if col_name in self.feature_builder.label_encoders:
                le = self.feature_builder.label_encoders[col_name]
                encoded = le.transform([val])[0] if val in le.classes_ else -1
            else:
                encoded = 0
            features[col_name] = encoded
        
        # Ad categoricals
        for col_name in ['advertiser_type', 'price_level']:
            vals = candidate_ads[col_name].values
            if col_name in self.feature_builder.label_encoders:
                le = self.feature_builder.label_encoders[col_name]
                encoded = [le.transform([v])[0] if v in le.classes_ else -1 for v in vals]
            else:
                encoded = [0] * len(vals)
            features[col_name] = encoded
        
        # Time of day
        if 'time_of_day' in self.feature_builder.label_encoders:
            le = self.feature_builder.label_encoders['time_of_day']
            encoded = le.transform([time_of_day])[0] if time_of_day in le.classes_ else -1
        else:
            encoded = 0
        features['time_of_day'] = encoded
        
        # Tags
        common_tags = [
            'foodie', 'family_friendly', 'nightlife', 'culture',
            'outdoor', 'budget', 'luxury', 'wellness'
        ]
        
        for tag in common_tags:
            features[f'tag_{tag}'] = candidate_ads['category_tags'].apply(
                lambda x: 1 if tag in parse_tags_string(x) else 0
            ).values
        
        # Interactions
        features['price_match'] = 0
        features['family_friendly_match'] = (
            features['is_family'] * features['tag_family_friendly']
        )
        
        # Ensure all training features are present
        if self.feature_builder.feature_names:
            for fname in self.feature_builder.feature_names:
                if fname not in features.columns:
                    features[fname] = 0
            
            features = features[self.feature_builder.feature_names]
        
        return features


class GradientBoostingRanker:
    """Gradient boosting (XGBoost or LightGBM) CTR model."""
    
    def __init__(self, use_xgboost: bool = True, **kwargs):
        """
        Args:
            use_xgboost: Use XGBoost if True, else LightGBM
            **kwargs: Model hyperparameters
        """
        self.use_xgboost = use_xgboost
        
        if use_xgboost and not HAS_XGB:
            warnings.warn("XGBoost not available, falling back to LightGBM")
            self.use_xgboost = False
        
        if not self.use_xgboost and not HAS_LGB:
            raise ImportError("Neither XGBoost nor LightGBM available")
        
        self.model = None
        self.feature_builder = FeatureBuilder(categorical_encoding='label')
        self.kwargs = kwargs
        
    def fit(
        self,
        exposure_log: pd.DataFrame,
        guests_df: pd.DataFrame,
        ads_df: pd.DataFrame
    ):
        """Fit gradient boosting model."""
        X, y = self.feature_builder.build_training_frame(
            exposure_log, guests_df, ads_df
        )
        
        if self.use_xgboost:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                **self.kwargs
            }
            self.model = xgb.XGBClassifier(**params)
        else:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'verbose': -1,
                **self.kwargs
            }
            self.model = lgb.LGBMClassifier(**params)
        
        self.model.fit(X, y)
    
    def predict_proba(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        time_of_day: str = 'afternoon',
        day_of_stay: int = 1
    ) -> np.ndarray:
        """Predict click probabilities."""
        features = self._build_prediction_features(
            guest_context, candidate_ads, time_of_day, day_of_stay
        )
        
        probs = self.model.predict_proba(features)[:, 1]
        return probs
    
    def rank(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        k: int = 3,
        time_of_day: str = 'afternoon',
        day_of_stay: int = 1
    ) -> List[str]:
        """Rank candidate ads."""
        probs = self.predict_proba(
            guest_context, candidate_ads, time_of_day, day_of_stay
        )
        
        top_k_indices = np.argsort(probs)[::-1][:k]
        return candidate_ads.iloc[top_k_indices]['ad_id'].tolist()
    
    def _build_prediction_features(
        self,
        guest_context: pd.Series,
        candidate_ads: pd.DataFrame,
        time_of_day: str,
        day_of_stay: int
    ) -> pd.DataFrame:
        """Build features for prediction (same as LogisticRegressionRanker)."""
        n = len(candidate_ads)
        
        features = pd.DataFrame()
        
        features['position'] = 1
        features['day_of_stay'] = day_of_stay
        features['total_guests'] = guest_context['total_guests']
        features['nights'] = guest_context['nights']
        features['price_per_night'] = guest_context['price_per_night']
        
        features['distance_km'] = candidate_ads['distance_km'].values
        features['base_utility'] = candidate_ads['base_utility'].values
        features['revenue_per_conversion'] = candidate_ads['revenue_per_conversion'].values
        
        features['is_family'] = int(guest_context.get('is_family', False))
        features['is_business'] = int(guest_context.get('is_business', False))
        features['is_weekend_stay'] = int(guest_context.get('is_weekend_stay', False))
        
        for col_name in ['purpose_of_stay', 'country', 'source']:
            val = guest_context.get(col_name, 'unknown')
            if col_name in self.feature_builder.label_encoders:
                le = self.feature_builder.label_encoders[col_name]
                encoded = le.transform([val])[0] if val in le.classes_ else -1
            else:
                encoded = 0
            features[col_name] = encoded
        
        for col_name in ['advertiser_type', 'price_level']:
            vals = candidate_ads[col_name].values
            if col_name in self.feature_builder.label_encoders:
                le = self.feature_builder.label_encoders[col_name]
                encoded = [le.transform([v])[0] if v in le.classes_ else -1 for v in vals]
            else:
                encoded = [0] * len(vals)
            features[col_name] = encoded
        
        if 'time_of_day' in self.feature_builder.label_encoders:
            le = self.feature_builder.label_encoders['time_of_day']
            encoded = le.transform([time_of_day])[0] if time_of_day in le.classes_ else -1
        else:
            encoded = 0
        features['time_of_day'] = encoded
        
        common_tags = [
            'foodie', 'family_friendly', 'nightlife', 'culture',
            'outdoor', 'budget', 'luxury', 'wellness'
        ]
        
        for tag in common_tags:
            features[f'tag_{tag}'] = candidate_ads['category_tags'].apply(
                lambda x: 1 if tag in parse_tags_string(x) else 0
            ).values
        
        features['price_match'] = 0
        features['family_friendly_match'] = (
            features['is_family'] * features['tag_family_friendly']
        )
        
        if self.feature_builder.feature_names:
            for fname in self.feature_builder.feature_names:
                if fname not in features.columns:
                    features[fname] = 0
            
            features = features[self.feature_builder.feature_names]
        
        return features

