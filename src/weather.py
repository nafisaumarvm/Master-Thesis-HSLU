"""
Weather-based targeting and contextual features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from .utils import set_random_seed


# Weather conditions
WEATHER_CONDITIONS = ['sunny', 'cloudy', 'rainy', 'snowy']

# Weather-appropriate advertiser types
WEATHER_PREFERENCES = {
    'sunny': {
        'preferred': ['tour', 'attraction', 'outdoor', 'transport'],
        'boosted_tags': ['outdoor', 'adventure', 'touristy'],
        'boost_factor': 1.5
    },
    'rainy': {
        'preferred': ['museum', 'gallery', 'cafe', 'spa', 'restaurant'],
        'boosted_tags': ['culture', 'quiet', 'wellness'],
        'boost_factor': 1.3
    },
    'cloudy': {
        'preferred': ['museum', 'gallery', 'restaurant', 'bar', 'shopping'],
        'boosted_tags': ['culture', 'foodie', 'shopping'],
        'boost_factor': 1.1
    },
    'snowy': {
        'preferred': ['spa', 'restaurant', 'cafe', 'bar'],
        'boosted_tags': ['wellness', 'quiet', 'romantic'],
        'boost_factor': 1.2
    }
}


def generate_weather_context(
    dates: pd.Series,
    location: str = 'generic',
    seed: int = 42
) -> pd.Series:
    """
    Generate weather conditions for given dates.
    
    Args:
        dates: Series of datetime objects
        location: Location identifier (for future seasonality)
        seed: Random seed
        
    Returns:
        Series of weather conditions
    """
    rng = set_random_seed(seed)
    
    # Simple weather generation based on month
    weather = []
    
    for date in dates:
        month = date.month if hasattr(date, 'month') else 6
        
        # Seasonal probabilities
        if month in [12, 1, 2]:  # Winter
            probs = [0.2, 0.3, 0.3, 0.2]  # sunny, cloudy, rainy, snowy
        elif month in [3, 4, 5]:  # Spring
            probs = [0.3, 0.3, 0.3, 0.1]
        elif month in [6, 7, 8]:  # Summer
            probs = [0.5, 0.3, 0.15, 0.05]
        else:  # Fall
            probs = [0.25, 0.35, 0.35, 0.05]
        
        weather_condition = rng.choice(WEATHER_CONDITIONS, p=probs)
        weather.append(weather_condition)
    
    return pd.Series(weather, index=dates.index)


def apply_weather_boost(
    ad_scores: pd.Series,
    ads_df: pd.DataFrame,
    weather: str
) -> pd.Series:
    """
    Apply weather-based boost to ad scores.
    
    Args:
        ad_scores: Current ad scores (indexed by ad_id)
        ads_df: Advertiser dataframe
        weather: Current weather condition
        
    Returns:
        Boosted ad scores
    """
    from .utils import parse_tags_string
    
    if weather not in WEATHER_PREFERENCES:
        return ad_scores
    
    prefs = WEATHER_PREFERENCES[weather]
    boosted_scores = ad_scores.copy()
    
    for ad_id in ad_scores.index:
        ad_info = ads_df[ads_df['ad_id'] == ad_id]
        
        if len(ad_info) == 0:
            continue
        
        ad_info = ad_info.iloc[0]
        
        # Check if advertiser type matches weather
        if ad_info['advertiser_type'] in prefs['preferred']:
            boosted_scores[ad_id] *= prefs['boost_factor']
        
        # Check if tags match weather
        ad_tags = parse_tags_string(ad_info['category_tags'])
        if any(tag in prefs['boosted_tags'] for tag in ad_tags):
            boosted_scores[ad_id] *= 1.1
    
    return boosted_scores


def filter_ads_by_weather(
    ads_df: pd.DataFrame,
    weather: str,
    min_relevance: float = 0.5
) -> pd.DataFrame:
    """
    Filter ads suitable for current weather.
    
    Args:
        ads_df: Advertiser catalogue
        weather: Current weather condition
        min_relevance: Minimum relevance threshold
        
    Returns:
        Filtered advertiser dataframe
    """
    if weather not in WEATHER_PREFERENCES:
        return ads_df
    
    prefs = WEATHER_PREFERENCES[weather]
    
    # Calculate weather relevance score
    def weather_relevance(row):
        score = 0.0
        
        if row['advertiser_type'] in prefs['preferred']:
            score += 0.7
        
        from .utils import parse_tags_string
        ad_tags = parse_tags_string(row['category_tags'])
        matching_tags = sum(1 for tag in ad_tags if tag in prefs['boosted_tags'])
        score += 0.3 * (matching_tags / max(len(prefs['boosted_tags']), 1))
        
        return max(score, 0.3)  # Minimum relevance
    
    ads_df = ads_df.copy()
    ads_df['weather_relevance'] = ads_df.apply(weather_relevance, axis=1)
    
    # Filter by minimum relevance
    filtered = ads_df[ads_df['weather_relevance'] >= min_relevance]
    
    return filtered


def add_weather_features(
    exposure_log: pd.DataFrame,
    weather_by_session: Optional[Dict[str, str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Add weather features to exposure log.
    
    Args:
        exposure_log: Exposure log
        weather_by_session: Optional dict mapping session_id to weather
        seed: Random seed
        
    Returns:
        Exposure log with weather features
    """
    rng = set_random_seed(seed)
    
    log = exposure_log.copy()
    
    if weather_by_session is None:
        # Generate random weather per session
        unique_sessions = log['session_id'].unique()
        weather_by_session = {
            session: rng.choice(WEATHER_CONDITIONS)
            for session in unique_sessions
        }
    
    log['weather'] = log['session_id'].map(weather_by_session)
    
    # Add weather-specific features
    log['is_weather_suitable'] = log.apply(
        lambda row: (
            row.get('advertiser_type', '') in 
            WEATHER_PREFERENCES.get(row['weather'], {}).get('preferred', [])
        ) if pd.notna(row.get('weather')) else False,
        axis=1
    )
    
    return log





