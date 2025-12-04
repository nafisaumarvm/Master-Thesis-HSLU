"""
Medium-Priority Enhancements for In-Room TV Advertising System

Implements:
5. Weather/time-of-day distributions
6. Advertiser inventory constraints
7. Creative fatigue effects
8. Guest opt-out behavior (optional)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# ============================================================================
# 5. WEATHER AND TIME-OF-DAY DISTRIBUTIONS
# ============================================================================

def generate_weather_distribution(
    num_days: int,
    location: str = 'temperate',
    season: str = 'mixed',
    rng: np.random.Generator = None
) -> pd.DataFrame:
    """
    Generate realistic weather patterns for simulation.
    
    Medium Enhancement #5: Weather distributions
    
    Args:
        num_days: Number of days to simulate
        location: Climate type ('temperate', 'tropical', 'mediterranean')
        season: Season ('winter', 'spring', 'summer', 'fall', 'mixed')
        rng: Random number generator
        
    Returns:
        DataFrame with columns:
        - date: Date
        - weather: Category ('sunny', 'partly_cloudy', 'rainy', 'stormy')
        - temperature: Degrees Celsius
        - precipitation_prob: Probability of rain [0, 1]
        
    Validates context interactions like:
    - Rainy + Museum ad → +61% attention
    - Sunny + Outdoor tour → +57% attention
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Weather probabilities by location and season
    weather_probs = {
        'temperate': {
            'winter': {'sunny': 0.30, 'partly_cloudy': 0.35, 'rainy': 0.25, 'stormy': 0.10},
            'spring': {'sunny': 0.45, 'partly_cloudy': 0.30, 'rainy': 0.20, 'stormy': 0.05},
            'summer': {'sunny': 0.65, 'partly_cloudy': 0.25, 'rainy': 0.08, 'stormy': 0.02},
            'fall': {'sunny': 0.40, 'partly_cloudy': 0.35, 'rainy': 0.20, 'stormy': 0.05},
            'mixed': {'sunny': 0.45, 'partly_cloudy': 0.32, 'rainy': 0.18, 'stormy': 0.05}
        },
        'mediterranean': {
            'summer': {'sunny': 0.85, 'partly_cloudy': 0.12, 'rainy': 0.02, 'stormy': 0.01},
            'winter': {'sunny': 0.50, 'partly_cloudy': 0.30, 'rainy': 0.15, 'stormy': 0.05},
            'mixed': {'sunny': 0.65, 'partly_cloudy': 0.22, 'rainy': 0.10, 'stormy': 0.03}
        },
        'tropical': {
            'summer': {'sunny': 0.40, 'partly_cloudy': 0.30, 'rainy': 0.25, 'stormy': 0.05},
            'winter': {'sunny': 0.60, 'partly_cloudy': 0.25, 'rainy': 0.12, 'stormy': 0.03},
            'mixed': {'sunny': 0.50, 'partly_cloudy': 0.28, 'rainy': 0.18, 'stormy': 0.04}
        }
    }
    
    # Get probabilities
    loc_probs = weather_probs.get(location, weather_probs['temperate'])
    season_probs = loc_probs.get(season, loc_probs['mixed'])
    
    # Generate weather sequence
    weather_categories = list(season_probs.keys())
    probs = list(season_probs.values())
    
    weather_seq = rng.choice(weather_categories, size=num_days, p=probs)
    
    # Generate temperatures (vary by weather)
    temp_base = {
        'winter': 8, 'spring': 15, 'summer': 25, 'fall': 12, 'mixed': 18
    }
    base_temp = temp_base.get(season, 18)
    
    temp_modifiers = {
        'sunny': 3,
        'partly_cloudy': 0,
        'rainy': -3,
        'stormy': -5
    }
    
    temperatures = []
    precip_probs = []
    
    for w in weather_seq:
        temp = base_temp + temp_modifiers[w] + rng.normal(0, 2)
        temperatures.append(round(temp, 1))
        
        # Precipitation probability
        precip_map = {
            'sunny': 0.05,
            'partly_cloudy': 0.25,
            'rainy': 0.70,
            'stormy': 0.95
        }
        precip_probs.append(precip_map[w])
    
    # Create DataFrame
    dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'weather': weather_seq,
        'temperature': temperatures,
        'precipitation_prob': precip_probs
    })


def get_weather_ad_boost(
    weather: str,
    ad_category: str
) -> float:
    """
    Calculate ad attention boost based on weather-category match.
    
    Returns multiplier in [0.5, 2.0]:
    - 1.0 = neutral
    - >1.0 = boosted (good match)
    - <1.0 = penalty (bad match)
    
    Examples:
    - Rainy + Museum → 1.6× boost
    - Sunny + Beach tour → 1.8× boost
    - Rainy + Outdoor tour → 0.6× penalty
    """
    # Weather-category affinity matrix
    affinities = {
        'sunny': {
            'tour': 1.8,
            'attraction_outdoor': 1.7,
            'beach': 1.9,
            'spa': 0.9,
            'restaurant': 1.1,
            'museum': 0.7,
            'cafe': 1.2
        },
        'partly_cloudy': {
            'tour': 1.2,
            'attraction_outdoor': 1.1,
            'spa': 1.1,
            'restaurant': 1.0,
            'museum': 1.0,
            'cafe': 1.0
        },
        'rainy': {
            'museum': 1.6,
            'spa': 1.5,
            'restaurant': 1.3,
            'cafe': 1.4,
            'tour': 0.6,
            'attraction_outdoor': 0.5
        },
        'stormy': {
            'museum': 1.7,
            'spa': 1.6,
            'restaurant': 1.2,
            'cafe': 1.1,
            'tour': 0.3,
            'attraction_outdoor': 0.3
        }
    }
    
    # Normalize category names
    cat_normalized = ad_category.lower()
    for key in ['tour', 'museum', 'spa', 'restaurant', 'cafe', 'beach', 'attraction']:
        if key in cat_normalized:
            cat_normalized = key
            break
    
    return affinities.get(weather, {}).get(cat_normalized, 1.0)


# ============================================================================
# 6. ADVERTISER INVENTORY CONSTRAINTS
# ============================================================================

class AdvertiserInventoryManager:
    """
    Manage advertiser budget and exposure constraints.
    
    Medium Enhancement #6: Realistic advertiser constraints
    
    Advertisers have:
    - Daily exposure caps (e.g., restaurant: max 40 ads/day)
    - Budget limits (max spend per week/month)
    - Time-of-day preferences (e.g., restaurants avoid morning)
    """
    
    def __init__(self, advertisers_df: pd.DataFrame):
        """
        Initialize with advertiser catalog.
        
        Expected columns:
        - advertiser_id
        - category
        - daily_cap (optional, will generate if missing)
        - budget_per_week (optional)
        """
        self.advertisers = advertisers_df.copy()
        
        # Set default caps if not provided
        if 'daily_cap' not in self.advertisers.columns:
            self.advertisers['daily_cap'] = self.advertisers['category'].apply(
                self._default_daily_cap
            )
        
        # Initialize tracking
        self.daily_exposures = {aid: 0 for aid in self.advertisers['advertiser_id']}
        self.weekly_exposures = {aid: 0 for aid in self.advertisers['advertiser_id']}
        self.current_day = 0
        self.current_week = 0
    
    def _default_daily_cap(self, category: str) -> int:
        """Set reasonable daily exposure caps by category."""
        caps = {
            'restaurant': 40,
            'tour': 30,
            'attraction': 50,
            'spa': 25,
            'cafe': 35,
            'bar': 30,
            'experience': 20,
            'event': 15
        }
        
        for key, cap in caps.items():
            if key in category.lower():
                return cap
        
        return 30  # Default
    
    def can_show_ad(self, advertiser_id: str, time_of_day: str = 'evening') -> bool:
        """
        Check if advertiser can show ad given constraints.
        
        Args:
            advertiser_id: Advertiser ID
            time_of_day: Time category
            
        Returns:
            True if ad can be shown, False if cap reached
        """
        # Check daily cap
        if self.daily_exposures.get(advertiser_id, 0) >= self._get_cap(advertiser_id):
            return False
        
        # Check time-of-day restrictions (e.g., restaurants avoid morning)
        category = self._get_category(advertiser_id)
        if not self._time_allowed(category, time_of_day):
            return False
        
        return True
    
    def _get_cap(self, advertiser_id: str) -> int:
        """Get daily cap for advertiser."""
        row = self.advertisers[self.advertisers['advertiser_id'] == advertiser_id]
        if len(row) > 0:
            return row.iloc[0]['daily_cap']
        return 30
    
    def _get_category(self, advertiser_id: str) -> str:
        """Get category for advertiser."""
        row = self.advertisers[self.advertisers['advertiser_id'] == advertiser_id]
        if len(row) > 0:
            return row.iloc[0].get('category', 'other')
        return 'other'
    
    def _time_allowed(self, category: str, time_of_day: str) -> bool:
        """Check if category should show ads at this time."""
        restrictions = {
            'restaurant': {'morning': 0.3, 'afternoon': 0.8, 'evening': 1.0, 'late_night': 0.5},
            'tour': {'morning': 1.0, 'afternoon': 0.9, 'evening': 0.2, 'late_night': 0.0},
            'cafe': {'morning': 1.0, 'afternoon': 0.8, 'evening': 0.4, 'late_night': 0.2},
            'spa': {'morning': 0.7, 'afternoon': 1.0, 'evening': 0.8, 'late_night': 0.3}
        }
        
        for key, probs in restrictions.items():
            if key in category.lower():
                # Probabilistic restriction
                return np.random.random() < probs.get(time_of_day, 1.0)
        
        return True  # No restriction
    
    def record_exposure(self, advertiser_id: str):
        """Record that ad was shown."""
        self.daily_exposures[advertiser_id] = self.daily_exposures.get(advertiser_id, 0) + 1
        self.weekly_exposures[advertiser_id] = self.weekly_exposures.get(advertiser_id, 0) + 1
    
    def new_day(self):
        """Reset daily counters."""
        self.daily_exposures = {aid: 0 for aid in self.advertisers['advertiser_id']}
        self.current_day += 1
        
        if self.current_day % 7 == 0:
            self.new_week()
    
    def new_week(self):
        """Reset weekly counters."""
        self.weekly_exposures = {aid: 0 for aid in self.advertisers['advertiser_id']}
        self.current_week += 1
    
    def get_available_advertisers(
        self,
        candidate_ids: List[str],
        time_of_day: str = 'evening'
    ) -> List[str]:
        """
        Filter candidate advertisers by constraints.
        
        Args:
            candidate_ids: List of candidate advertiser IDs
            time_of_day: Time category
            
        Returns:
            Filtered list of available advertiser IDs
        """
        return [aid for aid in candidate_ids if self.can_show_ad(aid, time_of_day)]


# ============================================================================
# 7. CREATIVE FATIGUE EFFECT
# ============================================================================

def compute_creative_fatigue(
    exposures_to_same_ad: int,
    fatigue_threshold: int = 5,
    fatigue_rate: float = 0.15
) -> float:
    """
    Model creative fatigue: ads become less effective with repeated exposure.
    
    Medium Enhancement #7: Creative fatigue
    
    Known from advertising science:
    - Initial exposures: High effectiveness
    - 3-7 exposures: Optimal (awareness builds)
    - 8+ exposures: Fatigue sets in (wear-out effect)
    
    Args:
        exposures_to_same_ad: Number of times guest saw this specific ad
        fatigue_threshold: When fatigue begins (default: 5)
        fatigue_rate: How fast effectiveness drops (default: 0.15 per exposure)
        
    Returns:
        Effectiveness multiplier in [0.3, 1.0]:
        - 1.0 = no fatigue
        - 0.3 = severe fatigue (minimum effectiveness)
        
    Theory:
        Based on Pechmann & Stewart (1988) wear-out effects and
        Naik et al. (1998) advertising pulsing strategies.
        
    Example:
        >>> compute_creative_fatigue(3)   # Below threshold
        1.0
        >>> compute_creative_fatigue(8)   # Fatigued
        0.55
        >>> compute_creative_fatigue(15)  # Severe fatigue
        0.30
    """
    if exposures_to_same_ad <= fatigue_threshold:
        return 1.0  # No fatigue yet
    
    excess = exposures_to_same_ad - fatigue_threshold
    effectiveness = 1.0 - (fatigue_rate * excess)
    
    # Floor at 0.3 (never completely ineffective)
    return max(0.3, effectiveness)


def apply_fatigue_to_awareness_growth(
    base_alpha: float,
    exposures_to_ad: int
) -> float:
    """
    Apply creative fatigue to awareness growth rate.
    
    Instead of constant α, it decreases with over-exposure:
        α_effective = α_base × fatigue_multiplier
        
    This makes the awareness model more realistic:
    - Early exposures: Fast awareness growth
    - Later exposures: Diminishing returns + fatigue
    """
    fatigue_mult = compute_creative_fatigue(exposures_to_ad)
    return base_alpha * fatigue_mult


# ============================================================================
# 8. GUEST OPT-OUT BEHAVIOR (OPTIONAL)
# ============================================================================

def simulate_guest_attention_drop(
    intrusion_cost: float,
    segment: str,
    rng: np.random.Generator
) -> bool:
    """
    Optional Enhancement #8: Guest channel-switching behavior
    
    Guests may switch channels or turn off TV if ads are too intrusive.
    
    Args:
        intrusion_cost: Current intrusion cost [0, 1]
        segment: Guest segment
        rng: Random number generator
        
    Returns:
        True if guest switches away (attention lost)
        False if guest continues watching
    """
    # Segment-specific tolerance thresholds
    tolerance = {
        'Business Traveler': 0.15,  # Low tolerance
        'Luxury Leisure': 0.25,
        'Cultural Tourist': 0.30,
        'Weekend Explorer': 0.30,
        'Adventure Seeker': 0.35,
        'Budget Family': 0.40,      # High tolerance
        'Extended Stay': 0.45
    }
    
    threshold = tolerance.get(segment, 0.30)
    
    if intrusion_cost > threshold:
        # Probability of switching away increases with excess intrusion
        excess = intrusion_cost - threshold
        switch_prob = min(0.8, excess * 2.0)
        return rng.random() < switch_prob
    
    return False


def compute_attention_dilution(
    num_ads_in_session: int,
    optimal_ads_per_session: int = 2
) -> float:
    """
    Model attention dilution: too many ads in one session reduces effectiveness.
    
    Optimal: 2-3 ads per TV session
    Over-saturated: 5+ ads → attention drops
    
    Returns:
        Attention multiplier [0.4, 1.0]
    """
    if num_ads_in_session <= optimal_ads_per_session:
        return 1.0
    
    excess = num_ads_in_session - optimal_ads_per_session
    dilution = 1.0 - (0.15 * excess)
    
    return max(0.4, dilution)


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

def run_enhanced_simulation_example():
    """
    Example showing how to use all medium enhancements together.
    """
    print("Medium Enhancements Integration Example")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    # 1. Generate weather
    print("\n1. Weather Distribution (5 days):")
    weather_df = generate_weather_distribution(5, location='temperate', season='summer', rng=rng)
    print(weather_df.to_string(index=False))
    
    # 2. Weather-ad boost
    print("\n2. Weather-Ad Affinity:")
    test_cases = [
        ('sunny', 'tour'),
        ('rainy', 'museum'),
        ('rainy', 'tour'),
        ('sunny', 'spa')
    ]
    for weather, category in test_cases:
        boost = get_weather_ad_boost(weather, category)
        print(f"  {weather:15s} + {category:10s} → {boost:.2f}× boost")
    
    # 3. Advertiser constraints
    print("\n3. Advertiser Inventory Management:")
    sample_ads = pd.DataFrame({
        'advertiser_id': ['rest_001', 'tour_001', 'spa_001'],
        'category': ['restaurant', 'tour', 'spa'],
        'daily_cap': [40, 30, 25]
    })
    
    manager = AdvertiserInventoryManager(sample_ads)
    
    # Simulate showing ads
    for i in range(45):
        if manager.can_show_ad('rest_001', 'evening'):
            manager.record_exposure('rest_001')
    
    print(f"  Restaurant exposures: {manager.daily_exposures['rest_001']} (cap: 40)")
    print(f"  Can show more? {manager.can_show_ad('rest_001', 'evening')}")
    
    # 4. Creative fatigue
    print("\n4. Creative Fatigue:")
    for exp in [1, 3, 5, 8, 12, 20]:
        effectiveness = compute_creative_fatigue(exp)
        print(f"  After {exp:2d} exposures → {effectiveness:.2f}× effectiveness")
    
    # 5. Opt-out behavior
    print("\n5. Guest Opt-Out Simulation:")
    for intrusion in [0.1, 0.3, 0.5, 0.8]:
        switched = simulate_guest_attention_drop(intrusion, 'Business Traveler', rng)
        print(f"  Intrusion {intrusion:.1f} → Switched away: {switched}")
    
    print("\n✅ All medium enhancements demonstrated!")


if __name__ == "__main__":
    run_enhanced_simulation_example()





