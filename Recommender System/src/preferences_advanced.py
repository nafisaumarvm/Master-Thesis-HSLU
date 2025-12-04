"""
Advanced Preference Modeling with Critical Improvements

Implements 5 key enhancements:
1. Awareness decay (forgetting)
2. Segment-specific learning rates
3. Preference drift over stay
4. Multi-objective optimization
5. Contextual interactions (weather × time × segment)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from .preferences import (
    compute_base_utility,
    SEGMENT_CATEGORY_AFFINITIES
)


# 1. SEGMENT-SPECIFIC AWARENESS PARAMETERS
SEGMENT_AWARENESS_PARAMS = {
    'luxury_leisure': {
        'alpha': 0.40,  # Fast learners, responsive to ads
        'delta': 0.05,  # Slow decay (remember well)
        'beta': 0.60    # Strong awareness effect
    },
    'cultural_tourist': {
        'alpha': 0.35,  # Moderate learning
        'delta': 0.03,  # Low decay (culture seekers remember)
        'beta': 0.50    # Moderate effect
    },
    'business_traveler': {
        'alpha': 0.15,  # Slow learning (low interest)
        'delta': 0.10,  # Fast decay (busy, forget quickly)
        'beta': 0.30    # Weak awareness effect
    },
    'weekend_explorer': {
        'alpha': 0.45,  # Fast learning (explorers)
        'delta': 0.08,  # Moderate decay
        'beta': 0.55    # Strong effect
    },
    'budget_family': {
        'alpha': 0.30,  # Moderate learning
        'delta': 0.04,  # Low decay (families plan)
        'beta': 0.45    # Moderate effect
    },
    'adventure_seeker': {
        'alpha': 0.50,  # Very fast learning (seekers)
        'delta': 0.06,  # Moderate decay
        'beta': 0.65    # Very strong effect
    },
    'extended_stay': {
        'alpha': 0.25,  # Slower learning (long-term)
        'delta': 0.02,  # Very slow decay (remember)
        'beta': 0.40    # Moderate effect
    },
    'bargain_hunter': {
        'alpha': 0.20,  # Slow learning (price-focused)
        'delta': 0.12,  # Fast decay (transactional)
        'beta': 0.25    # Weak effect
    }
}

# Default for unknown segments
DEFAULT_AWARENESS_PARAMS = {
    'alpha': 0.30,
    'delta': 0.05,
    'beta': 0.50
}


def update_awareness_advanced(
    current_awareness: float,
    was_exposed: bool,
    segment: str,
    custom_params: Optional[Dict] = None
) -> float:
    """
    Update awareness with segment-specific parameters and decay.
    
    Formula:
    - If exposed: ρ(t+1) = ρ(t) + α[segment] * (1 - ρ(t))
    - If not exposed: ρ(t+1) = ρ(t) * (1 - δ[segment])
    
    Parameters
    ----------
    current_awareness : float
        Current awareness ρ(t) ∈ [0, 1]
    was_exposed : bool
        Whether ad was shown
    segment : str
        Guest segment
    custom_params : dict, optional
        Override default params
        
    Returns
    -------
    float
        Updated awareness ρ(t+1)
    """
    # Get segment-specific parameters
    if custom_params:
        params = custom_params
    else:
        params = SEGMENT_AWARENESS_PARAMS.get(segment, DEFAULT_AWARENESS_PARAMS)
    
    alpha = params['alpha']  # Growth rate
    delta = params['delta']  # Decay rate
    
    if was_exposed:
        # Growth (van Leeuwen formula)
        new_awareness = current_awareness + alpha * (1 - current_awareness)
    else:
        # Decay (forgetting)
        new_awareness = current_awareness * (1 - delta)
    
    return np.clip(new_awareness, 0.0, 1.0)


# 2. PREFERENCE DRIFT OVER STAY
def compute_preference_drift(
    base_affinity: float,
    day_of_stay: int,
    total_nights: int,
    drift_type: str = 'exploration_fatigue'
) -> float:
    """
    Model preference drift across stay duration.
    
    Patterns:
    - Day 1-2: High exploration (boost novelty)
    - Day 3-5: Routine establishment (boost familiar)
    - Day 7+: Fatigue (reduce engagement)
    
    Parameters
    ----------
    base_affinity : float
        Base preference for category
    day_of_stay : int
        Current day (1-indexed)
    total_nights : int
        Total stay length
    drift_type : str
        'exploration_fatigue' or 'linear_decline'
        
    Returns
    -------
    float
        Adjusted affinity
    """
    if drift_type == 'exploration_fatigue':
        # Early exploration, mid routine, late fatigue
        stay_fraction = day_of_stay / max(total_nights, 1)
        
        if stay_fraction <= 0.2:  # First 20% (days 1-2 for 10-night stay)
            # High exploration phase
            drift = 0.15 * (1 - base_affinity)  # Boost novel items
        elif stay_fraction <= 0.6:  # Middle 60% (days 3-6)
            # Routine phase
            drift = 0.10 * base_affinity  # Boost familiar items
        else:  # Last 40% (days 7+)
            # Fatigue phase
            drift = -0.20 * stay_fraction  # Reduce all engagement
        
    elif drift_type == 'linear_decline':
        # Simple linear fatigue
        drift = -0.03 * (day_of_stay / max(total_nights, 1))
    
    else:
        drift = 0.0
    
    return np.clip(base_affinity + drift, 0.0, 1.0)


# 3. CONTEXTUAL INTERACTIONS
def compute_context_interactions(
    segment: str,
    weather: str,
    time_of_day: str,
    category: str,
    day_of_stay: int = 1
) -> float:
    """
    Compute interaction effects between context dimensions.
    
    Examples:
    - Rainy + Budget Family + Museum → High boost
    - Sunny + Luxury + Spa Outdoor → High boost
    - Evening + Weekend Explorer + Nightlife → High boost
    
    Parameters
    ----------
    segment : str
        Guest segment
    weather : str
        Weather condition
    time_of_day : str
        Time of day
    category : str
        Advertiser category
    day_of_stay : int
        Day in stay
        
    Returns
    -------
    float
        Interaction boost (additive to utility)
    """
    boost = 0.0
    
    # Weather × Segment × Category interactions
    if weather == 'rainy':
        if segment in ['budget_family', 'cultural_tourist']:
            if category in ['museum', 'gallery', 'cafe']:
                boost += 0.40  # Perfect match: indoor activities for families
        
        if segment == 'luxury_leisure':
            if category == 'spa':
                boost += 0.50  # Luxury spa on rainy day
    
    elif weather == 'sunny':
        if segment in ['adventure_seeker', 'weekend_explorer']:
            if category in ['tour', 'attraction', 'experience']:
                boost += 0.45  # Outdoor activities for explorers
        
        if segment == 'luxury_leisure':
            if category in ['tour', 'attraction']:
                boost += 0.35  # Luxury outdoor experiences
    
    # Time × Segment × Category interactions
    if time_of_day == 'morning':
        if segment == 'business_traveler':
            if category == 'cafe':
                boost += 0.40  # Morning coffee for business
    
    elif time_of_day == 'evening':
        if segment in ['weekend_explorer', 'cultural_tourist']:
            if category in ['restaurant', 'bar', 'nightlife']:
                boost += 0.35  # Evening social activities
        
        if segment == 'luxury_leisure':
            if category == 'restaurant':
                boost += 0.40  # Fine dining
    
    elif time_of_day == 'late_night':
        if segment == 'weekend_explorer':
            if category == 'nightlife':
                boost += 0.50  # Late night for weekend explorers
    
    # Early stay × Exploration
    if day_of_stay <= 2:
        if segment in ['adventure_seeker', 'cultural_tourist', 'weekend_explorer']:
            if category in ['attraction', 'tour', 'museum']:
                boost += 0.20  # Early exploration boost
    
    # Late stay × Routine
    if day_of_stay >= 7:
        if category in ['cafe', 'restaurant']:
            boost += 0.10  # Familiar routine items
    
    return boost


# 4. MULTI-OBJECTIVE REWARD
def compute_multi_objective_reward(
    click: bool,
    revenue: float,
    awareness_before: float,
    awareness_after: float,
    frequency: int,
    category_diversity: float,
    lambda_revenue: float = 1.0,
    lambda_awareness: float = 0.3,
    lambda_intrusion: float = 0.1,
    lambda_diversity: float = 0.2,
    f_max: int = 5
) -> Dict[str, float]:
    """
    Compute multi-objective reward for Pareto optimization.
    
    Objectives:
    1. Revenue maximization
    2. Awareness building
    3. Intrusion minimization
    4. Diversity promotion
    
    Parameters
    ----------
    click : bool
        Whether ad was clicked
    revenue : float
        Revenue from click
    awareness_before : float
        Awareness before exposure
    awareness_after : float
        Awareness after exposure
    frequency : int
        How many times ad shown to this guest
    category_diversity : float
        Diversity score (0-1, higher = more diverse)
    lambda_* : float
        Objective weights
    f_max : int
        Maximum acceptable frequency
        
    Returns
    -------
    dict
        Individual objectives and weighted total
    """
    # Objective 1: Revenue
    revenue_obj = click * revenue
    
    # Objective 2: Awareness building
    awareness_gain = awareness_after - awareness_before
    awareness_obj = awareness_gain
    
    # Objective 3: Intrusion cost (negative)
    intrusion = max(0, frequency - f_max)
    intrusion_obj = -intrusion
    
    # Objective 4: Diversity (positive)
    diversity_obj = category_diversity
    
    # Weighted combination
    total_reward = (
        lambda_revenue * revenue_obj +
        lambda_awareness * awareness_obj +
        lambda_intrusion * intrusion_obj +
        lambda_diversity * diversity_obj
    )
    
    return {
        'revenue': revenue_obj,
        'awareness': awareness_obj,
        'intrusion': intrusion_obj,
        'diversity': diversity_obj,
        'total': total_reward
    }


def generate_pareto_frontier(
    exposure_log: pd.DataFrame,
    lambda_revenue_range: List[float] = None,
    lambda_awareness_range: List[float] = None
) -> pd.DataFrame:
    """
    Generate Pareto frontier for revenue vs. awareness trade-off.
    
    Parameters
    ----------
    exposure_log : pd.DataFrame
        Exposure log with clicks, revenue, awareness
    lambda_revenue_range : list
        Range of revenue weights to test
    lambda_awareness_range : list
        Range of awareness weights to test
        
    Returns
    -------
    pd.DataFrame
        Pareto frontier points
    """
    if lambda_revenue_range is None:
        lambda_revenue_range = np.linspace(0, 1, 11)
    
    if lambda_awareness_range is None:
        lambda_awareness_range = np.linspace(0, 1, 11)
    
    frontier = []
    
    for lambda_rev in lambda_revenue_range:
        for lambda_aware in lambda_awareness_range:
            # Normalize so they sum to 1
            total = lambda_rev + lambda_aware
            if total == 0:
                continue
            
            lambda_rev_norm = lambda_rev / total
            lambda_aware_norm = lambda_aware / total
            
            # Compute total objective
            total_revenue = exposure_log['revenue'].sum()
            total_awareness_gain = (
                exposure_log['awareness_after'] - exposure_log['awareness_before']
            ).sum()
            
            objective = (
                lambda_rev_norm * total_revenue +
                lambda_aware_norm * total_awareness_gain
            )
            
            frontier.append({
                'lambda_revenue': lambda_rev_norm,
                'lambda_awareness': lambda_aware_norm,
                'total_revenue': total_revenue,
                'total_awareness_gain': total_awareness_gain,
                'objective': objective
            })
    
    return pd.DataFrame(frontier)


# 5. ADVANCED UTILITY COMPUTATION
def compute_advanced_utility(
    guest_segment: str,
    advertiser_category: str,
    guest_context: Dict,
    advertiser_attrs: Dict,
    day_of_stay: int = 1,
    total_nights: int = 1,
    include_drift: bool = True,
    include_interactions: bool = True
) -> Dict[str, float]:
    """
    Compute utility with all advanced features.
    
    Includes:
    - Base utility (van Leeuwen)
    - Preference drift
    - Contextual interactions
    
    Parameters
    ----------
    guest_segment : str
        Guest segment
    advertiser_category : str
        Advertiser category
    guest_context : dict
        Context (weather, time, etc.)
    advertiser_attrs : dict
        Advertiser attributes
    day_of_stay : int
        Current day in stay
    total_nights : int
        Total stay length
    include_drift : bool
        Whether to add preference drift
    include_interactions : bool
        Whether to add context interactions
        
    Returns
    -------
    dict
        Utility components
    """
    # Base utility (original van Leeuwen)
    base_utility = compute_base_utility(
        guest_segment,
        advertiser_category,
        guest_context,
        advertiser_attrs
    )
    
    # Get base affinity
    affinities = SEGMENT_CATEGORY_AFFINITIES.get(guest_segment, {})
    base_affinity = affinities.get(advertiser_category, 0.5)
    
    # Preference drift
    drift_boost = 0.0
    if include_drift:
        adjusted_affinity = compute_preference_drift(
            base_affinity,
            day_of_stay,
            total_nights
        )
        drift_boost = adjusted_affinity - base_affinity
    
    # Context interactions
    interaction_boost = 0.0
    if include_interactions:
        interaction_boost = compute_context_interactions(
            guest_segment,
            guest_context.get('weather', 'sunny'),
            guest_context.get('time_of_day', 'afternoon'),
            advertiser_category,
            day_of_stay
        )
    
    # Total utility
    total_utility = base_utility + drift_boost + interaction_boost
    
    return {
        'base_utility': base_utility,
        'drift_boost': drift_boost,
        'interaction_boost': interaction_boost,
        'total_utility': total_utility
    }


def get_awareness_params_summary() -> pd.DataFrame:
    """
    Get summary of segment-specific awareness parameters.
    
    Returns
    -------
    pd.DataFrame
        Summary table
    """
    data = []
    for segment, params in SEGMENT_AWARENESS_PARAMS.items():
        data.append({
            'segment': segment,
            'alpha_growth': params['alpha'],
            'delta_decay': params['delta'],
            'beta_effect': params['beta'],
            'half_life_days': -np.log(0.5) / params['delta'] if params['delta'] > 0 else np.inf
        })
    
    return pd.DataFrame(data)


# Example usage
if __name__ == '__main__':
    print("ADVANCED PREFERENCE MODEL - KEY FEATURES")
    print("="*70)
    
    print("\n1. SEGMENT-SPECIFIC AWARENESS PARAMETERS:")
    params_df = get_awareness_params_summary()
    print(params_df.to_string(index=False))
    
    print("\n2. AWARENESS DYNAMICS WITH DECAY:")
    segment = 'luxury_leisure'
    awareness = 0.5
    print(f"   Initial awareness: {awareness:.3f}")
    
    # Exposure
    awareness = update_awareness_advanced(awareness, True, segment)
    print(f"   After exposure: {awareness:.3f}")
    
    # No exposure (decay)
    for i in range(3):
        awareness = update_awareness_advanced(awareness, False, segment)
        print(f"   After day {i+1} without exposure: {awareness:.3f}")
    
    print("\n3. PREFERENCE DRIFT OVER STAY:")
    base_affinity = 0.7
    total_nights = 10
    for day in [1, 3, 7, 10]:
        adjusted = compute_preference_drift(base_affinity, day, total_nights)
        print(f"   Day {day:2d}: {base_affinity:.3f} → {adjusted:.3f} (drift: {adjusted-base_affinity:+.3f})")
    
    print("\n4. CONTEXTUAL INTERACTIONS:")
    interactions = [
        ('rainy', 'budget_family', 'museum', 'morning'),
        ('sunny', 'adventure_seeker', 'tour', 'afternoon'),
        ('sunny', 'luxury_leisure', 'spa', 'evening')
    ]
    for weather, segment, category, time in interactions:
        boost = compute_context_interactions(segment, weather, time, category)
        print(f"   {weather:6s} + {segment:20s} + {category:10s}: +{boost:.3f}")
    
    print("\n5. MULTI-OBJECTIVE REWARD:")
    reward = compute_multi_objective_reward(
        click=True,
        revenue=50.0,
        awareness_before=0.3,
        awareness_after=0.51,
        frequency=3,
        category_diversity=0.8
    )
    for obj, value in reward.items():
        print(f"   {obj:15s}: {value:.3f}")





