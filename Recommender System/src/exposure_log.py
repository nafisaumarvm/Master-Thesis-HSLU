"""
Exposure log generation with logging policies and position bias.

Van Leeuwen (2024) methodology:
- Utility-based choice model
- Counterfactual logging (full candidate sets)
- Awareness dynamics
- Position bias
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Callable, Dict
from scipy.special import expit as sigmoid

from .utils import set_random_seed, logit, parse_tags_string
from . import preferences as pref_module


# Position bias weights
POSITION_WEIGHTS = {
    1: 1.0,
    2: 0.7,
    3: 0.5
}

# Global position bias strength parameter
BETA_POSITION = 0.8


def popularity_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    k: int = 3,
    **kwargs
) -> List[str]:
    """
    Popularity-based policy: rank by base_utility.
    
    Args:
        guest_id: Guest identifier (unused but kept for interface consistency)
        candidate_ads_df: Dataframe of candidate ads
        k: Number of ads to select
        
    Returns:
        List of selected ad_ids
    """
    if len(candidate_ads_df) == 0:
        return []
    
    # Sort by base_utility (descending)
    sorted_ads = candidate_ads_df.sort_values('base_utility', ascending=False)
    
    # Select top-k
    selected = sorted_ads.head(k)
    
    return selected['ad_id'].tolist()


def random_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    k: int = 3,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    """
    Random uniform policy.
    
    Args:
        guest_id: Guest identifier
        candidate_ads_df: Dataframe of candidate ads
        k: Number of ads to select
        seed: Random seed
        
    Returns:
        List of selected ad_ids
    """
    if len(candidate_ads_df) == 0:
        return []
    
    rng = np.random.default_rng(seed)
    
    # Random sample
    n_sample = min(k, len(candidate_ads_df))
    selected_indices = rng.choice(len(candidate_ads_df), size=n_sample, replace=False)
    selected = candidate_ads_df.iloc[selected_indices]
    
    return selected['ad_id'].tolist()


def epsilon_greedy_policy(
    guest_id: str,
    candidate_ads_df: pd.DataFrame,
    k: int = 3,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
    **kwargs
) -> List[str]:
    """
    Epsilon-greedy policy: explore with probability epsilon.
    
    Args:
        guest_id: Guest identifier
        candidate_ads_df: Dataframe of candidate ads
        k: Number of ads to select
        epsilon: Exploration probability
        seed: Random seed
        
    Returns:
        List of selected ad_ids
    """
    if len(candidate_ads_df) == 0:
        return []
    
    rng = np.random.default_rng(seed)
    
    if rng.random() < epsilon:
        # Explore: random selection
        return random_policy(guest_id, candidate_ads_df, k, seed=seed)
    else:
        # Exploit: popularity
        return popularity_policy(guest_id, candidate_ads_df, k)


def filter_candidate_ads(
    ads_df: pd.DataFrame,
    time_of_day: str,
    max_distance_km: float = 8.0
) -> pd.DataFrame:
    """
    Filter ads by time of day and distance.
    
    Args:
        ads_df: Full advertiser catalogue
        time_of_day: One of 'morning', 'afternoon', 'evening', 'late_night'
        max_distance_km: Maximum distance threshold
        
    Returns:
        Filtered dataframe
    """
    # Filter by distance
    filtered = ads_df[ads_df['distance_km'] <= max_distance_km].copy()
    
    # Filter by opening dayparts
    def is_open(dayparts_str: str) -> bool:
        dayparts = parse_tags_string(dayparts_str)
        return time_of_day in dayparts
    
    filtered = filtered[filtered['opening_dayparts'].apply(is_open)]
    
    return filtered


def generate_exposure_log(
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame,
    guest_ad_prefs_df: pd.DataFrame,
    n_sessions_per_stay: int = 4,
    logging_policy: str = 'popularity',
    k_ads_per_session: int = 3,
    beta_position: float = BETA_POSITION,
    max_distance_km: float = 8.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate exposure log with logging policy and position bias.
    
    Args:
        guests_df: Unified guest dataframe
        ads_df: Advertiser catalogue
        guest_ad_prefs_df: Guest-ad preferences (intrinsic scores)
        n_sessions_per_stay: Number of TV sessions per guest stay
        logging_policy: 'popularity', 'random', or 'epsilon_greedy'
        k_ads_per_session: Number of ads to show per session
        beta_position: Position bias strength
        max_distance_km: Maximum distance for candidate ads
        seed: Random seed
        
    Returns:
        Exposure log dataframe with columns:
        - guest_id, stay_id, session_id, ad_id, position
        - logging_policy, propensity, click, revenue
        - time_of_day, day_of_stay, source
    """
    rng = set_random_seed(seed)
    
    # Select policy function
    policy_functions = {
        'popularity': popularity_policy,
        'random': random_policy,
        'epsilon_greedy': epsilon_greedy_policy
    }
    
    if logging_policy not in policy_functions:
        raise ValueError(f"Unknown logging policy: {logging_policy}")
    
    policy_fn = policy_functions[logging_policy]
    
    # Time of day options
    time_of_day_options = ['morning', 'afternoon', 'evening', 'late_night']
    
    exposure_records = []
    session_counter = 0
    
    for idx, guest in guests_df.iterrows():
        guest_id = guest['guest_id']
        stay_id = guest['stay_id']
        source = guest.get('source', 'unknown')
        
        # Get nights and ensure it's a valid integer
        nights = guest.get('nights', guest.get('stay_nights', 2))
        if pd.isna(nights) or nights <= 0:
            nights = 2  # Default to 2 nights if missing or invalid
        nights = int(nights)
        
        # Generate sessions for this stay
        for session_idx in range(n_sessions_per_stay):
            session_id = f"SESSION_{session_counter:08d}"
            session_counter += 1
            
            # Sample time of day
            time_of_day = rng.choice(time_of_day_options)
            
            # Sample day of stay (clip to nights)
            day_of_stay = min(int(rng.integers(1, nights + 2)), nights)
            
            # Filter candidate ads
            candidate_ads = filter_candidate_ads(ads_df, time_of_day, max_distance_km)
            
            if len(candidate_ads) == 0:
                # No candidates available
                continue
            
            # Apply logging policy to select ads
            selected_ad_ids = policy_fn(
                guest_id,
                candidate_ads,
                k=k_ads_per_session,
                seed=seed + session_counter
            )
            
            if len(selected_ad_ids) == 0:
                continue
            
            # Compute propensities
            if logging_policy == 'random':
                # Uniform propensity
                propensity = 1.0 / len(candidate_ads)
            elif logging_policy == 'popularity':
                # Propensity proportional to base_utility rank
                total_candidates = len(candidate_ads)
                propensities = {}
                for rank, ad_id in enumerate(selected_ad_ids, start=1):
                    propensities[ad_id] = 1.0 / total_candidates
            else:
                # Default: uniform over selected
                propensity = 1.0 / len(selected_ad_ids)
            
            # Generate impressions with position bias
            for position, ad_id in enumerate(selected_ad_ids, start=1):
                if position > len(POSITION_WEIGHTS):
                    break
                
                # Get base click probability
                pref_match = guest_ad_prefs_df[
                    (guest_ad_prefs_df['guest_id'] == guest_id) &
                    (guest_ad_prefs_df['ad_id'] == ad_id)
                ]
                
                if len(pref_match) == 0:
                    # Not in preferences; use low default
                    base_click_prob = 0.05
                else:
                    base_click_prob = pref_match.iloc[0]['base_click_prob']
                
                # Apply position bias
                position_weight = POSITION_WEIGHTS.get(position, 0.3)
                click_logit = logit(base_click_prob) + beta_position * position_weight
                click_prob = sigmoid(click_logit)
                
                # Sample click
                click = rng.binomial(1, click_prob)
                
                # Compute revenue
                ad_row = ads_df[ads_df['ad_id'] == ad_id].iloc[0]
                revenue = click * ad_row['revenue_per_conversion']
                
                # Determine propensity for this ad
                if logging_policy == 'popularity':
                    prop = propensities.get(ad_id, 1.0 / len(candidate_ads))
                else:
                    prop = propensity
                
                exposure_records.append({
                    'guest_id': guest_id,
                    'stay_id': stay_id,
                    'session_id': session_id,
                    'ad_id': ad_id,
                    'position': position,
                    'logging_policy': logging_policy,
                    'propensity': prop,
                    'click': click,
                    'revenue': revenue,
                    'time_of_day': time_of_day,
                    'day_of_stay': day_of_stay,
                    'source': source
                })
    
    return pd.DataFrame(exposure_records)


def add_additional_context_to_log(
    exposure_log: pd.DataFrame,
    guests_df: pd.DataFrame,
    ads_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Enrich exposure log with additional guest and ad features.
    
    Args:
        exposure_log: Exposure log
        guests_df: Guest dataframe
        ads_df: Advertiser dataframe
        
    Returns:
        Enriched exposure log
    """
    # Merge guest features
    guest_features = guests_df[[
        'guest_id', 'purpose_of_stay', 'is_family', 
        'is_business', 'price_per_night', 'country'
    ]].copy()
    
    enriched = exposure_log.merge(guest_features, on='guest_id', how='left')
    
    # Merge ad features
    ad_features = ads_df[[
        'ad_id', 'advertiser_type', 'price_level', 
        'distance_km', 'base_utility'
    ]].copy()
    
    enriched = enriched.merge(ad_features, on='ad_id', how='left')
    
    return enriched


def generate_exposure_log_van_leeuwen(
    guests_df: pd.DataFrame,
    advertisers_df: pd.DataFrame,
    utility_matrix: pd.DataFrame,
    n_sessions_per_stay: int = 4,
    k_ads_per_session: int = 3,
    logging_policy: str = 'softmax',
    alpha_awareness: float = 0.3,
    beta_awareness: float = 0.5,
    position_weights: Optional[Dict[int, float]] = None,
    temperature: float = 1.0,
    seed: int = 42
) -> tuple:
    """
    Generate exposure log following van Leeuwen (2024) methodology.
    
    Key features:
    - Utility-based choice model
    - Full counterfactual logging (candidate sets)
    - Awareness dynamics (ρ grows with exposure)
    - Position bias
    - Logging policy probabilities for IPS
    
    Parameters
    ----------
    guests_df : pd.DataFrame
        Guest data with segments and context
    advertisers_df : pd.DataFrame
        Advertiser catalogue
    utility_matrix : pd.DataFrame
        Pre-computed base utilities (guest_id, ad_id, base_utility)
    n_sessions_per_stay : int
        Sessions per guest stay
    k_ads_per_session : int
        How many ads to show per session
    logging_policy : str
        'softmax' (contextual), 'epsilon-greedy', or 'uniform'
    alpha_awareness : float
        Awareness growth rate α ∈ (0,1)
    beta_awareness : float
        Awareness effect on utility β
    position_weights : dict, optional
        Position bias weights
    temperature : float
        Softmax temperature (higher = more random)
    seed : int
        Random seed
        
    Returns
    -------
    exposure_log : pd.DataFrame
        Impression-level logs (one row per shown ad)
    counterfactual_log : pd.DataFrame
        Session-level logs with full candidate sets
    """
    rng = set_random_seed(seed)
    
    if position_weights is None:
        position_weights = POSITION_WEIGHTS
    
    # Track awareness per (guest, advertiser)
    awareness_state = {}  # (guest_id, ad_id) -> awareness
    
    # Time of day options
    time_options = ['morning', 'afternoon', 'evening', 'late_night']
    
    impressions = []
    counterfactuals = []
    session_counter = 0
    
    for _, guest in guests_df.iterrows():
        guest_id = guest['guest_id']
        stay_id = guest['stay_id']
        segment = guest['segment']
        
        # Get nights
        nights = guest.get('nights', guest.get('stay_nights', 2))
        if pd.isna(nights) or nights <= 0:
            nights = 2
        nights = int(nights)
        
        # Guest context
        guest_context = {
            'party_size': guest.get('total_guests', 2),
            'trip_purpose': guest.get('purpose_of_stay', 'leisure'),
            'weather': guest.get('weather', 'sunny')
        }
        
        # Get guest utilities
        guest_utils = utility_matrix[utility_matrix['guest_id'] == guest_id]
        if len(guest_utils) == 0:
            continue  # Skip if no utilities
        
        # Sessions for this stay
        for session_idx in range(n_sessions_per_stay):
            session_id = f"SESSION_{session_counter:08d}"
            session_counter += 1
            
            # Session context
            time_of_day = rng.choice(time_options)
            day_of_stay = min(int(rng.integers(1, nights + 2)), nights)
            
            # Update guest context
            guest_context['time_of_day'] = time_of_day
            guest_context['day_of_stay'] = day_of_stay
            
            # Get candidate ads for this session
            candidate_ads = advertisers_df.copy()
            
            # Compute utilities for all candidates with current awareness
            candidate_utils = []
            for _, ad in candidate_ads.iterrows():
                ad_id = ad['ad_id']
                
                # Get base utility
                util_row = guest_utils[guest_utils['ad_id'] == ad_id]
                if len(util_row) == 0:
                    # Compute on-the-fly if missing
                    base_util = pref_module.compute_base_utility(
                        segment,
                        ad['advertiser_type'],
                        guest_context,
                        {
                            'price_level': ad.get('price_level', 'medium'),
                            'distance_km': ad.get('distance_km', 5.0)
                        }
                    )
                else:
                    base_util = util_row.iloc[0]['base_utility']
                
                # Get current awareness
                awareness_key = (guest_id, ad_id)
                current_awareness = awareness_state.get(awareness_key, 0.0)
                
                # Utility with awareness (no position yet)
                util_with_awareness = base_util + beta_awareness * current_awareness
                
                candidate_utils.append({
                    'ad_id': ad_id,
                    'base_utility': base_util,
                    'awareness': current_awareness,
                    'utility': util_with_awareness
                })
            
            cand_df = pd.DataFrame(candidate_utils)
            
            # Logging policy: select k ads
            if logging_policy == 'softmax':
                # Softmax over utilities
                scores = cand_df['utility'].values / temperature
                scores = scores - scores.max()  # Numerical stability
                exp_scores = np.exp(scores)
                probs = exp_scores / exp_scores.sum()
                
                chosen_indices = rng.choice(
                    len(cand_df),
                    size=min(k_ads_per_session, len(cand_df)),
                    replace=False,
                    p=probs
                )
            elif logging_policy == 'epsilon-greedy':
                epsilon = 0.1
                if rng.random() < epsilon:
                    # Random
                    chosen_indices = rng.choice(
                        len(cand_df),
                        size=min(k_ads_per_session, len(cand_df)),
                        replace=False
                    )
                else:
                    # Top-k by utility
                    chosen_indices = cand_df['utility'].nlargest(k_ads_per_session).index.values
            else:  # uniform
                chosen_indices = rng.choice(
                    len(cand_df),
                    size=min(k_ads_per_session, len(cand_df)),
                    replace=False
                )
            
            chosen_ads = cand_df.iloc[chosen_indices]['ad_id'].tolist()
            
            # Store counterfactual info (for IPS)
            counterfactuals.append({
                'session_id': session_id,
                'guest_id': guest_id,
                'stay_id': stay_id,
                'segment': segment,
                'day_of_stay': day_of_stay,
                'time_of_day': time_of_day,
                'n_candidates': len(cand_df),
                'candidate_ads': ','.join(cand_df['ad_id'].tolist()),
                'shown_ads': ','.join(chosen_ads),
                'logging_policy': logging_policy
            })
            
            # Generate clicks for shown ads
            for rank, ad_id in enumerate(chosen_ads, 1):
                ad_info = candidate_ads[candidate_ads['ad_id'] == ad_id].iloc[0]
                cand_info = cand_df[cand_df['ad_id'] == ad_id].iloc[0]
                
                base_util = cand_info['base_utility']
                current_awareness = cand_info['awareness']
                
                # Click probability with position bias
                click_prob = pref_module.compute_choice_probability(
                    base_util,
                    current_awareness,
                    rank,
                    awareness_weight=beta_awareness,
                    position_weights=position_weights
                )
                
                # Sample click
                clicked = int(rng.random() < click_prob)
                
                # Update awareness (exposure increases awareness)
                awareness_key = (guest_id, ad_id)
                new_awareness = pref_module.update_awareness(
                    current_awareness,
                    was_exposed=True,
                    alpha=alpha_awareness
                )
                awareness_state[awareness_key] = new_awareness
                
                # Revenue if clicked
                revenue = ad_info['revenue_per_conversion'] * clicked if clicked else 0.0
                
                # Log impression
                impressions.append({
                    'session_id': session_id,
                    'guest_id': guest_id,
                    'stay_id': stay_id,
                    'ad_id': ad_id,
                    'advertiser_type': ad_info['advertiser_type'],
                    'segment': segment,
                    'position': rank,
                    'day_of_stay': day_of_stay,
                    'time_of_day': time_of_day,
                    'base_utility': base_util,
                    'awareness_before': current_awareness,
                    'awareness_after': new_awareness,
                    'click_prob': click_prob,
                    'click': clicked,
                    'revenue': revenue,
                    'distance_km': ad_info.get('distance_km', 5.0),
                    'price_level': ad_info.get('price_level', 'medium')
                })
    
    exposure_log = pd.DataFrame(impressions)
    counterfactual_log = pd.DataFrame(counterfactuals)
    
    return exposure_log, counterfactual_log

