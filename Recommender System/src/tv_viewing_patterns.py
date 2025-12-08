# TV Viewing Pattern Modeling for In-Room Hotel Advertising


import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, time


# Calculate probability that guest is watching TV

def get_tv_viewing_probability(
    segment: str,
    time_of_day: str,
    day_of_stay: int,
    is_weekend: bool = False
) -> float:
    
    # Base TV usage rates by segment (from industry benchmarks)
    base_rates = {
        'Extended Stay': 0.82,
        'Luxury Leisure': 0.75,
        'Budget Family': 0.71,
        'Cultural Tourist': 0.68,
        'Weekend Explorer': 0.64,
        'Adventure Seeker': 0.62,
        'Business Traveler': 0.48
    }
    
    # Time-of-day multipliers (peak = evening)
    time_multipliers = {
        'morning': {'base': 0.40, 'weekend': 0.55},      # 6-11am
        'afternoon': {'base': 0.25, 'weekend': 0.35},    # 12-5pm
        'evening': {'base': 1.20, 'weekend': 1.15},      # 6-11pm (PEAK)
        'late_night': {'base': 0.50, 'weekend': 0.60}    # 11pm-2am
    }
    
    # Day-of-stay pattern (fatigue effect)
    day_factors = {
        1: 1.15,    # Arrival day: tired, settling in
        2: 0.85,    # Day 2: exploring
        3: 0.80,    # Day 3: still active
        4: 0.90,    # Day 4: starting to relax more
        5: 1.00,    # Day 5: routine
        6: 1.05,    # Day 6+: fatigue, more in-room time
        7: 1.10,
    }
    
    # Get base rate for segment
    base = base_rates.get(segment, 0.65)
    
    # Apply time-of-day multiplier
    time_key = 'weekend' if is_weekend else 'base'
    time_mult = time_multipliers.get(time_of_day, {'base': 1.0, 'weekend': 1.0})[time_key]
    
    # Apply day-of-stay factor
    day_factor = day_factors.get(min(day_of_stay, 7), 1.10)
    
    # Segment-specific adjustments
    if segment == 'Business Traveler':
        # Much lower TV usage, especially weekdays
        if not is_weekend and time_of_day in ['morning', 'afternoon']:
            time_mult *= 0.3  # Working during day
    
    if segment == 'Budget Family':
        # Higher morning usage (kids watching)
        if time_of_day == 'morning':
            time_mult *= 1.4
    
    if segment == 'Adventure Seeker':
        # Lower daytime usage (out exploring)
        if time_of_day in ['morning', 'afternoon']:
            time_mult *= 0.6
    
    # Calculate final probability
    prob = base * time_mult * day_factor
    
    # Clip to [0, 1]
    return np.clip(prob, 0.0, 1.0)


def get_time_of_day_from_hour(hour: int) -> str:
    # Convert hour (0-23) to time-of-day category
    if 6 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 23:
        return 'evening'
    else:
        return 'late_night'


def simulate_tv_session_duration(segment: str, time_of_day: str, rng: np.random.Generator) -> int:
    # Simulate TV viewing session duration in minutes
    # Average durations by time of day (in minutes)
    avg_durations = {
        'morning': 25,
        'afternoon': 15,
        'evening': 45,    # Peak viewing time
        'late_night': 20
    }
    
    # Segment multipliers
    segment_multipliers = {
        'Extended Stay': 1.3,
        'Luxury Leisure': 1.1,
        'Budget Family': 1.2,
        'Cultural Tourist': 1.0,
        'Weekend Explorer': 0.8,
        'Adventure Seeker': 0.7,
        'Business Traveler': 0.6
    }
    
    avg = avg_durations.get(time_of_day, 30)
    mult = segment_multipliers.get(segment, 1.0)
    
    # Sample from gamma distribution (realistic duration distribution)
    shape = 2.0
    scale = (avg * mult) / shape
    duration = rng.gamma(shape, scale)
    
    # Clip to reasonable range (5-120 minutes)
    return int(np.clip(duration, 5, 120))


# Calculate guest annoyance/intrusion cost

def compute_intrusion_cost(
    frequency_per_guest: float,
    optimal_frequency: float = 5.0,
    penalty_weight: float = 0.05
) -> float:
    if frequency_per_guest <= 3.0:
        # Under-exposed: minimal cost
        return 0.0
    
    elif frequency_per_guest <= 7.0:
        # Optimal range: small linear cost
        excess = frequency_per_guest - optimal_frequency
        cost = penalty_weight * abs(excess)
        return np.clip(cost, 0.0, 0.15)
    
    else:
        # Over-exposed: quadratic penalty (grows rapidly)
        excess = frequency_per_guest - 7.0
        cost = 0.15 + penalty_weight * (excess ** 1.5)
        return np.clip(cost, 0.0, 1.0)


def compute_segment_specific_intrusion_cost(
    frequency_per_guest: float,
    segment: str
) -> float:
    # Segment-specific intrusion tolerance
    segment_params = {
        'Business Traveler': {'optimal': 3.0, 'penalty': 0.08},  # Low tolerance
        'Luxury Leisure': {'optimal': 4.5, 'penalty': 0.06},
        'Cultural Tourist': {'optimal': 5.0, 'penalty': 0.05},
        'Weekend Explorer': {'optimal': 4.0, 'penalty': 0.05},
        'Adventure Seeker': {'optimal': 5.0, 'penalty': 0.05},
        'Budget Family': {'optimal': 6.0, 'penalty': 0.04},     # Higher tolerance
        'Extended Stay': {'optimal': 7.0, 'penalty': 0.03}       # Highest tolerance
    }
    
    params = segment_params.get(segment, {'optimal': 5.0, 'penalty': 0.05})
    return compute_intrusion_cost(
        frequency_per_guest,
        optimal_frequency=params['optimal'],
        penalty_weight=params['penalty']
    )


# Measure awareness elasticity

def compute_awareness_elasticity(
    exposures: np.ndarray,
    awareness: np.ndarray,
    segment: Optional[str] = None
) -> Dict[str, float]:
    # Measure awareness elasticity

    # Sort by exposures
    sorted_idx = np.argsort(exposures)
    exp_sorted = exposures[sorted_idx]
    awa_sorted = awareness[sorted_idx]
    
    # Calculate marginal elasticity
    marginal = []
    for i in range(1, len(exp_sorted)):
        delta_exp = exp_sorted[i] - exp_sorted[i-1]
        delta_awa = awa_sorted[i] - awa_sorted[i-1]
        
        if delta_exp > 0:
            elasticity = delta_awa / delta_exp
            marginal.append(elasticity)
    
    result = {
        'marginal_elasticity': marginal,
        'average_elasticity': np.mean(marginal) if marginal else 0.0
    }
    
    # Key elasticity points
    if len(marginal) >= 1:
        result['elasticity_at_1'] = marginal[0]  # First exposure
    
    if len(marginal) >= 5:
        result['elasticity_at_5'] = marginal[4]  # Fifth exposure
    
    # Elasticity ratio (shows diminishing returns)
    if 'elasticity_at_1' in result and 'elasticity_at_5' in result:
        if result['elasticity_at_5'] > 0:
            result['diminishing_returns_ratio'] = result['elasticity_at_1'] / result['elasticity_at_5']
    
    if segment:
        result['segment'] = segment
    
    return result


# Adstock advertising model

def adstock_model(
    exposures: np.ndarray,
    decay_rate: float = 0.7,
    saturation: float = 0.9
) -> np.ndarray:
    # Adstock advertising model (Broadbent, 1984)

    adstock = np.zeros(len(exposures))
    
    for t in range(len(exposures)):
        if t == 0:
            adstock[t] = exposures[t]
        else:
            # Adstock accumulates with decay
            adstock[t] = exposures[t] + decay_rate * adstock[t-1]
    
    # Apply saturation (diminishing returns)
    adstock = saturation * (1 - np.exp(-adstock / saturation))
    
    return adstock


def ebbinghaus_forgetting_curve(
    time_since_exposure: np.ndarray,
    initial_awareness: float = 1.0,
    decay_constant: float = 1.25
) -> np.ndarray:
    # Ebbinghaus forgetting curve (1885)

    retention = initial_awareness / (1 + decay_constant * time_since_exposure)
    return retention


def compare_with_theory(
    observed_awareness: np.ndarray,
    exposures: np.ndarray,
    alpha: float,
    delta: float
) -> Dict[str, float]:
    # Compare our awareness model with classical advertising theory

    # Fit Adstock model
    # Map our α, δ to Adstock parameters
    decay_rate = 1 - delta  # Our decay → Adstock carryover
    adstock_pred = adstock_model(exposures, decay_rate=decay_rate)
    
    # Correlation with observed
    if len(observed_awareness) > 1 and len(adstock_pred) > 1:
        corr_adstock = np.corrcoef(observed_awareness, adstock_pred[:len(observed_awareness)])[0, 1]
    else:
        corr_adstock = 0.0
    
    # Mean absolute error
    mae_adstock = np.mean(np.abs(observed_awareness - adstock_pred[:len(observed_awareness)]))
    
    return {
        'adstock_correlation': corr_adstock,
        'adstock_mae': mae_adstock,
        'theoretical_alignment': 'strong' if corr_adstock > 0.85 else 'moderate'
    }


# Generate realistic TV viewing schedule

def generate_tv_viewing_schedule(
    guest_segment: str,
    stay_nights: int,
    arrival_date: pd.Timestamp,
    rng: np.random.Generator
) -> pd.DataFrame:
    schedule = []
    
    for day in range(stay_nights):
        current_date = arrival_date + pd.Timedelta(days=day)
        is_weekend = current_date.dayofweek >= 5
        day_of_stay = day + 1
        
        # Check each time period
        for hour in [8, 14, 20, 23]:  # Morning, afternoon, evening, late night
            time_category = get_time_of_day_from_hour(hour)
            
            # Probability guest watches TV
            tv_prob = get_tv_viewing_probability(
                guest_segment, time_category, day_of_stay, is_weekend
            )
            
            # Simulate TV on/off
            if rng.random() < tv_prob:
                duration = simulate_tv_session_duration(guest_segment, time_category, rng)
                
                session_time = current_date.replace(hour=hour, minute=0)
                schedule.append({
                    'datetime': session_time,
                    'time_of_day': time_category,
                    'duration_min': duration,
                    'tv_on_prob': tv_prob,
                    'day_of_stay': day_of_stay
                })
    
    return pd.DataFrame(schedule)


if __name__ == "__main__":

    # Test 1: TV viewing probability
    print("\n TV Viewing Probabilities:")
    segments = ['Business Traveler', 'Luxury Leisure', 'Extended Stay']
    times = ['morning', 'evening']
    
    for seg in segments:
        for t in times:
            prob = get_tv_viewing_probability(seg, t, day_of_stay=3, is_weekend=False)
            print(f"  {seg:20s} | {t:10s} | Day 3: {prob:.2%}")
    
    # Test 2: Intrusion cost
    print("\n Intrusion Cost Function:")
    for freq in [2.0, 4.2, 7.5, 10.0]:
        cost = compute_intrusion_cost(freq)
        print(f"  Frequency {freq:4.1f} → Intrusion cost: {cost:.3f}")
    
    # Test 3: Awareness elasticity
    print("\n Awareness Elasticity:")
    exp = np.array([0, 1, 2, 3, 4, 5])
    awa = np.array([0.0, 0.30, 0.51, 0.66, 0.76, 0.82])
    elasticity = compute_awareness_elasticity(exp, awa, segment='Luxury Leisure')
    print(f"Average elasticity: {elasticity['average_elasticity']:.3f}")
    print(f"Elasticity at 1st exposure: {elasticity.get('elasticity_at_1', 0):.3f}")
    print(f"Elasticity at 5th exposure: {elasticity.get('elasticity_at_5', 0):.3f}")
    print(f"Diminishing returns ratio: {elasticity.get('diminishing_returns_ratio', 0):.2f}×")
    





