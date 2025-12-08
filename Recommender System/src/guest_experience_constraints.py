"""
Guest Experience Constraints for In-Room TV Advertising

Based on interviews and surveys with hotel guests and staff, this module
implements real-world constraints to ensure non-intrusive advertising that
respects guest privacy and experience.

Key Findings from Research:
- Guests accept 1-2 ads per day maximum
- Ads should never interrupt media consumption
- Initial viewing period of 1 minute required before skip
- Content must be filtered (no politics, competitors, etc.)
- Privacy is paramount (federated learning approach)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class TVState(Enum):
    """TV states based on guest journey moments."""
    INITIAL_ENTRY = "initial_entry"  # First time entering room
    STARTUP = "startup"  # TV turned on
    WATCHING_CONTENT = "watching_content"  # Media consumption (NO ADS!)
    IDLE = "idle"  # TV on but no content
    OFF = "off"  # TV off


class ContentCategory(Enum):
    """Ad content categories for filtering."""
    SAFE = "safe"  # Tourism, dining, wellness, shopping
    RESTRICTED = "restricted"  # Politics, religion, adult content
    COMPETITOR = "competitor"  # Competing hotels, platforms
    PROHIBITED = "prohibited"  # Never allowed


class GuestExperienceConstraints:
    """
    Implements guest experience constraints based on survey findings.
    
    Research-Backed Constraints:
    1. Maximum 1-2 ads per guest per day
    2. Ads only at initial entry or TV startup (never during content)
    3. Required 1-minute viewing before skip option
    4. Content filtering (politics, competitors blocked)
    5. Privacy-preserving federated learning
    """
    
    def __init__(
        self,
        max_ads_per_day: int = 2,
        min_viewing_seconds: int = 60,
        allowed_states: List[TVState] = None
    ):
        """
        Initialize guest experience constraints.
        
        Args:
            max_ads_per_day: Maximum ads per guest per day (default: 2, from surveys)
            min_viewing_seconds: Required viewing time before skip (default: 60s)
            allowed_states: TV states when ads can be shown
        """
        self.max_ads_per_day = max_ads_per_day
        self.min_viewing_seconds = min_viewing_seconds
        
        # Ads only at these moments (NEVER during content consumption)
        self.allowed_states = allowed_states or [
            TVState.INITIAL_ENTRY,
            TVState.STARTUP,
            TVState.IDLE
        ]
        
        # Track ad exposure per guest per day
        self.exposure_log = {}  # {(guest_id, date): count}
        
        # Content filtering lists
        self.prohibited_keywords = {
            'politics', 'political', 'election', 'party', 'government',
            'religion', 'religious', 'church', 'temple', 'mosque',
            'competitor', 'booking.com', 'airbnb', 'expedia', 'hotels.com',
            'adult', 'casino', 'gambling', 'alcohol', 'tobacco'
        }
        
        self.competitor_brands = {
            'booking.com', 'airbnb', 'expedia', 'hotels.com', 'agoda',
            'tripadvisor', 'trivago', 'kayak', 'priceline'
        }
        
        # Safe categories from tourism
        self.safe_categories = {
            'restaurants', 'experiences', 'shopping', 'wellness',
            'tours', 'culture', 'museums', 'attractions', 'transportation'
        }
        
        # Prohibited categories (hotels shouldn't show competitor hotels)
        self.prohibited_categories = {
            'accommodation', 'hotels', 'lodging', 'hotel'
        }
    
    def can_show_ad(
        self,
        guest_id: str,
        current_date: datetime,
        tv_state: TVState
    ) -> Tuple[bool, str]:
        """
        Check if ad can be shown to guest.
        
        Args:
            guest_id: Unique guest identifier
            current_date: Current datetime
            tv_state: Current TV state
        
        Returns:
            (can_show, reason) tuple
        """
        # Check 1: TV state must be appropriate
        if tv_state not in self.allowed_states:
            if tv_state == TVState.WATCHING_CONTENT:
                return False, "Cannot interrupt media consumption"
            elif tv_state == TVState.OFF:
                return False, "TV is off"
            else:
                return False, f"TV state {tv_state.value} not allowed"
        
        # Check 2: Daily frequency cap
        date_key = current_date.date()
        exposure_key = (guest_id, date_key)
        
        current_count = self.exposure_log.get(exposure_key, 0)
        if current_count >= self.max_ads_per_day:
            return False, f"Daily cap reached ({self.max_ads_per_day} ads/day)"
        
        return True, "Approved"
    
    def record_exposure(
        self,
        guest_id: str,
        current_date: datetime,
        advertiser_id: str,
        viewing_seconds: int
    ) -> Dict:
        """
        Record ad exposure with viewing metrics.
        
        Args:
            guest_id: Unique guest identifier
            current_date: Exposure datetime
            advertiser_id: Advertiser identifier
            viewing_seconds: Actual viewing time
        
        Returns:
            Exposure record with metrics
        """
        date_key = current_date.date()
        exposure_key = (guest_id, date_key)
        
        # Increment daily count
        self.exposure_log[exposure_key] = self.exposure_log.get(exposure_key, 0) + 1
        
        # Calculate metrics
        completed_required = viewing_seconds >= self.min_viewing_seconds
        completion_rate = min(viewing_seconds / self.min_viewing_seconds, 1.0)
        
        return {
            'guest_id': guest_id,
            'date': date_key,
            'advertiser_id': advertiser_id,
            'viewing_seconds': viewing_seconds,
            'required_seconds': self.min_viewing_seconds,
            'completed_required': completed_required,
            'completion_rate': completion_rate,
            'daily_exposure_count': self.exposure_log[exposure_key],
            'timestamp': current_date
        }
    
    def filter_content(
        self,
        advertisers_df: pd.DataFrame,
        strict_mode: bool = True
    ) -> pd.DataFrame:
        """
        Filter advertisers based on content appropriateness.
        
        Args:
            advertisers_df: DataFrame of advertisers
            strict_mode: If True, apply strict filtering
        
        Returns:
            Filtered DataFrame with only appropriate content
        """
        filtered = advertisers_df.copy()
        
        # Add content safety score
        filtered['content_safety_score'] = 1.0
        
        # Check name and description for prohibited keywords
        def check_content_safety(row):
            text = ' '.join([
                str(row.get('name', '')),
                str(row.get('description', '')),
                str(row.get('category', ''))
            ]).lower()
            
            # Check for prohibited keywords
            for keyword in self.prohibited_keywords:
                if keyword in text:
                    return 0.0  # Prohibited
            
            # Check for competitor brands
            for brand in self.competitor_brands:
                if brand in text:
                    return 0.0  # Competitor
            
            # Check if category is prohibited (e.g., Accommodation - hotels shouldn't show competitor hotels)
            category = str(row.get('category', '')).lower()
            if any(prohibited in category for prohibited in self.prohibited_categories):
                return 0.0  # Prohibited
            
            # Check if category is safe
            if any(safe in category for safe in self.safe_categories):
                return 1.0  # Safe
            
            # Default: moderately safe
            return 0.5 if not strict_mode else 0.0
        
        filtered['content_safety_score'] = filtered.apply(check_content_safety, axis=1)
        
        # Filter: keep only safe content
        if strict_mode:
            filtered = filtered[filtered['content_safety_score'] >= 1.0].copy()
        else:
            filtered = filtered[filtered['content_safety_score'] >= 0.5].copy()
        
        print(f"Content filtering: {len(advertisers_df)} → {len(filtered)} advertisers")
        print(f"   Removed: {len(advertisers_df) - len(filtered)} (competitors, prohibited content)")
        
        return filtered
    
    def get_daily_statistics(self, current_date: datetime) -> Dict:
        """Get statistics for current day."""
        date_key = current_date.date()
        
        day_exposures = [
            count for (guest, date), count in self.exposure_log.items()
            if date == date_key
        ]
        
        if not day_exposures:
            return {
                'date': date_key,
                'unique_guests': 0,
                'total_exposures': 0,
                'avg_per_guest': 0.0,
                'guests_at_cap': 0
            }
        
        return {
            'date': date_key,
            'unique_guests': len(day_exposures),
            'total_exposures': sum(day_exposures),
            'avg_per_guest': np.mean(day_exposures),
            'guests_at_cap': sum(1 for c in day_exposures if c >= self.max_ads_per_day)
        }


class FederatedLearningFramework:
    """
    Privacy-preserving federated learning for TV ad recommendations.
    
    Based on survey findings that guest privacy is paramount, this implements
    a federated approach where:
    - Guest data stays local (on hotel TV system)
    - Only model updates (gradients) are shared
    - No individual guest data leaves the hotel
    - Differential privacy can be applied
    
    This is a simulation framework for thesis demonstration.
    """
    
    def __init__(
        self,
        num_hotels: int = 3,
        local_epochs: int = 5,
        privacy_budget: float = 1.0
    ):
        """
        Initialize federated learning framework.
        
        Args:
            num_hotels: Number of participating hotels (virtual clients)
            local_epochs: Training epochs per hotel before aggregation
            privacy_budget: Differential privacy epsilon (smaller = more private)
        """
        self.num_hotels = num_hotels
        self.local_epochs = local_epochs
        self.privacy_budget = privacy_budget
        
        # Track which data belongs to which hotel (simulation)
        self.hotel_assignments = {}
        
        # Global model state (preference parameters)
        self.global_params = {}
        
        # Local models per hotel
        self.local_params = {i: {} for i in range(num_hotels)}
    
    def assign_guest_to_hotel(self, guest_id: str) -> int:
        """Assign guest to a virtual hotel (for simulation)."""
        if guest_id not in self.hotel_assignments:
            # Deterministic assignment based on guest_id hash
            hotel_id = hash(guest_id) % self.num_hotels
            self.hotel_assignments[guest_id] = hotel_id
        
        return self.hotel_assignments[guest_id]
    
    def local_training_step(
        self,
        hotel_id: int,
        local_data: pd.DataFrame,
        current_params: Dict
    ) -> Dict:
        """
        Simulate local training at one hotel.
        
        In real deployment:
        - This runs on hotel's local TV system
        - Guest data never leaves hotel
        - Only gradient updates are computed
        
        Args:
            hotel_id: Hotel identifier
            local_data: Local guest data (stays at hotel)
            current_params: Current model parameters
        
        Returns:
            Updated parameters (gradients) for aggregation
        """
        # Simulate local training
        # In practice: run gradient descent on local data
        updated_params = current_params.copy()
        
        # Add noise for differential privacy
        noise_scale = 1.0 / self.privacy_budget
        for key in updated_params:
            if isinstance(updated_params[key], (int, float)):
                noise = np.random.normal(0, noise_scale * 0.01)
                updated_params[key] = updated_params[key] + noise
        
        return updated_params
    
    def federated_averaging(self, local_updates: List[Dict]) -> Dict:
        """
        Aggregate local updates using FedAvg algorithm.
        
        Args:
            local_updates: List of parameter updates from each hotel
        
        Returns:
            Aggregated global parameters
        """
        if not local_updates:
            return {}
        
        # Average parameters across hotels
        aggregated = {}
        all_keys = set()
        for update in local_updates:
            all_keys.update(update.keys())
        
        for key in all_keys:
            values = [
                update[key] for update in local_updates
                if key in update and isinstance(update[key], (int, float))
            ]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def train_federated_round(
        self,
        guest_data: pd.DataFrame
    ) -> Dict:
        """
        Execute one round of federated learning.
        
        Steps:
        1. Each hotel trains locally on its guests' data
        2. Hotels send only parameter updates (not data)
        3. Central server aggregates updates
        4. Global model is updated
        
        Args:
            guest_data: All guest data (partitioned by hotel)
        
        Returns:
            Training statistics
        """
        # Partition data by hotel (simulation)
        guest_data['hotel_id'] = guest_data['guest_id'].apply(
            lambda x: hash(str(x)) % self.num_hotels
        )
        
        local_updates = []
        
        # Each hotel trains locally
        for hotel_id in range(self.num_hotels):
            hotel_guests = guest_data[guest_data['hotel_id'] == hotel_id]
            
            if len(hotel_guests) == 0:
                continue
            
            # Local training (data stays at hotel!)
            update = self.local_training_step(
                hotel_id,
                hotel_guests,
                self.global_params
            )
            
            local_updates.append(update)
        
        # Aggregate updates (only gradients shared, not data!)
        self.global_params = self.federated_averaging(local_updates)
        
        return {
            'participating_hotels': len(local_updates),
            'global_params': self.global_params,
            'privacy_preserved': True,
            'data_centralized': False  # Guest data stays local!
        }


def simulate_guest_journey_with_constraints(
    guest_id: str,
    stay_days: int,
    constraints: GuestExperienceConstraints
) -> List[Dict]:
    """
    Simulate a guest's TV journey with experience constraints.
    
    Args:
        guest_id: Guest identifier
        stay_days: Length of stay in days
        constraints: Experience constraints
    
    Returns:
        List of events (TV states, ad opportunities, exposures)
    """
    events = []
    current_date = datetime.now()
    
    for day in range(stay_days):
        day_date = current_date + timedelta(days=day)
        
        # Typical TV usage pattern per day
        tv_events = [
            (8, TVState.INITIAL_ENTRY if day == 0 else TVState.STARTUP),  # Morning
            (9, TVState.WATCHING_CONTENT),  # Breakfast TV
            (10, TVState.OFF),  # Out for the day
            (18, TVState.STARTUP),  # Evening return
            (19, TVState.WATCHING_CONTENT),  # Dinner TV
            (21, TVState.IDLE),  # Channel surfing
            (22, TVState.WATCHING_CONTENT),  # Movie
            (23, TVState.OFF)  # Sleep
        ]
        
        ads_shown_today = 0
        
        for hour, tv_state in tv_events:
            event_time = day_date.replace(hour=hour, minute=0, second=0)
            
            # Check if ad can be shown
            can_show, reason = constraints.can_show_ad(guest_id, event_time, tv_state)
            
            event = {
                'guest_id': guest_id,
                'timestamp': event_time,
                'day_of_stay': day + 1,
                'tv_state': tv_state.value,
                'ad_opportunity': can_show,
                'reason': reason
            }
            
            # If opportunity and haven't hit daily cap, show ad
            if can_show and ads_shown_today < constraints.max_ads_per_day:
                # Simulate viewing (random between 30-120 seconds)
                viewing_time = np.random.randint(30, 121)
                
                exposure = constraints.record_exposure(
                    guest_id,
                    event_time,
                    f"advertiser_{np.random.randint(1, 100)}",
                    viewing_time
                )
                
                event.update({
                    'ad_shown': True,
                    'viewing_seconds': viewing_time,
                    'completed_required': exposure['completed_required']
                })
                
                ads_shown_today += 1
            else:
                event['ad_shown'] = False
            
            events.append(event)
    
    return events


if __name__ == "__main__":
    print("="*70)
    print("GUEST EXPERIENCE CONSTRAINTS - DEMONSTRATION")
    print("Based on Hotel Interview & Survey Findings")
    print("="*70)
    
    # Initialize constraints
    constraints = GuestExperienceConstraints(
        max_ads_per_day=2,
        min_viewing_seconds=60
    )
    
    print("\n📋 IMPLEMENTED CONSTRAINTS:")
    print(f"   Max ads per day: {constraints.max_ads_per_day}")
    print(f"   Required viewing: {constraints.min_viewing_seconds} seconds")
    print(f"   Allowed moments: {[s.value for s in constraints.allowed_states]}")
    print(f"   Prohibited: Politics, competitors, adult content")
    
    # Simulate guest journey
    print("\n🎬 SIMULATING 3-DAY GUEST STAY:")
    print("-"*70)
    
    events = simulate_guest_journey_with_constraints("guest_001", 3, constraints)
    
    # Show ad opportunities vs actual shows
    opportunities = sum(1 for e in events if e['ad_opportunity'])
    shown = sum(1 for e in events if e.get('ad_shown', False))
    
    print(f"   Total TV state changes: {len(events)}")
    print(f"   Ad opportunities: {opportunities}")
    print(f"   Ads actually shown: {shown}")
    print(f"   Compliance rate: {shown/3/2*100:.1f}% of max (2/day × 3 days)")
    
    # Show sample events
    print("\n📺 SAMPLE EVENTS:")
    for event in events[:10]:
        status = "✅ AD SHOWN" if event.get('ad_shown') else "❌ BLOCKED" if event['ad_opportunity'] else "⏸️  NO OPPORTUNITY"
        print(f"   Day {event['day_of_stay']}, {event['timestamp'].hour:02d}:00 | {event['tv_state']:20s} | {status}")
        if not event['ad_opportunity'] and event['tv_state'] == 'watching_content':
            print(f"      Reason: {event['reason']}")
    
    # Content filtering demo
    print("\n🛡️  CONTENT FILTERING DEMONSTRATION:")
    print("-"*70)
    
    # Create sample advertisers with some prohibited content
    sample_ads = pd.DataFrame([
        {'name': 'Swiss Museum', 'description': 'Cultural experience', 'category': 'Experiences'},
        {'name': 'Political Campaign', 'description': 'Vote for party', 'category': 'Politics'},
        {'name': 'Booking.com Competitor', 'description': 'Book hotels', 'category': 'Accommodation'},
        {'name': 'Local Restaurant', 'description': 'Swiss cuisine', 'category': 'Restaurants'},
        {'name': 'Religious Tour', 'description': 'Church visit', 'category': 'Tours'},
    ])
    
    filtered = constraints.filter_content(sample_ads, strict_mode=True)
    
    print(f"   Original: {len(sample_ads)} advertisers")
    print(f"   After filtering: {len(filtered)} advertisers")
    print(f"   Removed: {sample_ads[~sample_ads.index.isin(filtered.index)]['name'].tolist()}")
    
    print("\n✅ GUEST EXPERIENCE CONSTRAINTS OPERATIONAL!")




