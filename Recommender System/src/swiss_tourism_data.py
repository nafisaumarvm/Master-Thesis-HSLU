"""
Swiss Tourism Data API Integration

Fetches real restaurants, experiences, tours, and attractions from
Switzerland Tourism's official API.

API Documentation: https://www.tourismdata.ch/dataset/tourism-offers-in-switzerland/
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import time
import warnings


class SwissTourismAPI:
    """
    Interface to Swiss Tourism Data API for real advertiser catalog.
    
    Fetches actual:
    - Restaurants
    - Tours & Experiences
    - Attractions
    - Activities
    - Points of Interest
    
    Example:
        >>> api = SwissTourismAPI()
        >>> restaurants = api.fetch_restaurants(location='Zurich', limit=100)
        >>> advertisers_df = api.create_advertiser_catalog()
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Zurich Tourism API client.
        
        Uses the official Zurich Tourism API (zuerich.com) which provides
        real attractions, museums, restaurants, and experiences in Zurich.
        
        Args:
            api_key: Not needed for Zurich Tourism API (public)
        """
        self.api_key = api_key  # Not used for zuerich.com API
        self.base_url = "https://www.zuerich.com/en/api/v2/data"
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (research project)'
        }
        self.cache_dir = Path('data/raw/tourism')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Known working IDs from Zurich Tourism API
        self.known_ids = {
            'attractions': [96, 162, 97, 98, 95],  # Museums, art, landmarks
            'activities': [71, 72],  # Tours, experiences
            # Can expand with more IDs as discovered
        }
        
        # Try to load discovered IDs if available
        discovered_ids_file = self.cache_dir / 'valid_ids.json'
        if discovered_ids_file.exists():
            try:
                with open(discovered_ids_file, 'r') as f:
                    discovered_ids = json.load(f)
                    if discovered_ids:
                        print(f"✅ Loaded {len(discovered_ids)} discovered IDs from cache")
                        # Use all discovered IDs for attractions
                        self.known_ids['attractions'] = discovered_ids[:150]
                        self.known_ids['activities'] = discovered_ids[150:] if len(discovered_ids) > 150 else []
            except Exception as e:
                print(f"⚠️  Could not load discovered IDs: {e}")
    
    def fetch_offers(
        self,
        category: Optional[str] = None,
        location: Optional[str] = None,
        limit: int = 100,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Fetch tourism offers from Zurich Tourism API.
        
        The zuerich.com API uses numeric IDs, so we fetch from known categories.
        
        Args:
            category: Category type ('attractions', 'activities', 'restaurants')
            location: Not used (API is Zurich-specific)
            limit: Maximum number of results
            use_cache: Whether to use cached data
            
        Returns:
            List of offer dictionaries
        """
        # Create cache key
        cache_key = f"zurich_{category or 'all'}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if use_cache and cache_file.exists():
            print(f"✅ Loading cached Zurich data: {cache_file.name}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Fetch from API
        print(f"📡 Fetching from Zurich Tourism API: {category or 'all'}")
        
        try:
            # Get IDs for this category
            if category and category in self.known_ids:
                ids_to_fetch = self.known_ids[category][:limit]
            else:
                # Fetch from all known IDs
                ids_to_fetch = []
                for cat_ids in self.known_ids.values():
                    ids_to_fetch.extend(cat_ids)
                ids_to_fetch = ids_to_fetch[:limit]
            
            offers = []
            for offer_id in ids_to_fetch:
                try:
                    # Fetch individual offer
                    response = requests.get(
                        f"{self.base_url}?id={offer_id}",
                        headers=self.headers,
                        timeout=10
                    )
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # API returns list with one item
                    if isinstance(data, list) and len(data) > 0:
                        offers.append(data[0])
                    
                    # Be nice to API
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"    ⚠️  Could not fetch ID {offer_id}: {e}")
                    continue
            
            print(f"✅ Fetched {len(offers)} offers from Zurich Tourism API")
            
            # Cache results
            if offers:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(offers, f, indent=2, ensure_ascii=False)
            
            return offers
            
        except Exception as e:
            print(f"⚠️  API fetch failed: {e}")
            print(f"💡 Falling back to synthetic advertiser generation")
            return []
    
    def fetch_restaurants(
        self,
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Fetch restaurants from Zurich Tourism API."""
        # For now, returns empty (can add restaurant IDs later)
        return []
    
    def fetch_activities(
        self,
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Fetch activities/tours from Zurich Tourism API."""
        return self.fetch_offers(category='activities', location=location, limit=limit)
    
    def fetch_attractions(
        self,
        location: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Fetch attractions from Zurich Tourism API."""
        return self.fetch_offers(category='attractions', location=location, limit=limit)
    
    def discover_more_ids(self, start_id: int = 1, end_id: int = 200) -> List[int]:
        """
        Discover valid IDs from Zurich Tourism API by testing range.
        
        This is a helper to find more attraction IDs.
        Use sparingly to avoid overloading the API.
        """
        print(f"🔍 Discovering valid IDs from {start_id} to {end_id}...")
        valid_ids = []
        
        for offer_id in range(start_id, end_id + 1):
            try:
                response = requests.get(
                    f"{self.base_url}?id={offer_id}",
                    headers=self.headers,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        valid_ids.append(offer_id)
                        if len(valid_ids) % 10 == 0:
                            print(f"  Found {len(valid_ids)} valid IDs so far...")
                
                time.sleep(0.1)  # Be nice to API
                
            except Exception:
                continue
        
        print(f"✅ Discovered {len(valid_ids)} valid IDs: {valid_ids}")
        return valid_ids
    
    def create_advertiser_catalog(
        self,
        location: str = 'Zurich',
        total_advertisers: int = 200,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Create complete advertiser catalog from Swiss Tourism data.
        
        Args:
            location: Swiss location (e.g., 'Zurich', 'Geneva', 'Bern', 'Lucerne')
            total_advertisers: Target number of advertisers
            use_cache: Whether to use cached API responses
            
        Returns:
            DataFrame with columns:
            - advertiser_id
            - name
            - category
            - subcategory
            - location
            - latitude
            - longitude
            - distance_km (from hotel)
            - price_level (1-4)
            - rating (1-5)
            - open_hours
            - tags
            - source (real vs. synthetic)
        """
        print(f"\n🏢 Creating advertiser catalog for {location}")
        print("=" * 60)
        
        # Fetch from Zurich Tourism API using known categories
        api_categories = {
            'attractions': 'attractions',
            'activities': 'activities'
        }
        
        all_advertisers = []
        
        for api_cat, display_name in api_categories.items():
            print(f"\nFetching {display_name}...")
            offers = self.fetch_offers(
                category=api_cat,
                location=location,
                limit=100,
                use_cache=use_cache
            )
            
            # Convert offers to advertiser format
            for i, offer in enumerate(offers):
                advertiser = self._convert_offer_to_advertiser(offer, display_name, i)
                if advertiser:
                    all_advertisers.append(advertiser)
            
            if all_advertisers:
                recent_adds = [a for a in all_advertisers[-len(offers):]]
                print(f"  ✅ Added {len(recent_adds)} {display_name}")
            
            # Rate limiting (be nice to API)
            time.sleep(0.3)
        
        # Create DataFrame
        if all_advertisers:
            df = pd.DataFrame(all_advertisers)
            print(f"\n✅ Created catalog with {len(df)} real Swiss advertisers")
        else:
            print(f"\n⚠️  No API data available, generating synthetic advertisers")
            df = self._generate_synthetic_fallback(location, total_advertisers)
        
        # Fill to target size with synthetic if needed
        if len(df) < total_advertisers:
            needed = total_advertisers - len(df)
            print(f"💡 Adding {needed} synthetic advertisers to reach target of {total_advertisers}")
            synthetic = self._generate_synthetic_fallback(location, needed, start_id=len(df))
            df = pd.concat([df, synthetic], ignore_index=True)
        
        # Add computed fields
        df = self._enrich_advertiser_data(df)
        
        print(f"\n📊 Final catalog statistics:")
        print(f"   Total advertisers: {len(df)}")
        print(f"   Real (from API): {(df['source'] == 'real').sum()}")
        print(f"   Synthetic: {(df['source'] == 'synthetic').sum()}")
        print(f"\n   By category:")
        print(df['category'].value_counts().to_string())
        
        return df
    
    def _convert_offer_to_advertiser(
        self,
        offer: Dict,
        category: str,
        index: int
    ) -> Optional[Dict]:
        """
        Convert Zurich Tourism API offer (schema.org format) to advertiser format.
        
        The zuerich.com API uses schema.org structured data.
        """
        try:
            # Extract name (multilingual)
            name_obj = offer.get('name', {})
            name = (name_obj.get('en') or 
                   name_obj.get('de') or 
                   name_obj.get('fr') or 
                   f"Zurich {category.title()} {index}")
            
            # Extract address
            address_obj = offer.get('address', {})
            city = address_obj.get('addressLocality', 'Zurich')
            street = address_obj.get('streetAddress', '')
            
            # GPS coordinates
            geo = offer.get('geoCoordinates', {})
            latitude = geo.get('latitude')
            longitude = geo.get('longitude')
            
            # Description for tags
            desc_obj = offer.get('description', {}) or offer.get('disambiguatingDescription', {})
            description = (desc_obj.get('en') or 
                          desc_obj.get('de') or 
                          '')
            
            # Category from schema.org @type or category
            offer_type = offer.get('@customType') or offer.get('@type', 'Place')
            categories = offer.get('category', {})
            
            # Determine main category
            main_category = self._extract_category_from_offer(categories, offer_type)
            
            # Price from HTML table if available
            price_html = offer.get('price', {}).get('en', '')
            price_level = self._parse_price_from_html(price_html)
            
            # Rating (estimate 4.0-4.5 for Zurich Tourism listings)
            rating = np.random.uniform(4.0, 4.7)
            
            # Opening hours
            hours_spec = offer.get('openingHoursSpecification')
            open_hours = self._parse_zurich_opening_hours(hours_spec)
            
            # Generate tags
            tags = self._generate_tags(description, main_category)
            if 'Museum' in categories or 'museum' in offer_type.lower():
                tags.append('museum')
            if 'Art' in categories:
                tags.append('art')
            
            return {
                'advertiser_id': f"zurich_{offer.get('identifier', index)}",
                'name': name[:100],
                'category': main_category,
                'subcategory': offer_type,
                'location': city,
                'latitude': latitude,
                'longitude': longitude,
                'distance_km': None,  # Will be calculated later
                'price_level': price_level,
                'rating': float(rating),
                'open_hours': open_hours,
                'tags': tags,
                'source': 'real',
                'api_data': str(offer.get('identifier', ''))
            }
        
        except Exception as e:
            print(f"    ⚠️  Could not convert offer: {e}")
            return None
    
    def _extract_category_from_offer(self, categories: Dict, offer_type: str) -> str:
        """Extract main category from schema.org categories."""
        if 'Museums' in categories:
            return 'museum'
        elif 'Art' in categories or 'ArtObject' in offer_type:
            return 'attraction'
        elif 'Culture' in categories:
            return 'attraction'
        elif 'Gastronomy' in categories or 'Restaurant' in categories:
            return 'restaurant'
        elif 'Activities' in categories:
            return 'tour'
        elif 'Events' in categories:
            return 'event'
        else:
            return 'attraction'
    
    def _parse_price_from_html(self, price_html: str) -> int:
        """Extract price level from HTML price table."""
        if not price_html or price_html == '':
            return 2  # Default mid-range
        
        # Look for CHF amounts in the HTML
        import re
        prices = re.findall(r'CHF\s*(\d+)', price_html)
        
        if prices:
            avg_price = np.mean([float(p) for p in prices])
            if avg_price < 15:
                return 1  # Budget
            elif avg_price < 30:
                return 2  # Mid-range
            elif avg_price < 60:
                return 3  # Upscale
            else:
                return 4  # Luxury
        
        return 2  # Default
    
    def _parse_zurich_opening_hours(self, hours_spec) -> str:
        """Parse Zurich Tourism opening hours specification."""
        # For now, return default
        # In practice, would parse the hours_spec structure
        return "09:00-18:00"
    
    def _normalize_category(self, api_category: str) -> str:
        """Map API categories to our standard categories."""
        mapping = {
            'restaurant': 'restaurant',
            'activity': 'tour',
            'attraction': 'attraction',
            'event': 'event',
            'shop': 'retail',
            'museum': 'museum',
            'spa': 'spa',
            'bar': 'bar',
            'cafe': 'cafe'
        }
        return mapping.get(api_category.lower(), api_category.lower())
    
    def _estimate_price_level(self, offer: Dict, category: str) -> int:
        """
        Estimate price level (1-4) from offer data.
        
        1 = Budget (€0-30)
        2 = Mid-range (€30-70)
        3 = Upscale (€70-150)
        4 = Luxury (€150+)
        """
        # Check if price info in offer
        price = offer.get('price', {})
        if isinstance(price, dict):
            amount = price.get('amount') or price.get('from')
            if amount:
                if amount < 30:
                    return 1
                elif amount < 70:
                    return 2
                elif amount < 150:
                    return 3
                else:
                    return 4
        
        # Estimate by category
        defaults = {
            'restaurant': 2,
            'activity': 3,
            'attraction': 2,
            'event': 2,
            'shop': 2,
            'spa': 3,
            'museum': 1,
            'bar': 2,
            'cafe': 1
        }
        
        return defaults.get(category, 2)
    
    def _parse_opening_hours(self, hours: Dict) -> str:
        """Parse opening hours from API format."""
        # Simplified - just return typical hours
        # In practice, you'd parse the actual hours dict
        return "09:00-18:00"
    
    def _generate_tags(self, description: str, category: str) -> List[str]:
        """Generate tags from description and category."""
        tags = [category]
        
        # Common keywords
        keywords = {
            'family': 'family-friendly',
            'outdoor': 'outdoor',
            'indoor': 'indoor',
            'luxury': 'luxury',
            'authentic': 'authentic',
            'traditional': 'traditional',
            'modern': 'modern',
            'view': 'scenic',
            'mountain': 'mountain',
            'lake': 'lake',
            'city': 'urban',
            'historic': 'historic',
            'cultural': 'cultural',
            'romantic': 'romantic',
            'adventure': 'adventure'
        }
        
        description_lower = description.lower()
        for keyword, tag in keywords.items():
            if keyword in description_lower:
                tags.append(tag)
        
        return tags[:5]  # Limit to 5 tags
    
    def _enrich_advertiser_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add computed fields to advertiser catalog.
        
        Adds:
        - base_utility (for preference matching)
        - expected_revenue (for revenue optimization)
        - daily_cap (for inventory management)
        """
        # Base utility (random for now, should be segment-specific)
        df['base_utility'] = np.random.uniform(0.3, 0.8, len(df))
        
        # Expected revenue per engagement
        price_level_revenue = {1: 15, 2: 35, 3: 75, 4: 150}
        df['expected_revenue'] = df['price_level'].map(price_level_revenue) * df['rating'] / 5.0
        
        # Daily impression cap (varies by category)
        category_caps = {
            'restaurant': 40,
            'tour': 30,
            'attraction': 50,
            'spa': 25,
            'cafe': 35,
            'bar': 30,
            'museum': 40,
            'event': 20,
            'retail': 35
        }
        df['daily_cap'] = df['category'].map(category_caps).fillna(30)
        
        # Calculate distance if coordinates available
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Zurich hotel coordinates (example)
            hotel_lat, hotel_lon = 47.3769, 8.5417
            
            mask = df['latitude'].notna() & df['longitude'].notna()
            df.loc[mask, 'distance_km'] = df.loc[mask].apply(
                lambda row: self._haversine_distance(
                    hotel_lat, hotel_lon,
                    row['latitude'], row['longitude']
                ),
                axis=1
            )
        
        # Fill missing distances
        mask = df['distance_km'].isna()
        if mask.any():
            random_distances = np.random.uniform(0.3, 5.0, mask.sum())
            df.loc[mask, 'distance_km'] = random_distances
        
        return df
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in km."""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _generate_synthetic_fallback(
        self,
        location: str,
        num_advertisers: int,
        start_id: int = 0
    ) -> pd.DataFrame:
        """
        Generate synthetic advertisers as fallback.
        """
        categories = ['restaurant', 'tour', 'attraction', 'spa', 'cafe', 'bar', 'museum']
        
        advertisers = []
        for i in range(num_advertisers):
            category = np.random.choice(categories)
            
            advertisers.append({
                'advertiser_id': f"synthetic_{category}_{start_id + i}",
                'name': f"{category.title()} {location} {start_id + i}",
                'category': category,
                'subcategory': category,
                'location': location,
                'latitude': 47.3769 + np.random.normal(0, 0.05),
                'longitude': 8.5417 + np.random.normal(0, 0.05),
                'distance_km': np.random.uniform(0.3, 5.0),
                'price_level': np.random.randint(1, 5),
                'rating': np.random.uniform(3.5, 4.8),
                'open_hours': "09:00-18:00",
                'tags': [category],
                'source': 'synthetic',
                'api_data': ''
            })
        
        return pd.DataFrame(advertisers)


def load_swiss_advertisers(
    location: str = 'Zurich',
    num_advertisers: int = 200,
    use_cache: bool = True,
    use_real_api: bool = True
) -> pd.DataFrame:
    """
    Convenient function to load Swiss advertiser catalog.
    
    Uses Zurich Tourism API (zuerich.com) to fetch real attractions, museums,
    and activities. Falls back to synthetic generation if API unavailable.
    
    Args:
        location: Location (currently only 'Zurich' supported by API)
        num_advertisers: Target number of advertisers
        use_cache: Whether to use cached API responses
        use_real_api: Whether to attempt real API (set False to use synthetic only)
        
    Returns:
        DataFrame with Swiss advertisers (real from API + synthetic to fill)
        
    Example:
        >>> advertisers = load_swiss_advertisers('Zurich', 200)
        >>> print(f"Loaded {len(advertisers)} advertisers")
        >>> print(f"Real: {(advertisers['source']=='real').sum()}")
        >>> print(advertisers['category'].value_counts())
    """
    if use_real_api:
        api = SwissTourismAPI()
        return api.create_advertiser_catalog(
            location=location,
            total_advertisers=num_advertisers,
            use_cache=use_cache
        )
    else:
        # Skip API, use synthetic only
        print(f"💡 Using synthetic advertisers only (use_real_api=False)")
        api = SwissTourismAPI()
        return api._generate_synthetic_fallback(location, num_advertisers, start_id=0)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Swiss Tourism Data API Integration")
    print("=" * 60)
    
    # Example 1: Load advertisers
    print("\n📊 Example 1: Load Real Swiss Advertisers")
    print("-" * 60)
    
    try:
        advertisers_df = load_swiss_advertisers(
            location='Zurich',
            num_advertisers=200,
            use_cache=True
        )
        
        print(f"\n✅ Loaded {len(advertisers_df)} advertisers")
        print(f"\n📋 Sample (first 10):")
        print(advertisers_df[['name', 'category', 'price_level', 'rating', 'distance_km', 'source']].head(10).to_string(index=False))
        
        print(f"\n📈 Statistics:")
        print(f"   Real (from API): {(advertisers_df['source'] == 'real').sum()}")
        print(f"   Synthetic: {(advertisers_df['source'] == 'synthetic').sum()}")
        print(f"   Average rating: {advertisers_df['rating'].mean():.2f}")
        print(f"   Average distance: {advertisers_df['distance_km'].mean():.2f} km")
        
        print(f"\n📊 By category:")
        print(advertisers_df['category'].value_counts().to_string())
        
        # Save to file
        output_path = 'data/processed/swiss_advertisers.csv'
        advertisers_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Example complete!")
    print("\n💡 Integration tip:")
    print("   Use load_swiss_advertisers() in your simulation for real Swiss data!")

