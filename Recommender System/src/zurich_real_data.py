# Zurich Real Data Loader

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings


class ZurichDataLoader:
    # Load and parse real Zurich advertiser data from JSON files
    
    def __init__(self, data_dir: Optional[Path] = None):
        # Initialize Zurich data loader
        if data_dir is None:
            data_dir = Path(__file__).parent.parent
        
        self.data_dir = Path(data_dir)
        self.json_files = {
            # 'hotels': 'Hotels Zurich.json',  # EXCLUDED: Hotels shouldn't advertise competing hotels
            'attractions': 'Tourist Attractions Zurich.json',
            'shopping': 'Shopping Zurich.json',
            'spa_wellness': 'Spa & Wellness Zurich.json',
            'nightlife': 'Nightlife Zurich.json',
            'cultural': 'Cultural Activities Zurich.json',
            'gastro_luzern': 'Gastro Luzern.json'  # Lucerne restaurants
        }
        
        self.category_mapping = {
            # Experiences & Culture
            'Museums': 'Experiences',
            'Art': 'Experiences',
            'Culture': 'Experiences',
            'Attractions': 'Experiences',
            'Works of Art': 'Experiences',
            
            # Restaurants (from Nightlife)
            'Restaurants': 'Restaurants',
            'Gastronomy': 'Restaurants',
            'Cuisine': 'Restaurants',
            
            # Nightlife & Bars
            'Bars & Lounges': 'Nightlife',
            'Nightlife': 'Nightlife',
            'Clubs & Discos': 'Nightlife',
            'Restaurant & Bar': 'Nightlife',
            'Cocktail Bar': 'Nightlife',
            'Music Bar / Live Music': 'Nightlife',
            
            # Wellness & Spa
            'Wellness': 'Wellness',
            'Spa': 'Wellness',
            
            # Shopping
            'Shopping': 'Shopping',
            'Products': 'Shopping',
            'Fashion & Accessoires': 'Shopping',
            
            # Accommodation
            'Hotels': 'Accommodation',
            'Lodging': 'Accommodation',
            
            # Tours & Activities
            'Activities': 'Tours',
            'Tours': 'Tours',
            'Experiences': 'Tours',
        }
    
    def _extract_text(self, field: Dict, lang: str = 'en') -> str:
        # Extract text from multilingual field, with fallback
        if field is None:
            return ""
        
        if isinstance(field, dict):
            for try_lang in [lang, 'en', 'de', 'fr', 'it']:
                if try_lang in field and field[try_lang]:
                    return field[try_lang]
            return ""
        
        return str(field)
    
    def _map_category(self, category_dict: Dict) -> str:
        # Map JSON categories to our system's categories
        if not category_dict:
            return 'Experiences'  # Default
        
        for cat_name in category_dict.keys():
            if cat_name in self.category_mapping:
                return self.category_mapping[cat_name]
        
        if 'Culture' in category_dict or 'Museums' in category_dict:
            return 'Experiences'
        elif 'Gastronomy' in category_dict:
            return 'Restaurants'
        elif 'Shopping' in category_dict:
            return 'Shopping'
        elif 'Wellness' in category_dict:
            return 'Wellness'
        elif 'Hotels' in category_dict:
            return 'Accommodation'
        
        return 'Experiences'
    
    def _parse_price(self, price_dict: Dict, lang: str = 'en') -> float:
        # Extract price from price field
        if not price_dict:
            return np.nan
        
        price_text = self._extract_text(price_dict, lang)
        if not price_text:
            return np.nan
        
        import re
        numbers = re.findall(r'\d+\.?\d*', price_text)
        if numbers:
            try:
                prices = [float(n) for n in numbers[:2]]
                return np.mean(prices)
            except:
                return np.nan
        
        return np.nan
    
    def _parse_luzern_entry(self, entry: Dict, source_category: str) -> Dict:
        # Parse a Gastro Luzern entry (different format)
        # Extract basic info
        advertiser_id = entry.get('id', '')
        name = entry.get('title', 'Unknown')
        
        description = ''
        texts = entry.get('texts', [])
        for text in texts:
            if text.get('rel') == 'teaser' and text.get('type') == 'text/plain':
                description = text.get('value', '')
                break
        
        category = 'Restaurants'
        
        geo = entry.get('geo', {}).get('main', {})
        latitude = geo.get('latitude', np.nan)
        longitude = geo.get('longitude', np.nan)
        
        # Address
        street = entry.get('street', '')
        postal = entry.get('zip', '')
        city = entry.get('city', 'Luzern')
        
        # Other fields
        phone = entry.get('phone', '')
        email = entry.get('email', '')
        web = entry.get('web', '')
        
        return {
            'advertiser_id': f"LU_{advertiser_id}",
            'name': name,
            'name_de': name,  # Luzern data is primarily German
            'category': category,
            'source_category': source_category,
            'description': description,
            'type': 'Restaurant',
            'custom_type': 'Gastro',
            'latitude': latitude,
            'longitude': longitude,
            'street_address': street,
            'postal_code': postal,
            'city': city,
            'price_chf': np.nan,  # Not available in Luzern data
            'opening_hours': '',  # Not easily parseable
            'num_photos': 0,  # Would need to count media_objects
            'source': 'real_luzern',
            'data_source': 'Luzern Tourism JSON',
            'raw_categories': ', '.join(entry.get('categories', [])),
        }
    
    def _parse_entry(self, entry: Dict, source_category: str) -> Dict:
        # Parse a single JSON entry into our advertiser format
        # Extract basic info
        advertiser_id = entry.get('identifier', '')
        name_en = self._extract_text(entry.get('name'), 'en')
        name_de = self._extract_text(entry.get('name'), 'de')
        
        # Description (short version)
        description = self._extract_text(
            entry.get('disambiguatingDescription'), 'en'
        )
        
        # Category
        category = self._map_category(entry.get('category', {}))
        
        # Location
        geo = entry.get('geoCoordinates', {})
        latitude = geo.get('latitude', np.nan)
        longitude = geo.get('longitude', np.nan)
        
        # Address
        addr = entry.get('address', {})
        street = addr.get('streetAddress', '') if isinstance(addr, dict) else ''
        postal = addr.get('postalCode', '') if isinstance(addr, dict) else ''
        city = addr.get('addressLocality', 'Zürich') if isinstance(addr, dict) else 'Zürich'
        
        # Price
        price_chf = self._parse_price(entry.get('price', {}))
        
        # Type
        entry_type = entry.get('@type', 'Unknown')
        custom_type = entry.get('@customType', '')
        
        # Opening hours (simplified)
        opening_hours = entry.get('openingHours', '')
        if isinstance(opening_hours, list):
            opening_hours = ', '.join(opening_hours[:2])  # First 2 entries
        
        # Photos
        photos = entry.get('photo', [])
        num_photos = len(photos) if isinstance(photos, list) else 0
        
        return {
            'advertiser_id': f"ZH_{advertiser_id}",
            'name': name_en or name_de or 'Unknown',
            'name_de': name_de,
            'category': category,
            'source_category': source_category,
            'description': description,
            'type': entry_type,
            'custom_type': custom_type or entry_type,
            'latitude': latitude,
            'longitude': longitude,
            'street_address': street,
            'postal_code': postal,
            'city': city,
            'price_chf': price_chf,
            'opening_hours': str(opening_hours),
            'num_photos': num_photos,
            'source': 'real_zurich',
            'data_source': 'Zurich Tourism JSON',
            'raw_categories': ', '.join(entry.get('category', {}).keys())[:200],
        }
    
    def load_all_data(self, max_per_category: Optional[int] = None) -> pd.DataFrame:
        # Load all Zurich data from JSON files

        all_advertisers = []
        
        print("Loading Real Zurich Data from JSON Files")
        
        for source_key, filename in self.json_files.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"File not found: {filename}")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, dict) and 'items' in data:
                    # Luzern format (wrapped in dict with 'items' key)
                    entries = data['items']
                    print(f"   Found {data.get('overallcount', len(entries))} total in database (loading {len(entries)} from file)")
                elif isinstance(data, list):
                    # Zurich format (direct list)
                    entries = data
                else:
                    print(f"Unexpected format in {filename}: {type(data)}")
                    continue
                
                # Parse entries
                count = 0
                for entry in entries:
                    if max_per_category and count >= max_per_category:
                        break
                    
                    try:
                        # Use appropriate parser based on source
                        if source_key == 'gastro_luzern':
                            parsed = self._parse_luzern_entry(entry, source_key)
                        else:
                            parsed = self._parse_entry(entry, source_key)
                        all_advertisers.append(parsed)
                        count += 1
                    except Exception as e:
                        # Skip problematic entries
                        continue
                
                print(f"{filename:40s}: {count:4d} entries loaded")
            
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_advertisers)
        
        if len(df) == 0:
            print("No data loaded!")
            return df
        
        # Enrich with additional features
        df = self._enrich_data(df)
        
        print(f"TOTAL LOADED: {len(df)} real Zurich advertisers")
        for cat, count in df['category'].value_counts().items():
            print(f"   {cat:20s}: {count:4d}")
        
        return df
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add derived features to advertiser data
        # Budget tier (based on price)
        df['budget_tier'] = 'medium'
        df.loc[df['price_chf'] < 20, 'budget_tier'] = 'low'
        df.loc[df['price_chf'] > 50, 'budget_tier'] = 'high'
        df.loc[df['price_chf'].isna(), 'budget_tier'] = 'medium'  # Unknown = medium
        
        # Bid amount (simulated, based on category and tier)
        base_bids = {
            'Restaurants': 3.5,
            'Experiences': 4.0,
            'Nightlife': 3.0,
            'Wellness': 5.0,
            'Shopping': 3.0,
            'Tours': 4.5,
            'Accommodation': 6.0,
        }
        
        df['base_bid'] = df['category'].map(base_bids).fillna(3.5)
        
        # Add some realistic variance
        np.random.seed(42)
        bid_variance = np.random.uniform(0.8, 1.2, len(df))
        df['bid_amount'] = df['base_bid'] * bid_variance
        
        # Target segments (based on category and price tier)
        def assign_target_segments(row):
            segments = []
            
            # Budget-based
            if row['budget_tier'] == 'high' or row['category'] == 'Wellness':
                segments.append('Luxury Leisure')
            if row['budget_tier'] == 'low' or row['category'] == 'Shopping':
                segments.append('Bargain Hunter')
            if row['category'] in ['Restaurants', 'Nightlife']:
                segments.append('Family/Group')
            
            # Default segments
            if row['category'] in ['Accommodation']:
                segments.extend(['Business Traveler', 'Luxury Leisure'])
            if row['category'] in ['Experiences', 'Tours']:
                segments.extend(['Luxury Leisure', 'Family/Group'])
            
            # Remove duplicates
            segments = list(set(segments))
            
            # If none, add default
            if not segments:
                segments = ['Luxury Leisure', 'Family/Group']
            
            return ', '.join(segments[:3])  # Max 3 segments
        
        df['target_segments'] = df.apply(assign_target_segments, axis=1)
        
        # Weather preference
        def assign_weather_pref(row):
            if row['category'] in ['Shopping', 'Cultural', 'Museums']:
                return 'rainy'
            elif row['category'] in ['Tours', 'Experiences']:
                return 'sunny'
            elif row['category'] in ['Wellness', 'Spa']:
                return 'any'
            else:
                return 'any'
        
        df['weather_preference'] = df.apply(assign_weather_pref, axis=1)
        
        # Quality score (based on available info completeness)
        df['quality_score'] = (
            (~df['description'].isna()).astype(int) * 0.3 +
            (~df['latitude'].isna()).astype(int) * 0.2 +
            (~df['price_chf'].isna()).astype(int) * 0.2 +
            (df['num_photos'] > 0).astype(int) * 0.2 +
            (df['opening_hours'].str.len() > 10).astype(int) * 0.1
        )
        
        return df


def load_zurich_advertisers(
    n_advertisers: int = None,
    data_dir: Optional[Path] = None,
    seed: int = 42
) -> pd.DataFrame:
    # Load real Swiss advertisers for recommendation system
    loader = ZurichDataLoader(data_dir)
    
    # Load all data
    df = loader.load_all_data()
    
    if len(df) == 0:
        raise ValueError("No Swiss data could be loaded! Check JSON files.")
    
    # Use all data if n_advertisers is None, otherwise sample
    if n_advertisers is None:
        print(f"\nUsing ALL {len(df)} real Swiss establishments")    
    elif len(df) > n_advertisers:
        df = df.sample(n=n_advertisers, random_state=seed).reset_index(drop=True)
        print(f"\nSampled {n_advertisers} advertisers from {len(loader.load_all_data())} available")
    elif len(df) < n_advertisers:
        print(f"\nOnly {len(df)} advertisers available (requested {n_advertisers})")
    
    # Add advertiser index
    df['advertiser_idx'] = range(len(df))
    
    return df


if __name__ == "__main__":
    # Test loading
    # Load all data
    loader = ZurichDataLoader()
    df = loader.load_all_data()
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    print(f"\nSample Entries:")
    for i, row in df.head(5).iterrows():
        print(f"   {row['name'][:50]:50s} | {row['category']:15s} | {row['city']}")
    
    print(f"\nGPS Coverage:")
    print(f"   Valid coordinates: {(~df['latitude'].isna()).sum()} / {len(df)}")
    
    print(f"\nPrice Coverage:")
    print(f"   Valid prices: {(~df['price_chf'].isna()).sum()} / {len(df)}")
    print(f"   Average price: CHF {df['price_chf'].mean():.2f}")
    
    print(f"\nTarget Segments:")
    all_segments = set()
    for segments in df['target_segments'].str.split(', '):
        all_segments.update(segments)
    print(f"   Unique segments: {sorted(all_segments)}")
    
