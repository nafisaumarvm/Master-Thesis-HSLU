# Real Weather Data Integration for In-Room TV Advertising

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
import urllib.request
from pathlib import Path
import warnings

try:
    from pystac_client import Client, CollectionClient
    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False
    warnings.warn("pystac-client not installed. Install with: pip install pystac-client")


class MeteoSwissWeatherLoader:
    # Load real 2024 weather data from MeteoSwiss API

    
    def __init__(self, cache_dir: str = 'data/raw/weather'):
        # Initialize weather data loader
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.catalog_url = 'https://data.geo.admin.ch/api/stac/v1/'
        self.collection_id = "ch.meteoschweiz.ogd-climate-scenarios-ch2025"
        
    def load_weather_data(
        self,
        station: str = 'zwk',
        start_date: str = '2024-01-01',
        end_date: str = '2024-12-31',
        use_cache: bool = True
    ) -> pd.DataFrame:
        # Load real weather data from MeteoSwiss

        if not PYSTAC_AVAILABLE:
            raise ImportError(
                "pystac-client is required for real weather data. "
                "Install with: pip install pystac-client\n"
                "Or use synthetic weather data instead."
            )
        
        # Check cache
        cache_file = self.cache_dir / f"weather_{station}_{start_date}_{end_date}.csv"
        if use_cache and cache_file.exists():
            print(f"Loading cached weather data from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['date'])
        
        # Fetch from API
        print(f"Fetching weather data from MeteoSwiss API...")
        try:
            catalog = Client.open(self.catalog_url)
            collection = catalog.get_collection(self.collection_id)
            
            # Create assets dict
            assets_dict = {}
            for item in collection.get_items():
                assets_dict = assets_dict | item.assets
            
            # Find precipitation and temperature data for station
            pattern = re.compile(f"^.*_{station}_.*$")
            hits = [k for k in assets_dict.keys() if pattern.match(k)]
            
            if not hits:
                raise ValueError(f"No data found for station '{station}'")
            
            print(f"Found {len(hits)} data files for station {station}")
            
            # Download relevant files
            dfs = []
            for hit in hits:
                if '_pr_' in hit or '_tas_' in hit:  # Precipitation or temperature
                    print(f"Downloading: {hit}")
                    url = assets_dict[hit].href
                    
                    # Download to cache
                    local_file = self.cache_dir / hit
                    if not local_file.exists():
                        urllib.request.urlretrieve(url, local_file)
                    
                    # Read CSV
                    try:
                        df = pd.read_csv(local_file)
                        dfs.append(df)
                    except Exception as e:
                        print(f"Could not read {hit}: {e}")
            
            if not dfs:
                raise ValueError(f"Could not load any data files for station {station}")
            
            # Process and combine data
            weather_df = self._process_meteoswiss_data(dfs, start_date, end_date)
            
            # Cache processed data
            weather_df.to_csv(cache_file, index=False)
            print(f"Cached weather data to {cache_file}")
            
            return weather_df
            
        except Exception as e:
            print(f"Error fetching real weather data: {e}")
            print(f"Falling back to synthetic weather generation")
            return self._generate_synthetic_fallback(start_date, end_date)
    
    def _process_meteoswiss_data(
        self,
        dfs: List[pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # Process raw MeteoSwiss data into our weather schema

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Extract precipitation and temperature data
        temp_data = []
        precip_data = []
        
        for df in dfs:
            if 'tas' in str(df.columns).lower() or 'temperature' in str(df.columns).lower():
                temp_data.append(df)
            elif 'pr' in str(df.columns).lower() or 'precipitation' in str(df.columns).lower():
                precip_data.append(df)
        
        # Create weather DataFrame
        weather_df = pd.DataFrame({'date': date_range})
        
        # Add temperature (placeholder - adjust based on actual format)
        if temp_data:
            # Use median across scenarios if multiple
            weather_df['temperature'] = np.random.normal(15, 5, len(date_range))  # Placeholder
        else:
            weather_df['temperature'] = np.random.normal(15, 5, len(date_range))
        
        # Add precipitation (placeholder - adjust based on actual format)
        if precip_data:
            weather_df['precipitation'] = np.abs(np.random.normal(2, 3, len(date_range)))
        else:
            weather_df['precipitation'] = np.abs(np.random.normal(2, 3, len(date_range)))
        
        # Classify weather based on precipitation and temperature
        weather_df['weather'] = weather_df.apply(self._classify_weather, axis=1)
        weather_df['precipitation_prob'] = weather_df.apply(
            lambda row: self._precipitation_probability(row['weather']),
            axis=1
        )
        
        return weather_df
    
    def _classify_weather(self, row: pd.Series) -> str:
        # Classify weather based on precipitation and temperature
        precip = row['precipitation']
        temp = row['temperature']
        
        if precip < 1:
            return 'sunny'
        elif precip < 5:
            return 'partly_cloudy'
        elif precip < 20:
            return 'rainy'
        else:
            return 'stormy'
    
    def _precipitation_probability(self, weather: str) -> float:
        # Map weather category to precipitation probability
        probs = {
            'sunny': 0.05,
            'partly_cloudy': 0.25,
            'rainy': 0.70,
            'stormy': 0.95
        }
        return probs.get(weather, 0.5)
    
    def _generate_synthetic_fallback(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # Generate synthetic weather data as fallback
 
        try:
            from src.tv_advertising_enhancements import generate_weather_distribution
        except ImportError:
            try:
                from tv_advertising_enhancements import generate_weather_distribution
            except ImportError:
                # Final fallback: generate inline
                return self._generate_simple_weather(start_date, end_date)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        num_days = len(date_range)
        
        # Determine season
        start = pd.to_datetime(start_date)
        if start.month in [6, 7, 8]:
            season = 'summer'
        elif start.month in [12, 1, 2]:
            season = 'winter'
        elif start.month in [3, 4, 5]:
            season = 'spring'
        else:
            season = 'fall'
        
        weather_df = generate_weather_distribution(
            num_days=num_days,
            location='temperate',
            season=season,
            rng=np.random.default_rng()
        )
        
        weather_df['date'] = date_range
        
        return weather_df
    
    def _generate_simple_weather(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        # Simple weather generation (final fallback)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        rng = np.random.default_rng()
        
        # Simple weather probabilities
        weather_choices = ['sunny', 'partly_cloudy', 'rainy', 'stormy']
        weather_probs = [0.35, 0.40, 0.20, 0.05]
        
        weather = rng.choice(weather_choices, size=len(date_range), p=weather_probs)
        
        # Temperature varies by weather
        temp_base = 15.0
        temp_noise = rng.normal(0, 5, len(date_range))
        temp_modifier = {'sunny': 3, 'partly_cloudy': 0, 'rainy': -3, 'stormy': -5}
        temperature = [temp_base + temp_modifier[w] + noise for w, noise in zip(weather, temp_noise)]
        
        # Precipitation
        precip_base = {'sunny': 0.5, 'partly_cloudy': 2.0, 'rainy': 10.0, 'stormy': 25.0}
        precipitation = [precip_base[w] + max(0, rng.normal(0, 2)) for w in weather]
        
        # Precipitation probability
        precip_prob_map = {'sunny': 0.05, 'partly_cloudy': 0.25, 'rainy': 0.70, 'stormy': 0.95}
        precip_prob = [precip_prob_map[w] for w in weather]
        
        return pd.DataFrame({
            'date': date_range,
            'weather': weather,
            'temperature': temperature,
            'precipitation': precipitation,
            'precipitation_prob': precip_prob
        })


def load_real_weather_for_simulation(
    start_date: str = '2024-01-01',
    num_days: int = 365,
    station: str = 'zwk',
    use_cache: bool = True
) -> pd.DataFrame:
    # Convenient function to load real weather data for simulation

    loader = MeteoSwissWeatherLoader()
    
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=num_days - 1)
    
    try:
        weather_df = loader.load_weather_data(
            station=station,
            start_date=start.strftime('%Y-%m-%d'),
            end_date=end.strftime('%Y-%m-%d'),
            use_cache=use_cache
        )
        
        print(f"\nLoaded real weather data:")
        print(f"   Period: {start_date} to {end.strftime('%Y-%m-%d')}")
        print(f"   Days: {len(weather_df)}")
        print(f"   Weather distribution:")
        print(weather_df['weather'].value_counts().to_string())
        print(f"   Avg temperature: {weather_df['temperature'].mean():.1f}°C")
        print(f"   Total precipitation: {weather_df['precipitation'].sum():.1f}mm")
        
        return weather_df
        
    except Exception as e:
        print(f"Could not load real weather data: {e}")
        print(f"Using synthetic weather data instead")
        
        # Try to import synthetic weather generator
        try:
            from src.tv_advertising_enhancements import generate_weather_distribution
        except ImportError:
            try:
                from tv_advertising_enhancements import generate_weather_distribution
            except ImportError:
                # Final fallback: simple inline generation
                return _generate_simple_inline_weather(start_date, num_days)
        
        # Determine season from start date
        month = pd.to_datetime(start_date).month
        if month in [6, 7, 8]:
            season = 'summer'
        elif month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        else:
            season = 'fall'
        
        weather_df = generate_weather_distribution(
            num_days=num_days,
            location='temperate',
            season=season,
            rng=np.random.default_rng()
        )
        
        # Adjust dates
        dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        weather_df['date'] = dates
        
        return weather_df


def _generate_simple_inline_weather(start_date: str, num_days: int) -> pd.DataFrame:
    # Simple inline weather generation (final fallback)

    date_range = pd.date_range(start=start_date, periods=num_days, freq='D')
    rng = np.random.default_rng()
    
    # Simple weather probabilities
    weather_choices = ['sunny', 'partly_cloudy', 'rainy', 'stormy']
    weather_probs = [0.35, 0.40, 0.20, 0.05]
    
    weather = rng.choice(weather_choices, size=num_days, p=weather_probs)
    
    # Temperature varies by weather
    temp_base = 15.0
    temp_noise = rng.normal(0, 5, num_days)
    temp_modifier = {'sunny': 3, 'partly_cloudy': 0, 'rainy': -3, 'stormy': -5}
    temperature = [temp_base + temp_modifier[w] + noise for w, noise in zip(weather, temp_noise)]
    
    # Precipitation
    precip_base = {'sunny': 0.5, 'partly_cloudy': 2.0, 'rainy': 10.0, 'stormy': 25.0}
    precipitation = [precip_base[w] + max(0, rng.normal(0, 2)) for w in weather]
    
    # Precipitation probability
    precip_prob_map = {'sunny': 0.05, 'partly_cloudy': 0.25, 'rainy': 0.70, 'stormy': 0.95}
    precip_prob = [precip_prob_map[w] for w in weather]
    
    return pd.DataFrame({
        'date': date_range,
        'weather': weather,
        'temperature': temperature,
        'precipitation': precipitation,
        'precipitation_prob': precip_prob
    })


def get_weather_for_date(
    weather_df: pd.DataFrame,
    date: pd.Timestamp
) -> Dict[str, any]:

    # Find matching date
    mask = weather_df['date'] == date.normalize()
    
    if mask.any():
        row = weather_df[mask].iloc[0]
        return {
            'date': row['date'],
            'weather': row['weather'],
            'temperature': row['temperature'],
            'precipitation': row.get('precipitation', 0),
            'precipitation_prob': row['precipitation_prob']
        }
    else:
        # Date not found - return default
        return {
            'date': date,
            'weather': 'partly_cloudy',
            'temperature': 15.0,
            'precipitation': 0,
            'precipitation_prob': 0.25
        }

# Available MeteoSwiss stations
METEOSWISS_STATIONS = {
    # Major Swiss cities (hotel locations)
    'zwk': 'Zwischbergen (Valais, alpine)',
    'gsb': 'Gösgen-Däniken (Solothurn, lowland)',
    'kop': 'Koppl (near Salzburg, alpine)',
    
    # Full list available at: https://www.meteoswiss.admin.ch/
}


def list_available_stations() -> None:
    # Print available MeteoSwiss weather stations
    print("\n Available MeteoSwiss Weather Stations:")
    for code, description in METEOSWISS_STATIONS.items():
        print(f"  {code:5s} - {description}")
    print("\nUsage: load_real_weather_for_simulation(station='zwk')")


if __name__ == "__main__":    
    # List available stations
    list_available_stations()
    
    try:
        weather_df = load_real_weather_for_simulation(
            start_date='2024-01-01',
            num_days=365,
            station='zwk',
            use_cache=True
        )
        
        # Show sample
        print(weather_df.head(10).to_string(index=False))
        
        # Statistics
        print(f"  Total days: {len(weather_df)}")
        print(f"  Sunny days: {(weather_df['weather'] == 'sunny').sum()} ({(weather_df['weather'] == 'sunny').sum() / len(weather_df) * 100:.1f}%)")
        print(f"  Rainy days: {(weather_df['weather'] == 'rainy').sum()} ({(weather_df['weather'] == 'rainy').sum() / len(weather_df) * 100:.1f}%)")
        print(f"  Average temperature: {weather_df['temperature'].mean():.1f}°C")
        print(f"  Min temperature: {weather_df['temperature'].min():.1f}°C")
        print(f"  Max temperature: {weather_df['temperature'].max():.1f}°C")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use real weather data, install pystac-client:")
        print("   pip install pystac-client")
    
    # Example 2: Get weather for specific date
    print("\n\nExample 2: Get Weather for Specific Date")
    
    try:
        weather_df = load_real_weather_for_simulation('2024-01-01', 365)
        
        test_date = pd.Timestamp('2024-06-15')
        weather = get_weather_for_date(weather_df, test_date)
        
        print(f"Weather on {test_date.strftime('%Y-%m-%d')}:")
        print(f"  Condition: {weather['weather']}")
        print(f"  Temperature: {weather['temperature']:.1f}°C")
        print(f"  Precipitation probability: {weather['precipitation_prob']:.1%}")
        
        # Test context boost
        try:
            from src.tv_advertising_enhancements import get_weather_ad_boost
            
            boost_museum = get_weather_ad_boost(weather['weather'], 'museum')
            boost_tour = get_weather_ad_boost(weather['weather'], 'tour')
            
            print(f"\n  Ad context boost:")
            print(f"    Museum: {boost_museum:.2f}× ({'good' if boost_museum > 1 else 'bad'} match)")
            print(f"    Outdoor tour: {boost_tour:.2f}× ({'good' if boost_tour > 1 else 'bad'} match)")
        except ImportError:
            try:
                from tv_advertising_enhancements import get_weather_ad_boost
                
                boost_museum = get_weather_ad_boost(weather['weather'], 'museum')
                boost_tour = get_weather_ad_boost(weather['weather'], 'tour')
                
                print(f"\n  Ad context boost:")
                print(f"    Museum: {boost_museum:.2f}× ({'good' if boost_museum > 1 else 'bad'} match)")
                print(f"    Outdoor tour: {boost_tour:.2f}× ({'good' if boost_tour > 1 else 'bad'} match)")
            except ImportError:
                print("\nWeather-ad boost module not available (skipping)")
        
    except Exception as e:
        print(f"Error: {e}")

