'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { 
  AD_CATEGORIES, 
  SEGMENT_TYPES, 
  OBJECTIVES, 
  WEATHER_CONDITIONS, 
  AGE_GROUPS, 
  LANGUAGES, 
  QR_PLACEMENTS,
  COUNTRIES,
  SWISS_CITIES,
  GERMAN_CITIES,
  AUSTRIAN_CITIES,
  FRENCH_CITIES,
  ITALIAN_CITIES
} from '@/lib/types';

/**
 * Create Ad Campaign Page
 * Form for advertisers to create new campaigns with advanced targeting options
 */

export default function CreateAdCampaign() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Form state
  const [headline, setHeadline] = useState('');
  const [ctaText, setCtaText] = useState('');
  const [fileUrl, setFileUrl] = useState('');
  const [qrCodeUrl, setQrCodeUrl] = useState('');
  const [qrPlacement, setQrPlacement] = useState<string>('Bottom Right');
  const [category, setCategory] = useState<string>('');
  const [segments, setSegments] = useState<string[]>([]);
  const [weatherTargeting, setWeatherTargeting] = useState<string[]>(['any']);
  const [ageTargeting, setAgeTargeting] = useState<string[]>([]);
  const [languageTargeting, setLanguageTargeting] = useState<string[]>([]);
  const [country, setCountry] = useState('Switzerland');
  const [city, setCity] = useState('');
  const [locationType, setLocationType] = useState<'radius' | 'district'>('district');
  const [locationRadius, setLocationRadius] = useState(5);
  const [objective, setObjective] = useState<'awareness' | 'footfall' | 'performance'>('awareness');

  // Get cities based on selected country
  const getCitiesForCountry = (country: string) => {
    switch (country) {
      case 'Switzerland': return SWISS_CITIES;
      case 'Germany': return GERMAN_CITIES;
      case 'Austria': return AUSTRIAN_CITIES;
      case 'France': return FRENCH_CITIES;
      case 'Italy': return ITALIAN_CITIES;
      default: return SWISS_CITIES;
    }
  };

  const availableCities = getCitiesForCountry(country);

  // Reset city when country changes
  const handleCountryChange = (newCountry: string) => {
    setCountry(newCountry);
    setCity(''); // Reset city selection
  };

  // Cost calculation (synthetic)
  const baseCostPer100 = 5.50; // CHF
  const costMultiplier = 
    (segments.length * 0.1) + 
    (weatherTargeting.length > 1 ? 0.15 : 0) +
    (ageTargeting.length * 0.05) +
    (locationType === 'radius' ? 0.2 : 0);
  
  const costPer100Impressions = (baseCostPer100 * (1 + costMultiplier)).toFixed(2);
  const costPerCTR = ((parseFloat(costPer100Impressions) / 2.5) * 100).toFixed(2); // Assuming 2.5% CTR

  // Estimated reach (synthetic)
  const calculateEstimatedReach = () => {
    let base = 150;
    base += segments.length * 120;
    base += weatherTargeting.length > 1 ? 50 : 0;
    base += ageTargeting.length * 30;
    base += languageTargeting.length * 40;
    base += locationType === 'radius' ? locationRadius * 20 : 100;
    return base + Math.floor(Math.random() * 150);
  };

  const estimatedReach = calculateEstimatedReach();
  const estimatedHotels = Math.max(2, Math.floor(segments.length * 1.5) + (locationType === 'district' ? 3 : 1));

  const handleSegmentToggle = (segment: string) => {
    if (segments.includes(segment)) {
      setSegments(segments.filter(s => s !== segment));
    } else {
      setSegments([...segments, segment]);
    }
  };

  const handleWeatherToggle = (weather: string) => {
    if (weather === 'any') {
      setWeatherTargeting(['any']);
    } else {
      const filtered = weatherTargeting.filter(w => w !== 'any');
      if (filtered.includes(weather)) {
        const newWeather = filtered.filter(w => w !== weather);
        setWeatherTargeting(newWeather.length > 0 ? newWeather : ['any']);
      } else {
        setWeatherTargeting([...filtered, weather]);
      }
    }
  };

  const handleAgeToggle = (age: string) => {
    if (ageTargeting.includes(age)) {
      setAgeTargeting(ageTargeting.filter(a => a !== age));
    } else {
      setAgeTargeting([...ageTargeting, age]);
    }
  };

  const handleLanguageToggle = (lang: string) => {
    if (languageTargeting.includes(lang)) {
      setLanguageTargeting(languageTargeting.filter(l => l !== lang));
    } else {
      setLanguageTargeting([...languageTargeting, lang]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!headline || !ctaText || !category || segments.length === 0 || !country || !city || !objective) {
      setError('Please fill in all required fields');
      return;
    }

    if (ageTargeting.length === 0) {
      setError('Please select at least one age group');
      return;
    }

    if (languageTargeting.length === 0) {
      setError('Please select at least one language');
      return;
    }

    setLoading(true);

    try {
      const userStr = sessionStorage.getItem('user');
      if (!userStr) {
        setError('User not found');
        setLoading(false);
        return;
      }

      const user = JSON.parse(userStr);

      const res = await fetch('/api/ads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          advertiser_id: user.id,
          headline,
          cta_text: ctaText,
          file_url: fileUrl || '/uploads/placeholder.jpg',
          qr_code_url: qrCodeUrl,
          qr_placement: qrPlacement,
          category,
          segments,
          weather_targeting: weatherTargeting,
          age_targeting: ageTargeting,
          language_targeting: languageTargeting,
          country,
          city,
          location: `${city}, ${country}`,
          location_type: locationType,
          location_radius: locationType === 'radius' ? locationRadius : undefined,
          objective,
        }),
      });

      if (res.ok) {
        router.push('/advertiser/dashboard');
      } else {
        setError('Failed to create campaign');
        setLoading(false);
      }
    } catch (err) {
      setError('An error occurred');
      setLoading(false);
    }
  };

  const categoryLabels: Record<string, string> = {
    restaurants: 'Restaurants & Dining',
    experiences: 'Experiences & Activities',
    museums: 'Museums & Culture',
    mobility: 'Mobility & Transport',
    retail: 'Retail & Shopping',
    events: 'Events & Entertainment',
    nightlife: 'Nightlife & Bars'
  };

  return (
    <DashboardLayout requiredRole="advertiser">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Create New Campaign</h1>
          <p className="mt-2 text-gray-600">
            Design a privacy-preserving campaign with precise targeting
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Creative Content */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Ad Creative</h2>

            <div className="space-y-6">
              <div>
                <label className="label">
                  Headline *
                  <InfoTooltip text="A compelling headline that describes your offering (max 60 characters)" />
                </label>
                <input
                  type="text"
                  value={headline}
                  onChange={(e) => setHeadline(e.target.value)}
                  maxLength={60}
                  placeholder="e.g., Authentic Swiss & Italian Cuisine"
                  className="input"
                  required
                />
                <p className="mt-1 text-xs text-gray-500">{headline.length}/60 characters</p>
              </div>

              <div>
                <label className="label">
                  Call-to-Action Text *
                  <InfoTooltip text="What action should guests take? (max 30 characters)" />
                </label>
                <input
                  type="text"
                  value={ctaText}
                  onChange={(e) => setCtaText(e.target.value)}
                  maxLength={30}
                  placeholder="e.g., Reserve Your Table"
                  className="input"
                  required
                />
                <p className="mt-1 text-xs text-gray-500">{ctaText.length}/30 characters</p>
              </div>

              <div>
                <label className="label">
                  Creative Image/Video URL
                  <InfoTooltip text="URL to your ad creative. Leave blank to use placeholder." />
                </label>
                <input
                  type="url"
                  value={fileUrl}
                  onChange={(e) => setFileUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                  className="input"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Recommended: 1920x1080px, JPG or PNG
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="label">
                    QR Code Link
                    <InfoTooltip text="Direct link guests will visit when scanning the QR code" />
                  </label>
                  <input
                    type="url"
                    value={qrCodeUrl}
                    onChange={(e) => setQrCodeUrl(e.target.value)}
                    placeholder="https://yourbusiness.com/special-offer"
                    className="input"
                  />
                </div>

                <div>
                  <label className="label">
                    QR Code Position on Ad *
                    <InfoTooltip text="Where the QR code will appear on your ad creative" />
                  </label>
                  <select
                    value={qrPlacement}
                    onChange={(e) => setQrPlacement(e.target.value)}
                    className="input"
                    required
                  >
                    {QR_PLACEMENTS.map((placement) => (
                      <option key={placement} value={placement}>
                        {placement}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Category (Single Selection) */}
          <div className="card">
            <div className="flex items-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Content Category *</h2>
              <InfoTooltip text="Select the category that best describes your business" />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {AD_CATEGORIES.map((cat) => (
                <label
                  key={cat}
                  className={`flex items-center p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    category === cat
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="category"
                    value={cat}
                    checked={category === cat}
                    onChange={(e) => setCategory(e.target.value)}
                    className="w-4 h-4 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-3 text-sm font-medium text-gray-900">
                    {categoryLabels[cat] || cat}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Target Segments */}
          <div className="card">
            <div className="flex items-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Target Segments *</h2>
              <InfoTooltip text="Select guest segments most relevant to your offering" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {SEGMENT_TYPES.map((segment) => (
                <label
                  key={segment}
                  className={`flex items-center p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    segments.includes(segment)
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={segments.includes(segment)}
                    onChange={() => handleSegmentToggle(segment)}
                    className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                  />
                  <span className="ml-3 text-sm font-medium text-gray-900">{segment}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Advanced Targeting */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Advanced Targeting</h2>

            <div className="space-y-8">
              {/* Weather Targeting */}
              <div>
                <div className="flex items-center mb-4">
                  <label className="label mb-0">Weather Conditions</label>
                  <InfoTooltip text="Show ads based on current weather conditions" />
                </div>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-3">
                  {WEATHER_CONDITIONS.map((weather) => (
                    <label
                      key={weather}
                      className={`flex items-center justify-center p-3 border-2 rounded-lg cursor-pointer transition-all ${
                        weatherTargeting.includes(weather)
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={weatherTargeting.includes(weather)}
                        onChange={() => handleWeatherToggle(weather)}
                        className="sr-only"
                      />
                      <span className="text-sm font-medium text-gray-900 capitalize">{weather}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Age Targeting */}
              <div>
                <div className="flex items-center mb-4">
                  <label className="label mb-0">Age Groups *</label>
                  <InfoTooltip text="Target specific age demographics" />
                </div>
                <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
                  {AGE_GROUPS.map((age) => (
                    <label
                      key={age}
                      className={`flex items-center justify-center p-3 border-2 rounded-lg cursor-pointer transition-all ${
                        ageTargeting.includes(age)
                          ? 'border-purple-500 bg-purple-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={ageTargeting.includes(age)}
                        onChange={() => handleAgeToggle(age)}
                        className="sr-only"
                      />
                      <span className="text-sm font-medium text-gray-900">{age}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Language Targeting */}
              <div>
                <div className="flex items-center mb-4">
                  <label className="label mb-0">Languages *</label>
                  <InfoTooltip text="Target guests by their preferred language" />
                </div>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  {LANGUAGES.map((lang) => (
                    <label
                      key={lang}
                      className={`flex items-center justify-center p-3 border-2 rounded-lg cursor-pointer transition-all ${
                        languageTargeting.includes(lang)
                          ? 'border-green-500 bg-green-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={languageTargeting.includes(lang)}
                        onChange={() => handleLanguageToggle(lang)}
                        className="sr-only"
                      />
                      <span className="text-sm font-medium text-gray-900">{lang}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Location & Objective */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Location Targeting *</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="label">
                    Country *
                    <InfoTooltip text="Select the country where your business operates" />
                  </label>
                  <select
                    value={country}
                    onChange={(e) => handleCountryChange(e.target.value)}
                    className="input"
                    required
                  >
                    {COUNTRIES.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="label">
                    City *
                    <InfoTooltip text="Select the city where your business is located" />
                  </label>
                  <select
                    value={city}
                    onChange={(e) => setCity(e.target.value)}
                    className="input"
                    required
                  >
                    <option value="">Select city</option>
                    {availableCities.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="label">
                    Targeting Type *
                  </label>
                  <div className="space-y-3">
                    <label className="flex items-start p-3 border-2 rounded-lg cursor-pointer hover:border-gray-300">
                      <input
                        type="radio"
                        name="locationType"
                        value="district"
                        checked={locationType === 'district'}
                        onChange={(e) => setLocationType(e.target.value as 'district')}
                        className="mt-1 w-4 h-4 text-primary-600 focus:ring-primary-500"
                      />
                      <div className="ml-3">
                        <span className="text-sm font-medium text-gray-900">Whole City</span>
                        <p className="text-xs text-gray-600 mt-1">Reach all hotels in {city || 'the city'}</p>
                      </div>
                    </label>

                    <label className="flex items-start p-3 border-2 rounded-lg cursor-pointer hover:border-gray-300">
                      <input
                        type="radio"
                        name="locationType"
                        value="radius"
                        checked={locationType === 'radius'}
                        onChange={(e) => setLocationType(e.target.value as 'radius')}
                        className="mt-1 w-4 h-4 text-primary-600 focus:ring-primary-500"
                      />
                      <div className="ml-3 flex-1">
                        <span className="text-sm font-medium text-gray-900">Radius from Business</span>
                        <p className="text-xs text-gray-600 mt-1">Target hotels within a specific radius</p>
                        {locationType === 'radius' && (
                          <div className="mt-3">
                            <input
                              type="range"
                              min="1"
                              max="20"
                              value={locationRadius}
                              onChange={(e) => setLocationRadius(parseInt(e.target.value))}
                              className="w-full"
                            />
                            <p className="text-sm text-gray-700 mt-1">{locationRadius} km radius</p>
                          </div>
                        )}
                      </div>
                    </label>
                  </div>
                </div>
              </div>
            </div>

            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Campaign Objective *</h3>
              <div className="space-y-3">
                {OBJECTIVES.map((obj) => (
                  <label key={obj} className="flex items-start p-3 border-2 rounded-lg cursor-pointer hover:border-gray-300">
                    <input
                      type="radio"
                      name="objective"
                      value={obj}
                      checked={objective === obj}
                      onChange={(e) => setObjective(e.target.value as any)}
                      className="mt-1 w-4 h-4 text-primary-600 focus:ring-primary-500"
                    />
                    <div className="ml-3">
                      <span className="text-sm font-medium text-gray-900 capitalize">{obj}</span>
                      <p className="text-xs text-gray-600 mt-1">
                        {obj === 'awareness' && 'Build brand recognition'}
                        {obj === 'footfall' && 'Drive visits to your location'}
                        {obj === 'performance' && 'Maximize conversions and bookings'}
                      </p>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Cost & Estimated Reach */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Cost Overview */}
            <div className="card bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Cost Overview</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Per 100 Impressions</p>
                    <p className="text-xs text-gray-500 mt-1">Based on your targeting</p>
                  </div>
                  <p className="text-2xl font-bold text-green-600">
                    CHF {costPer100Impressions}
                  </p>
                </div>
                <div className="pt-4 border-t border-green-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Per Click (CTR)</p>
                      <p className="text-xs text-gray-500 mt-1">Assuming 2.5% CTR</p>
                    </div>
                    <p className="text-2xl font-bold text-green-600">
                      CHF {costPerCTR}
                    </p>
                  </div>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-600">
                * Prices dynamically calculated based on targeting complexity
              </p>
            </div>

            {/* Estimated Reach */}
            <div className="card bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Estimated Reach</h3>
              <div className="space-y-4">
                <div>
                  <p className="text-sm text-gray-600">Weekly Impressions</p>
                  <p className="text-2xl font-bold text-primary-600">
                    {estimatedReach.toLocaleString()}
                  </p>
                </div>
                <div className="pt-4 border-t border-primary-200">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Available Hotels</p>
                      <p className="text-2xl font-bold text-primary-600">{estimatedHotels}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600">Est. CTR</p>
                      <p className="text-2xl font-bold text-primary-600">
                        {(Math.random() * 2 + 1.5).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-600">
                * Estimates based on similar campaigns and targeting
              </p>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="card bg-red-50 border-red-200 text-red-700">
              {error}
            </div>
          )}

          {/* Submit */}
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => router.back()}
              className="btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Creating...' : 'Create Campaign'}
            </button>
          </div>
        </form>
      </div>
    </DashboardLayout>
  );
}
