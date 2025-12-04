'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { Ad } from '@/lib/types';

/**
 * Hotel Ad Monitoring Page
 * View and manage ads currently running in the hotel
 * Ability to remove ads that violate policies
 */

export default function HotelAdsMonitoring() {
  const [ads, setAds] = useState<Ad[]>([]);
  const [loading, setLoading] = useState(true);
  const [removing, setRemoving] = useState<number | null>(null);

  useEffect(() => {
    loadAds();
  }, []);

  const loadAds = async () => {
    try {
      // Get all ads (in production, filter by hotel location/settings)
      const res = await fetch('/api/ads');
      const data = await res.json();
      setAds(data.filter((ad: Ad) => ad.status === 'active'));
      setLoading(false);
    } catch (error) {
      console.error('Failed to load ads:', error);
      setLoading(false);
    }
  };

  const handleRemoveAd = async (adId: number, headline: string) => {
    if (!confirm(`Are you sure you want to remove "${headline}"? This ad violates your hotel policies.`)) {
      return;
    }

    setRemoving(adId);

    try {
      const res = await fetch(`/api/ads?adId=${adId}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setAds(ads.filter(ad => ad.id !== adId));
        alert('Ad successfully removed from your hotel');
      } else {
        alert('Failed to remove ad');
      }
    } catch (error) {
      console.error('Failed to remove ad:', error);
      alert('Failed to remove ad');
    } finally {
      setRemoving(null);
    }
  };

  if (loading) {
    return (
      <DashboardLayout requiredRole="hotel">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading ads...</div>
        </div>
      </DashboardLayout>
    );
  }

  const externalAds = ads.filter(ad => !ad.created_by_hotel);
  const hotelAds = ads.filter(ad => ad.created_by_hotel);

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Ad Monitoring</h1>
          <p className="mt-2 text-gray-600">
            Review and manage ads currently displayed in your hotel
          </p>
        </div>

        {/* Info Banner */}
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-900">Ad Quality Control</h3>
              <div className="mt-2 text-sm text-blue-800">
                <p>
                  All ads are pre-filtered by your allowed categories and exposure rules. 
                  You can remove any ad that violates your policies or doesn't align with your brand values.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* External Advertiser Ads */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <h2 className="text-xl font-bold text-gray-900">External Advertiser Ads</h2>
              <InfoTooltip text="Ads from external businesses targeting your guests" />
            </div>
            <span className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium">
              {externalAds.length} active
            </span>
          </div>

          {externalAds.length === 0 ? (
            <div className="card text-center py-12">
              <p className="text-gray-600">No external ads currently running</p>
            </div>
          ) : (
            <div className="space-y-4">
              {externalAds.map((ad) => (
                <div key={ad.id} className="card hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-semibold text-gray-900">{ad.headline}</h3>
                        <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                          Active
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{ad.cta_text}</p>

                      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <p className="text-xs text-gray-500">Category</p>
                          <p className="text-sm font-medium text-gray-900 capitalize">{ad.category}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Location</p>
                          <p className="text-sm font-medium text-gray-900">{ad.location}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Impressions</p>
                          <p className="text-sm font-medium text-gray-900">{ad.impressions.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">QR Placement</p>
                          <p className="text-sm font-medium text-gray-900">{ad.qr_placement || 'N/A'}</p>
                        </div>
                      </div>

                      <div className="mt-4">
                        <p className="text-xs text-gray-500 mb-2">Target Segments</p>
                        <div className="flex flex-wrap gap-2">
                          {ad.segments.map((seg, idx) => (
                            <span key={idx} className="px-2 py-1 text-xs bg-primary-50 text-primary-700 rounded">
                              {seg}
                            </span>
                          ))}
                        </div>
                      </div>

                      {ad.weather_targeting && ad.weather_targeting.length > 0 && (
                        <div className="mt-3">
                          <p className="text-xs text-gray-500 mb-2">Weather Targeting</p>
                          <div className="flex flex-wrap gap-2">
                            {ad.weather_targeting.map((weather, idx) => (
                              <span key={idx} className="px-2 py-1 text-xs bg-blue-50 text-blue-700 rounded capitalize">
                                {weather}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    <button
                      onClick={() => handleRemoveAd(ad.id, ad.headline)}
                      disabled={removing === ad.id}
                      className="ml-4 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {removing === ad.id ? 'Removing...' : 'Remove Ad'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Hotel-Created Content */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <h2 className="text-xl font-bold text-gray-900">Your Hotel Content</h2>
              <InfoTooltip text="Content created by your hotel for upselling services" />
            </div>
            <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
              {hotelAds.length} active
            </span>
          </div>

          {hotelAds.length === 0 ? (
            <div className="card text-center py-12">
              <p className="text-gray-600 mb-4">No hotel content created yet</p>
              <a href="/hotel/content" className="btn-primary inline-flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Create Hotel Content
              </a>
            </div>
          ) : (
            <div className="space-y-4">
              {hotelAds.map((ad) => (
                <div key={ad.id} className="card bg-primary-50 border-primary-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-semibold text-gray-900">{ad.headline}</h3>
                        <span className="px-2 py-1 text-xs font-medium bg-primary-200 text-primary-900 rounded">
                          Hotel Content
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{ad.cta_text}</p>

                      <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div>
                          <p className="text-xs text-gray-500">Category</p>
                          <p className="text-sm font-medium text-gray-900 capitalize">{ad.category}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">Impressions</p>
                          <p className="text-sm font-medium text-gray-900">{ad.impressions.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500">QR Placement</p>
                          <p className="text-sm font-medium text-gray-900">{ad.qr_placement || 'N/A'}</p>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => handleRemoveAd(ad.id, ad.headline)}
                      disabled={removing === ad.id}
                      className="ml-4 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {removing === ad.id ? 'Removing...' : 'Pause'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Policy Guidelines */}
        <div className="card bg-yellow-50 border-yellow-200">
          <h3 className="text-sm font-medium text-yellow-900 mb-2">When to Remove an Ad</h3>
          <ul className="text-sm text-yellow-800 space-y-1 list-disc list-inside">
            <li>Content violates your brand values or hotel policies</li>
            <li>Inappropriate or misleading messaging</li>
            <li>Poor quality creative or unclear call-to-action</li>
            <li>Business has received negative guest feedback</li>
            <li>Ad doesn't align with your allowed categories</li>
          </ul>
        </div>
      </div>
    </DashboardLayout>
  );
}





