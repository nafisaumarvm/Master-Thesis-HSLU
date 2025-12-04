'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import { Ad } from '@/lib/types';

/**
 * Advertiser Campaigns Page
 * Detailed view of all advertising campaigns
 */

export default function AdvertiserCampaigns() {
  const [ads, setAds] = useState<Ad[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'active' | 'paused'>('all');

  useEffect(() => {
    loadAds();
  }, []);

  const loadAds = async () => {
    try {
      const userStr = sessionStorage.getItem('user');
      if (!userStr) return;

      const user = JSON.parse(userStr);
      const res = await fetch(`/api/ads?userId=${user.id}`);
      const data = await res.json();
      setAds(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load ads:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardLayout requiredRole="advertiser">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading campaigns...</div>
        </div>
      </DashboardLayout>
    );
  }

  const filteredAds = ads.filter(ad => {
    if (filter === 'all') return true;
    return ad.status === filter;
  });

  return (
    <DashboardLayout requiredRole="advertiser">
      <div className="space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">All Campaigns</h1>
            <p className="mt-2 text-gray-600">
              Detailed performance metrics for your advertising campaigns
            </p>
          </div>

          {/* Filter */}
          <div className="flex space-x-2">
            <button
              onClick={() => setFilter('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === 'all'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              All ({ads.length})
            </button>
            <button
              onClick={() => setFilter('active')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === 'active'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Active ({ads.filter(a => a.status === 'active').length})
            </button>
          </div>
        </div>

        {/* Campaigns Table */}
        {filteredAds.length === 0 ? (
          <div className="card text-center py-12">
            <p className="text-gray-600">No campaigns found</p>
          </div>
        ) : (
          <div className="card overflow-hidden p-0">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Campaign
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Objective
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Impressions
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      CTR %
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Segments
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredAds.map((ad) => (
                    <tr key={ad.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4">
                        <div>
                          <div className="text-sm font-medium text-gray-900">{ad.headline}</div>
                          <div className="text-sm text-gray-500">{ad.cta_text}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs font-medium rounded ${
                          ad.status === 'active'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {ad.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col space-y-1">
                          <span className="px-2 py-1 text-xs bg-purple-50 text-purple-700 rounded capitalize inline-block">
                            {ad.objective}
                          </span>
                          <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded capitalize inline-block">
                            {ad.category}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {ad.impressions.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {ad.ctr_estimate}%
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        <div className="max-w-xs truncate">
                          {ad.segments.join(', ')}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(ad.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Performance Summary */}
        {filteredAds.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card">
              <h3 className="text-sm font-medium text-gray-600">Total Impressions</h3>
              <p className="mt-2 text-2xl font-bold text-gray-900">
                {filteredAds.reduce((sum, ad) => sum + ad.impressions, 0).toLocaleString()}
              </p>
            </div>
            <div className="card">
              <h3 className="text-sm font-medium text-gray-600">Average CTR</h3>
              <p className="mt-2 text-2xl font-bold text-gray-900">
                {(filteredAds.reduce((sum, ad) => sum + ad.ctr_estimate, 0) / filteredAds.length).toFixed(2)}%
              </p>
            </div>
            <div className="card">
              <h3 className="text-sm font-medium text-gray-600">Active Campaigns</h3>
              <p className="mt-2 text-2xl font-bold text-gray-900">
                {filteredAds.filter(ad => ad.status === 'active').length}
              </p>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}

