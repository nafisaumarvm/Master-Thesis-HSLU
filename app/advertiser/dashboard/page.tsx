'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import DashboardLayout from '@/components/DashboardLayout';
import { Ad } from '@/lib/types';

/**
 * Advertiser Dashboard
 * Overview of campaigns and performance metrics
 */

export default function AdvertiserDashboard() {
  const [ads, setAds] = useState<Ad[]>([]);
  const [loading, setLoading] = useState(true);

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
          <div className="text-gray-500">Loading dashboard...</div>
        </div>
      </DashboardLayout>
    );
  }

  const totalImpressions = ads.reduce((sum, ad) => sum + ad.impressions, 0);
  const averageCTR = ads.length > 0 
    ? (ads.reduce((sum, ad) => sum + ad.ctr_estimate, 0) / ads.length).toFixed(2)
    : '0.00';
  const activeAds = ads.filter(ad => ad.status === 'active').length;
  
  // Cost calculations (synthetic)
  const avgCostPer100 = 6.80; // CHF
  const totalCost = ((totalImpressions / 100) * avgCostPer100).toFixed(2);
  const costPerClick = totalImpressions > 0 && parseFloat(averageCTR) > 0
    ? ((parseFloat(totalCost) / (totalImpressions * (parseFloat(averageCTR) / 100)))).toFixed(2)
    : '0.00';

  return (
    <DashboardLayout requiredRole="advertiser">
      <div className="space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Campaign Overview</h1>
            <p className="mt-2 text-gray-600">
              Manage your privacy-preserving hotel advertising campaigns
            </p>
          </div>
          <Link href="/advertiser/create" className="btn-primary">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span>Create New Ad</span>
            </div>
          </Link>
        </div>

        {/* Cost Overview */}
        <div className="card bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Campaign Costs</h3>
              <p className="mt-1 text-sm text-gray-600">
                Total spend across all campaigns
              </p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-green-600">CHF {totalCost}</p>
              <p className="text-xs text-gray-500 mt-1">
                Avg. CHF {avgCostPer100.toFixed(2)}/100 impressions
              </p>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-green-200 grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-600">Cost per Click</p>
              <p className="text-lg font-semibold text-gray-900">CHF {costPerClick}</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Total Impressions</p>
              <p className="text-lg font-semibold text-gray-900">{totalImpressions.toLocaleString()}</p>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Campaigns</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">{activeAds}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Impressions</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {totalImpressions.toLocaleString()}
                </p>
              </div>
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg. CTR</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">{averageCTR}%</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Hotels Reached</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {ads.length > 0 ? Math.floor(Math.random() * 15) + 5 : 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Campaign List */}
        {ads.length === 0 ? (
          <div className="card text-center py-12">
            <div className="flex justify-center mb-4">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5.882V19.24a1.76 1.76 0 01-3.417.592l-2.147-6.15M18 13a3 3 0 100-6M5.436 13.683A4.001 4.001 0 017 6h1.832c4.1 0 7.625-1.234 9.168-3v14c-1.543-1.766-5.067-3-9.168-3H7a3.988 3.988 0 01-1.564-.317z" />
                </svg>
              </div>
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">No campaigns yet</h3>
            <p className="text-gray-600 mb-6">
              Create your first campaign to reach hotel guests with contextually relevant content
            </p>
            <Link href="/advertiser/create" className="btn-primary inline-flex items-center">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Create Your First Campaign
            </Link>
          </div>
        ) : (
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Your Campaigns</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {ads.map((ad) => (
                <div key={ad.id} className="card hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h3 className="text-lg font-semibold text-gray-900">{ad.headline}</h3>
                        <span className={`px-2 py-1 text-xs font-medium rounded ${
                          ad.status === 'active' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {ad.status}
                        </span>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{ad.cta_text}</p>
                    </div>
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-gray-500">Impressions</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {ad.impressions.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">CTR Estimate</p>
                      <p className="text-lg font-semibold text-gray-900">{ad.ctr_estimate}%</p>
                    </div>
                  </div>

                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="flex flex-wrap gap-2">
                          <span className="px-2 py-1 text-xs bg-primary-50 text-primary-700 rounded">
                            {ad.location}
                          </span>
                          <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded capitalize">
                            {ad.category}
                          </span>
                          <span className="px-2 py-1 text-xs bg-purple-50 text-purple-700 rounded capitalize">
                            {ad.objective}
                          </span>
                        </div>
                      </div>

                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs text-gray-500">Target Segments</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {ad.segments.map((seg, idx) => (
                        <span key={idx} className="text-xs text-gray-700">
                          {seg}{idx < ad.segments.length - 1 ? ',' : ''}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Privacy Note */}
        <div className="card bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-900">Privacy-First Advertising</h3>
              <div className="mt-2 text-sm text-blue-800">
                <p>
                  All campaigns are privacy-preserving and GDPR-compliant. You receive aggregated, 
                  anonymized insights only. Individual guest data is never collected or shared.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

