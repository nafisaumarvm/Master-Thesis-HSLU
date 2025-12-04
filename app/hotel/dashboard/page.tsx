'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { Analytics, Segment } from '@/lib/types';

/**
 * Hotel Admin Dashboard
 * Main overview page with segments, analytics, and privacy controls
 */

export default function HotelDashboard() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      const userStr = sessionStorage.getItem('user');
      if (!userStr) return;

      const user = JSON.parse(userStr);

      // Load analytics
      const analyticsRes = await fetch(`/api/hotel/analytics?userId=${user.id}`);
      const analyticsData = await analyticsRes.json();
      setAnalytics(analyticsData);

      // Load segments
      const segmentsRes = await fetch('/api/segments');
      const segmentsData = await segmentsRes.json();
      setSegments(segmentsData);

      setLoading(false);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardLayout requiredRole="hotel">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading dashboard...</div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard Overview</h1>
          <p className="mt-2 text-gray-600">
            Privacy-preserving guest segment insights and advertising performance
          </p>
        </div>

        {/* Revenue Card - Highlighted */}
        <div className="card bg-gradient-to-r from-green-50 to-emerald-50 border-green-200">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center">
                <p className="text-sm font-medium text-gray-600">Revenue Generated</p>
                <InfoTooltip text="Your hotel receives 30% of all advertising spend from campaigns displayed in your property" />
              </div>
              <p className="mt-2 text-4xl font-bold text-green-600">
                CHF {analytics?.revenue_generated?.toFixed(2) || '0.00'}
              </p>
              <p className="mt-2 text-xs text-gray-600">
                From CHF {analytics?.total_ad_spend?.toFixed(2) || '0.00'} total ad spend
              </p>
            </div>
            <div className="w-16 h-16 bg-green-100 rounded-lg flex items-center justify-center">
              <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-green-200">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-600">Your Revenue Share</span>
              <span className="font-semibold text-green-700">{analytics?.revenue_share_percentage || 30}%</span>
            </div>
          </div>
        </div>

        {/* Analytics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Impressions This Week</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {analytics?.impressions_this_week.toLocaleString() || 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              Non-intrusive ad exposure on welcome screens
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Estimated Conversions</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {analytics?.estimated_conversions || 0}
                </p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              Guests who engaged with advertised services
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Top Category</p>
                <p className="mt-2 text-2xl font-bold text-gray-900 capitalize">
                  {analytics?.top_category || 'N/A'}
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              Best performing content category
            </div>
          </div>

          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Occupancy Impact</p>
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {analytics?.occupancy_multiplier.toFixed(2)}x
                </p>
              </div>
              <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500">
              Current occupancy rate multiplier
            </div>
          </div>
        </div>

        {/* Guest Segments Overview */}
        <div>
          <div className="flex items-center mb-4">
            <h2 className="text-xl font-bold text-gray-900">Guest Segments</h2>
            <InfoTooltip text="Synthetic, anonymized guest segments based on booking patterns and stay characteristics. No personally identifiable information is collected or stored." />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {segments.map((segment) => (
              <div key={segment.id} className="card hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900">{segment.name}</h3>
                    <p className="mt-1 text-sm text-gray-600">{segment.description}</p>
                  </div>
                  <div className="ml-4">
                    <div className="text-2xl font-bold text-primary-600">
                      {segment.percentage}%
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{ width: `${segment.percentage}%` }}
                    ></div>
                  </div>
                </div>

                <div className="mt-4 space-y-1">
                  {segment.characteristics.slice(0, 3).map((char, idx) => (
                    <div key={idx} className="flex items-center text-xs text-gray-600">
                      <svg className="w-3 h-3 mr-2 text-primary-500" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      {char}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Revenue Breakdown */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold text-gray-900">Revenue Breakdown</h2>
            <InfoTooltip text="Your hotel earns 30% of all advertising revenue from campaigns displayed in your property" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="card">
              <div className="flex items-center justify-between mb-2">
                <p className="text-sm font-medium text-gray-600">This Week</p>
                <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded">30% share</span>
              </div>
              <p className="text-2xl font-bold text-green-600">
                CHF {analytics?.revenue_generated?.toFixed(2) || '0.00'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                From {analytics?.impressions_this_week.toLocaleString() || 0} impressions
              </p>
            </div>

            <div className="card">
              <p className="text-sm font-medium text-gray-600 mb-2">Projected Monthly</p>
              <p className="text-2xl font-bold text-gray-900">
                CHF {((analytics?.revenue_generated || 0) * 4.33).toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Based on current weekly average
              </p>
            </div>

            <div className="card">
              <p className="text-sm font-medium text-gray-600 mb-2">Top Earning Category</p>
              <p className="text-2xl font-bold text-primary-600 capitalize">
                {analytics?.top_category || 'N/A'}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Highest performing category
              </p>
            </div>
          </div>

          <div className="mt-6 card bg-blue-50 border-blue-200">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-900">Revenue Share Model</h3>
                <div className="mt-2 text-sm text-blue-800">
                  <p>
                    Your hotel automatically receives <strong>30% of all advertising spend</strong> from campaigns 
                    displayed on your Smart TVs. Advertisers pay based on impressions and targeting complexity. 
                    Revenue is calculated automatically and transparently.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="card bg-gradient-to-r from-primary-50 to-blue-50 border-primary-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Privacy-First Advertising</h3>
              <p className="mt-1 text-sm text-gray-600">
                All ad exposure is contextually appropriate, non-intrusive, and fully GDPR-compliant.
                Guest data is automatically deleted post-checkout.
              </p>
            </div>
            <div className="ml-4">
              <svg className="w-12 h-12 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}

