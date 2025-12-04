'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { HotelSettings } from '@/lib/types';

/**
 * Hotel Privacy Center
 * Privacy settings and compliance information
 */

export default function HotelPrivacy() {
  const [settings, setSettings] = useState<HotelSettings | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const userStr = sessionStorage.getItem('user');
      if (!userStr) return;

      const user = JSON.parse(userStr);
      const res = await fetch(`/api/hotel/settings?userId=${user.id}`);
      const data = await res.json();
      setSettings(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load settings:', error);
      setLoading(false);
    }
  };

  if (loading || !settings) {
    return (
      <DashboardLayout requiredRole="hotel">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading privacy settings...</div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Privacy Center</h1>
          <p className="mt-2 text-gray-600">
            Our privacy-first approach to personalized advertising
          </p>
        </div>

        {/* GDPR Compliance Badge */}
        <div className="card bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center">
                <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
            </div>
            <div className="ml-6">
              <h2 className="text-xl font-bold text-gray-900">GDPR Compliant by Design</h2>
              <p className="mt-2 text-gray-700">
                Our system is built from the ground up to respect guest privacy and comply with 
                GDPR, CCPA, and other international privacy regulations.
              </p>
            </div>
          </div>
        </div>

        {/* Privacy Rules */}
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Privacy Protections</h2>

          <div className="space-y-6">
            {/* Anonymization */}
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center">
                  <h3 className="text-lg font-semibold text-gray-900">Automatic Anonymization</h3>
                  <InfoTooltip text="All guest data is immediately anonymized. No names, room numbers, or PII is ever stored or shared." />
                  <span className="ml-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                    Active
                  </span>
                </div>
                <p className="mt-2 text-gray-600">
                  Guest segments are created using anonymized patterns only. We never collect, store, or 
                  share personally identifiable information such as names, email addresses, or room numbers.
                </p>
                <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-700">
                    <strong>What we collect:</strong> Anonymized booking patterns (e.g., "weekday arrival", 
                    "business traveler profile")
                  </p>
                  <p className="text-sm text-gray-700 mt-1">
                    <strong>What we DON'T collect:</strong> Names, emails, phone numbers, room numbers, 
                    payment info, browsing history
                  </p>
                </div>
              </div>
            </div>

            {/* Auto-Delete */}
            <div className="flex items-start pt-6 border-t border-gray-200">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </div>
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center">
                  <h3 className="text-lg font-semibold text-gray-900">Post-Checkout Data Deletion</h3>
                  <InfoTooltip text="All guest data is automatically deleted immediately after checkout. Zero retention." />
                  <span className="ml-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                    Active
                  </span>
                </div>
                <p className="mt-2 text-gray-600">
                  Guest data is automatically and permanently deleted as soon as checkout is completed. 
                  This ensures zero data retention and maximum privacy protection.
                </p>
                <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-700">
                    <strong>Retention Period:</strong> {settings.privacy_rules.dataRetentionDays} days 
                    post-checkout
                  </p>
                  <p className="text-sm text-gray-700 mt-1">
                    <strong>Deletion Method:</strong> Automatic, permanent, and irreversible
                  </p>
                </div>
              </div>
            </div>

            {/* Anonymized Insights */}
            <div className="flex items-start pt-6 border-t border-gray-200">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center">
                  <h3 className="text-lg font-semibold text-gray-900">Anonymized Insights Only</h3>
                  <InfoTooltip text="Analytics are aggregated and anonymized. Individual guest behavior is never tracked or reported." />
                  <span className="ml-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                    {settings.privacy_rules.anonymizedInsights ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <p className="mt-2 text-gray-600">
                  Analytics and insights are provided at an aggregated, segment level only. Individual 
                  guest behavior is never tracked, stored, or reported to advertisers.
                </p>
                <div className="mt-3 p-3 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-700">
                    <strong>Advertisers see:</strong> Aggregated segment impressions (e.g., "Business 
                    Travelers: 500 impressions")
                  </p>
                  <p className="text-sm text-gray-700 mt-1">
                    <strong>Advertisers DON'T see:</strong> Individual guest data, room numbers, viewing 
                    times, or any PII
                  </p>
                </div>
              </div>
            </div>

            {/* No Third-Party Tracking */}
            <div className="flex items-start pt-6 border-t border-gray-200">
              <div className="flex-shrink-0">
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                  </svg>
                </div>
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center">
                  <h3 className="text-lg font-semibold text-gray-900">No Third-Party Tracking</h3>
                  <span className="ml-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                    Guaranteed
                  </span>
                </div>
                <p className="mt-2 text-gray-600">
                  We never use third-party tracking pixels, cookies, or any external analytics tools. 
                  All data stays within the system and is never shared with external parties.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Compliance Documentation */}
        <div className="card bg-blue-50 border-blue-200">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Compliance & Documentation</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-2">GDPR Compliance</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Data minimization
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Right to erasure (automatic)
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Purpose limitation
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Privacy by design
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-2">Guest Rights</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  No PII collected
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  No third-party sharing
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Transparent processing
                </li>
                <li className="flex items-start">
                  <svg className="w-4 h-4 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  Auto-deletion guaranteed
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Contact */}
        <div className="card">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Privacy Questions?</h2>
          <p className="text-gray-600">
            If you have questions about our privacy practices or need to review our data processing 
            agreements, please contact our privacy team.
          </p>
          <button className="mt-4 btn-secondary">
            Contact Privacy Team
          </button>
        </div>
      </div>
    </DashboardLayout>
  );
}





