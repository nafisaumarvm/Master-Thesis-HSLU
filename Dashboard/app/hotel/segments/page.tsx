'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { Segment } from '@/lib/types';

/**
 * Hotel Segments Page
 * Detailed view of guest segments with characteristics
 */

export default function HotelSegments() {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSegments();
  }, []);

  const loadSegments = async () => {
    try {
      const res = await fetch('/api/segments');
      const data = await res.json();
      setSegments(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load segments:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <DashboardLayout requiredRole="hotel">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading segments...</div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Guest Segments</h1>
          <p className="mt-2 text-gray-600">
            Detailed breakdown of anonymized guest segments and their characteristics
          </p>
        </div>

        {/* Privacy Notice */}
        <div className="card bg-blue-50 border-blue-200">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-900">Privacy-Preserving Segmentation</h3>
              <div className="mt-2 text-sm text-blue-800">
                <p>
                  Segments are created using anonymized booking and stay patterns. No personally identifiable 
                  information (PII) is collected, stored, or shared with advertisers. All data is aggregated 
                  and automatically deleted after guest checkout.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Segments List */}
        <div className="space-y-6">
          {segments.map((segment) => (
            <div key={segment.id} className="card">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center">
                    <h2 className="text-xl font-bold text-gray-900">{segment.name}</h2>
                    <InfoTooltip text={`Represents ${segment.percentage}% of your guest base based on booking patterns and stay characteristics.`} />
                  </div>
                  <p className="mt-2 text-gray-600">{segment.description}</p>
                </div>
                <div className="ml-6">
                  <div className="flex flex-col items-center">
                    <div className="text-4xl font-bold text-primary-600">
                      {segment.percentage}%
                    </div>
                    <div className="text-xs text-gray-500 mt-1">of guests</div>
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-primary-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${segment.percentage}%` }}
                  ></div>
                </div>
              </div>

              {/* Characteristics */}
              <div className="mt-6">
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Key Characteristics:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {segment.characteristics.map((char, idx) => (
                    <div key={idx} className="flex items-start">
                      <svg className="w-5 h-5 mr-2 text-primary-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm text-gray-700">{char}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Targeting Info */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <p className="text-xs text-gray-500">
                  Advertisers can target this segment with contextually relevant, non-intrusive content 
                  based on your allowed categories and exposure rules.
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Summary Statistics */}
        <div className="card bg-gradient-to-r from-gray-50 to-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Segment Distribution</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Total Segments</span>
              <span className="font-semibold text-gray-900">{segments.length}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Coverage</span>
              <span className="font-semibold text-gray-900">100%</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Data Retention</span>
              <span className="font-semibold text-green-600">0 days post-checkout</span>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}





