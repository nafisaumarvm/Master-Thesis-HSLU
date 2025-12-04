'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { SEGMENT_TYPES, QR_PLACEMENTS } from '@/lib/types';

/**
 * Hotel Content Creation Page
 * Create promotional content for hotel services (upselling)
 * E.g., spa treatments, room service, hotel amenities
 */

export default function HotelContentCreation() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Form state
  const [headline, setHeadline] = useState('');
  const [ctaText, setCtaText] = useState('');
  const [fileUrl, setFileUrl] = useState('');
  const [qrCodeUrl, setQrCodeUrl] = useState('');
  const [qrPlacement, setQrPlacement] = useState<string>('Welcome Screen');
  const [serviceType, setServiceType] = useState<string>('');
  const [segments, setSegments] = useState<string[]>([]);
  const [description, setDescription] = useState('');

  const handleSegmentToggle = (segment: string) => {
    if (segments.includes(segment)) {
      setSegments(segments.filter(s => s !== segment));
    } else {
      setSegments([...segments, segment]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!headline || !ctaText || !serviceType || segments.length === 0) {
      setError('Please fill in all required fields');
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
          hotel_id: user.id,
          headline,
          cta_text: ctaText,
          file_url: fileUrl || '/uploads/hotel-service.jpg',
          qr_code_url: qrCodeUrl,
          qr_placement: qrPlacement,
          category: serviceType,
          segments,
          weather_targeting: ['any'],
          age_targeting: ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
          language_targeting: ['German', 'French', 'Italian', 'English'],
          location: 'Hotel Property',
          location_type: 'radius',
          location_radius: 0,
          objective: 'performance',
          created_by_hotel: true,
        }),
      });

      if (res.ok) {
        router.push('/hotel/ads');
      } else {
        setError('Failed to create content');
        setLoading(false);
      }
    } catch (err) {
      setError('An error occurred');
      setLoading(false);
    }
  };

  const hotelServiceTypes = [
    { value: 'wellness', label: 'Spa & Wellness', example: 'Massage treatments, sauna access' },
    { value: 'restaurants', label: 'Restaurant & Bar', example: 'Special menus, room service' },
    { value: 'experiences', label: 'Hotel Experiences', example: 'Tours, activities, concierge services' },
    { value: 'retail', label: 'Hotel Shop', example: 'Gift shop, souvenirs' },
    { value: 'events', label: 'Hotel Events', example: 'Wine tasting, cooking classes' },
  ];

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Create Hotel Content</h1>
          <p className="mt-2 text-gray-600">
            Promote your hotel services and amenities to increase guest engagement
          </p>
        </div>

        {/* Info Banner */}
        <div className="card bg-primary-50 border-primary-200">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-primary-900">Upselling Opportunities</h3>
              <div className="mt-2 text-sm text-primary-800">
                <p>
                  Use this tool to promote hotel services directly to your guests. Examples: spa treatments, 
                  restaurant reservations, room upgrades, local tours booked through concierge.
                </p>
              </div>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Service Type */}
          <div className="card">
            <div className="flex items-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Service Type *</h2>
              <InfoTooltip text="Select the type of hotel service you want to promote" />
            </div>

            <div className="space-y-3">
              {hotelServiceTypes.map((service) => (
                <label
                  key={service.value}
                  className={`flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all ${
                    serviceType === service.value
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="serviceType"
                    value={service.value}
                    checked={serviceType === service.value}
                    onChange={(e) => setServiceType(e.target.value)}
                    className="mt-1 w-5 h-5 text-primary-600 focus:ring-primary-500"
                  />
                  <div className="ml-3">
                    <span className="text-sm font-medium text-gray-900">{service.label}</span>
                    <p className="text-xs text-gray-600 mt-1">{service.example}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Content Details */}
          <div className="card">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Content Details</h2>

            <div className="space-y-6">
              <div>
                <label className="label">
                  Headline *
                  <InfoTooltip text="Catchy headline for your service (max 60 characters)" />
                </label>
                <input
                  type="text"
                  value={headline}
                  onChange={(e) => setHeadline(e.target.value)}
                  maxLength={60}
                  placeholder="e.g., Relax with a 90-Minute Spa Treatment"
                  className="input"
                  required
                />
                <p className="mt-1 text-xs text-gray-500">{headline.length}/60 characters</p>
              </div>

              <div>
                <label className="label">
                  Call-to-Action *
                  <InfoTooltip text="What should guests do? (max 30 characters)" />
                </label>
                <input
                  type="text"
                  value={ctaText}
                  onChange={(e) => setCtaText(e.target.value)}
                  maxLength={30}
                  placeholder="e.g., Book Now - 20% Off"
                  className="input"
                  required
                />
                <p className="mt-1 text-xs text-gray-500">{ctaText.length}/30 characters</p>
              </div>

              <div>
                <label className="label">
                  Description
                  <InfoTooltip text="Optional: Brief description of the service" />
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  placeholder="Describe your service offering..."
                  className="input"
                />
              </div>

              <div>
                <label className="label">
                  Image/Video URL
                  <InfoTooltip text="URL to your service image. Leave blank to use placeholder." />
                </label>
                <input
                  type="url"
                  value={fileUrl}
                  onChange={(e) => setFileUrl(e.target.value)}
                  placeholder="https://hotel.com/spa-image.jpg"
                  className="input"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="label">
                    Booking/Info Link (QR Code)
                    <InfoTooltip text="Link guests will visit to book or learn more" />
                  </label>
                  <input
                    type="url"
                    value={qrCodeUrl}
                    onChange={(e) => setQrCodeUrl(e.target.value)}
                    placeholder="https://hotel.com/spa/book"
                    className="input"
                  />
                </div>

                <div>
                  <label className="label">
                    QR Code Placement *
                    <InfoTooltip text="Where should the QR code appear?" />
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

          {/* Target Guests */}
          <div className="card">
            <div className="flex items-center mb-6">
              <h2 className="text-xl font-bold text-gray-900">Target Guest Segments *</h2>
              <InfoTooltip text="Which types of guests are most likely to be interested?" />
            </div>

            <div className="space-y-3">
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

          {/* Examples */}
          <div className="card bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Content Examples</h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-start">
                <svg className="w-5 h-5 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <strong>Spa Treatment:</strong> "Unwind with a Signature Massage" → "Book Treatment"
                </div>
              </div>
              <div className="flex items-start">
                <svg className="w-5 h-5 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <strong>Restaurant Special:</strong> "Swiss Wine Tasting Tonight" → "Reserve Your Spot"
                </div>
              </div>
              <div className="flex items-start">
                <svg className="w-5 h-5 mr-2 text-green-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <strong>Room Upgrade:</strong> "Upgrade to Suite - 30% Off" → "Upgrade Now"
                </div>
              </div>
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
              {loading ? 'Creating...' : 'Create Content'}
            </button>
          </div>
        </form>
      </div>
    </DashboardLayout>
  );
}





