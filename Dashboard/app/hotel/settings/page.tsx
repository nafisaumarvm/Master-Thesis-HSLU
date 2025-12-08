'use client';

import { useEffect, useState } from 'react';
import DashboardLayout from '@/components/DashboardLayout';
import InfoTooltip from '@/components/InfoTooltip';
import { HotelSettings, ExposureRules } from '@/lib/types';
import { AD_CATEGORIES } from '@/lib/types';

/**
 * Hotel Settings Page
 * Configure allowed content types and ad exposure rules
 */

export default function HotelSettingsPage() {
  const [settings, setSettings] = useState<HotelSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');

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

  const handleCategoryToggle = (category: string) => {
    if (!settings) return;

    const newCategories = settings.allowed_categories.includes(category)
      ? settings.allowed_categories.filter(c => c !== category)
      : [...settings.allowed_categories, category];

    setSettings({
      ...settings,
      allowed_categories: newCategories,
    });
  };

  const handleExposureRuleChange = (rule: keyof ExposureRules) => {
    if (!settings) return;

    setSettings({
      ...settings,
      exposure_rules: {
        ...settings.exposure_rules,
        [rule]: !settings.exposure_rules[rule],
      },
    });
  };

  const handleSave = async () => {
    if (!settings) return;

    setSaving(true);
    setSaveMessage('');

    try {
      const userStr = sessionStorage.getItem('user');
      if (!userStr) return;

      const user = JSON.parse(userStr);

      const res = await fetch('/api/hotel/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: user.id,
          allowed_categories: settings.allowed_categories,
          exposure_rules: settings.exposure_rules,
          privacy_rules: settings.privacy_rules,
        }),
      });

      if (res.ok) {
        setSaveMessage('Settings saved successfully!');
        setTimeout(() => setSaveMessage(''), 3000);
      } else {
        setSaveMessage('Failed to save settings');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      setSaveMessage('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  if (loading || !settings) {
    return (
      <DashboardLayout requiredRole="hotel">
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading settings...</div>
        </div>
      </DashboardLayout>
    );
  }

  const categoryLabels: Record<string, string> = {
    restaurants: 'Local Restaurants',
    experiences: 'Museums & Attractions',
    museums: 'Cultural Experiences',
    mobility: 'Mobility & Transport',
    retail: 'Retail & Luxury',
    wellness: 'Spa & Wellness',
    events: 'Events & Entertainment',
    nightlife: 'Nightlife'
  };

  return (
    <DashboardLayout requiredRole="hotel">
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Ad Settings</h1>
          <p className="mt-2 text-gray-600">
            Configure which content types are allowed and when ads may be shown
          </p>
        </div>

        {/* Allowed Content Types */}
        <div className="card">
          <div className="flex items-center mb-6">
            <h2 className="text-xl font-bold text-gray-900">Allowed Content Categories</h2>
            <InfoTooltip text="Select which types of advertising content are appropriate for your guests. Only selected categories will be shown." />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {AD_CATEGORIES.map((category) => (
              <label
                key={category}
                className={`flex items-center p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  settings.allowed_categories.includes(category)
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="checkbox"
                  checked={settings.allowed_categories.includes(category)}
                  onChange={() => handleCategoryToggle(category)}
                  className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
                />
                <div className="ml-3">
                  <span className="font-medium text-gray-900">
                    {categoryLabels[category] || category}
                  </span>
                </div>
              </label>
            ))}
          </div>

          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-sm text-yellow-800">
              <strong>Note:</strong> Only high-quality, contextually relevant advertisers in selected 
              categories will be shown to your guests.
            </p>
          </div>
        </div>

        {/* Ad Exposure Rules */}
        <div className="card">
          <div className="flex items-center mb-6">
            <h2 className="text-xl font-bold text-gray-900">Ad Exposure Rules</h2>
            <InfoTooltip text="Control when and where ads may appear on guest Smart TVs. Default is welcome screen only to ensure minimal disruption." />
          </div>

          <div className="space-y-4">
            <label className="flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all border-gray-200 hover:border-gray-300">
              <input
                type="checkbox"
                checked={settings.exposure_rules.welcomeScreen}
                onChange={() => handleExposureRuleChange('welcomeScreen')}
                className="mt-1 w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
              <div className="ml-3">
                <span className="font-medium text-gray-900">Show on Welcome Screen</span>
                <p className="text-sm text-gray-600 mt-1">
                  Display ads when guests first turn on the TV. This is the least intrusive option.
                </p>
                <span className="inline-block mt-2 px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded">
                  Recommended
                </span>
              </div>
            </label>

            <label className="flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all border-gray-200 hover:border-gray-300">
              <input
                type="checkbox"
                checked={settings.exposure_rules.idleMode}
                onChange={() => handleExposureRuleChange('idleMode')}
                className="mt-1 w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
              <div className="ml-3">
                <span className="font-medium text-gray-900">Show During Idle Mode</span>
                <p className="text-sm text-gray-600 mt-1">
                  Display ads when the TV has been inactive for 5+ minutes.
                </p>
              </div>
            </label>

            <label className="flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all border-gray-200 hover:border-gray-300">
              <input
                type="checkbox"
                checked={settings.exposure_rules.sidebarBanner}
                onChange={() => handleExposureRuleChange('sidebarBanner')}
                className="mt-1 w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
              <div className="ml-3">
                <span className="font-medium text-gray-900">Sidebar Banner (Hotel Info)</span>
                <p className="text-sm text-gray-600 mt-1">
                  Show small banner when guests browse hotel information screens.
                </p>
              </div>
            </label>

            <label className="flex items-start p-4 border-2 rounded-lg cursor-pointer transition-all border-primary-500 bg-primary-50">
              <input
                type="checkbox"
                checked={settings.exposure_rules.neverInterrupt}
                onChange={() => handleExposureRuleChange('neverInterrupt')}
                className="mt-1 w-5 h-5 text-primary-600 rounded focus:ring-primary-500"
              />
              <div className="ml-3">
                <span className="font-medium text-gray-900">Never Interrupt Content</span>
                <p className="text-sm text-gray-600 mt-1">
                  Ads will NEVER interrupt guest TV watching, movies, or streaming content.
                </p>
                <span className="inline-block mt-2 px-2 py-1 text-xs font-medium bg-primary-100 text-primary-800 rounded">
                  Always Enabled
                </span>
              </div>
            </label>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex items-center justify-between">
          <div>
            {saveMessage && (
              <div className={`text-sm font-medium ${
                saveMessage.includes('success') ? 'text-green-600' : 'text-red-600'
              }`}>
                {saveMessage}
              </div>
            )}
          </div>
          <button
            onClick={handleSave}
            disabled={saving}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </DashboardLayout>
  );
}





