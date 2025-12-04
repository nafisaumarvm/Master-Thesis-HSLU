/**
 * TypeScript types for the application
 * Ensures type safety across the codebase
 */

export type UserRole = 'hotel' | 'advertiser';

export interface User {
  id: number;
  email: string;
  role: UserRole;
  name: string;
  created_at: string;
}

export interface HotelSettings {
  id: number;
  user_id: number;
  allowed_categories: string[];
  exposure_rules: ExposureRules;
  privacy_rules: PrivacyRules;
}

export interface ExposureRules {
  welcomeScreen: boolean;
  idleMode: boolean;
  sidebarBanner: boolean;
  neverInterrupt: boolean;
}

export interface PrivacyRules {
  anonymizedInsights: boolean;
  autoDeletePostCheckout: boolean;
  dataRetentionDays: number;
  gdprCompliant: boolean;
}

export interface Segment {
  id: number;
  name: string;
  description: string;
  percentage: number;
  characteristics: string[];
}

export interface Ad {
  id: number;
  advertiser_id: number;
  hotel_id?: number; // For hotel-created content
  headline: string;
  cta_text: string;
  file_url: string;
  qr_code_url?: string;
  qr_placement?: string;
  category: string; // Single category now
  segments: string[];
  weather_targeting?: string[];
  age_targeting?: string[];
  language_targeting?: string[];
  country?: string; // Country selection
  city?: string; // City within country
  location: string; // Full location string for display
  location_type: 'radius' | 'district'; // New: radius or district
  location_radius?: number; // In km
  objective: 'awareness' | 'footfall' | 'performance';
  status: string;
  impressions: number;
  ctr_estimate: number;
  created_at: string;
  created_by_hotel?: boolean; // Flag for hotel-created content
}

export interface Analytics {
  id: number;
  user_id: number;
  impressions_this_week: number;
  estimated_conversions: number;
  top_category: string;
  occupancy_multiplier: number;
  revenue_generated?: number; // Hotel's share from ads (CHF)
  revenue_share_percentage?: number; // Percentage hotel receives (e.g., 30%)
  total_ad_spend?: number; // Total advertiser spend (CHF)
  last_updated: string;
}

export const AD_CATEGORIES = [
  'restaurants',
  'experiences',
  'museums',
  'mobility',
  'retail',
  'events',
  'nightlife'
] as const;

export const SEGMENT_TYPES = [
  'Business Travelers',
  'Leisure Travelers',
  'Families with Children',
  'Groups (3+ people)',
  'Couples',
  'Solo Travelers'
] as const;

export const OBJECTIVES = [
  'awareness',
  'footfall',
  'performance'
] as const;

export const WEATHER_CONDITIONS = [
  'sunny',
  'rainy',
  'snowy',
  'cloudy',
  'any'
] as const;

export const AGE_GROUPS = [
  '18-24',
  '25-34',
  '35-44',
  '45-54',
  '55-64',
  '65+'
] as const;

export const LANGUAGES = [
  'German',
  'French',
  'Italian',
  'English',
  'Other'
] as const;

export const COUNTRIES = [
  'Switzerland',
  'Germany',
  'Austria',
  'France',
  'Italy'
] as const;

export const SWISS_CITIES = [
  'Zürich',
  'Geneva',
  'Basel',
  'Lausanne',
  'Bern',
  'Lucerne',
  'St. Gallen',
  'Lugano',
  'Interlaken',
  'Zermatt'
] as const;

export const GERMAN_CITIES = [
  'Munich',
  'Berlin',
  'Frankfurt',
  'Hamburg',
  'Stuttgart'
] as const;

export const AUSTRIAN_CITIES = [
  'Vienna',
  'Salzburg',
  'Innsbruck',
  'Graz'
] as const;

export const FRENCH_CITIES = [
  'Paris',
  'Lyon',
  'Marseille',
  'Strasbourg'
] as const;

export const ITALIAN_CITIES = [
  'Milan',
  'Rome',
  'Florence',
  'Venice'
] as const;

export const QR_PLACEMENTS = [
  'Top Right',
  'Top Left',
  'Bottom Right',
  'Bottom Left',
  'Center Bottom'
] as const;
