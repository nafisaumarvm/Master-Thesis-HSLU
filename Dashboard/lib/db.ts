/**
 * Database management with JSON-based storage
 * No native dependencies required - works on all platforms
 * All data is synthetic and privacy-preserving
 */

import bcrypt from 'bcryptjs';
import fs from 'fs';
import path from 'path';

const dbPath = path.join(process.cwd(), 'data', 'database.json');

interface Database {
  users: any[];
  hotel_settings: any[];
  ads: any[];
  segments: any[];
  analytics: any[];
}

let db: Database | null = null;

// Ensure data directory exists
function ensureDataDir() {
  const dataDir = path.join(process.cwd(), 'data');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
}

// Load database from file
function loadDatabase(): Database {
  ensureDataDir();
  
  if (!fs.existsSync(dbPath)) {
    const emptyDb: Database = {
      users: [],
      hotel_settings: [],
      ads: [],
      segments: [],
      analytics: []
    };
    fs.writeFileSync(dbPath, JSON.stringify(emptyDb, null, 2));
    return emptyDb;
  }
  
  const data = fs.readFileSync(dbPath, 'utf-8');
  return JSON.parse(data);
}

// Save database to file
function saveDatabase(database: Database) {
  ensureDataDir();
  fs.writeFileSync(dbPath, JSON.stringify(database, null, 2));
}

export function getDb(): Database {
  if (!db) {
    db = loadDatabase();
    initializeDatabase();
  }
  return db;
}

function initializeDatabase() {
  if (!db) return;

  // Check if data already exists
  if (db.users.length > 0) return;

  console.log('🔄 Seeding database with synthetic data...');

  // Seed users
  const hotelPassword = bcrypt.hashSync('hotel123', 10);
  const advertiserPassword = bcrypt.hashSync('advertiser123', 10);

  const hotelUser = {
    id: 1,
    email: 'hotel@example.com',
    password_hash: hotelPassword,
    role: 'hotel',
    name: 'Grand Hotel Plaza',
    created_at: new Date().toISOString()
  };

  const advertiserUser = {
    id: 2,
    email: 'advertiser@example.com',
    password_hash: advertiserPassword,
    role: 'advertiser',
    name: 'Local Business Co.',
    created_at: new Date().toISOString()
  };

  db.users.push(hotelUser, advertiserUser);

  // Seed hotel settings with privacy-first defaults
  db.hotel_settings.push({
    id: 1,
    user_id: hotelUser.id,
    allowed_categories: ['restaurants', 'experiences', 'wellness'],
    exposure_rules: {
      welcomeScreen: true,
      idleMode: false,
      sidebarBanner: false,
      neverInterrupt: true
    },
    privacy_rules: {
      anonymizedInsights: true,
      autoDeletePostCheckout: true,
      dataRetentionDays: 0,
      gdprCompliant: true
    }
  });

  // Seed segments (diverse traveler types)
  const segments = [
    {
      id: 1,
      name: 'Business Travelers',
      description: 'Professionals and corporate guests',
      percentage: 28.0,
      characteristics: ['Mon-Thu arrivals', 'Single occupancy', 'Short stays (1-3 nights)', 'Conference attendees', 'High per-day spending', 'Late check-ins']
    },
    {
      id: 2,
      name: 'Leisure Travelers',
      description: 'Individual tourists and vacationers',
      percentage: 22.0,
      characteristics: ['Weekend arrivals', 'Medium stays (2-4 nights)', 'Interest in sightseeing', 'Restaurant reservations', 'Local experiences', 'Flexible schedules']
    },
    {
      id: 3,
      name: 'Families with Children',
      description: 'Parents traveling with kids',
      percentage: 18.0,
      characteristics: ['School holidays', 'Multiple rooms', 'Family-friendly activities', 'Early dining', 'Connecting rooms', 'Kid-focused amenities']
    },
    {
      id: 4,
      name: 'Groups (3+ people)',
      description: 'Friend groups, tours, and team bookings',
      percentage: 15.0,
      characteristics: ['Multiple room bookings', 'Shared experiences', 'Evening activities', 'Group dining', 'Event attendees', 'Social atmosphere']
    },
    {
      id: 5,
      name: 'Couples',
      description: 'Romantic getaways and couples trips',
      percentage: 12.0,
      characteristics: ['Weekend breaks', 'Romance packages', 'Spa interest', 'Fine dining', 'Privacy preference', 'Special occasions']
    },
    {
      id: 6,
      name: 'Solo Travelers',
      description: 'Independent travelers and digital nomads',
      percentage: 5.0,
      characteristics: ['Flexible schedules', 'Single occupancy', 'Co-working spaces', 'Social lounges', 'Extended stays', 'Local immersion']
    }
  ];

  db.segments.push(...segments);

  // Seed sample ads (Swiss market focused)
  const sampleAds = [
    {
      id: 1,
      advertiser_id: advertiserUser.id,
      headline: 'Authentic Swiss & Italian Cuisine',
      cta_text: 'Reserve Your Table',
      file_url: '/uploads/sample-restaurant.jpg',
      qr_code_url: 'https://restaurant-zurich.ch/reserve',
      qr_placement: 'Bottom Right',
      category: 'restaurants',
      segments: ['Business Travelers', 'Leisure Travelers', 'Couples'],
      weather_targeting: ['any'],
      age_targeting: ['25-34', '35-44', '45-54'],
      language_targeting: ['German', 'English', 'Italian'],
      country: 'Switzerland',
      city: 'Zürich',
      location: 'Zürich, Switzerland',
      location_type: 'radius',
      location_radius: 2,
      objective: 'footfall',
      status: 'active',
      impressions: 1247,
      ctr_estimate: 2.8,
      created_at: new Date().toISOString(),
      created_by_hotel: false
    }
  ];

  db.ads.push(...sampleAds);

  // Seed analytics with revenue calculation
  const totalImpressions = 2139;
  const avgCostPer100 = 6.80; // CHF
  const totalAdSpend = (totalImpressions / 100) * avgCostPer100; // ~CHF 145.45
  const revenueSharePercentage = 30; // Hotel receives 30%
  const hotelRevenue = totalAdSpend * (revenueSharePercentage / 100); // ~CHF 43.64

  db.analytics.push({
    id: 1,
    user_id: hotelUser.id,
    impressions_this_week: totalImpressions,
    estimated_conversions: 43,
    top_category: 'restaurants',
    occupancy_multiplier: 1.15,
    revenue_generated: parseFloat(hotelRevenue.toFixed(2)),
    revenue_share_percentage: revenueSharePercentage,
    total_ad_spend: parseFloat(totalAdSpend.toFixed(2)),
    last_updated: new Date().toISOString()
  });

  // Save to disk
  saveDatabase(db);

  console.log('✅ Database seeded with synthetic data');
}

// Helper functions for CRUD operations

export function findUser(email: string) {
  const database = getDb();
  return database.users.find(u => u.email === email);
}

export function findUserById(id: number) {
  const database = getDb();
  return database.users.find(u => u.id === id);
}

export function getHotelSettings(userId: number) {
  const database = getDb();
  return database.hotel_settings.find(s => s.user_id === userId);
}

export function updateHotelSettings(userId: number, settings: any) {
  const database = getDb();
  const index = database.hotel_settings.findIndex(s => s.user_id === userId);
  if (index !== -1) {
    database.hotel_settings[index] = {
      ...database.hotel_settings[index],
      ...settings,
      user_id: userId
    };
    saveDatabase(database);
    return true;
  }
  return false;
}

export function getAnalytics(userId: number) {
  const database = getDb();
  return database.analytics.find(a => a.user_id === userId);
}

export function getAllSegments() {
  const database = getDb();
  return database.segments;
}

export function getAds(advertiserId?: number) {
  const database = getDb();
  if (advertiserId) {
    return database.ads.filter(a => a.advertiser_id === advertiserId);
  }
  return database.ads;
}

export function createAd(ad: any) {
  const database = getDb();
  const newId = database.ads.length > 0 ? Math.max(...database.ads.map(a => a.id)) + 1 : 1;
  const newAd = {
    ...ad,
    id: newId,
    status: 'active',
    impressions: 0,
    created_at: new Date().toISOString()
  };
  database.ads.push(newAd);
  saveDatabase(database);
  return newAd;
}

export function deleteAd(adId: number) {
  const database = getDb();
  const index = database.ads.findIndex(a => a.id === adId);
  if (index !== -1) {
    database.ads.splice(index, 1);
    saveDatabase(database);
    return true;
  }
  return false;
}

export default getDb;
