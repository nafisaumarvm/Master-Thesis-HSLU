import { NextRequest, NextResponse } from 'next/server';
import { getAds, createAd, deleteAd } from '@/lib/db';
import { Ad } from '@/lib/types';

/**
 * Ads API
 * GET: Retrieve ads (filtered by advertiser if userId provided)
 * POST: Create a new ad
 * DELETE: Delete an ad
 */

export async function GET(request: NextRequest) {
  try {
    const userId = request.nextUrl.searchParams.get('userId');
    
    let ads: any[];
    
    if (userId) {
      ads = getAds(parseInt(userId));
    } else {
      ads = getAds();
    }

    const response: Ad[] = ads.map(ad => ({
      id: ad.id,
      advertiser_id: ad.advertiser_id,
      headline: ad.headline,
      cta_text: ad.cta_text,
      file_url: ad.file_url,
      categories: ad.categories,
      segments: ad.segments,
      location: ad.location,
      objective: ad.objective,
      status: ad.status,
      impressions: ad.impressions,
      ctr_estimate: ad.ctr_estimate,
      created_at: ad.created_at,
    }));

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error('Get ads error:', error);
    return NextResponse.json({ error: 'Failed to retrieve ads' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const { 
      advertiser_id, 
      hotel_id,
      headline, 
      cta_text, 
      file_url, 
      qr_code_url,
      qr_placement,
      category, 
      segments, 
      weather_targeting,
      age_targeting,
      language_targeting,
      country,
      city,
      location,
      location_type,
      location_radius,
      objective,
      created_by_hotel
    } = data;

    if (!headline || !cta_text || !category || !segments || !location || !objective) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    if (!advertiser_id && !hotel_id) {
      return NextResponse.json({ error: 'Either advertiser_id or hotel_id is required' }, { status: 400 });
    }

    // Generate estimated impressions based on segments (synthetic)
    const estimatedImpressions = Math.floor(Math.random() * 500) + 200;
    const estimatedCTR = (Math.random() * 3 + 1).toFixed(2);

    const newAd = createAd({
      advertiser_id: advertiser_id || null,
      hotel_id: hotel_id || null,
      headline,
      cta_text,
      file_url: file_url || '/uploads/placeholder.jpg',
      qr_code_url: qr_code_url || '',
      qr_placement: qr_placement || 'Welcome Screen',
      category,
      segments,
      weather_targeting: weather_targeting || ['any'],
      age_targeting: age_targeting || [],
      language_targeting: language_targeting || [],
      country: country || 'Switzerland',
      city: city || '',
      location,
      location_type: location_type || 'district',
      location_radius: location_radius || null,
      objective,
      ctr_estimate: parseFloat(estimatedCTR),
      created_by_hotel: created_by_hotel || false
    });

    return NextResponse.json({ 
      success: true, 
      id: newAd.id,
      estimated_impressions: estimatedImpressions 
    }, { status: 201 });
  } catch (error) {
    console.error('Create ad error:', error);
    return NextResponse.json({ error: 'Failed to create ad' }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const adId = request.nextUrl.searchParams.get('adId');
    
    if (!adId) {
      return NextResponse.json({ error: 'Ad ID required' }, { status: 400 });
    }

    const success = deleteAd(parseInt(adId));

    if (success) {
      return NextResponse.json({ success: true }, { status: 200 });
    } else {
      return NextResponse.json({ error: 'Ad not found' }, { status: 404 });
    }
  } catch (error) {
    console.error('Delete ad error:', error);
    return NextResponse.json({ error: 'Failed to delete ad' }, { status: 500 });
  }
}
