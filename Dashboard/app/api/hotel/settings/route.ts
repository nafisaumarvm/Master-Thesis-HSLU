import { NextRequest, NextResponse } from 'next/server';
import { getHotelSettings, updateHotelSettings } from '@/lib/db';
import { HotelSettings } from '@/lib/types';

/**
 * Hotel Settings API
 * GET: Retrieve hotel settings
 * PUT: Update hotel settings
 */

export async function GET(request: NextRequest) {
  try {
    const userId = request.nextUrl.searchParams.get('userId');
    
    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 });
    }

    const settings = getHotelSettings(parseInt(userId));

    if (!settings) {
      return NextResponse.json({ error: 'Settings not found' }, { status: 404 });
    }

    const response: HotelSettings = {
      id: settings.id,
      user_id: settings.user_id,
      allowed_categories: settings.allowed_categories,
      exposure_rules: settings.exposure_rules,
      privacy_rules: settings.privacy_rules,
    };

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error('Get settings error:', error);
    return NextResponse.json({ error: 'Failed to retrieve settings' }, { status: 500 });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const { userId, allowed_categories, exposure_rules, privacy_rules } = await request.json();

    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 });
    }

    const success = updateHotelSettings(userId, {
      allowed_categories,
      exposure_rules,
      privacy_rules
    });

    if (success) {
      return NextResponse.json({ success: true }, { status: 200 });
    } else {
      return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
    }
  } catch (error) {
    console.error('Update settings error:', error);
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
