import { NextRequest, NextResponse } from 'next/server';
import { getAnalytics } from '@/lib/db';
import { Analytics } from '@/lib/types';

/**
 * Hotel Analytics API
 * GET: Retrieve analytics data for hotel dashboard
 */

export async function GET(request: NextRequest) {
  try {
    const userId = request.nextUrl.searchParams.get('userId');
    
    if (!userId) {
      return NextResponse.json({ error: 'User ID required' }, { status: 400 });
    }

    const analytics = getAnalytics(parseInt(userId));

    if (!analytics) {
      // Return default analytics if none exist
      return NextResponse.json({
        impressions_this_week: 0,
        estimated_conversions: 0,
        top_category: 'N/A',
        occupancy_multiplier: 1.0,
        revenue_generated: 0,
        revenue_share_percentage: 30,
        total_ad_spend: 0,
      }, { status: 200 });
    }

    // Calculate revenue dynamically based on impressions
    const avgCostPer100 = 6.80; // CHF
    const totalAdSpend = (analytics.impressions_this_week / 100) * avgCostPer100;
    const revenueSharePercentage = 30; // Hotel receives 30%
    const hotelRevenue = totalAdSpend * (revenueSharePercentage / 100);

    return NextResponse.json({
      ...analytics,
      revenue_generated: parseFloat(hotelRevenue.toFixed(2)),
      revenue_share_percentage: revenueSharePercentage,
      total_ad_spend: parseFloat(totalAdSpend.toFixed(2)),
    }, { status: 200 });
  } catch (error) {
    console.error('Get analytics error:', error);
    return NextResponse.json({ error: 'Failed to retrieve analytics' }, { status: 500 });
  }
}
