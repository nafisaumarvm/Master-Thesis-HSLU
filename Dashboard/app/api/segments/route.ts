import { NextRequest, NextResponse } from 'next/server';
import { getAllSegments } from '@/lib/db';
import { Segment } from '@/lib/types';

/**
 * Segments API
 * GET: Retrieve all guest segments (synthetic data)
 */

export async function GET(request: NextRequest) {
  try {
    const segments = getAllSegments();

    const response: Segment[] = segments.map(seg => ({
      id: seg.id,
      name: seg.name,
      description: seg.description,
      percentage: seg.percentage,
      characteristics: seg.characteristics,
    }));

    return NextResponse.json(response, { status: 200 });
  } catch (error) {
    console.error('Get segments error:', error);
    return NextResponse.json({ error: 'Failed to retrieve segments' }, { status: 500 });
  }
}
