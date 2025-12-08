'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Navigation from './Navigation';

/**
 * Dashboard Layout Component
 * Wraps authenticated pages with navigation and checks for valid session
 */

interface DashboardLayoutProps {
  children: React.ReactNode;
  requiredRole?: 'hotel' | 'advertiser';
}

export default function DashboardLayout({ children, requiredRole }: DashboardLayoutProps) {
  const router = useRouter();

  useEffect(() => {
    const userStr = sessionStorage.getItem('user');
    if (!userStr) {
      router.push('/');
      return;
    }

    if (requiredRole) {
      const user = JSON.parse(userStr);
      if (user.role !== requiredRole) {
        // Redirect to correct dashboard
        router.push(user.role === 'hotel' ? '/hotel/dashboard' : '/advertiser/dashboard');
      }
    }
  }, [router, requiredRole]);

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}





