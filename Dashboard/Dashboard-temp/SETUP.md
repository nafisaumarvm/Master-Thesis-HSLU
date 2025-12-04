# Quick Setup Guide

## Step-by-Step Installation

### 1. Install Dependencies
```bash
cd "/Users/nafisaumar/Documents/Master Thesis/Dashboard"
npm install
```

This will install all required packages:
- Next.js and React
- TailwindCSS
- TypeScript
- bcryptjs for password hashing
- No native dependencies - works on all platforms!

### 2. Start the Development Server
```bash
npm run dev
```

The application will:
- Start on http://localhost:3000
- Automatically create the JSON database
- Seed the database with demo data
- Initialize two demo accounts

### 3. Login to the Dashboard

Open http://localhost:3000 in your browser.

**For Hotel Admin Dashboard:**
```
Email: hotel@example.com
Password: hotel123
```

**For Advertiser Dashboard:**
```
Email: advertiser@example.com
Password: advertiser123
```

## First Time User Guide

### As a Hotel Admin:

1. **Dashboard Overview**
   - View your guest segments and performance metrics
   - See current impressions and estimated conversions

2. **Configure Settings** (Go to Settings)
   - Select which ad categories to allow
   - Choose when ads can be shown (default: welcome screen only)
   - Review and confirm privacy settings

3. **Review Segments** (Go to Segments)
   - Explore the 5 guest segment types
   - Understand segment characteristics
   - All data is synthetic and anonymized

4. **Privacy Center** (Go to Privacy)
   - Review GDPR compliance measures
   - Understand data handling practices
   - Confirm automatic deletion policies

### As an Advertiser:

1. **Create Your First Campaign** (Click "Create New Ad")
   - Write a compelling headline (max 60 characters)
   - Add a call-to-action (max 30 characters)
   - (Optional) Add image/video URL
   - Select relevant content categories
   - Choose target guest segments
   - Set your location and campaign objective
   - Review estimated reach

2. **View Campaign Performance** (Go to Dashboard)
   - Monitor impressions and CTR
   - Track active campaigns
   - See estimated reach across hotels

3. **Manage Campaigns** (Go to Campaigns)
   - View all campaigns in a table format
   - Filter by status (active/paused)
   - Review detailed performance metrics

## Troubleshooting

### Port Already in Use
If port 3000 is already in use:
```bash
npm run dev -- -p 3001
```

### Database Issues
If you encounter database errors:
```bash
# Delete the database and restart
rm -rf data/database.json
npm run dev
```

### Clear Browser Cache
If you see old data or login issues:
1. Clear browser cache and cookies for localhost:3000
2. Clear sessionStorage: Open browser console and run:
   ```javascript
   sessionStorage.clear()
   ```
3. Refresh the page

### TypeScript Errors
If you see TypeScript errors during development:
```bash
# Rebuild the project
rm -rf .next
npm run dev
```

## Project Features Checklist

### ✅ Implemented Features

**Authentication:**
- ✅ Simple email/password login
- ✅ Role-based routing (hotel/advertiser)
- ✅ Session management with sessionStorage
- ✅ Secure password hashing (bcrypt)

**Hotel Admin Features:**
- ✅ Dashboard with analytics
- ✅ Guest segments overview
- ✅ Content category configuration
- ✅ Ad exposure rules settings
- ✅ Privacy center with GDPR info
- ✅ Real-time settings updates

**Advertiser Features:**
- ✅ Campaign dashboard
- ✅ Create new campaigns
- ✅ Target specific segments
- ✅ Location targeting
- ✅ Objective selection (awareness/footfall/performance)
- ✅ Estimated reach calculator
- ✅ Campaign performance metrics
- ✅ Campaign list view

**Privacy & Compliance:**
- ✅ No PII collection
- ✅ Synthetic data only
- ✅ Automatic anonymization
- ✅ Zero retention post-checkout
- ✅ GDPR compliance documentation
- ✅ Privacy-first design

**UI/UX:**
- ✅ Clean, minimalist interface
- ✅ Role-based navigation
- ✅ Contextual tooltips
- ✅ Responsive design
- ✅ TailwindCSS styling
- ✅ Loading states
- ✅ Error handling

**Technical:**
- ✅ Next.js 14 with App Router
- ✅ TypeScript for type safety
- ✅ SQLite database with automatic seeding
- ✅ API routes for all operations
- ✅ Modular component architecture

## Database Schema

The JSON database (`data/database.json`) includes these collections:

1. **users** - Hotel admins and advertisers
2. **hotel_settings** - Ad preferences and exposure rules
3. **ads** - Advertising campaigns
4. **segments** - Guest segment definitions
5. **analytics** - Performance metrics

The database is automatically created and seeded on first run.

## Technical Architecture

### Overview

The dashboard is built as a **full-stack TypeScript application** using Next.js 14's App Router architecture. It follows a **monolithic serverless architecture** where frontend and backend logic coexist in a single codebase, with API routes handling server-side operations.

### Backend Architecture

#### **API Layer (Next.js API Routes)**

The backend is implemented using Next.js API Routes, which are serverless functions that handle HTTP requests:

- **Location**: `/app/api/*/route.ts`
- **Pattern**: RESTful endpoints with HTTP method handlers (`GET`, `POST`, `PUT`, `DELETE`)
- **Key Endpoints**:
  - `/api/auth/login` - User authentication
  - `/api/hotel/settings` - Hotel configuration management
  - `/api/hotel/analytics` - Performance metrics retrieval
  - `/api/segments` - Guest segment data
  - `/api/ads` - Campaign CRUD operations

**Request/Response Flow:**
```
Client Request → Next.js API Route → Business Logic Layer → Data Access Layer → JSON Database
```

Each API route:
1. Validates incoming request data
2. Calls appropriate functions from `lib/db.ts` or `lib/auth.ts`
3. Handles errors with try-catch blocks
4. Returns JSON responses with appropriate HTTP status codes

#### **Business Logic Layer**

Located in `/lib/`, this layer contains core application logic:

- **`lib/auth.ts`**: Authentication utilities
  - Password hashing with bcryptjs (10 salt rounds)
  - User credential validation
  - Session management helpers
  
- **`lib/db.ts`**: Database operations
  - JSON file-based storage (`data/database.json`)
  - CRUD operations for all entities
  - Automatic database initialization and seeding
  - In-memory caching with file persistence

- **`lib/types.ts`**: TypeScript type definitions
  - Interfaces for all data models (User, Ad, Segment, Analytics, etc.)
  - Type-safe constants (categories, segments, objectives)
  - Ensures type safety across frontend and backend

#### **Data Storage Architecture**

**Current Implementation: JSON File-Based Storage**

- **Storage Format**: Single JSON file (`data/database.json`)
- **Structure**: Object with collections (users, hotel_settings, ads, segments, analytics)
- **Operations**: 
  - Read: Load entire file into memory
  - Write: Serialize entire database to disk
  - Transactions: Not supported (single-threaded Node.js prevents race conditions in development)

**Data Model:**
```typescript
{
  users: User[],
  hotel_settings: HotelSettings[],
  ads: Ad[],
  segments: Segment[],
  analytics: Analytics[]
}
```

**Advantages for Development:**
- Zero configuration
- No database server required
- Human-readable data format
- Easy debugging and data inspection
- Cross-platform compatibility

**Limitations:**
- Not suitable for concurrent writes in production
- No query optimization
- No relationships or foreign keys
- Entire database loaded into memory

### Model Pipeline & Design

#### **Segment-Based Targeting System**

The dashboard uses a **rule-based targeting model** (not machine learning) that matches ads to guest segments:

**1. Guest Segmentation Model**

- **Input**: Guest behavioral patterns (arrival day, stay duration, booking characteristics)
- **Output**: 6 predefined segments with percentage distributions:
  - Business Travelers (28%)
  - Leisure Travelers (22%)
  - Families with Children (18%)
  - Groups 3+ (15%)
  - Couples (12%)
  - Solo Travelers (5%)

**2. Ad Targeting Pipeline**

When an advertiser creates a campaign:

```
Campaign Creation
    ↓
Segment Selection (multi-select)
    ↓
Location Targeting (country/city/radius)
    ↓
Demographic Filters (age, language, weather)
    ↓
Category Matching (restaurants, experiences, etc.)
    ↓
Hotel Settings Validation (allowed categories, exposure rules)
    ↓
Estimated Impressions Calculation
    ↓
Campaign Activation
```

**3. Matching Algorithm (Conceptual)**

The system matches ads to guests through:

1. **Segment Matching**: Ad's target segments must overlap with guest's segment
2. **Category Filtering**: Ad category must be in hotel's allowed categories
3. **Location Matching**: Guest location must match ad's geographic targeting
4. **Demographic Filtering**: Age, language, weather conditions must align
5. **Exposure Rules**: Hotel settings determine when/where ads can appear

**4. Revenue Calculation Model**

- **Input**: Total impressions, cost per 100 impressions
- **Calculation**: 
  ```
  Total Ad Spend = (Impressions / 100) × Cost per 100
  Hotel Revenue = Total Ad Spend × Revenue Share Percentage (30%)
  ```
- **Output**: Revenue metrics displayed in hotel analytics

#### **Synthetic Data Generation**

All data is synthetically generated for privacy and demonstration:

- **Users**: Pre-seeded with hashed passwords
- **Segments**: Statically defined with realistic percentages
- **Analytics**: Calculated from synthetic impression data
- **Ads**: Sample campaigns with realistic targeting parameters

### Technical Languages & Structures

#### **Programming Languages**

- **TypeScript 5.3** (Primary)
  - Strict mode enabled
  - Full type safety across codebase
  - Interface-first design pattern
  - ES2020 target with modern JavaScript features

- **JavaScript** (Configuration)
  - `next.config.js` - Next.js configuration
  - `postcss.config.js` - CSS processing
  - `tailwind.config.js` - Styling configuration

#### **Frontend Structure**

**Framework: Next.js 14 (App Router)**
- File-based routing (`app/*/page.tsx`)
- Server Components by default
- Client Components with `'use client'` directive
- Built-in code splitting and optimization

**UI Library: React 18**
- Functional components only
- React Hooks for state management (`useState`, `useEffect`)
- No class components or external state management libraries

**Styling: TailwindCSS 3.4**
- Utility-first CSS framework
- PostCSS processing with Autoprefixer
- Responsive design with mobile-first approach
- Custom color palette and spacing system

**Component Architecture:**
```
app/
├── layout.tsx (Root layout)
├── page.tsx (Home/login)
├── hotel/
│   ├── dashboard/page.tsx
│   ├── settings/page.tsx
│   ├── segments/page.tsx
│   └── ...
└── advertiser/
    ├── dashboard/page.tsx
    ├── create/page.tsx
    └── ...
```

**Shared Components:**
- `components/DashboardLayout.tsx` - Role-based layout wrapper
- `components/Navigation.tsx` - Role-aware navigation menu
- `components/InfoTooltip.tsx` - Contextual help tooltips

#### **Backend Structure**

**API Routes Pattern:**
```typescript
// app/api/[resource]/route.ts
export async function GET(request: NextRequest) { ... }
export async function POST(request: NextRequest) { ... }
export async function PUT(request: NextRequest) { ... }
export async function DELETE(request: NextRequest) { ... }
```

**Data Access Pattern:**
```typescript
// lib/db.ts
export function getDb(): Database { ... }
export function findUser(email: string) { ... }
export function createAd(ad: Ad) { ... }
```

#### **Type System**

**Type Definitions (`lib/types.ts`):**
- Interfaces for all domain models
- Union types for enums (UserRole, Objective, etc.)
- Const arrays for valid values (AD_CATEGORIES, SEGMENT_TYPES)
- Type exports used across frontend and backend

**Type Safety:**
- No `any` types (strict TypeScript)
- Generic types where appropriate
- Type inference for function returns
- Compile-time error checking

#### **Build & Development Tools**

**Build System:**
- Next.js built-in bundler (Webpack/Turbopack)
- Automatic code splitting
- Tree shaking for unused code
- CSS optimization and minification

**Development Tools:**
- TypeScript compiler for type checking
- ESLint with Next.js configuration
- Hot Module Replacement (HMR) for development
- Fast Refresh for React components

**Package Management:**
- npm (Node Package Manager)
- Lock file (`package-lock.json`) for dependency versioning
- No native dependencies (pure JavaScript/TypeScript)

### Design Patterns & Principles

#### **1. Separation of Concerns**
- **Pages**: Route handlers and data fetching
- **Components**: Reusable UI elements
- **API Routes**: Backend business logic
- **Lib**: Shared utilities and data access

#### **2. Type Safety First**
- TypeScript interfaces for all data models
- Type checking at compile time
- Runtime validation in API routes
- No dynamic type casting

#### **3. Privacy by Design**
- No PII collection in data models
- Synthetic data only
- Anonymized analytics
- Zero retention policy (conceptual)

#### **4. Component Composition**
- Small, focused components
- Props-based communication
- No prop drilling (direct parent-child relationships)
- Reusable layout components

#### **5. Error Handling**
- Try-catch blocks in all async operations
- User-friendly error messages
- Console logging for debugging
- Graceful degradation on failures

### Data Flow Architecture

#### **Client-Side Data Flow**
```
User Interaction
    ↓
React Event Handler
    ↓
API Call (fetch)
    ↓
State Update (useState)
    ↓
Component Re-render
```

#### **Server-Side Data Flow**
```
API Request
    ↓
Route Handler
    ↓
Business Logic (lib/*)
    ↓
Data Access (lib/db.ts)
    ↓
JSON File I/O
    ↓
Response Serialization
    ↓
JSON Response
```

### Security Architecture

**Current Implementation:**
- Password hashing with bcryptjs (10 rounds)
- Client-side session storage (sessionStorage)
- No SQL injection risk (JSON storage)
- Type validation via TypeScript

**Authentication Flow:**
```
Login Form → POST /api/auth/login → bcrypt.compare() → User Object (no password) → sessionStorage
```

**Authorization:**
- Role-based access control (hotel vs advertiser)
- Route-level protection via component checks
- API-level filtering (user-scoped queries)

### Performance Considerations

**Current Optimizations:**
- Server-side rendering (SSR) for initial page load
- Automatic code splitting by Next.js
- CSS purging with TailwindCSS
- In-memory database caching

**Bottlenecks:**
- JSON file I/O (synchronous operations)
- Full database load on each request
- No query optimization
- No caching layer

**Scalability Path:**
1. Migrate to PostgreSQL for concurrent access
2. Add Redis for caching
3. Implement database connection pooling
4. Add API response caching
5. Consider microservices for high scale

## Next Steps for Production

If deploying to production, consider:

1. **Authentication**: Implement proper session management (NextAuth.js)
2. **Database**: Migrate from JSON to PostgreSQL or MongoDB
3. **File Uploads**: Add proper image/video upload handling
4. **API Security**: Add rate limiting and CORS configuration
5. **Monitoring**: Set up error tracking (Sentry, etc.)
6. **Testing**: Add unit and integration tests
7. **CI/CD**: Set up deployment pipeline

## Support

For issues or questions:
1. Check the main README.md
2. Review code comments in source files
3. Use the tooltip (ℹ️) icons throughout the interface
4. Check browser console for detailed error messages

---

**Your dashboard is now ready to use!**

Start exploring the features or create your first campaign. All data is synthetic and safe to experiment with.

