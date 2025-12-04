# Architecture Documentation

## System Architecture

This document describes the technical architecture and design decisions for the Hotel Advertising Dashboard.

## Overview

The application follows a **privacy-first, human-centered design** approach, implementing a full-stack TypeScript solution with Next.js.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         React Components (Client-Side)                │  │
│  │  - Login Page                                         │  │
│  │  - Hotel Dashboard Pages (4 pages)                    │  │
│  │  - Advertiser Dashboard Pages (3 pages)               │  │
│  │  - Shared Components (Navigation, Layout, Tooltips)   │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↕                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Next.js API Routes                       │  │
│  │  - /api/auth/login                                    │  │
│  │  - /api/hotel/settings                                │  │
│  │  - /api/hotel/analytics                               │  │
│  │  - /api/segments                                      │  │
│  │  - /api/ads                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↕                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Business Logic Layer                           │  │
│  │  - Authentication (lib/auth.ts)                       │  │
│  │  - Database Management (lib/db.ts)                    │  │
│  │  - Type Definitions (lib/types.ts)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↕                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            SQLite Database                            │  │
│  │  - users, hotel_settings, ads, segments, analytics    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **UI Library**: React 18
- **Styling**: TailwindCSS 3.4
- **Language**: TypeScript 5.3
- **State Management**: React hooks (useState, useEffect)
- **Routing**: Next.js App Router (file-based)

### Backend
- **API**: Next.js API Routes (serverless functions)
- **Database**: SQLite (better-sqlite3)
- **Authentication**: bcryptjs for password hashing
- **Session**: sessionStorage (client-side)

### Development
- **Build Tool**: Next.js built-in (Webpack/Turbopack)
- **Type Checking**: TypeScript
- **Linting**: ESLint (Next.js config)
- **CSS Processing**: PostCSS with Autoprefixer

## Design Patterns

### 1. Component Architecture

**Separation of Concerns:**
- **Pages** (`app/*/page.tsx`): Route handlers and data fetching
- **Components** (`components/`): Reusable UI elements
- **Layout** (`app/layout.tsx`): Shared layout wrapper
- **API** (`app/api/`): Backend business logic

**Component Hierarchy:**
```
RootLayout
└── Navigation (authenticated only)
    └── DashboardLayout (role-specific)
        └── Page Components
            └── Shared Components (cards, forms, tooltips)
```

### 2. Data Flow

**Client → Server:**
1. User interaction triggers React event handler
2. Event handler calls API route via `fetch()`
3. API route processes request
4. Database query executed via `lib/db.ts`
5. Response returned to client

**Server → Client:**
1. Page component fetches data in `useEffect`
2. API route queries database
3. Data serialized to JSON
4. Client receives and sets state
5. React re-renders with new data

### 3. Authentication Flow

```
Login Page
    ↓ (email + password)
POST /api/auth/login
    ↓ (validate credentials)
lib/auth.ts → authenticateUser()
    ↓ (query database)
Return user object (no password)
    ↓ (store in sessionStorage)
Redirect to role-specific dashboard
```

### 4. Database Design

**Entity Relationship:**
```
users (1) ←→ (1) hotel_settings
users (1) ←→ (*) ads
users (1) ←→ (1) analytics
segments (*) [independent lookup table]
```

**Key Constraints:**
- Foreign keys enforce referential integrity
- UNIQUE constraints on email, user_id in settings
- CHECK constraints for role validation
- JSON fields for flexible array/object storage

## Privacy-First Design Decisions

### 1. No PII Collection
**Implementation:**
- Database schema excludes PII fields
- Segments use aggregated patterns only
- No tracking pixels or cookies

### 2. Automatic Data Deletion
**Implementation:**
- Conceptual: Data marked for deletion post-checkout
- In production: Cron job would delete records
- Retention period: 0 days (immediate)

### 3. Anonymized Insights
**Implementation:**
- Analytics aggregated at segment level
- No individual guest tracking
- Advertisers see only totals, never individuals

### 4. Synthetic Data
**Implementation:**
- All seeded data is synthetic
- Realistic patterns but not real people
- Safe for development and demonstration

## API Design

### RESTful Principles

**Endpoints:**
- `GET /api/segments` - List all segments
- `GET /api/hotel/settings?userId=X` - Get hotel settings
- `PUT /api/hotel/settings` - Update hotel settings
- `GET /api/hotel/analytics?userId=X` - Get analytics
- `GET /api/ads?userId=X` - List ads (filtered by user)
- `POST /api/ads` - Create new ad
- `POST /api/auth/login` - Authenticate user

**Response Format:**
```json
{
  "data": { ... },      // Success
  "error": "message"    // Error
}
```

**Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Server Error

## Security Considerations

### Current Implementation (Development)
- ✅ Password hashing with bcrypt
- ✅ SQL injection prevention (parameterized queries)
- ✅ Client-side session management
- ✅ Role-based access control

### Production Requirements
- ⚠️ Implement server-side sessions (NextAuth.js, JWT)
- ⚠️ Add CSRF protection
- ⚠️ Rate limiting on API routes
- ⚠️ HTTPS only
- ⚠️ Secure headers (helmet.js)
- ⚠️ Input validation and sanitization
- ⚠️ Environment variable management

## Performance Optimizations

### Current
- **SSR**: Server-side rendering for initial page load
- **Code Splitting**: Automatic with Next.js
- **Image Optimization**: Next.js Image component (not used, URLs only)
- **CSS**: TailwindCSS with purging

### Future Optimizations
- Add React.memo for expensive components
- Implement virtual scrolling for long lists
- Add pagination for campaign lists
- Cache database queries (Redis)
- Compress API responses
- Add service worker for offline support

## Database Seeding Strategy

**Seed Data Includes:**
1. **Users**: Hotel admin + advertiser demo accounts
2. **Hotel Settings**: Privacy-first defaults
3. **Segments**: 5 guest types with characteristics
4. **Ads**: 2 sample campaigns
5. **Analytics**: Synthetic performance data

**Seeding Logic:**
- Check if data exists before seeding
- Use transactions for consistency
- Idempotent: safe to run multiple times

## Error Handling Strategy

### Client-Side
- Try-catch blocks around API calls
- User-friendly error messages
- Loading states during async operations
- Graceful degradation if API fails

### Server-Side
- Try-catch in all API routes
- Database constraint violations handled
- Generic error messages (no stack traces to client)
- Console logging for debugging

## Testing Strategy (Future)

### Recommended Tests
1. **Unit Tests**
   - Database functions
   - Authentication logic
   - Component rendering

2. **Integration Tests**
   - API routes
   - Full user flows
   - Database interactions

3. **E2E Tests**
   - Login flow
   - Campaign creation
   - Settings update

4. **Tools**
   - Jest + React Testing Library
   - Cypress for E2E
   - MSW for API mocking

## Deployment Considerations

### Local Development
- ✅ Zero configuration
- ✅ SQLite in-memory or file
- ✅ Hot reload
- ✅ No environment variables

### Production
- Migrate to PostgreSQL/MySQL
- Add connection pooling
- Environment-based configuration
- CDN for static assets
- Database migrations strategy
- Backup and recovery plan

## Scalability Considerations

### Current Limitations
- SQLite → Single-user/development only
- sessionStorage → No multi-tab sync
- No caching layer
- File-based database

### Scaling Path
1. **Phase 1** (1-100 users):
   - Migrate to PostgreSQL
   - Add Redis for caching
   - Implement proper sessions

2. **Phase 2** (100-1000 users):
   - Database read replicas
   - API rate limiting
   - CDN for assets

3. **Phase 3** (1000+ users):
   - Microservices architecture
   - Message queue (RabbitMQ/Kafka)
   - Separate analytics service

## Code Quality Standards

### TypeScript
- Strict mode enabled
- No `any` types (use `unknown` or proper types)
- Interface-first design

### Components
- Functional components only
- Hooks for state management
- Props validation via TypeScript
- Descriptive component names

### Naming Conventions
- **Components**: PascalCase (`DashboardLayout`)
- **Functions**: camelCase (`authenticateUser`)
- **Files**: kebab-case for routes, PascalCase for components
- **CSS Classes**: Tailwind utilities (no custom classes except utilities)

### Comments
- JSDoc for functions
- Inline comments for complex logic
- README in each directory explaining purpose

## Future Enhancements

### Technical
- [ ] Add real-time updates (WebSockets)
- [ ] Implement file upload for ad creatives
- [ ] Add image optimization pipeline
- [ ] Implement A/B testing framework
- [ ] Add comprehensive logging

### Features
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Campaign scheduling
- [ ] Budget management
- [ ] Email notifications
- [ ] Bulk operations
- [ ] Export reports (CSV/PDF)

---

**This architecture prioritizes simplicity, privacy, and maintainability while remaining production-ready with minimal modifications.**





