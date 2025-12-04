# Technical Architecture

## Overview

The TV Welcome WebApp is a client-side single-page application (SPA) designed specifically for Smart TV browsers, particularly Samsung Smart TVs. The application provides a personalized welcome experience for hotel guests, featuring targeted advertising, streaming service access, and hotel information services. The architecture follows a modern, component-based frontend approach with a focus on TV-optimized user experience.

## Technology Stack

### Core Technologies

- **React 19.1.1**: Modern JavaScript library for building user interfaces using a component-based architecture
- **TypeScript 5.8.3**: Typed superset of JavaScript providing static type checking and enhanced developer experience
- **Vite (Rolldown-based)**: Next-generation build tool and development server offering fast Hot Module Replacement (HMR) and optimized production builds
- **Styled Components 6.1.19**: CSS-in-JS library enabling component-scoped styling with TypeScript support

### Development Tools

- **ESLint 9.36.0**: Code linting and quality assurance
- **TypeScript ESLint**: TypeScript-specific linting rules
- **React Hooks ESLint Plugin**: Enforces React Hooks best practices
- **Vite React Plugin**: React-specific optimizations for Vite build pipeline

## Architecture Patterns

### Component-Based Architecture

The application follows a modular component structure where each screen and feature is encapsulated as a reusable React component:

- **Ad Component**: Full-screen video advertisement with QR code integration
- **WelcomePage Component**: Main landing page with personalized greeting and navigation cards
- **StreamingSection Component**: Grid-based interface for streaming service selection
- **InfoSection Component**: Information cards displaying hotel services and amenities

### State Management

The application employs React's built-in state management using hooks:

- **Local Component State**: `useState` for component-specific state (navigation indices, timers, skip availability)
- **Route State**: Centralized routing state in `App.tsx` using a simple state machine pattern
- **Memoization**: `useMemo` for optimizing guest data computation

### Routing Architecture

The application implements a lightweight, custom routing system without external routing libraries:

```typescript
type AppRoute =
  | { screen: 'ad' }
  | { screen: 'welcome' }
  | { screen: 'streaming' }
  | { screen: 'info' }
```

This approach provides:
- Type-safe navigation
- Minimal bundle size
- Simple state transitions
- Predictable navigation flow

## Frontend Architecture

### Application Structure

```
src/
├── App.tsx                 # Root component with routing logic
├── main.tsx                # Application entry point
├── types.ts                # TypeScript type definitions
├── components/             # Feature components
│   ├── Ad.tsx              # Advertisement component
│   ├── WelcomePage.tsx     # Main welcome screen
│   ├── StreamingSection.tsx # Streaming services grid
│   └── InfoSection.tsx     # Hotel information cards
└── styles/
    └── GlobalStyles.ts     # Global CSS-in-JS styles
```

### Component Communication

- **Props-Based Data Flow**: Unidirectional data flow from parent to child components
- **Callback Functions**: Navigation and state updates handled via callback props
- **Event-Driven Navigation**: Keyboard event listeners for TV remote control

### Styling Architecture

The application uses **Styled Components** for CSS-in-JS styling, providing:

- **Component-Scoped Styles**: Each component defines its own styled elements
- **Dynamic Styling**: Conditional styling based on component state (e.g., focus states)
- **Theme Consistency**: Global styles defined in `GlobalStyles.ts` for TV-optimized defaults
- **Type Safety**: TypeScript integration for styled component props

Key styling considerations for TV optimization:
- Large font sizes (2.8rem - 4.8rem) for TV viewing distances
- High-contrast focus indicators (cyan outline: `#22d3ee`)
- Full-screen layouts (100vw × 100vh)
- Dark color scheme for reduced eye strain
- Hidden cursor for TV environments

## Backend Architecture

### Current Implementation

The current implementation is **frontend-only** with no dedicated backend server. Data is handled as follows:

- **Mock Data**: Guest information is hardcoded in `App.tsx` as a demonstration
- **Static Assets**: Video and image files served from the `public/` directory
- **External APIs**: QR code generation via third-party API (qrserver.com)

### Backend Integration Points

For production deployment, the following backend integration points are recommended:

1. **Property Management System (PMS) Integration**
   - Guest data retrieval (name, check-in date, preferences)
   - Real-time room status and service availability
   - Booking and reservation management

2. **Content Management System (CMS)**
   - Dynamic advertisement content delivery
   - Personalized content based on guest profile
   - Local business and attraction information

3. **API Architecture**
   - RESTful API endpoints for guest data
   - WebSocket connections for real-time updates
   - Authentication and session management

4. **Data Storage**
   - Guest preferences and history
   - Advertisement performance metrics
   - User interaction analytics

## Model Pipeline & Personalization

### Current State

The application currently uses **static mock data** for personalization:

```typescript
const guest: GuestData = {
  name: 'Peter',
  checkInDate: '2025-10-01',
  interests: ['spa', 'dining', 'city-tours']
}
```

### Recommended Personalization Pipeline

For a production system, the following pipeline would be implemented:

1. **Data Collection Layer**
   - Guest profile data from PMS
   - Historical interaction data
   - Preference inference from booking patterns

2. **Recommendation Engine**
   - Content-based filtering for advertisements
   - Collaborative filtering for service recommendations
   - Context-aware suggestions (time of day, weather, events)

3. **Content Selection**
   - Advertisement selection based on guest interests
   - Local attraction recommendations
   - Service prioritization (spa, dining, tours)

4. **A/B Testing Framework**
   - Advertisement variant testing
   - UI/UX optimization experiments
   - Conversion rate optimization

### Machine Learning Integration (Future)

Potential ML models for enhanced personalization:

- **Recommendation Models**: Matrix factorization or deep learning for content recommendations
- **Churn Prediction**: Identify guests likely to use services
- **Content Optimization**: Optimize advertisement timing and content
- **Natural Language Processing**: Sentiment analysis from guest feedback

## Design Patterns

### 1. Container/Presentational Pattern

- **Container Components**: `App.tsx` handles routing and state management
- **Presentational Components**: Feature components (`WelcomePage`, `StreamingSection`) focus on UI rendering

### 2. Custom Hooks Pattern

- Event listener management via `useEffect` hooks
- Timer and countdown logic encapsulated in components
- Keyboard navigation handlers as reusable patterns

### 3. Composition Pattern

- Components composed of smaller styled sub-components
- Flexible prop interfaces for component reuse
- Separation of concerns between layout and content

### 4. State Machine Pattern

- Simple routing state machine with discrete states
- Predictable state transitions
- Type-safe state definitions

## TV-Specific Optimizations

### Input Handling

- **Keyboard Event Listeners**: Arrow keys for navigation, Enter for selection
- **Focus Management**: Visual focus indicators for TV remote navigation
- **Accessibility**: Large touch targets (44px+), high contrast, clear focus states

### Performance Optimizations

- **Code Splitting**: Potential for route-based code splitting
- **Asset Optimization**: Video compression, image optimization
- **Lazy Loading**: Components loaded on-demand
- **Bundle Size**: Minimal dependencies to reduce bundle size

### Browser Compatibility

- **Samsung Smart TV Browser**: Primary target platform
- **Web Standards**: HTML5 video, CSS Grid, Flexbox
- **Polyfills**: Minimal polyfills required for modern TV browsers

## Build & Deployment

### Development Environment

- **Vite Dev Server**: Fast HMR with `--host` flag for LAN access
- **TypeScript Compilation**: Type checking during development
- **ESLint**: Real-time code quality feedback

### Production Build

- **TypeScript Compilation**: `tsc -b` for type checking
- **Vite Build**: Optimized production bundle with:
  - Code minification
  - Tree shaking
  - Asset optimization
  - Source maps for debugging

### Deployment Strategy

- **Static Hosting**: Can be deployed to any static hosting service (Netlify, Vercel, AWS S3)
- **HTTPS Requirement**: Required for TV browser compatibility
- **CDN Distribution**: Global content delivery for performance
- **Local Network Deployment**: For on-premise hotel deployments

## Data Flow

```
┌─────────────────┐
│   Guest Data    │  (Mock/API)
│   (PMS/CRM)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   App.tsx       │  (State Management)
│   (Router)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌──────────────┐
│ Ad      │ │ WelcomePage  │
│         │ │              │
└────┬────┘ └──────┬───────┘
     │             │
     │      ┌──────┴──────┐
     │      │             │
     ▼      ▼             ▼
┌─────────┐ ┌─────────────┐ ┌─────────────┐
│  Video  │ │ Streaming   │ │ Info        │
│  Ad     │ │ Section     │ │ Section     │
└─────────┘ └─────────────┘ └─────────────┘
```

## Security Considerations

### Current Implementation

- **Client-Side Only**: No sensitive data handling
- **Static Assets**: Public folder accessible to all

### Production Recommendations

- **HTTPS Enforcement**: Secure communication
- **Content Security Policy (CSP)**: Prevent XSS attacks
- **API Authentication**: Secure guest data access
- **Input Validation**: Sanitize user inputs
- **Rate Limiting**: Prevent abuse of external APIs

## Scalability & Extensibility

### Current Limitations

- Single-page application with limited scalability
- Mock data limits personalization
- No backend for dynamic content

### Scalability Path

1. **Microservices Architecture**: Separate services for recommendations, content, analytics
2. **Caching Strategy**: CDN caching, browser caching, API response caching
3. **Database Integration**: Guest data persistence, analytics storage
4. **Real-Time Updates**: WebSocket connections for live content updates

### Extension Points

- **New Sections**: Easy addition of new feature components
- **Plugin Architecture**: Modular content providers
- **Theme System**: Customizable styling per property
- **Multi-Language Support**: Internationalization (i18n) framework

## Testing Strategy

### Current State

- **Manual Testing**: TV browser testing on physical devices
- **Development Testing**: Local development server testing

### Recommended Testing Framework

- **Unit Testing**: Jest + React Testing Library
- **Component Testing**: Isolated component testing
- **Integration Testing**: Component interaction testing
- **E2E Testing**: Playwright or Cypress for TV browser simulation
- **Accessibility Testing**: Automated a11y checks

## Monitoring & Analytics

### Recommended Implementation

- **Performance Monitoring**: Core Web Vitals tracking
- **User Analytics**: Guest interaction tracking
- **Error Tracking**: Sentry or similar error monitoring
- **A/B Testing**: Feature flag system for experimentation
- **Conversion Tracking**: Advertisement effectiveness metrics

## Conclusion

The TV Welcome WebApp demonstrates a modern, component-based frontend architecture optimized for Smart TV environments. While currently a prototype with mock data, the architecture provides a solid foundation for production deployment with backend integration, personalization pipelines, and advanced recommendation systems. The use of TypeScript, React, and Styled Components ensures type safety, maintainability, and a responsive development experience.


