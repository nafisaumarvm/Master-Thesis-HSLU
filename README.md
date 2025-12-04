# Hotel Advertising Dashboard

A lightweight, production-ready dashboard web application for managing privacy-preserving, personalized in-room advertising for Smart TVs in hotels.

## 🎯 Overview

This dashboard implements a human-centered design approach to hotel advertising, based on research findings from interviews and surveys with hotels and advertisers. It prioritizes:

- **Privacy-First Design**: GDPR-compliant, no PII collection, automatic data deletion
- **Non-Intrusive Advertising**: Contextual ad placement (welcome screen only by default)
- **Operational Simplicity**: Minimal technical setup, simple management interface
- **Advertiser Quality**: High-quality environment with clear segmentation and measurable outcomes

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- No external services or databases required

### Installation

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Demo Credentials

**Hotel Admin:**
- Email: `hotel@example.com`
- Password: `hotel123`

**Advertiser:**
- Email: `advertiser@example.com`
- Password: `advertiser123`

## 📋 Features

### For Hotel Admins

#### 1. Dashboard Overview
- Real-time analytics (impressions, conversions, occupancy multiplier)
- Guest segment visualization
- Privacy-first metrics

#### 2. Guest Segments
- 5 synthetic, anonymized segments:
  - Business Travelers (35%)
  - Leisure Travelers (28%)
  - Conference Guests (15%)
  - Families (12%)
  - Long-Stay Guests (10%)
- Zero PII collection
- Automatic post-checkout deletion

#### 3. Ad Settings
**Allowed Content Categories:**
- ✓ Local Restaurants
- ✓ Museums & Attractions
- ✓ Mobility & Transport
- ✓ Retail & Luxury
- ✓ Spa & Wellness
- ✓ Events & Entertainment
- ✓ Nightlife

**Ad Exposure Rules:**
- ✅ Show on Welcome Screen (Recommended)
- ⬜ Show During Idle Mode
- ⬜ Sidebar Banner (Hotel Info)
- ✅ Never Interrupt Content (Always Enabled)

#### 4. Privacy Center
- GDPR compliance overview
- Automatic anonymization
- Post-checkout data deletion (0 days retention)
- No third-party tracking

### For Advertisers

#### 1. Campaign Dashboard
- Active campaigns overview
- Performance metrics (impressions, CTR, reach)
- Privacy-preserving analytics

#### 2. Create Campaign
- Upload creative (headline, CTA, image/video)
- Select content categories
- Target specific guest segments
- Choose campaign objective:
  - Awareness
  - Footfall
  - Performance/Conversions
- Location targeting
- Estimated reach calculator

#### 3. Campaign Management
- View all campaigns
- Filter by status (active/paused)
- Detailed performance metrics
- Aggregated, anonymized insights only

## 🏗️ Tech Stack

- **Frontend**: Next.js 14 (React 18)
- **Styling**: TailwindCSS
- **Backend**: Next.js API Routes
- **Database**: JSON-based (no native dependencies)
- **Authentication**: Simple email/password (bcryptjs)
- **TypeScript**: Full type safety

## 📁 Project Structure

```
/
├── app/
│   ├── api/                    # API routes
│   │   ├── auth/login/         # Authentication
│   │   ├── hotel/              # Hotel-specific APIs
│   │   ├── ads/                # Ad management
│   │   └── segments/           # Guest segments
│   ├── hotel/                  # Hotel admin pages
│   │   ├── dashboard/          # Main overview
│   │   ├── segments/           # Segment details
│   │   ├── settings/           # Ad settings
│   │   └── privacy/            # Privacy center
│   ├── advertiser/             # Advertiser pages
│   │   ├── dashboard/          # Campaign overview
│   │   ├── campaigns/          # Campaign list
│   │   └── create/             # Create new campaign
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Login page
│   └── globals.css             # Global styles
├── components/
│   ├── Navigation.tsx          # Role-based navigation
│   ├── DashboardLayout.tsx     # Authenticated layout
│   └── InfoTooltip.tsx         # Contextual help
├── lib/
│   ├── db.ts                   # Database setup & seeding
│   ├── auth.ts                 # Authentication utilities
│   └── types.ts                # TypeScript types
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── README.md
```

## 🔒 Privacy & Compliance

### GDPR Compliance

✅ **Data Minimization**: Only anonymized booking patterns collected  
✅ **Right to Erasure**: Automatic deletion post-checkout  
✅ **Purpose Limitation**: Data used only for ad targeting  
✅ **Privacy by Design**: No PII collection architecture  

### What We Collect
- Anonymized booking patterns (e.g., "weekday arrival")
- Aggregated segment data
- Non-identifiable stay characteristics

### What We DON'T Collect
- ❌ Names, emails, phone numbers
- ❌ Room numbers
- ❌ Payment information
- ❌ Browsing history
- ❌ Individual guest behavior

### Data Retention
- **0 days** post-checkout
- Automatic, permanent deletion
- No backups of guest data

## 🎨 Design Philosophy

### Human-Centered Principles

1. **Respect Guest Experience**: Ads only on welcome screen by default
2. **Operational Simplicity**: No technical setup, intuitive interface
3. **Privacy-First**: GDPR-compliant by design
4. **Value-Driven**: Only contextually relevant, useful content
5. **Advertiser Quality**: High-quality environment with clear metrics

### UI/UX Guidelines

- **Minimalist Design**: Clean, uncluttered interface
- **Limited Options**: Prevent overwhelming users
- **Microcopy Tooltips**: Contextual help throughout
- **Default Settings**: Privacy-preserving defaults
- **Clear Navigation**: Maximum 4 menu items per role

## 🧪 Development

### Build for Production

```bash
npm run build
npm start
```

### Database

The JSON database (`data/database.json`) is automatically created and seeded with synthetic data on first run.

To reset the database:
```bash
rm -rf data/database.json
npm run dev
```

### Environment

No environment variables required. Everything runs locally.

## 📊 Synthetic Data

All data in the system is synthetic and privacy-preserving:

- **Guest Segments**: Aggregated patterns, no real guests
- **Analytics**: Simulated metrics for demonstration
- **Ads**: Sample campaigns with randomized performance
- **Users**: Demo accounts only

## 🚢 Deployment

### Local Deployment
The application is designed to run locally with zero external dependencies.

### Production Considerations
For production deployment:

1. **Replace JSON database** with PostgreSQL/MySQL/MongoDB for multi-user support
2. **Add proper authentication** (e.g., NextAuth.js, Auth0)
3. **Implement file uploads** for ad creatives
4. **Add rate limiting** to API routes
5. **Enable HTTPS** in production
6. **Set up monitoring** and error tracking
7. **Implement proper session management**

## 📝 Research-Based Design

This dashboard implements findings from:

- Hotel manager interviews regarding operational constraints
- Guest surveys on privacy expectations
- Advertiser requirements for quality and measurement
- GDPR compliance requirements
- Human-centered design best practices

### Key Research Insights Implemented

✅ Hotels have limited bandwidth → Simple, set-and-forget interface  
✅ Privacy concerns → Zero PII, automatic deletion  
✅ Guest experience → Non-intrusive, contextual placement only  
✅ Advertiser needs → Clear segmentation, measurable outcomes  
✅ Operational reality → No technical setup, free to install  

## 🤝 Contributing

This is a research prototype. For production use, please review the security and scalability considerations mentioned above.

## 📄 License

This project is for educational and research purposes.

## 👥 User Roles Summary

| Feature | Hotel Admin | Advertiser |
|---------|-------------|------------|
| View guest segments | ✅ | ❌ |
| Configure allowed categories | ✅ | ❌ |
| Set exposure rules | ✅ | ❌ |
| View privacy settings | ✅ | ❌ |
| Create ad campaigns | ❌ | ✅ |
| Target segments | ❌ | ✅ |
| View campaign performance | ❌ | ✅ |
| Upload creatives | ❌ | ✅ |

## 🆘 Support

For questions or issues:
1. Check this README
2. Review the code comments
3. Inspect the tooltips in the UI
4. Check the browser console for errors

---

**Built with privacy, simplicity, and human-centered design at the core.**

