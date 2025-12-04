## Smart TV Welcome App (Prototype)

React + TypeScript app optimized for Samsung Smart TV browsers, with a personalized Ad page shown first, then a Welcome page linking to Streaming and Info sections. Built with Vite and `styled-components`.

### Architecture

- `src/components/AdPage.tsx`: full-screen personalized ad; Enter proceeds to Welcome
- `src/components/WelcomePage.tsx`: greeting and 3 cards (Streaming, Info, Exit)
- `src/components/StreamingSection.tsx`: grid of streaming services, navigable
- `src/components/InfoSection.tsx`: hotel/Airbnb info cards (Wi‑Fi, contact, offers)
- `src/styles/GlobalStyles.ts`: TV-friendly global styles (focus, fonts, 100% screen)
- `src/App.tsx`: simple route state machine: ad → welcome → streaming/info
- `src/types.ts`: shared types (`GuestData`, `AppRoute`)

### Navigation (Remote)

- Use Arrow keys to move selection
- Enter selects/continues
- Backspace/Escape returns from sections

### Run Locally

```bash
npm install
npm run dev -- --host
```

Then open the LAN URL on your Samsung TV browser (ensure TV and your machine are on the same network).

### Test on Samsung Smart TV

- Press Home and open the TV web browser
- Enter the dev server LAN URL (from terminal, `--host` exposes it)
- Use remote arrows and Enter to navigate; Back/Escape to go back in sections

### Personalization & Data Feeds

- Mock `GuestData` is defined in `App.tsx`. Replace with real data:
  - Connect to a PMS/CRM API and map to `GuestData`
  - Fetch ad content server-side; pass down as props to `AdPage`
  - Cache minimal data locally; avoid heavy storage on TV browsers

### Production Notes

- Prefer single-page app with minimal navigation; avoid heavy animations
- Keep focus states high-contrast and large; use 44px+ tap/target sizes
- For external streaming apps, most TVs restrict deep linking; prototype redirects via `window.location.href`
- Consider hosting on HTTPS with a friendly local captive portal for auto-launch

### Extensibility

- Add sections for Dining, Spa, Housekeeping Requests, or Checkout
- Add QR codes in InfoSection for mobile handoff to app/WhatsApp
- Replace mock lists with API-driven content

