# SkyGuard AI — Frontend Architecture Guide

This document outlines the architecture, directory structure, data patterns, accessibility features, authentication boundary, and future integration points for the SkyGuard AI frontend web application.

---

## 1. Frontend Folder Structure

```
skyguard_ai/
├── docs/                             # Project documentation and progress tracking
│   ├── FRONTEND_ARCHITECTURE.md      # This guide
│   └── DEVELOPMENT_PROGRESS.md       # Step-by-step progress tracking
├── src/
│   ├── assets/                       # Static media (logos, icons, illustrations)
│   ├── components/                   # Reusable UI building blocks
│   │   ├── common/                   # Foundation widgets (Button, Tooltip, Card, Badge, Modal, etc.)
│   │   ├── layout/                   # Application Shell (AppShell, Header, Sidebar, Footer)
│   │   └── state/                    # Visual state wrappers (Loading, Error, Empty, Offline)
│   ├── config/                       # Centralized application & API config
│   │   └── api.config.ts             # Base URLs, timeouts, polling intervals [BACKEND REQUIRED]
│   ├── context/                      # Application Context Providers
│   │   └── AuthContext.tsx           # Global authentication state provider
│   ├── hooks/                        # Custom React hooks
│   │   └── useSensorPolling.ts       # Polling hook shell [REALTIME: POLLING — FUTURE BACKEND INTEGRATION]
│   ├── mock/                         # Static mock fixtures [MOCK DATA — TEMPORARY]
│   │   ├── sensorData.ts             # API-contract-compliant sensor readings
│   │   ├── stationData.ts            # Meteorological monitoring stations
│   │   ├── anomalyData.ts            # ML anomaly event records
│   │   └── systemStatus.ts           # Operational system health states
│   ├── pages/                        # Page views corresponding to application routes
│   │   ├── LoginPage.tsx             # Standalone 2-column meteorological login console
│   │   ├── ForgotPasswordPage.tsx    # Password recovery request view [BACKEND REQUIRED — PASSWORD RESET]
│   │   ├── ResetPasswordPage.tsx     # Password reset placeholder view [BACKEND REQUIRED — PASSWORD RESET]
│   │   ├── DashboardPage.tsx
│   │   ├── MonitorPage.tsx
│   │   ├── AlertsPage.tsx
│   │   ├── SensorHealthPage.tsx
│   │   ├── AnalyticsPage.tsx
│   │   ├── StationsPage.tsx
│   │   ├── ReportsPage.tsx
│   │   ├── MaintenancePage.tsx
│   │   ├── SettingsPage.tsx
│   │   ├── ProfilePage.tsx
│   │   └── NotFoundPage.tsx
│   ├── router/                       # Routing configuration
│   │   ├── AppRouter.tsx             # React Router v6 route mapping
│   │   └── ProtectedRoute.tsx        # Authentication layout guard
│   ├── services/                     # Data & auth abstraction layer
│   │   ├── authService.ts            # Auth service boundary [MOCK AUTH — TEMPORARY]
│   │   ├── sensorService.ts          # Sensor data service interface
│   │   ├── stationService.ts         # Monitoring station service interface
│   │   └── anomalyService.ts        # Anomaly records service interface
│   ├── styles/                       # Global styles & design system tokens
│   │   ├── variables.css             # Color palette, spacing, typography, focus tokens
│   │   └── global.css                # CSS resets, utility classes, animations
│   ├── types/                        # TypeScript domain interfaces & contracts
│   │   ├── auth.ts                   # User & Auth state schemas
│   │   └── index.ts                  # API-contract-matching schemas
│   ├── App.tsx                       # Root component wrapping Router & AuthProvider
│   └── main.tsx                      # Vite React mounting entry point
├── index.html                        # HTML5 document root with SEO metadata
├── package.json                      # Project dependencies & scripts
├── tsconfig.json                     # TypeScript compiler configuration
└── vite.config.ts                    # Vite build tool setup with `@` path alias
```

---

## 2. Authentication Architecture

> [!IMPORTANT]
> **API Contract Status:** Authentication API endpoints are currently **NOT** defined in the API contract. The backend maturity is **LEVEL 0 — NO VERIFIED BACKEND**. No URLs, HTTP methods, or specific auth endpoints are assumed.

### Current Mock Architecture:

```
[LoginPage / UI]
      ↓ calls AuthContext.login()
[AuthContext.tsx]
      ↓ delegates to
[authService.ts]
      ↓ returns typed user session
[Mock Auth Adapter — MOCK AUTH — TEMPORARY]
      ↓ persists session
[sessionStorage / localStorage]
```

### Future Production Architecture:

```
[LoginPage / UI]  (NO UI REWRITE REQUIRED)
      ↓ calls AuthContext.login()
[AuthContext.tsx]
      ↓ delegates to
[authService.ts]
      ↓ authService will be replaced or extended with real auth implementation
[Real Backend Authentication — BACKEND REQUIRED — AUTHENTICATION]
      ↓ returns user session
[Frontend Session]
```

### Key Architectural Rules Enforced:
1. **Isolated Boundaries:** UI components (`LoginPage`, `Header`, `AppShell`) never touch `localStorage` or `sessionStorage` directly. They rely strictly on `AuthContext`.
2. **Demo Credentials:** Obvious temporary credentials (`demo@skyguard.local` / `demo1234`) are used for demonstration purposes. No passwords are saved in browser storage.
3. **Session Persistence:** `rememberMe = false` stores mock user data in `sessionStorage`; `rememberMe = true` stores mock user data in `localStorage`. Both are cleared on logout.
4. **Protected Routes:** `ProtectedRoute.tsx` guards all application routes (`/dashboard`, `/monitor`, `/alerts`, etc.), redirecting unauthenticated traffic to `/login` while preserving the origin target for post-login return.

---

## 3. Component & Page Definitions

- **Components (`components/common/`)**: Low-level foundational controls (`Button`, `Card`, `StatusBadge`, `Tooltip`, `Modal`, `Dropdown`, `Loading`, `Skeleton`, `EmptyState`, `ErrorState`, `ConfirmationDialog`).
- **Layout Shell (`components/layout/`)**: `AppShell`, `Header`, `Sidebar`, `Footer`. Includes user dropdown menu and logout confirmation dialog.
- **Pages (`pages/`)**: Route-level view containers.

---

## 4. API & Backend Integration Markers

All future integration points are explicitly annotated throughout the codebase:
- `[BACKEND REQUIRED]`: Features needing live backend servers
- `[MOCK AUTH — TEMPORARY]`: Frontend mock authentication adapters
- `[BACKEND REQUIRED — AUTHENTICATION]`: Future authentication implementation once backend contract is defined
- `[BACKEND REQUIRED — SESSION MANAGEMENT]`: Future session and token handling
- `[BACKEND REQUIRED — PASSWORD RESET]`: Future password recovery functionality

---

## 5. Dashboard Data Architecture (Step 4)

### Current Architecture:
```
[DashboardPage / UI]
      ↓ uses
[useDashboardData() Hook]
      ↓ delegates to
[Feature Services: currentReadingService, trendsService, anomalyService]
      ↓ consumes
[Mock Adapters — MOCK DATA — TEMPORARY]
      ↓ fixtures strictly matching
[EXACT API CONTRACT]
```

### Future Production Architecture:
```
[DashboardPage / UI]  (NO UI REWRITE REQUIRED)
      ↓ uses
[useDashboardData() Hook]
      ↓ delegates to
[Feature Services: currentReadingService, trendsService, anomalyService]
      ↓ executes HTTP fetch calls
[Live Backend APIs: Level 1+ Backend]
```

### Step 4 Backend Contract Boundaries:
1. `[API: GET /api/current-reading — BACKEND REQUIRED]`: Live sensor readings for the active AWS.
2. `[API: GET /api/trends — BACKEND REQUIRED]`: Historical telemetry time-series points (hours query param).
3. `[API: GET /api/anomalies/latest — BACKEND REQUIRED]`: Most recent anomaly incident for the active AWS.
4. `[API: GET /api/anomalies/recent — BACKEND REQUIRED]`: Compact recent anomaly incident history list.
5. `[SIH DEMO PREPARATION] [API: POST /api/inject-anomaly — BACKEND REQUIRED]`: Architecture seam for future anomaly injection demo trigger. (NOT called or implemented in Step 4).

---

## 6. Real-Time Telemetry Monitor Architecture (Step 5)

### Unified Polling & Telemetry Management:
The Real-Time Monitor (`/monitor`) shares the exact single polling engine (`useDashboardData.ts`) with the Main Dashboard, avoiding duplicate polling timers or overlapping network requests.

```
[MonitorPage / UI]
      ↓ uses
[useDashboardData(selectedStationId, { trendHours, pollingIntervalMs: 4000 })]
      ↓ manages
- Centralized 4000ms polling interval
- Pause / Resume state ([FRONTEND ONLY])
- In-memory sequence buffer (TelemetryHistoryRecord[], max 150 items)
- Freshness evaluation (LIVE <=15s, DATA DELAYED 15-60s, DATA STALE >60s)
      ↓ queries
[currentReadingService] & [trendsService]
      ↓ consumes
[Mock Adapters — MOCK DATA — TEMPORARY]
```

### Step 5 Backend Contract Boundaries:
1. `[API: GET /api/current-reading?station_id=... — BACKEND REQUIRED]`: Provides current Temperature, Pressure, Humidity, and overall hardware health.
2. `[API: GET /api/trends?station_id=...&hours=... — BACKEND REQUIRED]`: Historical telemetry feeds for 6h / 12h / 24h trend views.
3. `[FRONTEND ONLY]`: Pause/resume controls, sequential telemetry history buffering, and telemetry freshness calculations.


