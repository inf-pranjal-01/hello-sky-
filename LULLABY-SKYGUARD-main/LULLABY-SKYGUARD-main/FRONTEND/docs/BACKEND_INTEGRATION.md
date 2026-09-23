# SkyGuard AI — Backend Integration Specification

This document details the backend integration architecture, contract schemas, transport configuration, and the step-by-step handoff checklist for connecting the SkyGuard AI frontend with the future FastAPI backend.

---

## 1. Current Backend Status

**Status:** `LEVEL 3 — FULLY INTEGRATED (Steps 13A–13K Complete)`  
**Backend:** FastAPI live at `http://127.0.0.1:8000`  
**Default Frontend Mode:** `MOCK` (isolated in-memory fixtures, zero network requests)  
**Real Mode:** `VITE_API_MODE=real` routes all 10 service calls through the hardened `apiClient` transport.

All 10 backend endpoints have been verified against the live FastAPI backend. Automated integration tests (9 test suites) and a mock regression suite (19 assertions) all pass. TypeScript reports 0 errors and the production build completes cleanly.

---

## 2. API Mode Switch & Configuration

The frontend provides an explicit transport switch in `src/config/api.config.ts`:

- **Mock Mode (Default):** `VITE_API_MODE=mock` (or omitted) — all services return instant mock fixtures.
- **Real Backend Mode:** `VITE_API_MODE=real` — services route through `apiClient` using native `fetch` against the FastAPI base URL.
- **Base URL Configuration:** `VITE_API_BASE_URL` (default: `http://localhost:8000`).

```env
# Example .env.production for FastAPI integration
VITE_API_MODE=real
VITE_API_BASE_URL=http://localhost:8000
```

> [!SECURITY]
> Frontend environment variables (`VITE_*`) are bundled into client code and visible in the browser. Never include database passwords, private keys, or signing secrets in frontend configuration.

---

## 3. Approved API Contract (Exact 10 Endpoints)

The backend implements the following 10 endpoints:

### 1. `GET /api/stations`
Fetches the complete Automatic Weather Station network inventory.
- **Request:** None
- **Response Shape:**
  ```json
  [
    {
      "station_id": "ST-NDL-001",
      "name": "New Delhi Central Observatory",
      "lat": 28.6139,
      "lon": 77.2090,
      "status": "NORMAL",
      "elevation_m": 216,
      "region": "Northern Plains"
    }
  ]
  ```
- **Valid `status` values:** `"NORMAL" | "WARNING" | "CRITICAL" | "OFFLINE"`
- **Used by:** Station selector, sidebar, Stations / Spatial Validation page.

---

### 2. `GET /api/current-reading?station_id={station_id}`
Fetches real-time sensor telemetry and anomaly status for a specific station.
- **Real-Time Strategy:** Polling every 3–5 seconds (`API_CONFIG.realtime.pollingIntervalMs = 4000`)
- **Query Params:** `station_id: string` (required)
- **Response Shape:**
  ```json
  {
    "station_id": "ST-NDL-001",
    "timestamp": "2026-09-10T09:30:00.000Z",
    "temperature_c": {
      "value": 34.2,
      "normal_min": 15.0,
      "normal_max": 42.0
    },
    "pressure_hpa": {
      "value": 1008.4,
      "normal_min": 990.0,
      "normal_max": 1025.0
    },
    "humidity_pct": {
      "value": 62.1,
      "normal_min": 20.0,
      "normal_max": 85.0
    },
    "anomaly_score_pct": 12.5,
    "risk_level": "low",
    "sensor_health_pct": 98.0,
    "sensor_health_status": "HEALTHY"
  }
  ```
- **Valid `risk_level` values:** `"low" | "medium" | "high" | "critical"`
- **Valid `sensor_health_status` values:** `"HEALTHY" | "WARNING" | "CRITICAL" | "OFFLINE"`
- **Used by:** Dashboard Metric Cards, Monitor Page, Station Summary.

---

### 3. `GET /api/trends?station_id={station_id}&hours={hours}`
Fetches historical time-series points for telemetry trend charts.
- **Query Params:** `station_id: string` (required), `hours: number` (optional, default: 6)
- **Response Shape:**
  ```json
  {
    "station_id": "ST-NDL-001",
    "hours": 6,
    "points": [
      {
        "timestamp": "2026-09-10T03:30:00.000Z",
        "temperature_c": 28.5,
        "pressure_hpa": 1010.2,
        "humidity_pct": 74.0,
        "anomaly_score_pct": 5.0
      }
    ]
  }
  ```
- **Used by:** Dashboard Trend Charts, Telemetry History.

---

### 4. `GET /api/anomalies/latest?station_id={station_id}`
Fetches the most recent incident or anomaly event detected for a station.
- **Query Params:** `station_id: string` (required)
- **Response Shape:** Object or `null` if no active/recent anomaly exists:
  ```json
  {
    "anomaly_id": "ANOM-20260910-001",
    "timestamp": "2026-09-10T09:15:00.000Z",
    "station_id": "ST-NDL-001",
    "anomaly_score_pct": 87.4,
    "severity": "high",
    "type": "spike",
    "root_cause": "Rapid temperature spike exceeding 3σ historical rate of change",
    "description": "Temperature surged 6.2°C within a 3-minute window without corresponding humidity drop."
  }
  ```
- **Valid `severity` values:** `"low" | "medium" | "high" | "critical"`
- **Valid `type` values:** `"spike" | "frozen_value" | "drift" | "dropout" | "multivariate_inconsistency"`
- **Used by:** Latest Anomaly Banner, Alerts page.

---

### 5. `GET /api/anomalies/recent?station_id={station_id}&limit={limit}`
Fetches recent anomaly event history for a station.
- **Query Params:** `station_id: string` (required), `limit: number` (optional, default: 5)
- **Response Shape:**
  ```json
  [
    {
      "anomaly_id": "ANOM-20260910-001",
      "timestamp": "2026-09-10T09:15:00.000Z",
      "station_id": "ST-NDL-001",
      "anomaly_score_pct": 87.4,
      "severity": "high",
      "type": "spike",
      "root_cause": "Rapid temperature spike exceeding 3σ historical rate of change",
      "description": "Temperature surged 6.2°C within a 3-minute window."
    }
  ]
  ```
- **Used by:** Recent Anomalies List, Alerts page, Maintenance anomaly selector.

---

### 6. `GET /api/explain/{anomaly_id}`
Fetches SHAP / feature impact breakdown for a specific detected anomaly.
- **Path Param:** `anomaly_id: string` (required)
- **Response Shape:**
  ```json
  {
    "anomaly_id": "ANOM-20260910-001",
    "features": [
      { "name": "Temperature Rate of Change", "impact": 0.54 },
      { "name": "Pressure-Temp Covariance", "impact": 0.28 },
      { "name": "Diurnal Baseline Residual", "impact": 0.12 },
      { "name": "Humidity Gradient", "impact": -0.06 }
    ]
  }
  ```
- **Used by:** Anomaly Investigation Modal, Feature Impact Chart.

---

### 7. `GET /api/sensor-health?station_id={station_id}`
Fetches hardware reliability and sensing subsystem health status.
- **Query Params:** `station_id: string` (required)
- **Response Shape:**
  ```json
  {
    "station_id": "ST-NDL-001",
    "sensor_health_pct": 94.0,
    "sensor_health_status": "HEALTHY"
  }
  ```
- **Valid `sensor_health_status` values:** `"HEALTHY" | "WARNING" | "CRITICAL" | "OFFLINE"`
- **Used by:** Sensor Health Dashboard, Hardware Reliability Monitor.

---

### 8. `POST /api/repair-sensor`
Initiates sensor recovery mode for an Automatic Weather Station.
- **Request Body:**
  ```json
  { "station_id": "AWS-CHN-024" }
  ```
- **Response Shape:**
  ```json
  {
    "success": true,
    "station_id": "AWS-CHN-024",
    "status": "WARNING",
    "recovery_active": true,
    "message": "Sensor marked for repair recovery. Clean readings will be evaluated before returning it to HEALTHY."
  }
  ```
- **IMPORTANT:** `recovery_active: true` means recovery has been initiated. The backend requires 5 consecutive clean readings before returning to `HEALTHY`. The frontend MUST NOT claim repair is complete while `recovery_active = true`.
- **Used by:** Sensor Health page repair flow.

---

### 9. `POST /api/inject-anomaly`
Starts synchronized anomaly replay in the backend telemetry pipeline.
- **IMPORTANT:** This does **NOT** directly inject an anomaly into the requested station. The backend starts a synchronized historical replay. The anomaly may appear on a different station than requested.
- **Request Body:**
  ```json
  {
    "station_id": "AWS-CHN-024",
    "type": "spike"
  }
  ```
- **Supported types:** `"spike" | "frozen_value" | "drift" | "dropout" | "sensor_fail_low" | "multivariate_inconsistency"`
- **Response Shape:**
  ```json
  {
    "success": true,
    "anomaly_id": "ANOM-INJ-20260910-001",
    "message": "Anomaly replay started."
  }
  ```
- **409 Conflict:** Returned when a replay is already in progress.
- **Used by:** Demonstration Control Panel / SIH demo trigger.

---

### 10. `POST /api/maintenance-ticket`
Dispatches a field maintenance ticket for an anomalous sensor.
- **Request Body:**
  ```json
  { "anomaly_id": "anom_04537" }
  ```
- **Response Shape:**
  ```json
  {
    "ticket_id": "TCK-0004",
    "station_id": "AWS-CHN-024",
    "issue": "Sensor stuck / communication fault",
    "priority": "high",
    "created_at": "2026-09-10T18:11:37.446112+00:00"
  }
  ```
- **IMPORTANT:** The frontend sends ONLY `anomaly_id`. All other fields (`ticket_id`, `station_id`, `issue`, `priority`, `created_at`) are computed and returned by the backend. The frontend MUST NOT calculate any of these values.
- **Used by:** Maintenance Operations page, Anomaly Investigation deep-link.

---

## 4. Frontend-Only / Derived Functionality (No Backend Endpoints Needed)

The following features are computed client-side from the 9 contract endpoints above:

| Feature | Implementation Mode | Notes |
| :--- | :--- | :--- |
| **Analytics Dashboard** | `[FRONTEND ONLY]` | Aggregated statistics, distribution calculations, and anomaly histograms computed in the browser. |
| **Operational Reports** | `[FRONTEND ONLY]` | Multi-metric report compilation, health score synthesis, recommendations engine, and browser PDF printing (`window.print()`). |
| **Spatial Validation / Comparison** | `[FRONTEND DEMO LOGIC]` | Haversine distance calculations and peer variance analysis derived from station metadata and telemetry. |
| **Authentication & Session** | `[MOCK AUTH — TEMPORARY]` | Frontend session storage (`sessionStorage` / `localStorage`) for operator identity and role management. |
| **Sensor Health History Chart** | `[FRONTEND ONLY]` | Historical point generation for health trend display. |

---

## 5. Polling & Data Freshness Model

The frontend uses an HTTP polling architecture:
- **Interval:** 4000ms (configurable 3–5 seconds)
- **Freshness Rules:**
  - `LIVE`: Reading timestamp ≤ 15 seconds ago
  - `DATA DELAYED`: Reading timestamp > 15 seconds and ≤ 60 seconds ago
  - `DATA STALE`: Reading timestamp > 60 seconds ago
  - `MONITORING PAUSED`: Operator paused the feed via UI toggle
- **Failure Recovery:** If a polling request fails due to temporary network loss, the UI retains the last known valid telemetry reading, displays a `DATA DELAYED` / `DATA STALE` indicator with a retry option, and continues polling without crashing or looping indefinitely.

---

## 6. Runtime Response Validation

The transport layer (`src/services/validators.ts`) validates every incoming response:
- Type-checks all required fields (`station_id`, metric values, timestamps, severities, etc.).
- Normalizes out-of-range or missing optional fields with safe defaults.
- Converts malformed responses into controlled `ApiError` instances with code `VALIDATION_ERROR` to prevent unhandled UI crashes.

---

## 7. Backend Developer Handoff Checklist

When building the FastAPI backend, follow this verification checklist:

- [ ] FastAPI application initialized with CORS enabled for frontend origin (`http://localhost:5173` or production domain).
- [ ] `GET /api/stations` implemented matching contract schema.
- [ ] `GET /api/current-reading` implemented with required sensor sub-objects (`temperature_c`, `pressure_hpa`, `humidity_pct`).
- [ ] `GET /api/trends` implemented with time-series `points` array.
- [ ] `GET /api/anomalies/latest` implemented (returning object or `null`).
- [ ] `GET /api/anomalies/recent` implemented with `limit` query param.
- [ ] `GET /api/explain/{anomaly_id}` implemented with `features` impact array.
- [ ] `GET /api/sensor-health` implemented with `sensor_health_pct` and `sensor_health_status`.
- [ ] `POST /api/inject-anomaly` implemented accepting `station_id` and `type`.
- [ ] `POST /api/maintenance-ticket` implemented accepting `anomaly_id` and returning `ticket_id`.
- [ ] All response field names match the contract in exact casing (`snake_case`).
- [ ] Set `VITE_API_MODE=real` and `VITE_API_BASE_URL=http://localhost:8000` in `.env`.
- [ ] Run `npm run build` and verify end-to-end telemetry flows.
