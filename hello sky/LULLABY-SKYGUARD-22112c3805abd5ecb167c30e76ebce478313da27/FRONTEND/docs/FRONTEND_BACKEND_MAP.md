# SkyGuard AI — Frontend to Backend Endpoint Mapping

This document tracks all API boundaries, their maturity status, backend contracts, and frontend mapping points.

---

## Maturity Level: LEVEL 3 — FULLY INTEGRATED (Steps 13A–13K Complete)

The real FastAPI backend is live at `http://127.0.0.1:8000`. All 10 endpoints have been verified against the real backend. The frontend operates in two modes:

- **Mock Mode (Default):** `VITE_API_MODE=mock` — all services return instant mock fixtures, zero network requests.
- **Real Backend Mode:** `VITE_API_MODE=real` — all services route through the hardened `apiClient` transport against the live FastAPI backend.

---

## Approved API Endpoint Mapping Table (Exact 10 Endpoints)

| # | Frontend Component / Feature | Backend Endpoint Reference | Method | Status | Notes |
| :- | :--- | :--- | :-: | :--- | :--- |
| **1** | **Station Selector / Sidebar / Map** | `/api/stations` | `GET` | `INTEGRATED — VERIFIED` | Fetches AWS station network inventory (20 stations from real backend) |
| **2** | **Dashboard Metric Cards & Monitor** | `/api/current-reading` | `GET` | `INTEGRATED — VERIFIED` | Telemetry readings with value & normal ranges (4s polling) |
| **3** | **Dashboard & Monitor Trend Charts** | `/api/trends` | `GET` | `INTEGRATED — VERIFIED` | Historical time-series points (`hours` param) |
| **4** | **Latest Anomaly Card / Banner** | `/api/anomalies/latest` | `GET` | `INTEGRATED — VERIFIED` | Most recent incident for active AWS |
| **5** | **Recent Anomalies List & Maintenance** | `/api/anomalies/recent` | `GET` | `INTEGRATED — VERIFIED` | Compact historical incidents (`limit` param) |
| **6** | **Feature Impact / Explainability Modal** | `/api/explain/{anomaly_id}` | `GET` | `INTEGRATED — VERIFIED` | SHAP feature contribution breakdown (21 features from real backend) |
| **7** | **Sensor Health & Reliability Monitor** | `/api/sensor-health` | `GET` | `INTEGRATED — VERIFIED` | Backend-reported hardware health (`HEALTHY \| WARNING \| OFFLINE`) |
| **8** | **Operator Repair / Recovery Panel** | `/api/repair-sensor` | `POST` | `INTEGRATED — VERIFIED` | Initiates 5-clean-reading recovery cycle on backend |
| **9** | **SIH Anomaly Injection Control Panel** | `/api/inject-anomaly` | `POST` | `INTEGRATED — VERIFIED` | Starts synchronized anomaly replay on backend (not direct injection) |
| **10** | **Maintenance Ticket Dispatch** | `/api/maintenance-ticket` | `POST` | `INTEGRATED — VERIFIED` | Field technician maintenance ticket — all fields from backend |

---

## Disallowed / Non-Existent Endpoints Check

The following endpoints are **NOT** in the API contract and are strictly prohibited from being introduced:
- `/api/sensor-health/history` (historical sensor health is [FRONTEND ONLY] [MOCK DATA — TEMPORARY])
- `/api/health-history`
- `/api/sensor-diagnostics`
- `/api/hardware-health`
- `/api/battery`
- `/api/signal`
- `/api/calibration`
- `/api/degradation`
- `/api/notifications`
- `/api/search`
- `/api/system-status`
- `/auth/me`
- `/api/v1/auth/*`
- `/api/shap` (use `/api/explain/{anomaly_id}` for model explainability)
- `/api/spatial`
- `/api/spatial-validation`
- `/api/stations/nearby`
- `/api/stations/neighbors`
- `/api/network`
- `/api/network/telemetry`
- `/api/compare-stations`
- `/api/station-comparison`
- `/api/geospatial`
- `/api/station-map`
- `/api/regional-analysis`
- `/api/reports`
- `/api/reports/generate`
- `/api/reports/create`
- `/api/reports/export`
- `/api/reports/history`
- `/api/report`
- `/api/pdf`
- `/api/analytics`
- `/api/analytics/report`
- `/api/maintenance`
- `/api/maintenance-tickets`
- `/api/health`
- WebSocket / MQTT / SSE streaming endpoints

---

## Architecture Seams & Boundary Notes

- **Step 8 Analytics & Insights:** `[FRONTEND ONLY]` (All aggregations, histograms, and statistical distributions computed client-side).
- **Step 9 Spatial Validation:** `[FRONTEND DEMO LOGIC]` (Neighbor distance and peer variance calculated from station coordinates and telemetry).
- **Step 10 Operational Reports:** `[FRONTEND ONLY]` (Multi-metric synthesis, automated recommendations, and print export).
- **Step 11 Maintenance Operations:** `[API: POST /api/maintenance-ticket]` (Creates ticket with `{ anomaly_id: string }`).
- **Step 12 Transport Hardening:** `src/services/apiClient.ts` native `fetch` client with timeout, JSON parsing, `ApiError` hierarchy, and runtime schema validation guards (`src/services/validators.ts`).
