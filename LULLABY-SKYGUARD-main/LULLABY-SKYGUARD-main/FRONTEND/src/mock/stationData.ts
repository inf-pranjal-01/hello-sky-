import { Station } from '../types';

/**
 * [MOCK DATA]
 * Isolated meteorological station fixtures for mock mode.
 * Matches exact station API contract specification: [API: GET /api/stations — INTEGRATED]
 */
export const MOCK_STATIONS: Station[] = [
  {
    station_id: 'ST-NDL-001',
    name: 'New Delhi National Observatory',
    lat: 28.6139,
    lon: 77.2090,
    status: 'NORMAL',
    elevation_m: 216,
    region: 'Northern Plains',
    last_ping: '2026-09-08T20:00:00Z',
  },
  {
    station_id: 'ST-MUM-002',
    name: 'Mumbai Coastal Radar Station',
    lat: 19.0760,
    lon: 72.8777,
    status: 'WARNING',
    elevation_m: 14,
    region: 'Western Coastal',
    last_ping: '2026-09-08T20:00:00Z',
  },
  {
    station_id: 'ST-BLR-003',
    name: 'Bengaluru Highland Telemetry',
    lat: 12.9716,
    lon: 77.5946,
    status: 'NORMAL',
    elevation_m: 920,
    region: 'Deccan Plateau',
    last_ping: '2026-09-08T20:00:00Z',
  },
  {
    station_id: 'ST-HYD-004',
    name: 'Hyderabad Central AWS',
    lat: 17.3850,
    lon: 78.4867,
    status: 'CRITICAL',
    elevation_m: 542,
    region: 'Telangana Region',
    last_ping: '2026-09-08T19:45:00Z',
  },
  {
    station_id: 'ST-KOL-005',
    name: 'Kolkata Delta Meteorological Station',
    lat: 22.5726,
    lon: 88.3639,
    status: 'OFFLINE',
    elevation_m: 9,
    region: 'Eastern Coastal',
    last_ping: '2026-09-08T12:00:00Z',
  },
];
