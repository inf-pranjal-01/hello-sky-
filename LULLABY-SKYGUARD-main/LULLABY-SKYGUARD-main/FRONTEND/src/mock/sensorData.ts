import { SensorReading } from '../types';

/**
 * [MOCK FIXTURES]
 * Isolated sensor reading fixtures.
 * Matches exact current-reading API contract specification.
 */
export const MOCK_SENSOR_READINGS: SensorReading[] = [
  {
    station_id: 'ST-NDL-001',
    timestamp: '2026-09-08T20:00:00Z',
    temperature_c: 34.2,
    pressure_hpa: 1012.4,
    humidity_pct: 68.5,
    anomaly_score_pct: 12.4,
    risk_level: 'low',
    sensor_health_pct: 98.0,
    sensor_health_status: 'optimal',
  },
  {
    station_id: 'ST-MUM-002',
    timestamp: '2026-09-08T20:00:00Z',
    temperature_c: 31.8,
    pressure_hpa: 1008.2,
    humidity_pct: 84.1,
    anomaly_score_pct: 78.9,
    risk_level: 'high',
    sensor_health_pct: 76.5,
    sensor_health_status: 'warning',
  },
  {
    station_id: 'ST-BLR-003',
    timestamp: '2026-09-08T20:00:00Z',
    temperature_c: 24.5,
    pressure_hpa: 915.0,
    humidity_pct: 62.0,
    anomaly_score_pct: 5.1,
    risk_level: 'low',
    sensor_health_pct: 99.5,
    sensor_health_status: 'optimal',
  },
  {
    station_id: 'ST-HYD-004',
    timestamp: '2026-09-08T20:00:00Z',
    temperature_c: 36.1,
    pressure_hpa: 1005.6,
    humidity_pct: 45.2,
    anomaly_score_pct: 91.3,
    risk_level: 'critical',
    sensor_health_pct: 42.0,
    sensor_health_status: 'degraded',
  },
];
