import { CurrentSensorReading } from '../types';

/**
 * [MOCK DATA]
 * Isolated current sensor reading fixtures for mock mode.
 * Matches exact backend response shape: [API: GET /api/current-reading — INTEGRATED]
 */
export const MOCK_CURRENT_READINGS: Record<string, CurrentSensorReading> = {
  'ST-NDL-001': {
    station_id: 'ST-NDL-001',
    timestamp: new Date().toISOString(),
    temperature_c: {
      value: 28.4,
      normal_min: 18.0,
      normal_max: 35.0,
    },
    pressure_hpa: {
      value: 1008.4,
      normal_min: 990.0,
      normal_max: 1025.0,
    },
    humidity_pct: {
      value: 64.2,
      normal_min: 30.0,
      normal_max: 80.0,
    },
    anomaly_score_pct: 17,
    risk_level: 'low',
    sensor_health_pct: 94,
    sensor_health_status: 'HEALTHY',
  },
  'ST-MUM-002': {
    station_id: 'ST-MUM-002',
    timestamp: new Date().toISOString(),
    temperature_c: {
      value: 31.8,
      normal_min: 22.0,
      normal_max: 34.0,
    },
    pressure_hpa: {
      value: 1004.2,
      normal_min: 995.0,
      normal_max: 1020.0,
    },
    humidity_pct: {
      value: 84.1,
      normal_min: 40.0,
      normal_max: 80.0,
    },
    anomaly_score_pct: 78,
    risk_level: 'high',
    sensor_health_pct: 76,
    sensor_health_status: 'WARNING',
  },
  'ST-BLR-003': {
    station_id: 'ST-BLR-003',
    timestamp: new Date().toISOString(),
    temperature_c: {
      value: 24.5,
      normal_min: 16.0,
      normal_max: 32.0,
    },
    pressure_hpa: {
      value: 915.0,
      normal_min: 900.0,
      normal_max: 935.0,
    },
    humidity_pct: {
      value: 62.0,
      normal_min: 35.0,
      normal_max: 75.0,
    },
    anomaly_score_pct: 8,
    risk_level: 'low',
    sensor_health_pct: 99,
    sensor_health_status: 'HEALTHY',
  },
  'ST-HYD-004': {
    station_id: 'ST-HYD-004',
    timestamp: new Date().toISOString(),
    temperature_c: {
      value: 36.1,
      normal_min: 20.0,
      normal_max: 35.0,
    },
    pressure_hpa: {
      value: 995.6,
      normal_min: 990.0,
      normal_max: 1025.0,
    },
    humidity_pct: {
      value: 45.2,
      normal_min: 30.0,
      normal_max: 70.0,
    },
    anomaly_score_pct: 91,
    risk_level: 'critical',
    sensor_health_pct: 42,
    sensor_health_status: 'CRITICAL',
  },
  'ST-KOL-005': {
    station_id: 'ST-KOL-005',
    timestamp: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago (offline)
    temperature_c: {
      value: 29.1,
      normal_min: 20.0,
      normal_max: 36.0,
    },
    pressure_hpa: {
      value: 1010.5,
      normal_min: 995.0,
      normal_max: 1022.0,
    },
    humidity_pct: {
      value: 78.4,
      normal_min: 40.0,
      normal_max: 85.0,
    },
    anomaly_score_pct: 0,
    risk_level: 'low',
    sensor_health_pct: 0,
    sensor_health_status: 'OFFLINE',
  },
};
