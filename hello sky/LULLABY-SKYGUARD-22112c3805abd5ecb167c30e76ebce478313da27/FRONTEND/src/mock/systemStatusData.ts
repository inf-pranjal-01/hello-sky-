import { SystemStatusSummary } from '../types';

/**
 * [MOCK FIXTURES]
 * Isolated system status summary fixture.
 */
export const MOCK_SYSTEM_STATUS: SystemStatusSummary = {
  overall_status: 'WARNING',
  active_stations_count: 4,
  total_stations_count: 5,
  active_anomalies_count: 2,
  avg_sensor_health_pct: 79.0,
  last_updated: '2026-09-08T20:00:00Z',
};
