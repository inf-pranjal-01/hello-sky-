import { SensorHealth, HealthHistoryPoint, RepairSensorResponse } from '../types';
import { MOCK_SENSOR_HEALTH, generateMockHealthHistory } from '../mock/sensorHealthData';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { validateSensorHealth, validateRepairSensorResponse } from './validators';
import { trendsService } from './trendsService';

/**
 * Sensor Health Service
 * 
 * Boundary interface for AWS hardware reliability and sensing subsystem health.
 * - [MOCK DATA]: Returns isolated mock fixtures when API_CONFIG.mode === 'mock'
 * - [API: GET /api/sensor-health?station_id={station_id} — INTEGRATED]: Dispatches real HTTP call when API_CONFIG.mode === 'real'
 */
export const sensorHealthService = {
  /**
   * Fetch current overall sensor health status for an Automatic Weather Station.
   * Request contract: GET /api/sensor-health?station_id={stationId}
   */
  async getSensorHealth(stationId: string, options?: RequestOptions): Promise<SensorHealth> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const data = MOCK_SENSOR_HEALTH[stationId];
          if (!data) {
            resolve({
              station_id: stationId,
              sensor_health_pct: 95,
              sensor_health_status: 'HEALTHY',
            });
            return;
          }
          resolve({ ...data });
        }, 50);
      });
    }

    const rawData = await apiClient.get<unknown>(
      API_CONFIG.endpoints.sensorHealth,
      { station_id: stationId },
      options
    );
    return validateSensorHealth(rawData, stationId);
  },

  /**
   * Fetch historical sensor health records for trend visualization.
   * [FRONTEND DERIVED — DEMO HISTORY]
   * [NOT IN CURRENT API CONTRACT: Historical health endpoint]
   */
  async getHealthHistory(stationId: string, hours: number = 10): Promise<HealthHistoryPoint[]> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const history = generateMockHealthHistory(stationId, hours);
          resolve(history);
        }, 60);
      });
    }

    const trends = await trendsService.getTrends(stationId, hours);
    const healthFromStatus = (status: string | undefined): number => {
      if (status === 'OFFLINE') return 0;
      if (status === 'WARNING' || status === 'CRITICAL') return 50;
      return 100;
    };
    return trends.points.map((point) => ({
      timestamp: point.timestamp,
      health_pct: point.health_status
        ? healthFromStatus(String(point.health_status))
        : Math.max(0, 100 - (point.anomaly_score_pct ?? 0)),
      status: (point.health_status as HealthHistoryPoint['status']) || 'HEALTHY',
    }));
  },

  /**
   * Request a sensor repair / recovery sweep for an Automatic Weather Station.
   * Request contract: POST /api/repair-sensor  { station_id }
   * [API: POST /api/repair-sensor — INTEGRATED]
   */
  async markRepaired(stationId: string, options?: RequestOptions): Promise<RepairSensorResponse> {
    if (isMockMode()) {
      return {
        success: true,
        station_id: stationId,
        status: 'WARNING',
        recovery_active: true,
        message: 'Physical repair confirmed. Three consecutive clean readings are required before HEALTHY.',
      };
    }
    const rawData = await apiClient.post<unknown>(
      API_CONFIG.endpoints.repairSensor,
      { station_id: stationId, force_recovery: false },
      options
    );
    return validateRepairSensorResponse(rawData, stationId);
  },

  async forceRecover(stationId: string, options?: RequestOptions): Promise<RepairSensorResponse> {
    if (isMockMode()) {
      return {
        success: true,
        station_id: stationId,
        status: 'HEALTHY',
        recovery_active: false,
        message: 'Sensor force-recovered and health counters reset.',
      };
    }
    const rawData = await apiClient.post<unknown>(
      API_CONFIG.endpoints.repairSensor,
      { station_id: stationId, force_recovery: true },
      options
    );
    return validateRepairSensorResponse(rawData, stationId);
  },

  async repairSensor(stationId: string, options?: RequestOptions): Promise<RepairSensorResponse> {
    return this.markRepaired(stationId, options);
  },
};
