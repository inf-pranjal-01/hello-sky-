import {
  AnomalyRecord,
  SystemStatusSummary,
  LatestAnomaly,
  RecentAnomalyItem,
  AnomalyExplanation,
} from '../types';
import {
  MOCK_ANOMALIES,
  MOCK_LATEST_ANOMALIES,
  MOCK_RECENT_ANOMALIES,
  MOCK_ANOMALY_EXPLANATIONS,
} from '../mock/anomalyData';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { ApiError } from './apiError';
import {
  validateLatestAnomaly,
  validateRecentAnomalies,
  validateAnomalyExplanation,
} from './validators';
import { systemStatusService } from './systemStatusService';

/**
 * Anomaly Service & System Status
 * 
 * Boundary interface for anomaly records, system health summaries, and model explainability.
 * - [MOCK DATA]: Returns isolated mock fixtures when API_CONFIG.mode === 'mock'
 * - [API: GET /api/anomalies/latest — INTEGRATED]
 * - [API: GET /api/anomalies/recent — INTEGRATED]
 * - [API: GET /api/explain/{anomaly_id} — INTEGRATED]
 */
export const anomalyService = {
  /**
   * Fetch active anomalies list across the network.
   * [FRONTEND HELPER — MOCK DATA]
   */
  async getActiveAnomalies(): Promise<AnomalyRecord[]> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve([...MOCK_ANOMALIES]);
      }, 50);
    });
  },

  /**
   * Fetch overall system health status summary.
   * [FRONTEND HELPER — MOCK DATA]
   */
  async getSystemStatus(options?: RequestOptions): Promise<SystemStatusSummary> {
    return systemStatusService.getNetworkStatus(options);
  },

  /**
   * Fetch latest anomaly for a specific Automatic Weather Station (AWS).
   * Request contract: GET /api/anomalies/latest?station_id={stationId}
   */
  async getLatestAnomaly(stationId: string, options?: RequestOptions): Promise<LatestAnomaly | null> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const anomaly = MOCK_LATEST_ANOMALIES[stationId] ?? null;
          resolve(anomaly ? { ...anomaly } : null);
        }, 60);
      });
    }

    try {
      const rawData = await apiClient.get<unknown>(
        API_CONFIG.endpoints.latestAnomaly,
        { station_id: stationId },
        options
      );
      return validateLatestAnomaly(rawData, stationId);
    } catch (err: unknown) {
      // HTTP 404 means no anomaly exists for this station — that is a valid state
      if (err instanceof ApiError && err.status === 404) {
        return null;
      }
      // Network errors, timeouts, 500s, and validation errors must remain thrown
      throw err;
    }
  },

  /**
   * Fetch recent anomaly event history. When stationId is omitted or 'all', fetches network-wide.
   * Request contract: GET /api/anomalies/recent?station_id={stationId}&limit={limit}
   */
  async getRecentAnomalies(
    stationId?: string,
    limit: number = 50,
    options?: RequestOptions
  ): Promise<RecentAnomalyItem[]> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          if (stationId && stationId !== 'all') {
            const list = MOCK_RECENT_ANOMALIES[stationId] || [];
            resolve(list.slice(0, limit));
          } else {
            const all = Object.values(MOCK_RECENT_ANOMALIES).flat();
            resolve(all.slice(0, limit));
          }
        }, 60);
      });
    }

    const params: Record<string, string | number> = { limit };
    if (stationId && stationId !== 'all') {
      params.station_id = stationId;
    }

    const rawData = await apiClient.get<unknown>(
      API_CONFIG.endpoints.recentAnomalies,
      params,
      options
    );
    return validateRecentAnomalies(rawData, stationId && stationId !== 'all' ? stationId : undefined);
  },

  /**
   * Fetch anomaly feature contribution / explainability breakdown.
   * Request contract: GET /api/explain/{anomaly_id}
   */
  async getAnomalyExplanation(
    anomalyId: string,
    options?: RequestOptions
  ): Promise<AnomalyExplanation | null> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const explanation = MOCK_ANOMALY_EXPLANATIONS[anomalyId];
          if (explanation) {
            resolve({ ...explanation });
          } else {
            // Fallback generated explanation for injected/unregistered anomalies in demo mode
            resolve({
              anomaly_id: anomalyId,
              features: [
                { name: 'Primary Metric Variance', impact: 0.52 },
                { name: 'Diurnal Residual', impact: 0.28 },
                { name: 'Neighbor AWS Variance', impact: -0.12 },
                { name: 'Historical Mean Offset', impact: -0.05 },
              ],
            });
          }
        }, 80);
      });
    }

    const path = API_CONFIG.endpoints.explainAnomaly(anomalyId);
    const rawData = await apiClient.get<unknown>(path, undefined, options);
    return validateAnomalyExplanation(rawData, anomalyId);
  },
};
