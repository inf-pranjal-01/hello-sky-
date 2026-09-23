import { InjectAnomalyRequest, InjectAnomalyResponse, AnomalyType } from '../types';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { validateInjectAnomalyResponse } from './validators';
import { ApiError } from './apiError';

export const SUPPORTED_ANOMALY_TYPES: readonly AnomalyType[] = [
  'spike',
  'frozen_value',
  'drift',
  'dropout',
  'sensor_fail_low',
  'multivariate_inconsistency',
] as const;

/**
 * Anomaly Injection Service
 * 
 * Boundary interface for demonstration anomaly replay injection into the telemetry pipeline.
 * - [MOCK DATA]: Returns simulated confirmation when API_CONFIG.mode === 'mock'
 * - [API: POST /api/inject-anomaly — INTEGRATED]: Dispatches real HTTP call when API_CONFIG.mode === 'real'
 * 
 * Contract:
 *  POST /api/inject-anomaly
 *  Body: { station_id: string, type: "spike" | "frozen_value" | "drift" | "dropout" | "sensor_fail_low" | "multivariate_inconsistency" }
 *  Response: { success: boolean, anomaly_id: string, message: string }
 */
export const anomalyInjectionService = {
  async injectAnomaly(
    req: InjectAnomalyRequest,
    options?: RequestOptions
  ): Promise<InjectAnomalyResponse> {
    if (!req || typeof req.station_id !== 'string' || !req.station_id.trim()) {
      throw ApiError.validationError('Invalid anomaly injection request: missing or empty station_id.');
    }

    if (!SUPPORTED_ANOMALY_TYPES.includes(req.type)) {
      throw ApiError.validationError(
        `Invalid anomaly injection request: unsupported type '${String(req.type)}'. Supported types: ${SUPPORTED_ANOMALY_TYPES.join(', ')}`
      );
    }

    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve({
            success: true,
            anomaly_id: 'ANOM-MOCK-001',
            message: `Anomaly replay started for station ${req.station_id} with fault pattern ${req.type}.`,
          });
        }, 100);
      });
    }

    const rawData = await apiClient.post<unknown>(
      API_CONFIG.endpoints.injectAnomaly,
      req,
      options
    );
    return validateInjectAnomalyResponse(rawData);
  },
};
