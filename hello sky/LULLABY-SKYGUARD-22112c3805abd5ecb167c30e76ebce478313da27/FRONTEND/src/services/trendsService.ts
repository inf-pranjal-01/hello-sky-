import { TrendsResponse } from '../types';
import { generateMockTrends } from '../mock/trendData';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { validateTrends } from './validators';

/**
 * Trends Service
 * 
 * Boundary interface for historical time-series telemetry used in trend visualizations.
 * - [MOCK DATA]: Returns isolated mock fixtures when API_CONFIG.mode === 'mock'
 * - [API: GET /api/trends — INTEGRATED]: Dispatches real HTTP call when API_CONFIG.mode === 'real'
 */
export const trendsService = {
  /**
   * Fetch telemetry trends for a station over a specified duration in hours.
   * Request contract: GET /api/trends?station_id={stationId}&hours={hours}
   */
  async getTrends(stationId: string, hours: number = 6, options?: RequestOptions): Promise<TrendsResponse> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          const trends = generateMockTrends(stationId, hours);
          resolve(trends);
        }, 75);
      });
    }

    const rawData = await apiClient.get<unknown>(
      API_CONFIG.endpoints.trends,
      { station_id: stationId, hours },
      options
    );
    return validateTrends(rawData, hours);
  },
};
