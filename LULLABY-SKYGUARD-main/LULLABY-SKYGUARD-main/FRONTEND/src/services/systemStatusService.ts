import { SystemStreamStatus, SystemStatusSummary } from '../types';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { ApiError } from './apiError';
import { validateNetworkStatus } from './validators';
import { stationService } from './stationService';
import { summarizeNetworkStatus } from '../utils/networkStatus';
import { MOCK_ANOMALIES } from '../mock/anomalyData';

export const systemStatusService = {
  async get(options?: RequestOptions): Promise<SystemStreamStatus> {
    if (isMockMode()) {
      return { mode: 'live', replay_step_seconds: null, live_poll_interval_seconds: 30 * 60 };
    }
    const data = await apiClient.get<unknown>(API_CONFIG.endpoints.systemStatus, undefined, options);
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      throw ApiError.validationError('System status response must be an object.');
    }
    const status = data as Record<string, unknown>;
    if ((status.mode !== 'live' && status.mode !== 'replay') || typeof status.live_poll_interval_seconds !== 'number') {
      throw ApiError.validationError('System status response has an invalid mode or cadence.');
    }
    return {
      mode: status.mode,
      replay_step_seconds: typeof status.replay_step_seconds === 'number' ? status.replay_step_seconds : null,
      live_poll_interval_seconds: status.live_poll_interval_seconds,
      is_pre_warming: typeof status.is_pre_warming === 'boolean' ? status.is_pre_warming : false,
    };
  },

  async switchToLive(options?: RequestOptions): Promise<SystemStreamStatus> {
    if (isMockMode()) {
      return { mode: 'live', replay_step_seconds: null, live_poll_interval_seconds: 30 * 60 };
    }
    const data = await apiClient.post<unknown>(API_CONFIG.endpoints.systemMode, { mode: 'live' }, options);
    if (!data || typeof data !== 'object' || (data as Record<string, unknown>).mode !== 'live') {
      throw ApiError.validationError('Live-mode switch response is invalid.');
    }
    return { mode: 'live', replay_step_seconds: null, live_poll_interval_seconds: 30 * 60 };
  },

  async refreshLive(options?: RequestOptions): Promise<void> {
    if (isMockMode()) return;
    await apiClient.post<unknown>(API_CONFIG.endpoints.refreshLive, undefined, options);
  },

  async getNetworkStatus(options?: RequestOptions): Promise<SystemStatusSummary> {
    if (isMockMode()) {
      const stations = await stationService.getAllStations(options);
      return summarizeNetworkStatus(
        stations,
        MOCK_ANOMALIES.filter((item) => item.status === 'detected').length
      );
    }
    const data = await apiClient.get<unknown>(API_CONFIG.endpoints.networkStatus, undefined, options);
    return validateNetworkStatus(data);
  },

  async clearHistory(target: 'all' | 'replay' = 'all', options?: RequestOptions): Promise<{ success: boolean; message: string }> {
    if (isMockMode()) {
      return { success: true, message: 'Mock data purged.' };
    }
    const data = await apiClient.post<{ success: boolean; message: string }>(
      API_CONFIG.endpoints.clearHistory,
      { target },
      options
    );
    return data;
  },
};
