import { Station } from '../types';
import { MOCK_STATIONS } from '../mock/stationData';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { validateStations } from './validators';

/**
 * Station Service
 * 
 * Boundary interface for station metadata operations.
 * - [MOCK DATA]: Returns isolated mock fixtures when API_CONFIG.mode === 'mock'
 * - [API: GET /api/stations — INTEGRATED]: Dispatches real HTTP call when API_CONFIG.mode === 'real'
 */
export const stationService = {
  /**
   * Fetch all Automatic Weather Stations in the network.
   * [API: GET /api/stations]
   */
  async getAllStations(options?: RequestOptions): Promise<Station[]> {
    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve([...MOCK_STATIONS]);
        }, 50);
      });
    }

    const rawData = await apiClient.get<unknown>(API_CONFIG.endpoints.stations, undefined, options);
    return validateStations(rawData);
  },

  /**
   * Fetch a single station by its unique identifier.
   */
  async getStationById(stationId: string, options?: RequestOptions): Promise<Station | null> {
    const all = await this.getAllStations(options);
    return all.find((s) => s.station_id === stationId) || null;
  },
};
