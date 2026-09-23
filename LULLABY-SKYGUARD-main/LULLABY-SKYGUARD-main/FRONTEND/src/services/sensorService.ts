import { SensorReading } from '../types';
import { MOCK_SENSOR_READINGS } from '../mock/sensorData';

/**
 * Sensor Service
 * 
 * Boundary interface between UI/hooks and data sources.
 * Returns mock fixtures in mock mode: [MOCK FIXTURES]
 * 
 * In real mode, use currentReadingService.ts for the live GET /api/current-reading endpoint.
 */
export const sensorService = {
  async getLatestReadings(): Promise<SensorReading[]> {
    // Simulate minimal network latency (50ms) for async contract adherence
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve([...MOCK_SENSOR_READINGS]);
      }, 50);
    });
  },

  async getReadingByStationId(stationId: string): Promise<SensorReading | null> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const reading = MOCK_SENSOR_READINGS.find((r) => r.station_id === stationId) || null;
        resolve(reading);
      }, 50);
    });
  },
};
