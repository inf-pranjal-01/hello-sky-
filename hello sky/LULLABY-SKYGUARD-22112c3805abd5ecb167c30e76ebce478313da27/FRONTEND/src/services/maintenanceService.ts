import { MaintenanceTicketResponse } from '../types';
import { MOCK_RECENT_ANOMALIES, MOCK_LATEST_ANOMALIES } from '../mock/anomalyData';
import { API_CONFIG, isMockMode } from '../config/api.config';
import { apiClient, RequestOptions } from './apiClient';
import { validateMaintenanceTicketResponse } from './validators';
import { ApiError } from './apiError';

/**
 * Maintenance Operations Service
 * 
 * Service abstraction for the maintenance workflow.
 * - [MOCK DATA]: Returns simulated ticket response when API_CONFIG.mode === 'mock'
 * - [API: POST /api/maintenance-ticket — INTEGRATED]: Dispatches real HTTP call when API_CONFIG.mode === 'real'
 * 
 * Target Backend Endpoint:
 * POST /api/maintenance-ticket
 * Request: { "anomaly_id": "string" }
 * Response: { "ticket_id": "string", "station_id": "string", "issue": "string", "priority": "string", "created_at": "string" }
 */

let ticketSequence = 1;

export const maintenanceService = {
  /**
   * Dispatches a maintenance ticket for a detected sensor anomaly.
   * [API: POST /api/maintenance-ticket]
   */
  async createMaintenanceTicket(
    anomalyId: string,
    options?: RequestOptions
  ): Promise<MaintenanceTicketResponse> {
    if (!anomalyId || typeof anomalyId !== 'string' || anomalyId.trim().length === 0) {
      throw new ApiError('Invalid anomaly identifier. A valid anomaly must be selected.', {
        code: 'VALIDATION_ERROR',
        isValidationError: true,
      });
    }

    if (isMockMode()) {
      return new Promise((resolve) => {
        setTimeout(() => {
          // Find anomaly details across mock datasets
          let matchingAnomaly = Object.values(MOCK_LATEST_ANOMALIES).find(
            (a) => a != null && a.anomaly_id === anomalyId
          );

          if (!matchingAnomaly) {
            Object.values(MOCK_RECENT_ANOMALIES).forEach((list) => {
              const found = list.find((a) => a.anomaly_id === anomalyId);
              if (found) matchingAnomaly = found;
            });
          }

          const now = new Date();
          const datePart = now.toISOString().slice(0, 10).replace(/-/g, '');
          const seqNum = String(ticketSequence++).padStart(3, '0');
          const ticketId = `MT-${datePart}-${seqNum}`;

          const stationId = matchingAnomaly ? matchingAnomaly.station_id : 'ST-NDL-001';
          const priority = matchingAnomaly ? matchingAnomaly.severity.toUpperCase() : 'HIGH';
          const issue = matchingAnomaly
            ? `${matchingAnomaly.type.replace(/_/g, ' ')} detected: ${matchingAnomaly.root_cause}`
            : 'Suspicious sensor telemetry deviation requiring physical inspection';

          const ticketResponse: MaintenanceTicketResponse = {
            ticket_id: ticketId,
            station_id: stationId,
            issue: issue.charAt(0).toUpperCase() + issue.slice(1),
            priority,
            created_at: now.toISOString(),
          };

          resolve(ticketResponse);
        }, 350);
      });
    }

    const rawData = await apiClient.post<unknown>(
      API_CONFIG.endpoints.maintenanceTicket,
      { anomaly_id: anomalyId },
      options
    );
    return validateMaintenanceTicketResponse(rawData);
  },
};
