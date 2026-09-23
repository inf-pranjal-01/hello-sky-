import { useState, useEffect, useCallback, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Station,
  LatestAnomaly,
  RecentAnomalyItem,
  MaintenanceTicketResponse,
} from '../types';
import { useStation } from '../context/StationContext';
import { anomalyService } from '../services/anomalyService';
import { maintenanceService } from '../services/maintenanceService';
import { formatUserErrorMessage } from '../services/apiError';

export interface UseMaintenanceDataResult {
  selectedStation: Station | null;
  anomalies: (LatestAnomaly | RecentAnomalyItem)[];
  selectedAnomalyId: string | null;
  selectedAnomaly: LatestAnomaly | RecentAnomalyItem | null;
  isLoadingAnomalies: boolean;
  isConfirmModalOpen: boolean;
  isSubmitting: boolean;
  createdTicket: MaintenanceTicketResponse | null;
  submissionError: string | null;

  selectAnomaly: (anomalyId: string) => void;
  openConfirmModal: () => void;
  closeConfirmModal: () => void;
  confirmCreateTicket: () => Promise<void>;
  resetCreatedTicket: () => void;
  refreshAnomalies: () => Promise<void>;
}

/**
 * useMaintenanceData
 * 
 * Custom hook orchestrating the maintenance ticket dispatch workflow.
 * Manages anomaly selection, confirmation modal state, duplicate submission locks,
 * and POST /api/maintenance-ticket lifecycle.
 */
export function useMaintenanceData(): UseMaintenanceDataResult {
  const { selectedStation } = useStation();
  const [searchParams] = useSearchParams();
  const initialAnomalyId = searchParams.get('anomaly_id');

  const [rawAnomalies, setRawAnomalies] = useState<(LatestAnomaly | RecentAnomalyItem)[]>([]);
  const [selectedAnomalyId, setSelectedAnomalyId] = useState<string | null>(initialAnomalyId);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState<boolean>(true);

  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [createdTicket, setCreatedTicket] = useState<MaintenanceTicketResponse | null>(null);
  const [submissionError, setSubmissionError] = useState<string | null>(null);

  const stationId = selectedStation?.station_id ?? null;

  // Load anomalies for the selected station
  const loadAnomalies = useCallback(async () => {
    if (!stationId) {
      setRawAnomalies([]);
      setSelectedAnomalyId(null);
      setIsLoadingAnomalies(false);
      return;
    }

    setIsLoadingAnomalies(true);
    try {
      const [recent, latest] = await Promise.all([
        anomalyService.getRecentAnomalies(stationId, 10).catch(() => []),
        anomalyService.getLatestAnomaly(stationId).catch(() => null),
      ]);

      const combined: (LatestAnomaly | RecentAnomalyItem)[] = [...recent];
      if (latest && !combined.some((a) => a.anomaly_id === latest.anomaly_id)) {
        combined.unshift(latest);
      }

      // Sort newest first
      const sorted = combined.sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );

      setRawAnomalies(sorted);

      // Default select the requested or latest anomaly
      if (initialAnomalyId && sorted.some((a) => a.anomaly_id === initialAnomalyId)) {
        setSelectedAnomalyId(initialAnomalyId);
      } else if (sorted.length > 0 && !selectedAnomalyId) {
        setSelectedAnomalyId(sorted[0].anomaly_id);
      }
    } catch {
      setRawAnomalies([]);
    } finally {
      setIsLoadingAnomalies(false);
    }
  }, [stationId, initialAnomalyId]);

  useEffect(() => {
    loadAnomalies();
  }, [loadAnomalies]);

  // Derived selected anomaly object
  const selectedAnomaly = useMemo(() => {
    if (!selectedAnomalyId || rawAnomalies.length === 0) return null;
    return rawAnomalies.find((a) => a.anomaly_id === selectedAnomalyId) ?? null;
  }, [selectedAnomalyId, rawAnomalies]);

  // Select anomaly action
  const selectAnomaly = useCallback((anomalyId: string) => {
    setSelectedAnomalyId(anomalyId);
    setSubmissionError(null);
  }, []);

  // Modal actions
  const openConfirmModal = useCallback(() => {
    if (!selectedAnomalyId) return;
    setSubmissionError(null);
    setIsConfirmModalOpen(true);
  }, [selectedAnomalyId]);

  const closeConfirmModal = useCallback(() => {
    if (isSubmitting) return; // Prevent closing while submitting
    setIsConfirmModalOpen(false);
  }, [isSubmitting]);

  // Submit action: calls maintenanceService with duplicate submission lock
  const confirmCreateTicket = useCallback(async () => {
    if (!selectedAnomalyId || isSubmitting) return;

    setIsSubmitting(true);
    setSubmissionError(null);

    try {
      const response = await maintenanceService.createMaintenanceTicket(selectedAnomalyId);
      setCreatedTicket(response);
      setIsConfirmModalOpen(false);
    } catch (err) {
      setSubmissionError(
        formatUserErrorMessage(err, 'Failed to create maintenance ticket. Please verify connection and retry.')
      );
      setIsConfirmModalOpen(false);
    } finally {
      setIsSubmitting(false);
    }
  }, [selectedAnomalyId, isSubmitting]);

  // Reset ticket creation state to allow creating another ticket in the same session
  const resetCreatedTicket = useCallback(() => {
    setCreatedTicket(null);
    setSubmissionError(null);
  }, []);

  return {
    selectedStation,
    anomalies: rawAnomalies,
    selectedAnomalyId,
    selectedAnomaly,
    isLoadingAnomalies,
    isConfirmModalOpen,
    isSubmitting,
    createdTicket,
    submissionError,
    selectAnomaly,
    openConfirmModal,
    closeConfirmModal,
    confirmCreateTicket,
    resetCreatedTicket,
    refreshAnomalies: loadAnomalies,
  };
}
