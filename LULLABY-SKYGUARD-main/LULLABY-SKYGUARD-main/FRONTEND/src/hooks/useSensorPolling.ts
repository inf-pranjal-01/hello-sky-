import { useState, useEffect, useCallback, useRef } from 'react';
import { SystemStatusSummary } from '../types';
import { anomalyService } from '../services/anomalyService';
import { requestTelemetryRefresh } from '../utils/refreshEvents';

export interface UseSensorPollingResult {
  readings: never[];
  systemStatus: SystemStatusSummary | null;
  loading: boolean;
  error: string | null;
  isPolling: boolean;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * Header network-health poll. Station telemetry lives in useDashboardData;
 * this hook only keeps the real aggregate status badge honest.
 */
export function useSensorPolling(autoPoll: boolean = false): UseSensorPollingResult {
  const [systemStatus, setSystemStatus] = useState<SystemStatusSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<number | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const statusData = await anomalyService.getSystemStatus();
      setSystemStatus(statusData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch network status.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    if (autoPoll) {
      timerRef.current = window.setInterval(() => {
        fetchData();
      }, 5000);
    }
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
      }
    };
  }, [fetchData, autoPoll]);

  const refresh = useCallback(async () => {
    await fetchData();
    requestTelemetryRefresh();
  }, [fetchData]);

  return {
    readings: [],
    systemStatus,
    loading,
    error,
    isPolling: autoPoll,
    lastUpdated,
    refresh,
  };
}
