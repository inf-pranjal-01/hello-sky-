import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Station,
  ReportPeriod,
  StationOperationalReport,
  TrendsResponse,
  CurrentSensorReading,
  RecentAnomalyItem,
  LatestAnomaly,
  SensorHealth,
} from '../types';
import { useStation } from '../context/StationContext';
import { currentReadingService } from '../services/currentReadingService';
import { trendsService } from '../services/trendsService';
import { anomalyService } from '../services/anomalyService';
import { sensorHealthService } from '../services/sensorHealthService';
import { getMockNetworkReadings } from '../mock/stationNetworkData';
import { buildSpatialComparisonSummary } from '../utils/spatialCalculations';
import { buildStationOperationalReport } from '../utils/reportBuilder';

export interface UseReportDataResult {
  selectedStation: Station | null;
  periodHours: ReportPeriod;
  setPeriodHours: (hours: ReportPeriod) => void;
  selectedMetricTab: 'temperature_c' | 'pressure_hpa' | 'humidity_pct';
  setSelectedMetricTab: (metric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct') => void;

  report: StationOperationalReport | null;
  isLoading: boolean;
  isGenerating: boolean;
  error: string | null;

  generateReport: () => void;
  printReport: () => void;
  refresh: () => Promise<void>;
}

/**
 * useReportData
 * 
 * Custom hook orchestrating operational report aggregation across all domain services.
 * [FRONTEND ONLY — REPORT STATE & DATA ORCHESTRATION]
 */
export function useReportData(): UseReportDataResult {
  const { selectedStation, stations } = useStation();

  const [periodHours, setPeriodHours] = useState<ReportPeriod>(24);
  const [selectedMetricTab, setSelectedMetricTab] = useState<
    'temperature_c' | 'pressure_hpa' | 'humidity_pct'
  >('temperature_c');

  const [rawTrends, setRawTrends] = useState<TrendsResponse | null>(null);
  const [rawReading, setRawReading] = useState<CurrentSensorReading | null>(null);
  const [rawAnomalies, setRawAnomalies] = useState<RecentAnomalyItem[]>([]);
  const [rawLatestAnomaly, setRawLatestAnomaly] = useState<LatestAnomaly | null>(null);
  const [rawSensorHealth, setRawSensorHealth] = useState<SensorHealth | null>(null);

  const [generatedTimestamp, setGeneratedTimestamp] = useState<string>(() =>
    new Date().toISOString()
  );

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const stationId = selectedStation?.station_id ?? null;

  const loadReportRawData = useCallback(async () => {
    if (!stationId) {
      setRawTrends(null);
      setRawReading(null);
      setRawAnomalies([]);
      setRawLatestAnomaly(null);
      setRawSensorHealth(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const [reading, trends, recentAnoms, latestAnom, health] = await Promise.all([
        currentReadingService.getCurrentReading(stationId).catch(() => null),
        trendsService.getTrends(stationId, periodHours).catch(() => null),
        anomalyService.getRecentAnomalies(stationId, 10).catch(() => []),
        anomalyService.getLatestAnomaly(stationId).catch(() => null),
        sensorHealthService.getSensorHealth(stationId).catch(() => null),
      ]);

      setRawReading(reading);
      setRawTrends(trends);
      setRawAnomalies(recentAnoms);
      setRawLatestAnomaly(latestAnom);
      setRawSensorHealth(health);
      setGeneratedTimestamp(new Date().toISOString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to compile operational report data.');
    } finally {
      setIsLoading(false);
    }
  }, [stationId, periodHours]);

  useEffect(() => {
    loadReportRawData();
  }, [loadReportRawData]);

  // Spatial context derivation
  const spatialSummary = useMemo(() => {
    if (!selectedStation) return null;

    const networkReadings = getMockNetworkReadings(selectedStation.station_id, 'localized_deviation', stations);
    const neighborReadings = stations
      .filter((s) => s.station_id !== selectedStation.station_id)
      .map((s) => networkReadings[s.station_id])
      .filter((r): r is NonNullable<typeof r> => r != null);

    const currentTelemetry = rawReading
      ? {
          temperature_c: rawReading.temperature_c.value,
          pressure_hpa: rawReading.pressure_hpa.value,
          humidity_pct: rawReading.humidity_pct.value,
        }
      : networkReadings[selectedStation.station_id] ?? {
          temperature_c: 31.2,
          pressure_hpa: 1012.4,
          humidity_pct: 68.0,
        };

    return buildSpatialComparisonSummary(
      selectedStation.station_id,
      currentTelemetry,
      neighborReadings,
      'localized_deviation'
    );
  }, [selectedStation, stations, rawReading]);

  // Assembled operational report model
  const report = useMemo<StationOperationalReport | null>(() => {
    if (!selectedStation) return null;

    return buildStationOperationalReport({
      station: selectedStation,
      periodHours,
      currentReading: rawReading,
      trends: rawTrends,
      anomalies: rawAnomalies,
      latestAnomaly: rawLatestAnomaly,
      sensorHealth: rawSensorHealth,
      spatialSummary,
      generatedAt: generatedTimestamp,
    });
  }, [
    selectedStation,
    periodHours,
    rawReading,
    rawTrends,
    rawAnomalies,
    rawLatestAnomaly,
    rawSensorHealth,
    spatialSummary,
    generatedTimestamp,
  ]);

  // Generate Report action
  const generateReport = useCallback(() => {
    setIsGenerating(true);
    setGeneratedTimestamp(new Date().toISOString());
    setTimeout(() => {
      setIsGenerating(false);
    }, 250);
  }, []);

  // Print Report action
  const printReport = useCallback(() => {
    window.print();
  }, []);

  return {
    selectedStation,
    periodHours,
    setPeriodHours,
    selectedMetricTab,
    setSelectedMetricTab,
    report,
    isLoading,
    isGenerating,
    error,
    generateReport,
    printReport,
    refresh: loadReportRawData,
  };
}
