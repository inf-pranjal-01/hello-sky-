import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Station,
  StationOperationalStatus,
  NeighborStationItem,
  SpatialComparisonSummary,
  SpatialDemoScenario,
  CurrentSensorReading,
  LatestAnomaly,
} from '../types';
import { useStation } from '../context/StationContext';
import { currentReadingService } from '../services/currentReadingService';
import { anomalyService } from '../services/anomalyService';
import { calculateDistanceKm } from '../utils/geospatial';
import { getMockNetworkReadings } from '../mock/stationNetworkData';
import { buildSpatialComparisonSummary } from '../utils/spatialCalculations';

export type StationFilterStatus = 'ALL' | StationOperationalStatus;

export interface UseStationNetworkDataResult {
  // Station Context
  selectedStation: Station | null;
  stations: Station[];
  selectStation: (station: Station) => void;

  // Scenario state [SIH DEMO]
  scenario: SpatialDemoScenario;
  setScenario: (scenario: SpatialDemoScenario) => void;

  // Filtering
  statusFilter: StationFilterStatus;
  setStatusFilter: (filter: StationFilterStatus) => void;

  // Telemetry & Anomaly Context
  currentReading: CurrentSensorReading | null;
  latestAnomaly: LatestAnomaly | null;

  // Derived Spatial Data
  neighbors: NeighborStationItem[];
  filteredNeighbors: NeighborStationItem[];
  spatialSummary: SpatialComparisonSummary | null;

  // Status
  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

/**
 * useStationNetworkData
 * 
 * Custom hook orchestrating station network telemetry, spatial distance calculations,
 * and neighbor comparisons.
 * [FRONTEND ONLY — DERIVED FROM STATION COORDINATES & MOCK TELEMETRY]
 */
export function useStationNetworkData(): UseStationNetworkDataResult {
  const { stations, selectedStation, setSelectedStation } = useStation();

  const [scenario, setScenario] = useState<SpatialDemoScenario>('localized_deviation');
  const [statusFilter, setStatusFilter] = useState<StationFilterStatus>('ALL');

  const [currentReading, setCurrentReading] = useState<CurrentSensorReading | null>(null);
  const [latestAnomaly, setLatestAnomaly] = useState<LatestAnomaly | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const selectedStationId = selectedStation?.station_id ?? null;

  const loadData = useCallback(async () => {
    if (!selectedStationId) {
      setCurrentReading(null);
      setLatestAnomaly(null);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const [reading, anomaly] = await Promise.all([
        currentReadingService.getCurrentReading(selectedStationId).catch(() => null),
        anomalyService.getLatestAnomaly(selectedStationId).catch(() => null),
      ]);

      setCurrentReading(reading);
      setLatestAnomaly(anomaly);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load station telemetry.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedStationId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Mock network readings for the current scenario
  const networkReadings = useMemo(() => {
    return getMockNetworkReadings(selectedStationId, scenario, stations);
  }, [selectedStationId, scenario, stations]);

  // Derive neighbor stations with Haversine distance and mock telemetry
  const neighbors = useMemo<NeighborStationItem[]>(() => {
    if (!selectedStation || stations.length === 0) return [];

    return stations
      .filter((s) => s.station_id !== selectedStation.station_id)
      .map((s) => {
        const distance = calculateDistanceKm(
          selectedStation.lat,
          selectedStation.lon,
          s.lat,
          s.lon
        );

        return {
          station: s,
          distance_km: distance ?? 9999,
          reading: networkReadings[s.station_id],
        };
      })
      .sort((a, b) => a.distance_km - b.distance_km);
  }, [stations, selectedStation, networkReadings]);

  // Filter neighbors by status
  const filteredNeighbors = useMemo(() => {
    if (statusFilter === 'ALL') return neighbors;
    return neighbors.filter((n) => n.station.status === statusFilter);
  }, [neighbors, statusFilter]);

  // Derive spatial comparison summary
  const spatialSummary = useMemo<SpatialComparisonSummary | null>(() => {
    if (!selectedStation) return null;

    // Use current reading if available, or fall back to network mock reading
    const selectedTelemetry = currentReading
      ? {
          temperature_c: currentReading.temperature_c.value,
          pressure_hpa: currentReading.pressure_hpa.value,
          humidity_pct: currentReading.humidity_pct.value,
        }
      : networkReadings[selectedStation.station_id] ?? {
          temperature_c: 31.2,
          pressure_hpa: 1012.4,
          humidity_pct: 68.0,
        };

    const neighborReadingsList = neighbors
      .map((n) => n.reading)
      .filter((r): r is NonNullable<typeof r> => r != null);

    return buildSpatialComparisonSummary(
      selectedStation.station_id,
      selectedTelemetry,
      neighborReadingsList,
      scenario
    );
  }, [selectedStation, currentReading, networkReadings, neighbors, scenario]);

  return {
    selectedStation,
    stations,
    selectStation: setSelectedStation,
    scenario,
    setScenario,
    statusFilter,
    setStatusFilter,
    currentReading,
    latestAnomaly,
    neighbors,
    filteredNeighbors,
    spatialSummary,
    isLoading,
    error,
    lastUpdated,
    refresh: loadData,
  };
}
