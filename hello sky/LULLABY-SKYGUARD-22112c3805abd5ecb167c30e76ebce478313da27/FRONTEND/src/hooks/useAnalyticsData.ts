import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  TrendsResponse,
  CurrentSensorReading,
  RecentAnomalyItem,
  HealthHistoryPoint,
  AnalyticsSummary,
} from '../types';
import { trendsService } from '../services/trendsService';
import { currentReadingService } from '../services/currentReadingService';
import { anomalyService } from '../services/anomalyService';
import { sensorHealthService } from '../services/sensorHealthService';
import {
  calculateMetricStatistics,
  deriveHighestSeverity,
  deriveSeverityDistribution,
  deriveTypeDistribution,
  generateDeterministicInsights,
} from '../utils/analytics';
import { calculateFreshness, FreshnessState } from '../utils/freshness';

export interface UseAnalyticsDataResult {
  hours: number;
  setHours: (hours: number) => void;
  selectedMetric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct';
  setSelectedMetric: (metric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct') => void;

  trends: TrendsResponse | null;
  currentReading: CurrentSensorReading | null;
  anomalies: RecentAnomalyItem[];
  healthHistory: HealthHistoryPoint[];
  analyticsSummary: AnalyticsSummary | null;

  isLoading: boolean;
  error: string | null;
  lastUpdated: Date | null;
  freshness: FreshnessState;
  staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE';

  refresh: () => Promise<void>;
}

/**
 * useAnalyticsData
 * 
 * Custom analytics hook that aggregates telemetry and anomaly feeds to derive
 * operational analytics and insights strictly frontend-side.
 * [FRONTEND ONLY] [DERIVED ANALYTICS HOOK]
 */
export function useAnalyticsData(
  stationId: string | null | undefined,
  initialHours: number = 24
): UseAnalyticsDataResult {
  const [hours, setHours] = useState<number>(initialHours);
  const [selectedMetric, setSelectedMetric] = useState<
    'temperature_c' | 'pressure_hpa' | 'humidity_pct'
  >('temperature_c');

  const [trends, setTrends] = useState<TrendsResponse | null>(null);
  const [currentReading, setCurrentReading] = useState<CurrentSensorReading | null>(null);
  const [anomalies, setAnomalies] = useState<RecentAnomalyItem[]>([]);
  const [healthHistory, setHealthHistory] = useState<HealthHistoryPoint[]>([]);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // Fetch all prerequisite feeds
  const loadData = useCallback(async () => {
    if (!stationId) {
      setTrends(null);
      setCurrentReading(null);
      setAnomalies([]);
      setHealthHistory([]);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const [trendsData, readingData, recentAnoms, latestAnom, healthData] = await Promise.all([
        trendsService.getTrends(stationId, hours),
        currentReadingService.getCurrentReading(stationId).catch(() => null),
        anomalyService.getRecentAnomalies(stationId, 10).catch(() => []),
        anomalyService.getLatestAnomaly(stationId).catch(() => null),
        sensorHealthService.getHealthHistory(stationId, hours).catch(() => []),
      ]);

      setTrends(trendsData);
      setCurrentReading(readingData);

      // Combine recent and latest anomalies
      const combinedAnomalies = [...recentAnoms];
      if (latestAnom && !combinedAnomalies.some((a) => a.anomaly_id === latestAnom.anomaly_id)) {
        combinedAnomalies.unshift(latestAnom);
      }
      setAnomalies(
        combinedAnomalies.sort(
          (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        )
      );

      setHealthHistory(healthData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load analytics telemetry.');
    } finally {
      setIsLoading(false);
    }
  }, [stationId, hours]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Derived Analytics Calculations
  const analyticsSummary = useMemo<AnalyticsSummary | null>(() => {
    if (!trends || !trends.points || trends.points.length === 0) {
      return null;
    }

    const tempValues = trends.points.map((p) => p.temperature_c);
    const pressValues = trends.points.map((p) => p.pressure_hpa);
    const humValues = trends.points.map((p) => p.humidity_pct);

    const tempStats = calculateMetricStatistics(tempValues, 1);
    const pressStats = calculateMetricStatistics(pressValues, 1);
    const humStats = calculateMetricStatistics(humValues, 1);

    const severityDist = deriveSeverityDistribution(anomalies);
    const typeDist = deriveTypeDistribution(anomalies);
    const highestSev = deriveHighestSeverity(anomalies.map((a) => a.severity));

    const insights = generateDeterministicInsights({
      temperature: tempStats,
      pressure: pressStats,
      humidity: humStats,
      totalAnomalies: anomalies.length,
      highestSeverity: highestSev,
      hours,
    });

    return {
      temperature: tempStats,
      pressure: pressStats,
      humidity: humStats,
      totalAnomalies: anomalies.length,
      highestSeverity: highestSev,
      severityDistribution: severityDist,
      typeDistribution: typeDist,
      anomalyTimeline: anomalies,
      insights,
    };
  }, [trends, anomalies, hours]);

  const freshness = calculateFreshness(currentReading?.timestamp, false);
  const isDelayed = freshness.status === 'DATA DELAYED';
  const isStale = freshness.status === 'DATA STALE';
  const staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE' = isStale
    ? 'DATA STALE'
    : isDelayed
    ? 'DATA DELAYED'
    : 'LIVE';

  return {
    hours,
    setHours,
    selectedMetric,
    setSelectedMetric,
    trends,
    currentReading,
    anomalies,
    healthHistory,
    analyticsSummary,
    isLoading,
    error,
    lastUpdated,
    freshness,
    staleStatusText,
    refresh: loadData,
  };
}
