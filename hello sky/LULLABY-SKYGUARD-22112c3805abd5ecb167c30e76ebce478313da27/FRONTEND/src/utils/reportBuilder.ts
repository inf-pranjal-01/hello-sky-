import {
  Station,
  CurrentSensorReading,
  TrendsResponse,
  RecentAnomalyItem,
  LatestAnomaly,
  SensorHealth,
  SpatialComparisonSummary,
  StationOperationalReport,
  ReportPeriod,
  ReportMetadata,
} from '../types';
import {
  calculateMetricStatistics,
  deriveHighestSeverity,
  deriveSeverityDistribution,
  deriveTypeDistribution,
  generateDeterministicInsights,
} from './analytics';

/**
 * Builds conservative, evidence-based operational recommendations.
 * [FRONTEND ONLY — DERIVED RECOMMENDATIONS]
 */
export function generateOperationalRecommendations(params: {
  station: Station;
  totalAnomalies: number;
  highestSeverity: string;
  sensorHealth: SensorHealth | null;
  spatialSummary?: SpatialComparisonSummary | null;
}): string[] {
  const { totalAnomalies, highestSeverity, sensorHealth, spatialSummary } = params;
  const recommendations: string[] = [];

  // 1. Critical/High Anomaly Recommendation
  if (highestSeverity === 'critical') {
    recommendations.push(
      'Priority Alert Review: Inspect the active CRITICAL anomaly in the Incident Investigation workspace to verify transducer signal integrity.'
    );
  } else if (highestSeverity === 'high') {
    recommendations.push(
      'Incident Triage: Review recent HIGH severity anomaly records in the Alerts module to assess duration and affected meteorological variables.'
    );
  } else if (totalAnomalies > 0) {
    recommendations.push(
      'Telemetry Monitoring: Low/moderate anomaly flags detected; maintain automated signal tracking to observe trend progression.'
    );
  }

  // 2. Spatial Consistency Recommendation
  if (spatialSummary?.consistencyStatus === 'DEVIATION_DETECTED') {
    recommendations.push(
      'Spatial Cross-Validation: Cross-reference telemetry with nearby observatory stations in the Station Network workspace to distinguish localized sensor faults from regional weather fronts.'
    );
  }

  // 3. Sensor Health & Hardware Reliability Recommendation
  if (sensorHealth && sensorHealth.sensor_health_pct < 50) {
    recommendations.push(
      'Hardware Maintenance Action: Transducer health score is critically degraded. Schedule an on-site physical inspection and recalibration.'
    );
  } else if (sensorHealth && sensorHealth.sensor_health_pct < 80) {
    recommendations.push(
      'Condition Monitoring: Subsystem health indicates moderate wear. Continue scheduled monitoring and check for potential signal drift.'
    );
  }

  // 4. Nominal Recommendation
  if (recommendations.length === 0) {
    recommendations.push(
      'Nominal Operation: Station telemetry and hardware reliability indices remain within optimal thresholds. Continue autonomous telemetry acquisition.'
    );
  }

  return recommendations;
}

/**
 * Pure transformation function that aggregates station telemetry, anomalies,
 * sensor health, and spatial context into a normalized operational report model.
 * 
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export function buildStationOperationalReport(params: {
  station: Station;
  periodHours: ReportPeriod;
  currentReading: CurrentSensorReading | null;
  trends: TrendsResponse | null;
  anomalies: RecentAnomalyItem[];
  latestAnomaly: LatestAnomaly | null;
  sensorHealth: SensorHealth | null;
  spatialSummary?: SpatialComparisonSummary | null;
  generatedAt?: string;
}): StationOperationalReport {
  const {
    station,
    periodHours,
    currentReading,
    trends,
    anomalies,
    latestAnomaly,
    sensorHealth,
    spatialSummary,
    generatedAt = new Date().toISOString(),
  } = params;

  const points = trends?.points || [];

  const tempValues = points.map((p) => p.temperature_c);
  const pressValues = points.map((p) => p.pressure_hpa);
  const humValues = points.map((p) => p.humidity_pct);

  const tempStats = calculateMetricStatistics(tempValues, 1);
  const pressStats = calculateMetricStatistics(pressValues, 1);
  const humStats = calculateMetricStatistics(humValues, 1);

  // Combine and deduplicate anomalies
  const combinedAnomalies = [...anomalies];
  if (latestAnomaly && !combinedAnomalies.some((a) => a.anomaly_id === latestAnomaly.anomaly_id)) {
    combinedAnomalies.unshift(latestAnomaly);
  }

  const sevDist = deriveSeverityDistribution(combinedAnomalies);
  const typeDist = deriveTypeDistribution(combinedAnomalies);
  const highestSev = deriveHighestSeverity(combinedAnomalies.map((a) => a.severity));

  const insights = generateDeterministicInsights({
    temperature: tempStats,
    pressure: pressStats,
    humidity: humStats,
    totalAnomalies: combinedAnomalies.length,
    highestSeverity: highestSev,
    hours: periodHours,
    stationName: station.name,
  });

  const recommendations = generateOperationalRecommendations({
    station,
    totalAnomalies: combinedAnomalies.length,
    highestSeverity: highestSev,
    sensorHealth,
    spatialSummary,
  });

  const dateStr = generatedAt.slice(0, 10).replace(/-/g, '');
  const reportId = `REP-${station.station_id}-${dateStr}-${periodHours}H`;

  const metadata: ReportMetadata = {
    reportId,
    generatedAt,
    periodHours,
    systemVersion: 'SkyGuard AI v0.1.0-SIH',
  };

  return {
    metadata,
    station,
    currentReading,
    telemetry: {
      temperature: tempStats,
      pressure: pressStats,
      humidity: humStats,
      points,
    },
    anomalies: {
      total: combinedAnomalies.length,
      highestSeverity: highestSev,
      severityDistribution: sevDist,
      typeDistribution: typeDist,
      latestAnomaly,
      incidents: combinedAnomalies.sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ),
    },
    sensorHealth,
    spatial: spatialSummary,
    insights,
    recommendations,
  };
}
