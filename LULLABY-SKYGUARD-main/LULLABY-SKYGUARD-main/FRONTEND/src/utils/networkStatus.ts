import { Station, SystemOverallStatus, SystemStatusSummary } from '../types';

export function summarizeNetworkStatus(
  stations: Station[],
  activeAnomaliesCount = 0,
  avgSensorHealthPct?: number
): SystemStatusSummary {
  const statuses = stations.map((station) => station.status);
  let overall: SystemOverallStatus = 'NORMAL';
  if (statuses.includes('CRITICAL')) overall = 'CRITICAL';
  else if (statuses.includes('WARNING') || statuses.includes('OFFLINE')) overall = 'WARNING';

  const activeStations = stations.filter((station) => station.status !== 'OFFLINE').length;
  return {
    overall_status: overall,
    active_stations_count: activeStations,
    total_stations_count: stations.length,
    active_anomalies_count: activeAnomaliesCount,
    avg_sensor_health_pct: avgSensorHealthPct ?? 100,
    last_updated: new Date().toISOString(),
  };
}
