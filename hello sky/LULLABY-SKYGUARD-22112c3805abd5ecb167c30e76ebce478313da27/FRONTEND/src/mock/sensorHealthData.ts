import { SensorHealth, HealthHistoryPoint } from '../types';

/**
 * [MOCK DATA]
 * Isolated sensor health status fixtures for mock mode.
 * Matches exact API contract: [API: GET /api/sensor-health?station_id={station_id} — INTEGRATED]
 */
export const MOCK_SENSOR_HEALTH: Record<string, SensorHealth> = {
  'ST-NDL-001': {
    station_id: 'ST-NDL-001',
    sensor_health_pct: 94,
    sensor_health_status: 'HEALTHY',
  },
  'ST-MUM-002': {
    station_id: 'ST-MUM-002',
    sensor_health_pct: 76,
    sensor_health_status: 'WARNING',
  },
  'ST-BLR-003': {
    station_id: 'ST-BLR-003',
    sensor_health_pct: 99,
    sensor_health_status: 'HEALTHY',
  },
  'ST-HYD-004': {
    station_id: 'ST-HYD-004',
    sensor_health_pct: 42,
    sensor_health_status: 'CRITICAL',
  },
  'ST-KOL-005': {
    station_id: 'ST-KOL-005',
    sensor_health_pct: 0,
    sensor_health_status: 'OFFLINE',
  },
};

/**
 * Deterministic Health History Mock Generator
 * [FRONTEND ONLY — DEMO HISTORY]
 * Generates gradual, plausible health trajectories over time without random jumps on rerender.
 * Note: Historical health trajectory endpoint is NOT in the backend API contract.
 */
export const generateMockHealthHistory = (
  stationId: string,
  hours: number = 6
): HealthHistoryPoint[] => {
  const pointsCount = hours === 6 ? 12 : hours === 12 ? 24 : 36;
  const intervalMs = (hours * 3600 * 1000) / pointsCount;
  const now = Date.now();
  const currentHealth = MOCK_SENSOR_HEALTH[stationId] ?? {
    station_id: stationId,
    sensor_health_pct: 95,
    sensor_health_status: 'HEALTHY',
  };

  const points: HealthHistoryPoint[] = [];

  // Trajectory configurations per station status
  let baseScore = currentHealth.sensor_health_pct;
  let startScore = baseScore;

  if (currentHealth.sensor_health_status === 'HEALTHY') {
    startScore = Math.min(100, baseScore + 3);
  } else if (currentHealth.sensor_health_status === 'WARNING') {
    startScore = 88;
  } else if (currentHealth.sensor_health_status === 'CRITICAL') {
    startScore = 68;
  } else if (currentHealth.sensor_health_status === 'OFFLINE') {
    startScore = 40;
  }

  for (let i = pointsCount - 1; i >= 0; i--) {
    const time = new Date(now - i * intervalMs).toISOString();
    const progress = (pointsCount - 1 - i) / (pointsCount - 1 || 1); // 0.0 -> 1.0

    // Gradual interpolation with gentle sine oscillation
    const interpolated = startScore + (baseScore - startScore) * progress;
    const wave = Math.sin((i * Math.PI) / 4) * 1.2;
    let score = Math.round(interpolated + wave);

    if (currentHealth.sensor_health_status === 'OFFLINE' && progress > 0.6) {
      score = 0;
    } else {
      score = Math.max(0, Math.min(100, score));
    }

    let status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE' = 'HEALTHY';
    if (score === 0) status = 'OFFLINE';
    else if (score < 50) status = 'CRITICAL';
    else if (score < 80) status = 'WARNING';
    else status = 'HEALTHY';

    points.push({
      timestamp: time,
      health_pct: score,
      status,
    });
  }

  return points;
};
