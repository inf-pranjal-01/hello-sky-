import { TrendsResponse } from '../types';

/**
 * [MOCK DATA]
 * Generates realistic historical trend data points for a given station and duration.
 * Matches: [API: GET /api/trends — INTEGRATED]
 */
export function generateMockTrends(stationId: string, hours: number = 6): TrendsResponse {
  const points = [];
  const now = Date.now();
  const stepMinutes = hours <= 6 ? 15 : hours <= 12 ? 30 : 60;
  const totalSteps = (hours * 60) / stepMinutes;

  // Station specific baselines
  let baseTemp = 28.0;
  let basePress = 1008.0;
  let baseHum = 64.0;

  if (stationId === 'ST-MUM-002') {
    baseTemp = 31.0;
    basePress = 1004.0;
    baseHum = 82.0;
  } else if (stationId === 'ST-BLR-003') {
    baseTemp = 24.0;
    basePress = 915.0;
    baseHum = 60.0;
  } else if (stationId === 'ST-HYD-004') {
    baseTemp = 35.5;
    basePress = 996.0;
    baseHum = 44.0;
  }

  for (let i = totalSteps; i >= 0; i--) {
    const time = new Date(now - i * stepMinutes * 60 * 1000);
    // Plausible smooth meteorological variations
    const cycle = Math.sin((totalSteps - i) / 4);
    const noise = (Math.random() - 0.5) * 0.4;
    
    // If ST-HYD-004 in last 3 steps, simulate thermal & pressure drift
    const isAnomalyWindow = stationId === 'ST-HYD-004' && i <= 3;
    const tempDrift = isAnomalyWindow ? 2.5 : 0;
    const pressDrop = isAnomalyWindow ? -4.5 : 0;

    const temp = Number((baseTemp + cycle * 1.5 + noise + tempDrift).toFixed(1));
    const press = Number((basePress - cycle * 1.0 + noise * 0.5 + pressDrop).toFixed(1));
    const hum = Number(Math.min(100, Math.max(10, baseHum - cycle * 3.0 + noise * 1.2)).toFixed(1));

    points.push({
      timestamp: time.toISOString(),
      temperature_c: temp,
      pressure_hpa: press,
      humidity_pct: hum,
      anomaly_score_pct: isAnomalyWindow ? 91 : Math.floor(Math.random() * 15 + 5),
    });
  }

  return {
    station_id: stationId,
    hours,
    points,
  };
}
