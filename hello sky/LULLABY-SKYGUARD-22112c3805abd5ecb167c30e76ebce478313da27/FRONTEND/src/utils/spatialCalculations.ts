import {
  MetricSpatialComparison,
  SpatialConsistencyStatus,
  SpatialComparisonSummary,
  SpatialDemoScenario,
} from '../types';

/**
 * [FRONTEND DEMO LOGIC] [NOT PRODUCTION ML]
 * 
 * Configurable thresholds for detecting significant deviation between a selected station
 * and its neighboring stations.
 * Note: In a production system, thresholds will be dynamically parameterized by a trained
 * spatial model or meteorological variance rules.
 */
export const SPATIAL_DEVIATION_THRESHOLDS = {
  temperature_c: 3.0, // °C deviation threshold
  pressure_hpa: 2.0,   // hPa deviation threshold
  humidity_pct: 15.0,  // % humidity deviation threshold
};

/**
 * Calculates the arithmetic mean of a numeric array, handling edge cases.
 * Returns null if the array is empty or contains no valid numbers.
 */
export function calculateNeighborAverage(values: number[]): number | null {
  const valid = values.filter((v) => typeof v === 'number' && !isNaN(v) && isFinite(v));
  if (valid.length === 0) return null;
  const sum = valid.reduce((acc, val) => acc + val, 0);
  const avg = sum / valid.length;
  return Math.round(avg * 10) / 10;
}

/**
 * Calculates a metric's spatial comparison against neighboring stations.
 */
export function calculateMetricSpatialComparison(
  selectedValue: number | null | undefined,
  neighborValues: number[],
  threshold: number,
  unit: string
): MetricSpatialComparison {
  const defaultVal = 0;
  const safeSelected =
    selectedValue != null && !isNaN(selectedValue) && isFinite(selectedValue)
      ? selectedValue
      : defaultVal;

  const neighborAvg = calculateNeighborAverage(neighborValues);

  if (neighborAvg === null) {
    return {
      selected: Math.round(safeSelected * 10) / 10,
      neighborAverage: Math.round(safeSelected * 10) / 10,
      difference: 0,
      isSignificantDeviation: false,
      unit,
    };
  }

  const diff = Math.round((safeSelected - neighborAvg) * 10) / 10;
  const isSignificantDeviation = Math.abs(diff) >= threshold;

  return {
    selected: Math.round(safeSelected * 10) / 10,
    neighborAverage: neighborAvg,
    difference: diff,
    isSignificantDeviation,
    unit,
  };
}

/**
 * Derives the overall spatial consistency status from metric comparisons.
 */
export function deriveSpatialConsistency(
  tempComp: MetricSpatialComparison,
  pressComp: MetricSpatialComparison,
  humComp: MetricSpatialComparison,
  neighborCount: number
): SpatialConsistencyStatus {
  if (neighborCount === 0) {
    return 'INSUFFICIENT_DATA';
  }

  const hasDeviation =
    tempComp.isSignificantDeviation ||
    pressComp.isSignificantDeviation ||
    humComp.isSignificantDeviation;

  return hasDeviation ? 'DEVIATION_DETECTED' : 'CONSISTENT';
}

/**
 * Generates a neutral, deterministic spatial summary text without fabricated claims.
 */
export function generateMultivariateSummary(
  tempComp: MetricSpatialComparison,
  pressComp: MetricSpatialComparison,
  humComp: MetricSpatialComparison,
  consistencyStatus: SpatialConsistencyStatus,
  neighborCount: number
): string {
  if (consistencyStatus === 'INSUFFICIENT_DATA' || neighborCount === 0) {
    return 'Insufficient neighboring station telemetry is available in this geographical sector to establish a reliable spatial comparison baseline.';
  }

  if (consistencyStatus === 'CONSISTENT') {
    return `The selected station exhibits close agreement with ${neighborCount} nearby station${
      neighborCount > 1 ? 's' : ''
    } across temperature, barometric pressure, and humidity (deviations remain within nominal regional bounds).`;
  }

  // Deviation detected
  const deviatingMetrics: string[] = [];
  if (tempComp.isSignificantDeviation) {
    const sign = tempComp.difference > 0 ? '+' : '';
    deviatingMetrics.push(`temperature (${sign}${tempComp.difference} ${tempComp.unit})`);
  }
  if (pressComp.isSignificantDeviation) {
    const sign = pressComp.difference > 0 ? '+' : '';
    deviatingMetrics.push(`pressure (${sign}${pressComp.difference} ${pressComp.unit})`);
  }
  if (humComp.isSignificantDeviation) {
    const sign = humComp.difference > 0 ? '+' : '';
    deviatingMetrics.push(`humidity (${sign}${humComp.difference} ${humComp.unit})`);
  }

  return `The selected station shows significant localized divergence from neighboring station averages in: ${deviatingMetrics.join(
    ', '
  )}. This suggests a localized anomaly rather than a widespread meteorological event.`;
}

/**
 * Assembles the full spatial comparison summary.
 */
export function buildSpatialComparisonSummary(
  selectedStationId: string,
  selectedReading: { temperature_c: number; pressure_hpa: number; humidity_pct: number },
  neighborReadings: Array<{ temperature_c: number; pressure_hpa: number; humidity_pct: number }>,
  scenario: SpatialDemoScenario
): SpatialComparisonSummary {
  const neighborTemps = neighborReadings.map((r) => r.temperature_c);
  const neighborPress = neighborReadings.map((r) => r.pressure_hpa);
  const neighborHums = neighborReadings.map((r) => r.humidity_pct);

  const temperature = calculateMetricSpatialComparison(
    selectedReading.temperature_c,
    neighborTemps,
    SPATIAL_DEVIATION_THRESHOLDS.temperature_c,
    '°C'
  );

  const pressure = calculateMetricSpatialComparison(
    selectedReading.pressure_hpa,
    neighborPress,
    SPATIAL_DEVIATION_THRESHOLDS.pressure_hpa,
    'hPa'
  );

  const humidity = calculateMetricSpatialComparison(
    selectedReading.humidity_pct,
    neighborHums,
    SPATIAL_DEVIATION_THRESHOLDS.humidity_pct,
    '%'
  );

  const consistencyStatus = deriveSpatialConsistency(
    temperature,
    pressure,
    humidity,
    neighborReadings.length
  );

  const multivariateSummary = generateMultivariateSummary(
    temperature,
    pressure,
    humidity,
    consistencyStatus,
    neighborReadings.length
  );

  return {
    selectedStationId,
    neighborCount: neighborReadings.length,
    temperature,
    pressure,
    humidity,
    consistencyStatus,
    multivariateSummary,
    demoScenario: scenario,
  };
}
