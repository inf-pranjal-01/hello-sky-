import {
  MetricStatistics,
  SeverityDistribution,
  AnomalyTypeDistribution,
  AnomalySeverity,
  RecentAnomalyItem,
} from '../types';

/**
 * Calculate the arithmetic mean of a number array.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function calculateAverage(values: number[], decimals: number = 1): number {
  if (!values || values.length === 0) return 0;
  const valid = values.filter((v) => typeof v === 'number' && !isNaN(v));
  if (valid.length === 0) return 0;
  const sum = valid.reduce((acc, curr) => acc + curr, 0);
  return Number((sum / valid.length).toFixed(decimals));
}

/**
 * Calculate the minimum value in a number array.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function calculateMinimum(values: number[], decimals: number = 1): number {
  if (!values || values.length === 0) return 0;
  const valid = values.filter((v) => typeof v === 'number' && !isNaN(v));
  if (valid.length === 0) return 0;
  return Number(Math.min(...valid).toFixed(decimals));
}

/**
 * Calculate the maximum value in a number array.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function calculateMaximum(values: number[], decimals: number = 1): number {
  if (!values || values.length === 0) return 0;
  const valid = values.filter((v) => typeof v === 'number' && !isNaN(v));
  if (valid.length === 0) return 0;
  return Number(Math.max(...valid).toFixed(decimals));
}

/**
 * Calculate the numerical range (max - min) of a number array.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function calculateRange(values: number[], decimals: number = 1): number {
  if (!values || values.length <= 1) return 0;
  const min = calculateMinimum(values, decimals);
  const max = calculateMaximum(values, decimals);
  return Number((max - min).toFixed(decimals));
}

/**
 * Calculate complete descriptive statistics for a series.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function calculateMetricStatistics(
  values: number[],
  decimals: number = 1
): MetricStatistics {
  if (!values || values.length === 0) {
    return {
      average: 0,
      min: 0,
      max: 0,
      range: 0,
      count: 0,
    };
  }

  const valid = values.filter((v) => typeof v === 'number' && !isNaN(v));
  const avg = calculateAverage(valid, decimals);
  const min = calculateMinimum(valid, decimals);
  const max = calculateMaximum(valid, decimals);
  const range = Number((max - min).toFixed(decimals));

  return {
    average: avg,
    min,
    max,
    range,
    count: valid.length,
  };
}

/**
 * Determine highest severity rank among a collection of severities.
 * Severity precedence: critical > high > medium > low
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function deriveHighestSeverity(severities: AnomalySeverity[]): AnomalySeverity | 'none' {
  if (!severities || severities.length === 0) return 'none';
  if (severities.includes('critical')) return 'critical';
  if (severities.includes('high')) return 'high';
  if (severities.includes('medium')) return 'medium';
  if (severities.includes('low')) return 'low';
  return 'none';
}

/**
 * Calculate frequency distribution across anomaly severities.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function deriveSeverityDistribution(anomalies: RecentAnomalyItem[]): SeverityDistribution {
  const dist: SeverityDistribution = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  };

  if (!anomalies) return dist;

  for (const a of anomalies) {
    if (a.severity === 'critical') dist.critical++;
    else if (a.severity === 'high') dist.high++;
    else if (a.severity === 'medium') dist.medium++;
    else if (a.severity === 'low') dist.low++;
  }

  return dist;
}

/**
 * Calculate frequency distribution across standard anomaly types.
 * [FRONTEND ONLY] [DERIVED CALCULATION]
 */
export function deriveTypeDistribution(anomalies: RecentAnomalyItem[]): AnomalyTypeDistribution {
  const dist: AnomalyTypeDistribution = {
    spike: 0,
    frozen_value: 0,
    drift: 0,
    dropout: 0,
    sensor_fail_low: 0,
    multivariate_inconsistency: 0,
    physical_bounds: 0,
    statistical_anomaly: 0,
  };

  if (!anomalies) return dist;

  for (const a of anomalies) {
    if (a.type in dist) {
      dist[a.type]++;
    }
  }

  return dist;
}

/**
 * Generate deterministic, rules-based operational insights.
 * [FRONTEND ONLY] [DERIVED INSIGHTS ENGINE]
 */
export function generateDeterministicInsights(params: {
  temperature: MetricStatistics;
  pressure: MetricStatistics;
  humidity: MetricStatistics;
  totalAnomalies: number;
  highestSeverity: AnomalySeverity | 'none';
  hours: number;
  stationName?: string;
}): string[] {
  const { temperature, pressure, humidity, totalAnomalies, highestSeverity, hours } = params;
  const insights: string[] = [];

  // Insight 1: Telemetry Variation Summary
  if (temperature.count > 0) {
    insights.push(
      `Temperature averaged ${temperature.average.toFixed(1)} °C across the last ${hours} hours, fluctuating within a range of ${temperature.range.toFixed(1)} °C (${temperature.min.toFixed(1)} °C to ${temperature.max.toFixed(1)} °C).`
    );
  }

  // Insight 2: Anomaly Frequency & Severity
  if (totalAnomalies === 0) {
    insights.push(
      `No anomaly events were detected during the selected ${hours}-hour observation window. Autonomous telemetry stream is operating nominally.`
    );
  } else {
    insights.push(
      `${totalAnomalies} anomaly event${totalAnomalies > 1 ? 's were' : ' was'} recorded in the ${hours}-hour window, with the highest detected severity categorized as ${highestSeverity.toUpperCase()}.`
    );
  }

  // Insight 3: Metric Stability Comparison
  if (humidity.count > 0 && pressure.count > 0) {
    if (humidity.range > 20) {
      insights.push(
        `Relative humidity displayed significant variance (range of ${humidity.range.toFixed(1)} %), which is typical during diurnal thermal transitions.`
      );
    } else if (pressure.range > 8) {
      insights.push(
        `Barometric pressure displayed notable shifts (${pressure.range.toFixed(1)} hPa delta), indicating dynamic regional atmospheric pressure gradients.`
      );
    } else {
      insights.push(
        `Barometric pressure and relative humidity remained steady with narrow standard variance over the monitoring interval.`
      );
    }
  }

  return insights;
}
