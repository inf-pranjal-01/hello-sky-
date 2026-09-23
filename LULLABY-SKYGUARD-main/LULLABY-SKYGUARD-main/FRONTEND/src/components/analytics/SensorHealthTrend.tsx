import React from 'react';
import { HealthHistoryChart } from '../sensor-health/HealthHistoryChart';
import './SensorHealthTrend.css';

export interface SensorHealthTrendProps {
  stationId: string | null | undefined;
  className?: string;
}

/**
 * SensorHealthTrend
 *
 * Wrapper around HealthHistoryChart for the Analytics page.
 * Reuses the existing chart from the Sensor Health page without
 * duplicating the SVG/rendering logic.
 *
 * [FRONTEND DERIVED — DEMO HISTORY]
 */
export const SensorHealthTrend: React.FC<SensorHealthTrendProps> = ({
  stationId,
  className = '',
}) => {
  return (
    <section
      className={`sg-sensor-health-trend ${className}`}
      aria-label="Sensor health trend (analytics view)"
    >
      <HealthHistoryChart stationId={stationId} />
    </section>
  );
};
