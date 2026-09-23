import React from 'react';
import { Thermometer, Gauge, Droplets } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { MetricStatistics, CurrentSensorReading } from '../../types';
import './MetricStatisticsCard.css';

export interface MetricStatisticsCardProps {
  temperatureStats: MetricStatistics | null | undefined;
  pressureStats: MetricStatistics | null | undefined;
  humidityStats: MetricStatistics | null | undefined;
  currentReading: CurrentSensorReading | null | undefined;
  isLoading?: boolean;
}

export const MetricStatisticsCard: React.FC<MetricStatisticsCardProps> = ({
  temperatureStats,
  pressureStats,
  humidityStats,
  currentReading,
  isLoading = false,
}) => {
  if (isLoading || !temperatureStats || !pressureStats || !humidityStats) {
    return (
      <div className="sg-metric-stats-grid" aria-label="Loading metric statistical summaries">
        {[1, 2, 3].map((idx) => (
          <Card key={idx} variant="glass" className="sg-metric-stat-card">
            <Skeleton width="140px" height="1.2rem" />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.5rem' }}>
              <Skeleton width="100%" height="1.2rem" />
              <Skeleton width="100%" height="1.2rem" />
              <Skeleton width="100%" height="1.2rem" />
              <Skeleton width="100%" height="1.2rem" />
            </div>
          </Card>
        ))}
      </div>
    );
  }

  const metrics = [
    {
      id: 'temp',
      title: 'Temperature Statistics',
      icon: <Thermometer size={18} className="text-warning" />,
      unit: '°C',
      stats: temperatureStats,
      normalMin: currentReading?.temperature_c.normal_min ?? 15,
      normalMax: currentReading?.temperature_c.normal_max ?? 35,
    },
    {
      id: 'press',
      title: 'Pressure Statistics',
      icon: <Gauge size={18} className="text-accent" />,
      unit: 'hPa',
      stats: pressureStats,
      normalMin: currentReading?.pressure_hpa.normal_min ?? 980,
      normalMax: currentReading?.pressure_hpa.normal_max ?? 1030,
    },
    {
      id: 'hum',
      title: 'Humidity Statistics',
      icon: <Droplets size={18} className="text-optimal" />,
      unit: '%',
      stats: humidityStats,
      normalMin: currentReading?.humidity_pct.normal_min ?? 30,
      normalMax: currentReading?.humidity_pct.normal_max ?? 90,
    },
  ];

  return (
    <div className="sg-metric-stats-grid" role="region" aria-label="Monitored variables descriptive statistics">
      {metrics.map((m) => (
        <Card key={m.id} variant="glass" className="sg-metric-stat-card">
          <div className="sg-metric-stat-card__header">
            <div className="sg-metric-stat-card__title-group">
              {m.icon}
              <h3 className="sg-metric-stat-card__title">{m.title}</h3>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted, #94a3b8)' }}>
              {m.stats.count} samples
            </span>
          </div>

          <div className="sg-metric-stat-card__rows">
            <div className="sg-metric-stat-row">
              <span className="sg-metric-stat-row__label">Arithmetic Mean (Avg):</span>
              <span className="sg-metric-stat-row__val">
                {m.stats.average.toFixed(1)} {m.unit}
              </span>
            </div>

            <div className="sg-metric-stat-row">
              <span className="sg-metric-stat-row__label">Observed Minimum (Min):</span>
              <span className="sg-metric-stat-row__val">
                {m.stats.min.toFixed(1)} {m.unit}
              </span>
            </div>

            <div className="sg-metric-stat-row">
              <span className="sg-metric-stat-row__label">Observed Maximum (Max):</span>
              <span className="sg-metric-stat-row__val">
                {m.stats.max.toFixed(1)} {m.unit}
              </span>
            </div>

            <div className="sg-metric-stat-row">
              <span className="sg-metric-stat-row__label">Observed Range (Δ):</span>
              <span className="sg-metric-stat-row__val">
                {m.stats.range.toFixed(1)} {m.unit}
              </span>
            </div>
          </div>

          <div className="sg-metric-stat-card__baseline">
            <span className="sg-metric-stat-card__baseline-label">Baseline Threshold:</span>
            <span className="sg-metric-stat-card__baseline-val">
              {m.normalMin} – {m.normalMax} {m.unit}
            </span>
          </div>
        </Card>
      ))}
    </div>
  );
};
