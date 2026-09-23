import React from 'react';
import { Thermometer, Gauge, Droplets, CheckCircle, AlertCircle } from 'lucide-react';
import { SpatialComparisonSummary } from '../../types';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import './SpatialComparison.css';

export interface SpatialComparisonProps {
  summary: SpatialComparisonSummary | null;
  isLoading?: boolean;
}

export const SpatialComparison: React.FC<SpatialComparisonProps> = ({
  summary,
  isLoading = false,
}) => {
  if (isLoading || !summary) {
    return (
      <div className="sg-spatial-comp-grid">
        {[1, 2, 3].map((i) => (
          <Card key={i} variant="glass" className="sg-spatial-comp-card">
            <Skeleton width="140px" height="1.2rem" />
            <Skeleton width="100%" height="90px" style={{ marginTop: '0.75rem' }} />
          </Card>
        ))}
      </div>
    );
  }

  const { temperature, pressure, humidity, neighborCount } = summary;

  const metrics = [
    {
      id: 'temperature',
      label: 'Temperature Delta',
      icon: <Thermometer size={18} className="text-accent" aria-hidden="true" />,
      data: temperature,
      unit: '°C',
    },
    {
      id: 'pressure',
      label: 'Barometric Pressure Delta',
      icon: <Gauge size={18} className="text-accent" aria-hidden="true" />,
      data: pressure,
      unit: 'hPa',
    },
    {
      id: 'humidity',
      label: 'Relative Humidity Delta',
      icon: <Droplets size={18} className="text-accent" aria-hidden="true" />,
      data: humidity,
      unit: '%',
    },
  ];

  return (
    <div
      className="sg-spatial-comp-grid"
      role="region"
      aria-label="Spatial telemetry comparison with neighboring stations"
    >
      {metrics.map((m) => {
        const diff = m.data.difference;
        const diffSign = diff > 0 ? '+' : '';
        const isDev = m.data.isSignificantDeviation;

        return (
          <Card
            key={m.id}
            variant="glass"
            className={`sg-spatial-comp-card ${isDev ? 'sg-spatial-comp-card--deviation' : ''}`}
          >
            <div className="sg-spatial-comp-card__header">
              <div className="sg-spatial-comp-card__title-group">
                {m.icon}
                <h4 className="sg-spatial-comp-card__title">{m.label}</h4>
              </div>
              <span className="sg-spatial-comp-card__notice">
                vs {neighborCount} neighbor{neighborCount > 1 ? 's' : ''}
              </span>
            </div>

            {/* Readings Comparison Grid */}
            <div className="sg-spatial-comp-card__readings">
              <div className="sg-spatial-comp-card__col">
                <span className="sg-spatial-comp-card__col-label">Selected AWS</span>
                <span className="sg-spatial-comp-card__col-val sg-font-mono">
                  {m.data.selected.toFixed(1)} {m.unit}
                </span>
              </div>

              <div className="sg-spatial-comp-card__divider" aria-hidden="true" />

              <div className="sg-spatial-comp-card__col">
                <span className="sg-spatial-comp-card__col-label">Neighbor Mean</span>
                <span className="sg-spatial-comp-card__col-val sg-font-mono">
                  {m.data.neighborAverage.toFixed(1)} {m.unit}
                </span>
              </div>
            </div>

            {/* Spatial Difference & Assessment Badge */}
            <div className="sg-spatial-comp-card__footer">
              <div className="sg-spatial-comp-card__diff-group">
                <span className="sg-spatial-comp-card__diff-label">Divergence:</span>
                <span
                  className={`sg-spatial-comp-card__diff-val sg-font-mono ${
                    isDev ? 'sg-spatial-comp-card__diff-val--warning' : ''
                  }`}
                >
                  {diffSign}
                  {diff.toFixed(1)} {m.unit}
                </span>
              </div>

              <div
                className={`sg-spatial-comp-card__badge ${
                  isDev ? 'sg-spatial-comp-card__badge--dev' : 'sg-spatial-comp-card__badge--ok'
                }`}
              >
                {isDev ? (
                  <>
                    <AlertCircle size={13} aria-hidden="true" />
                    <span>Spatial Deviation</span>
                  </>
                ) : (
                  <>
                    <CheckCircle size={13} aria-hidden="true" />
                    <span>Regionally Consistent</span>
                  </>
                )}
              </div>
            </div>
          </Card>
        );
      })}
    </div>
  );
};
