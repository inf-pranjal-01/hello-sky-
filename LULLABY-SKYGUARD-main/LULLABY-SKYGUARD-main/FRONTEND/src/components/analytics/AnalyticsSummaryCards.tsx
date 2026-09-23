import React from 'react';
import {
  Thermometer,
  ArrowUpDown,
  Gauge,
  Droplets,
  AlertTriangle,
  Flame,
} from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { AnalyticsSummary } from '../../types';
import './AnalyticsSummaryCards.css';

export interface AnalyticsSummaryCardsProps {
  summary: AnalyticsSummary | null;
  isLoading?: boolean;
}

export const AnalyticsSummaryCards: React.FC<AnalyticsSummaryCardsProps> = ({
  summary,
  isLoading = false,
}) => {
  if (isLoading || !summary) {
    return (
      <div className="sg-analytics-kpi-grid" aria-label="Loading analytics summary indicators">
        {[1, 2, 3, 4, 5, 6].map((idx) => (
          <Card key={idx} variant="glass" className="sg-analytics-kpi-card">
            <div className="sg-analytics-kpi-card__header">
              <Skeleton width="100px" height="0.9rem" />
              <Skeleton width="28px" height="28px" borderRadius="50%" />
            </div>
            <div className="sg-analytics-kpi-card__body">
              <Skeleton width="60%" height="1.8rem" />
            </div>
          </Card>
        ))}
      </div>
    );
  }

  const { temperature, pressure, humidity, totalAnomalies, highestSeverity } = summary;

  let severityBadgeType: 'optimal' | 'moderate' | 'high' | 'critical' = 'optimal';
  if (highestSeverity === 'critical') severityBadgeType = 'critical';
  else if (highestSeverity === 'high') severityBadgeType = 'high';
  else if (highestSeverity === 'medium') severityBadgeType = 'moderate';
  else if (highestSeverity === 'low') severityBadgeType = 'optimal';

  return (
    <div className="sg-analytics-kpi-grid" role="region" aria-label="Analytics key performance indicators">
      {/* 1. Average Temperature */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Avg Temperature</span>
          <div className="sg-analytics-kpi-card__icon sg-analytics-kpi-card__icon--warning">
            <Thermometer size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          <div className="sg-analytics-kpi-card__value">
            {temperature.average.toFixed(1)}
            <span className="sg-analytics-kpi-card__unit">°C</span>
          </div>
          <span className="sg-analytics-kpi-card__subtext">{temperature.count} samples</span>
        </div>
      </Card>

      {/* 2. Temperature Range */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Thermal Range</span>
          <div className="sg-analytics-kpi-card__icon">
            <ArrowUpDown size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          <div className="sg-analytics-kpi-card__value">
            {temperature.range.toFixed(1)}
            <span className="sg-analytics-kpi-card__unit">°C</span>
          </div>
          <span className="sg-analytics-kpi-card__subtext">
            {temperature.min.toFixed(1)} – {temperature.max.toFixed(1)} °C
          </span>
        </div>
      </Card>

      {/* 3. Average Pressure */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Avg Pressure</span>
          <div className="sg-analytics-kpi-card__icon">
            <Gauge size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          <div className="sg-analytics-kpi-card__value">
            {pressure.average.toFixed(1)}
            <span className="sg-analytics-kpi-card__unit">hPa</span>
          </div>
          <span className="sg-analytics-kpi-card__subtext">Δ {pressure.range.toFixed(1)} hPa</span>
        </div>
      </Card>

      {/* 4. Average Humidity */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Avg Humidity</span>
          <div className="sg-analytics-kpi-card__icon sg-analytics-kpi-card__icon--optimal">
            <Droplets size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          <div className="sg-analytics-kpi-card__value">
            {humidity.average.toFixed(1)}
            <span className="sg-analytics-kpi-card__unit">%</span>
          </div>
          <span className="sg-analytics-kpi-card__subtext">Δ {humidity.range.toFixed(1)} %</span>
        </div>
      </Card>

      {/* 5. Anomalies Detected */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Anomalies Detected</span>
          <div
            className={`sg-analytics-kpi-card__icon ${
              totalAnomalies > 0
                ? 'sg-analytics-kpi-card__icon--critical'
                : 'sg-analytics-kpi-card__icon--optimal'
            }`}
          >
            <AlertTriangle size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          <div
            className={`sg-analytics-kpi-card__value ${
              totalAnomalies > 0 ? 'text-critical' : 'text-optimal'
            }`}
          >
            {totalAnomalies}
          </div>
          <span className="sg-analytics-kpi-card__subtext">
            {totalAnomalies === 0 ? 'Nominal' : 'Events flagged'}
          </span>
        </div>
      </Card>

      {/* 6. Highest Severity */}
      <Card variant="glass" className="sg-analytics-kpi-card">
        <div className="sg-analytics-kpi-card__header">
          <span className="sg-analytics-kpi-card__label">Highest Severity</span>
          <div
            className={`sg-analytics-kpi-card__icon ${
              highestSeverity === 'critical'
                ? 'sg-analytics-kpi-card__icon--critical'
                : highestSeverity === 'high' || highestSeverity === 'medium'
                ? 'sg-analytics-kpi-card__icon--warning'
                : 'sg-analytics-kpi-card__icon--optimal'
            }`}
          >
            <Flame size={16} />
          </div>
        </div>
        <div className="sg-analytics-kpi-card__body">
          {highestSeverity === 'none' ? (
            <div
              className="sg-analytics-kpi-card__value text-optimal"
              style={{ fontSize: '1.25rem', textTransform: 'uppercase' }}
            >
              NONE
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
              <StatusBadge
                status={severityBadgeType}
                label={highestSeverity.toUpperCase()}
                size="md"
              />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};
