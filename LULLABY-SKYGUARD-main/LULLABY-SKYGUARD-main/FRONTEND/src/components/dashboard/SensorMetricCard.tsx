import React from 'react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import './SensorMetricCard.css';

export interface SensorMetricCardProps {
  title: string;
  icon: React.ReactNode;
  value?: number;
  unit: string;
  normalMin?: number;
  normalMax?: number;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  accentColor?: string;
  className?: string;
  suggestedValue?: number | null;
}

export const SensorMetricCard: React.FC<SensorMetricCardProps> = ({
  title,
  icon,
  value,
  unit,
  normalMin,
  normalMax,
  isLoading = false,
  error = null,
  onRetry,
  accentColor = 'var(--accent-blue)',
  className = '',
  suggestedValue,
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-metric-card ${className}`}>
        <div className="sg-metric-card__header">
          <Skeleton width="40%" height="1rem" />
          <Skeleton width="1.5rem" height="1.5rem" borderRadius="50%" />
        </div>
        <div className="sg-metric-card__value-row">
          <Skeleton width="60%" height="2.5rem" />
        </div>
        <div className="sg-metric-card__range-row">
          <Skeleton width="80%" height="0.85rem" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-metric-card sg-metric-card--error ${className}`}>
        <div className="sg-metric-card__header">
          <span className="sg-metric-card__title">{title}</span>
          <span className="sg-metric-card__icon" style={{ color: 'var(--status-critical-text)' }}>{icon}</span>
        </div>
        <p className="sg-metric-card__error-msg">{error}</p>
        {onRetry && (
          <button type="button" className="sg-metric-card__retry-btn" onClick={onRetry}>
            Retry
          </button>
        )}
      </Card>
    );
  }

  const hasValue = typeof value === 'number' && !isNaN(value);
  const isOutOfRange =
    hasValue &&
    typeof normalMin === 'number' &&
    typeof normalMax === 'number' &&
    (value < normalMin || value > normalMax);

  // Compute visual percentage position in the normal range
  let rangePct = 50;
  if (hasValue && typeof normalMin === 'number' && typeof normalMax === 'number') {
    const span = normalMax - normalMin;
    if (span > 0) {
      rangePct = Math.min(100, Math.max(0, ((value - normalMin) / span) * 100));
    }
  }

  return (
    <Card variant="glass" className={`sg-metric-card ${isOutOfRange ? 'sg-metric-card--out-of-range' : ''} ${className}`}>
      <div className="sg-metric-card__header">
        <span className="sg-metric-card__title">{title}</span>
        <span className="sg-metric-card__icon" style={{ color: accentColor }} aria-hidden="true">
          {icon}
        </span>
      </div>

      <div className="sg-metric-card__value-row">
        <span className="sg-metric-card__number">
          {hasValue ? value.toFixed(1) : '—'}
        </span>
        <span className="sg-metric-card__unit" aria-label={unit}>
          {unit}
        </span>
        {isOutOfRange && (
          <span className="sg-metric-card__range-badge" title="Value outside seasonal normal baseline">
            Range Alert
          </span>
        )}
      </div>

      {typeof normalMin === 'number' && typeof normalMax === 'number' && (
        <div className="sg-metric-card__range-info">
          <div className="sg-metric-card__range-labels">
            <span className="sg-range-label">Normal Range:</span>
            <span className="sg-range-value">
              {normalMin.toFixed(1)} — {normalMax.toFixed(1)} {unit}
            </span>
          </div>

          <div
            className="sg-range-bar"
            role="progressbar"
            aria-valuenow={hasValue ? value : undefined}
            aria-valuemin={normalMin}
            aria-valuemax={normalMax}
            aria-label={`${title} normal operating range indicator`}
          >
            <div
              className={`sg-range-bar__fill ${isOutOfRange ? 'sg-range-bar__fill--out-of-range' : ''}`}
              style={{ width: `${rangePct}%`, backgroundColor: accentColor }}
            />
          </div>
        </div>
      )}

      {typeof suggestedValue === 'number' && (
        <div className="sg-metric-card__suggested">
          <span>Suggested</span>
          <strong>{suggestedValue.toFixed(1)}&thinsp;{unit}</strong>
        </div>
      )}
    </Card>
  );
};
