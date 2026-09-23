import React from 'react';
import { Sparkles } from 'lucide-react';
import { Skeleton } from '../common/Skeleton';
import { ExplanationFeature } from '../../types';
import './FeatureImpactChart.css';

export interface FeatureImpactChartProps {
  features: ExplanationFeature[];
  isLoading?: boolean;
  error?: string | null;
  className?: string;
}

export const FeatureImpactChart: React.FC<FeatureImpactChartProps> = ({
  features = [],
  isLoading = false,
  error = null,
  className = '',
}) => {
  if (isLoading) {
    return (
      <div className={`sg-feature-impact ${className}`} aria-label="Loading feature contributions">
        <div className="sg-feature-impact__header">
          <Skeleton width="180px" height="1.1rem" />
          <Skeleton width="90%" height="0.9rem" style={{ marginTop: '0.25rem' }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '0.75rem' }}>
          <Skeleton width="100%" height="1.5rem" />
          <Skeleton width="100%" height="1.5rem" />
          <Skeleton width="100%" height="1.5rem" />
          <Skeleton width="100%" height="1.5rem" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`sg-feature-impact ${className}`}>
        <div className="sg-feature-impact__header">
          <h4 className="sg-feature-impact__title">Feature Contribution Analysis</h4>
        </div>
        <div className="sg-feature-impact__empty" style={{ color: 'var(--color-status-critical, #ef4444)' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div className={`sg-feature-impact ${className}`}>
        <div className="sg-feature-impact__header">
          <h4 className="sg-feature-impact__title">Feature Contribution Analysis</h4>
        </div>
        <div className="sg-feature-impact__empty">
          No feature impact breakdown available for this anomaly event.
        </div>
      </div>
    );
  }

  // Find max absolute value for proportional scaling (minimum scale factor 0.5 to prevent single tiny bar filling 100%)
  const maxAbsImpact = Math.max(...features.map((f) => Math.abs(f.impact)), 0.5);

  return (
    <div className={`sg-feature-impact ${className}`} role="region" aria-label="Feature contribution breakdown">
      <div className="sg-feature-impact__header">
        <div className="sg-feature-impact__title-row">
          <h4 className="sg-feature-impact__title">
            <Sparkles size={16} className="text-accent" aria-hidden="true" />
            Feature Contribution Analysis
          </h4>
          <span className="sg-feature-impact__notice">
            ● LIVE BACKEND
          </span>
        </div>
        <p className="sg-feature-impact__subtitle">
          Estimated contribution of individual telemetry attributes towards the anomaly score.
          Positive values push toward anomaly; negative values push toward normal baseline.
        </p>
      </div>

      <div className="sg-feature-impact__legend" aria-hidden="true">
        <div className="sg-feature-impact__legend-item">
          <span className="sg-feature-impact__legend-dot sg-feature-impact__legend-dot--positive" />
          <span>Increases Anomaly Risk (+)</span>
        </div>
        <div className="sg-feature-impact__legend-item">
          <span className="sg-feature-impact__legend-dot sg-feature-impact__legend-dot--negative" />
          <span>Stabilizes Toward Normal (-)</span>
        </div>
      </div>

      <div className="sg-feature-impact__chart" role="list" aria-label="Feature impact list">
        {features.map((feature, idx) => {
          const isPositive = feature.impact >= 0;
          const absVal = Math.abs(feature.impact);
          const barPct = Math.min((absVal / maxAbsImpact) * 50, 50); // max 50% from center axis
          const sign = isPositive ? '+' : '';
          const displayImpact = `${sign}${(feature.impact * 100).toFixed(0)}%`;

          return (
            <div
              key={idx}
              className="sg-feature-impact__row"
              role="listitem"
              aria-label={`${feature.name}: ${displayImpact} contribution`}
            >
              <span className="sg-feature-impact__feature-name" title={feature.name}>
                {feature.name}
              </span>

              <div className="sg-feature-impact__bar-track" aria-hidden="true">
                <div className="sg-feature-impact__axis-center" />
                {isPositive ? (
                  <div
                    className="sg-feature-impact__bar sg-feature-impact__bar--positive"
                    style={{ width: `${barPct}%` }}
                  />
                ) : (
                  <div
                    className="sg-feature-impact__bar sg-feature-impact__bar--negative"
                    style={{ width: `${barPct}%` }}
                  />
                )}
              </div>

              <span
                className={`sg-feature-impact__value ${
                  isPositive ? 'sg-feature-impact__value--positive' : 'sg-feature-impact__value--negative'
                }`}
              >
                {displayImpact}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
