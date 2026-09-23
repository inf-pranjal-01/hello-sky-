import React from 'react';
import { CheckCircle2, AlertTriangle, HelpCircle, ShieldCheck, Sparkles } from 'lucide-react';
import { SpatialComparisonSummary } from '../../types';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import './SpatialConsistencyCard.css';

export interface SpatialConsistencyCardProps {
  summary: SpatialComparisonSummary | null;
  isLoading?: boolean;
}

export const SpatialConsistencyCard: React.FC<SpatialConsistencyCardProps> = ({
  summary,
  isLoading = false,
}) => {
  if (isLoading || !summary) {
    return (
      <Card variant="glass" className="sg-spatial-verdict-card">
        <Skeleton width="220px" height="1.4rem" />
        <Skeleton width="100%" height="80px" style={{ marginTop: '0.75rem' }} />
      </Card>
    );
  }

  const { consistencyStatus, multivariateSummary, neighborCount, demoScenario } = summary;

  const isConsistent = consistencyStatus === 'CONSISTENT';
  const isDeviation = consistencyStatus === 'DEVIATION_DETECTED';

  return (
    <Card
      variant="glass"
      className={`sg-spatial-verdict-card ${
        isConsistent
          ? 'sg-spatial-verdict-card--consistent'
          : isDeviation
          ? 'sg-spatial-verdict-card--deviation'
          : 'sg-spatial-verdict-card--insufficient'
      }`}
      role="region"
      aria-label="Spatial consistency analysis verdict"
    >
      <div className="sg-spatial-verdict__header">
        <div className="sg-spatial-verdict__icon-row">
          {isConsistent && <CheckCircle2 size={24} className="text-optimal" aria-hidden="true" />}
          {isDeviation && <AlertTriangle size={24} className="text-warning" aria-hidden="true" />}
          {!isConsistent && !isDeviation && (
            <HelpCircle size={24} className="text-muted" aria-hidden="true" />
          )}

          <div>
            <div className="sg-spatial-verdict__badge-row">
              <span className="sg-spatial-verdict__verdict-pill">
                SPATIAL PATTERN:{' '}
                {isConsistent
                  ? 'REGIONALLY CONSISTENT'
                  : isDeviation
                  ? 'LOCALIZED DEVIATION'
                  : 'INSUFFICIENT DATA'}
              </span>
              <span className="sg-spatial-verdict__scenario-badge">
                <Sparkles size={11} aria-hidden="true" />
                Scenario: {demoScenario === 'localized_deviation' ? 'Localized Anomaly' : 'Regional Consistency'}
              </span>
            </div>
            <h3 className="sg-spatial-verdict__title">
              {isConsistent && 'Environmental Telemetry Agrees with Regional Baseline'}
              {isDeviation && 'Suspicious Localized Divergence from Nearby Observatories'}
              {!isConsistent && !isDeviation && 'Insufficient Data for Spatial Verification'}
            </h3>
          </div>
        </div>

        <span className="sg-spatial-verdict__notice">
          [FRONTEND DEMO LOGIC — NOT PRODUCTION ML]
        </span>
      </div>

      {/* Multivariate Summary Text */}
      <p className="sg-spatial-verdict__summary-text">{multivariateSummary}</p>

      {/* Operational Storytelling Seam */}
      <div className="sg-spatial-verdict__context-box">
        <div className="sg-spatial-verdict__context-header">
          <ShieldCheck size={16} className="text-accent" aria-hidden="true" />
          <span className="sg-spatial-verdict__context-title">
            Meteorological Validation Intelligence
          </span>
        </div>
        <p className="sg-spatial-verdict__context-desc">
          {isConsistent
            ? `When multiple stations across a regional network observe correlated telemetry shifts, the pattern represents a genuine meteorological phenomenon (e.g. regional frontal passage or heatwave) rather than a sensor failure.`
            : `When an anomaly occurs at an isolated station while all ${neighborCount} neighboring stations remain within nominal baselines, the pattern suggests an isolated instrument failure, sensor drift, or localized hardware artifact requiring on-site investigation.`}
        </p>
      </div>
    </Card>
  );
};
