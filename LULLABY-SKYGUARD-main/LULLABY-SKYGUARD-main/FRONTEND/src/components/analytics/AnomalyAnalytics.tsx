import React from 'react';
import { AlertTriangle, Activity } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { AnalyticsSummary, SeverityDistribution, AnomalyTypeDistribution } from '../../types';
import './AnomalyAnalytics.css';

export interface AnomalyAnalyticsProps {
  analyticsSummary: AnalyticsSummary | null;
  isLoading?: boolean;
  className?: string;
}

// ── Severity distribution bar rows ────────────────────────────────────────────
interface SeverityRowProps {
  label: string;
  count: number;
  total: number;
  barClass: string;
}

const SeverityRow: React.FC<SeverityRowProps> = ({ label, count, total, barClass }) => {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div className="sg-anomaly-analytics-row" role="listitem">
      <div className="sg-anomaly-analytics-row__meta">
        <span className="sg-anomaly-analytics-row__label">{label}</span>
        <span className="sg-anomaly-analytics-row__count" aria-label={`${count} events`}>
          {count}
        </span>
      </div>
      <div
        className="sg-anomaly-analytics-row__track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${count} (${pct}%)`}
      >
        <div
          className={`sg-anomaly-analytics-row__bar ${barClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

// ── Type distribution bar rows ─────────────────────────────────────────────────
interface TypeRowProps {
  label: string;
  count: number;
  total: number;
}

const TypeRow: React.FC<TypeRowProps> = ({ label, count, total }) => {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div className="sg-anomaly-analytics-row" role="listitem">
      <div className="sg-anomaly-analytics-row__meta">
        <span className="sg-anomaly-analytics-row__label">{label}</span>
        <span className="sg-anomaly-analytics-row__count" aria-label={`${count} events`}>
          {count}
        </span>
      </div>
      <div
        className="sg-anomaly-analytics-row__track"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${count} (${pct}%)`}
      >
        <div
          className="sg-anomaly-analytics-row__bar sg-anomaly-analytics-row__bar--type"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};

// ── Loading skeleton ───────────────────────────────────────────────────────────
const AnomalyAnalyticsSkeleton: React.FC = () => (
  <div className="sg-anomaly-analytics-grid">
    {[0, 1].map((i) => (
      <Card key={i} variant="glass" className="sg-anomaly-analytics-card">
        <div className="sg-anomaly-analytics-card__header">
          <Skeleton width="160px" height="1.2rem" />
        </div>
        <div className="sg-anomaly-analytics-card__list">
          {[0, 1, 2, 3].map((j) => (
            <div key={j} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <Skeleton width="100%" height="0.8rem" />
              <Skeleton width="100%" height="8px" />
            </div>
          ))}
        </div>
      </Card>
    ))}
  </div>
);

// ── Main component ─────────────────────────────────────────────────────────────
export const AnomalyAnalytics: React.FC<AnomalyAnalyticsProps> = ({
  analyticsSummary,
  isLoading = false,
  className = '',
}) => {
  if (isLoading) {
    return <AnomalyAnalyticsSkeleton />;
  }

  const sevDist: SeverityDistribution = analyticsSummary?.severityDistribution ?? {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  };

  const typeDist: AnomalyTypeDistribution = analyticsSummary?.typeDistribution ?? {
    spike: 0,
    frozen_value: 0,
    drift: 0,
    dropout: 0,
    sensor_fail_low: 0,
    multivariate_inconsistency: 0,
  };

  const totalSev = sevDist.critical + sevDist.high + sevDist.medium + sevDist.low;
  const totalType =
    typeDist.spike +
    typeDist.frozen_value +
    typeDist.drift +
    typeDist.dropout +
    typeDist.sensor_fail_low +
    typeDist.multivariate_inconsistency;

  const hasData = totalSev > 0 || totalType > 0;

  return (
    <div className={`sg-anomaly-analytics-grid ${className}`}>
      {/* Severity Distribution Card */}
      <Card
        variant="glass"
        className="sg-anomaly-analytics-card"
        role="region"
        aria-label="Anomaly severity distribution"
      >
        <div className="sg-anomaly-analytics-card__header">
          <div className="sg-anomaly-analytics-card__title-group">
            <AlertTriangle size={16} aria-hidden="true" />
            <h3 className="sg-anomaly-analytics-card__title">Severity Distribution</h3>
          </div>
          {/* [FRONTEND ONLY] [DERIVED FROM EXISTING DATA] */}
        </div>

        {!hasData ? (
          <p className="sg-anomaly-analytics-card__empty">
            No anomalies detected in this observation window.
          </p>
        ) : (
          <div className="sg-anomaly-analytics-card__list" role="list" aria-label="Severity counts">
            <SeverityRow
              label="Critical"
              count={sevDist.critical}
              total={totalSev}
              barClass="sg-anomaly-analytics-row__bar--critical"
            />
            <SeverityRow
              label="High"
              count={sevDist.high}
              total={totalSev}
              barClass="sg-anomaly-analytics-row__bar--high"
            />
            <SeverityRow
              label="Medium"
              count={sevDist.medium}
              total={totalSev}
              barClass="sg-anomaly-analytics-row__bar--medium"
            />
            <SeverityRow
              label="Low"
              count={sevDist.low}
              total={totalSev}
              barClass="sg-anomaly-analytics-row__bar--low"
            />
          </div>
        )}
      </Card>

      {/* Type Distribution Card */}
      <Card
        variant="glass"
        className="sg-anomaly-analytics-card"
        role="region"
        aria-label="Anomaly type distribution"
      >
        <div className="sg-anomaly-analytics-card__header">
          <div className="sg-anomaly-analytics-card__title-group">
            <Activity size={16} aria-hidden="true" />
            <h3 className="sg-anomaly-analytics-card__title">Anomaly Type Breakdown</h3>
          </div>
          {/* [FRONTEND ONLY] [DERIVED FROM EXISTING DATA] */}
        </div>

        {!hasData ? (
          <p className="sg-anomaly-analytics-card__empty">
            No anomaly type data available for this window.
          </p>
        ) : (
          <div className="sg-anomaly-analytics-card__list" role="list" aria-label="Type counts">
            <TypeRow label="Spike" count={typeDist.spike} total={totalType} />
            <TypeRow label="Frozen Value" count={typeDist.frozen_value} total={totalType} />
            <TypeRow label="Drift" count={typeDist.drift} total={totalType} />
            <TypeRow label="Dropout" count={typeDist.dropout} total={totalType} />
            <TypeRow
              label="Multivariate Inconsistency"
              count={typeDist.multivariate_inconsistency}
              total={totalType}
            />
          </div>
        )}
      </Card>
    </div>
  );
};
