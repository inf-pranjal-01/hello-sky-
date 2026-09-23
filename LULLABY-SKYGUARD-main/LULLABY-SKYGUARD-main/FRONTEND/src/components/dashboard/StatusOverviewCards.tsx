import React from 'react';
import { ShieldAlert, HeartPulse } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import './StatusOverviewCards.css';

export interface AnomalyScoreCardProps {
  score?: number;
  riskLevel?: 'low' | 'medium' | 'high' | 'critical';
  modelStatus?: string;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export const AnomalyScoreCard: React.FC<AnomalyScoreCardProps> = ({
  score,
  riskLevel = 'low',
  modelStatus,
  isLoading = false,
  error = null,
  onRetry,
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className="sg-status-overview-card">
        <Skeleton width="50%" height="1rem" />
        <Skeleton width="40%" height="2.5rem" style={{ margin: '0.75rem 0' }} />
        <Skeleton width="70%" height="1.25rem" />
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className="sg-status-overview-card sg-status-overview-card--error">
        <span className="sg-status-overview-card__title">Anomaly Score</span>
        <p className="sg-status-overview-card__error-text">{error}</p>
        {onRetry && (
          <button type="button" className="sg-status-overview-card__retry-btn" onClick={onRetry}>
            Retry
          </button>
        )}
      </Card>
    );
  }

  const hasScore = typeof score === 'number' && !isNaN(score);
  const scoreVal = hasScore ? Math.round(score) : 0;

  let riskBadgeType: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (riskLevel === 'critical') riskBadgeType = 'critical';
  else if (riskLevel === 'high') riskBadgeType = 'high';
  else if (riskLevel === 'medium') riskBadgeType = 'moderate';

  return (
    <Card variant="glass" className="sg-status-overview-card">
      <div className="sg-status-overview-card__header">
        <span className="sg-status-overview-card__title">Anomaly Score</span>
        <ShieldAlert
          size={18}
          className={riskLevel === 'critical' || riskLevel === 'high' ? 'text-critical' : 'text-accent'}
          aria-hidden="true"
        />
      </div>

      <div className="sg-status-overview-card__body">
        <div className="sg-status-overview-card__number-row">
          <span className="sg-status-overview-card__number">{hasScore ? `${scoreVal}%` : '—'}</span>
          <div className="sg-status-overview-card__badges">
            <StatusBadge status={riskBadgeType} label={`${riskLevel.toUpperCase()} RISK`} size="sm" />
            {modelStatus && (
              <span className={`sg-model-badge sg-model-badge--${modelStatus.toLowerCase()}`}>
                MODEL: {modelStatus}
              </span>
            )}
          </div>
        </div>

        <p className="sg-status-overview-card__sub">
          {scoreVal > 75
            ? 'Severe multi-metric variance detected'
            : scoreVal > 40
            ? 'Minor telemetry irregularity detected'
            : 'Nominal operational envelope'}
        </p>

        {/* Visual score bar */}
        <div
          className="sg-score-meter"
          role="meter"
          aria-valuenow={scoreVal}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Anomaly risk score meter"
        >
          <div
            className={`sg-score-meter__fill sg-score-meter__fill--${riskLevel}`}
            style={{ width: `${scoreVal}%` }}
          />
        </div>
      </div>
    </Card>
  );
};

interface SensorHealthCardProps {
  healthPct?: number;
  healthStatus?: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
}

export const SensorHealthCard: React.FC<SensorHealthCardProps> = ({
  healthPct,
  healthStatus = 'HEALTHY',
  isLoading = false,
  error = null,
  onRetry,
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className="sg-status-overview-card">
        <Skeleton width="50%" height="1rem" />
        <Skeleton width="40%" height="2.5rem" style={{ margin: '0.75rem 0' }} />
        <Skeleton width="70%" height="1.25rem" />
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className="sg-status-overview-card sg-status-overview-card--error">
        <span className="sg-status-overview-card__title">Sensor Health</span>
        <p className="sg-status-overview-card__error-text">{error}</p>
        {onRetry && (
          <button type="button" className="sg-status-overview-card__retry-btn" onClick={onRetry}>
            Retry
          </button>
        )}
      </Card>
    );
  }

  const hasHealth = typeof healthPct === 'number' && !isNaN(healthPct);
  const healthVal = hasHealth ? Math.round(healthPct) : 0;

  // Map to StatusBadge status
  let badgeStatus: 'optimal' | 'warning' | 'critical' | 'offline' = 'optimal';
  if (healthStatus === 'WARNING') badgeStatus = 'warning';
  else if (healthStatus === 'CRITICAL') badgeStatus = 'critical';
  else if (healthStatus === 'OFFLINE') badgeStatus = 'offline';

  return (
    <Card variant="glass" className="sg-status-overview-card">
      <div className="sg-status-overview-card__header">
        <span className="sg-status-overview-card__title">Sensor Health</span>
        <HeartPulse
          size={18}
          className={healthStatus === 'HEALTHY' ? 'text-optimal' : 'text-critical'}
          aria-hidden="true"
        />
      </div>

      <div className="sg-status-overview-card__body">
        <div className="sg-status-overview-card__number-row">
          <span className="sg-status-overview-card__number">{hasHealth ? `${healthVal}%` : '—'}</span>
          <StatusBadge status={badgeStatus} label={healthStatus} size="sm" />
        </div>

        <p className="sg-status-overview-card__sub">
          {healthStatus === 'HEALTHY'
            ? 'Transducers operational & calibrated'
            : healthStatus === 'WARNING'
            ? 'Telemetry signal degradation'
            : healthStatus === 'CRITICAL'
            ? 'Hardware fault or transmission failure'
            : 'Excluded from baseline; raw telemetry remains visible'}
        </p>

        {/* Visual health bar */}
        <div
          className="sg-score-meter"
          role="meter"
          aria-valuenow={healthVal}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Sensor hardware health index"
        >
          <div
            className={`sg-score-meter__fill sg-score-meter__fill--health-${healthStatus.toLowerCase()}`}
            style={{ width: `${healthVal}%` }}
          />
        </div>
      </div>
    </Card>
  );
};
