import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { LatestAnomaly } from '../../types';
import './LatestAnomalyCard.css';

export interface LatestAnomalyCardProps {
  anomaly?: LatestAnomaly | null;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  className?: string;
  streamMode?: 'live' | 'replay';
}

export const LatestAnomalyCard: React.FC<LatestAnomalyCardProps> = ({
  anomaly,
  isLoading = false,
  error = null,
  onRetry,
  className = '',
  streamMode = 'live',
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-latest-anomaly-card ${className}`}>
        <div className="sg-latest-anomaly-header">
          <Skeleton width="140px" height="1.2rem" />
          <Skeleton width="80px" height="1.2rem" />
        </div>
        <div style={{ marginTop: '1rem' }}>
          <Skeleton width="60%" height="1.5rem" />
          <Skeleton width="90%" height="2rem" style={{ marginTop: '0.5rem' }} />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-latest-anomaly-card sg-latest-anomaly-card--error ${className}`}>
        <div className="sg-latest-anomaly-header">
          <h3>Latest Anomaly</h3>
        </div>
        <div className="sg-latest-anomaly-error">
          <p>{error}</p>
          {onRetry && (
            <button type="button" className="sg-latest-retry-btn" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      </Card>
    );
  }

  // Healthy empty state when there is no anomaly
  if (!anomaly) {
    return (
      <Card variant="glass" className={`sg-latest-anomaly-card ${className}`}>
        <div className="sg-latest-anomaly-header">
          <div className="sg-latest-title">
            <CheckCircle2 size={18} className="text-optimal" aria-hidden="true" />
            <h3>Latest Anomaly</h3>
          </div>
          <span className="sg-anomaly-notice">
            ● {streamMode === 'replay' ? 'REPLAY MODE' : 'LIVE BACKEND'}
          </span>
        </div>

        <div className="sg-latest-healthy-state">
          <div className="sg-healthy-badge">
            <CheckCircle2 size={32} className="text-optimal" aria-hidden="true" />
            <p className="sg-healthy-title">No recent anomalies detected.</p>
            <p className="sg-healthy-desc">
              All sensors are operating within expected meteorological tolerance baselines.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  // Map severity to StatusBadge
  let severityBadge: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (anomaly.severity === 'critical') severityBadge = 'critical';
  else if (anomaly.severity === 'high') severityBadge = 'high';
  else if (anomaly.severity === 'medium') severityBadge = 'moderate';

  const formattedTime = new Date(anomaly.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  const suggestedEntries = Object.entries(anomaly.suggested_values || {});

  return (
    <Card variant="glass" className={`sg-latest-anomaly-card sg-latest-anomaly-card--active ${className}`}>
      <div className="sg-latest-anomaly-header">
        <div className="sg-latest-title">
          <AlertTriangle size={18} className="text-critical" aria-hidden="true" />
          <h3>Latest Anomaly</h3>
        </div>
        <div className="sg-latest-badges">
          <StatusBadge status={severityBadge} label={`${anomaly.severity.toUpperCase()} SEVERITY`} size="sm" />
          <span className="sg-latest-score-pill">
            Score: {Math.round(anomaly.anomaly_score_pct)}%
          </span>
        </div>
      </div>

      <div className="sg-latest-anomaly-content">
        <div className="sg-latest-type-row">
          <span className="sg-latest-type-label">Anomaly Event:</span>
          <span className="sg-latest-type-val">{anomaly.type.replace('_', ' ').toUpperCase()}</span>
          <span className="sg-latest-timestamp">{formattedTime}</span>
        </div>

        <div className="sg-latest-field">
          <span className="sg-latest-field-label">Root Cause:</span>
          <p className="sg-latest-root-cause">{anomaly.root_cause}</p>
        </div>

        <div className="sg-latest-field">
          <span className="sg-latest-field-label">Description:</span>
          <p className="sg-latest-description">{anomaly.description}</p>
        </div>

        {anomaly.regime && (
          <div className="sg-latest-field">
            <span className="sg-latest-field-label">Regime:</span>
            <p className="sg-latest-description">{anomaly.regime.replace(/_/g, ' ')}</p>
          </div>
        )}

        {anomaly.network_corroboration && (
          <div className="sg-latest-field sg-latest-field--network">
            <span className="sg-latest-field-label">Network Evidence:</span>
            <div className="sg-latest-network-badge">
              <StatusBadge 
                status={anomaly.network_corroboration === 'LOCALIZED' ? 'warning' : anomaly.network_corroboration === 'REGIONAL' ? 'moderate' : 'optimal'} 
                label={anomaly.network_corroboration.replace(/_/g, ' ')} 
                size="sm" 
              />
            </div>
          </div>
        )}

        {anomaly.decision_basis && (
          <div className="sg-latest-field">
            <span className="sg-latest-field-label">Decision Basis:</span>
            <p className="sg-latest-description" style={{ fontSize: '0.75rem', letterSpacing: '0.05em', fontWeight: 600 }}>
              {anomaly.decision_basis.replace(/_/g, ' ')}
            </p>
          </div>
        )}

        {suggestedEntries.length > 0 && (
          <div className="sg-latest-field">
            <span className="sg-latest-field-label">Suggested replacement:</span>
            <p className="sg-latest-description">
              {suggestedEntries.map(([parameter, value]) =>
                `${parameter.replace(/_/g, ' ')} ${value}`).join(' · ')}
            </p>
          </div>
        )}

        <div className="sg-latest-footer">
          <span className="sg-anomaly-id">ID: {anomaly.anomaly_id}</span>
          <span className="sg-anomaly-notice">
            ● {streamMode === 'replay' ? 'REPLAY MODE' : 'LIVE BACKEND'}
          </span>
        </div>
      </div>
    </Card>
  );
};
