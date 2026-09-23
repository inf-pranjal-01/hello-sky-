import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList, ExternalLink } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { StatusBadge } from '../common/StatusBadge';
import { EmptyState } from '../common/EmptyState';
import { RecentAnomalyItem, AnomalySeverity } from '../../types';
import './AnomalyTimeline.css';

export interface AnomalyTimelineProps {
  anomalies: RecentAnomalyItem[];
  isLoading?: boolean;
  className?: string;
}

/**
 * Maps anomaly_type snake_case to a readable label.
 */
function formatAnomalyType(type: string): string {
  const map: Record<string, string> = {
    spike: 'Spike',
    frozen_value: 'Frozen Value',
    drift: 'Drift',
    dropout: 'Dropout',
    multivariate_inconsistency: 'Multivariate',
  };
  return map[type] ?? type.replace(/_/g, ' ');
}

/**
 * Maps AnomalySeverity to StatusBadge status values.
 * AnomalySeverity is lowercase: 'critical' | 'high' | 'medium' | 'low'
 */
function severityToStatus(
  severity: AnomalySeverity
): 'critical' | 'high' | 'moderate' | 'low' | 'optimal' {
  switch (severity) {
    case 'critical':
      return 'critical';
    case 'high':
      return 'high';
    case 'medium':
      return 'moderate';
    case 'low':
      return 'low';
    default:
      return 'low';
  }
}

const AnomalyTimelineSkeleton: React.FC = () => (
  <Card variant="glass" className="sg-anomaly-timeline-card">
    <div className="sg-anomaly-timeline__header">
      <Skeleton width="180px" height="1.2rem" />
    </div>
    <div className="sg-anomaly-timeline__table-wrapper">
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} className="sg-anomaly-timeline__skeleton-row">
          <Skeleton width="110px" height="0.8rem" />
          <Skeleton width="70px" height="1.4rem" />
          <Skeleton width="90px" height="0.8rem" />
          <Skeleton width="50px" height="0.8rem" />
          <Skeleton width="24px" height="24px" borderRadius="50%" />
        </div>
      ))}
    </div>
  </Card>
);

// [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
export const AnomalyTimeline: React.FC<AnomalyTimelineProps> = ({
  anomalies,
  isLoading = false,
  className = '',
}) => {
  const navigate = useNavigate();

  if (isLoading) return <AnomalyTimelineSkeleton />;

  return (
    <Card
      variant="glass"
      className={`sg-anomaly-timeline-card ${className}`}
      role="region"
      aria-label="Anomaly timeline"
    >
      <div className="sg-anomaly-timeline__header">
        <div className="sg-anomaly-timeline__title-group">
          <ClipboardList size={17} aria-hidden="true" />
          <h3 className="sg-anomaly-timeline__title">Anomaly Timeline</h3>
        </div>
        <span className="sg-anomaly-timeline__notice">[FRONTEND ONLY] [DERIVED FROM EXISTING DATA]</span>
      </div>

      {anomalies.length === 0 ? (
        <EmptyState
          title="No Anomalies Recorded"
          description="No anomaly events were detected during this observation window."
          actionLabel="Go to Alerts"
          onAction={() => navigate('/alerts')}
        />
      ) : (
        <div className="sg-anomaly-timeline__table-wrapper">
          <table className="sg-anomaly-timeline__table" aria-label="List of recent anomaly events">
            <thead>
              <tr>
                <th scope="col">Timestamp</th>
                <th scope="col">Severity</th>
                <th scope="col">Type</th>
                <th scope="col">Score</th>
                <th scope="col">
                  <span className="sg-sr-only">Open in Alerts</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {anomalies.map((anomaly) => (
                <tr
                  key={anomaly.anomaly_id}
                  className="sg-anomaly-timeline__row"
                  onClick={() => navigate('/alerts')}
                  tabIndex={0}
                  role="button"
                  aria-label={`View anomaly ${anomaly.anomaly_id} in Alerts`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      navigate('/alerts');
                    }
                  }}
                >
                  <td className="sg-anomaly-timeline__cell--time">
                    <span title={new Date(anomaly.timestamp).toLocaleString()}>
                      {new Date(anomaly.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}
                    </span>
                  </td>
                  <td>
                    <StatusBadge status={severityToStatus(anomaly.severity)} />
                  </td>
                  <td className="sg-anomaly-timeline__cell--type">
                    {formatAnomalyType(anomaly.type)}
                  </td>
                  <td className="sg-anomaly-timeline__cell--score sg-font-mono">
                    {anomaly.anomaly_score_pct != null
                      ? `${anomaly.anomaly_score_pct.toFixed(1)}%`
                      : '—'}
                  </td>
                  <td>
                    <ExternalLink
                      size={14}
                      className="sg-anomaly-timeline__link-icon"
                      aria-hidden="true"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
};
