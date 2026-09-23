import React from 'react';
import { useNavigate } from 'react-router-dom';
import { History, ArrowRight } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { RecentAnomalyItem } from '../../types';
import './RecentAnomaliesCard.css';

export interface RecentAnomaliesCardProps {
  anomalies?: RecentAnomalyItem[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  className?: string;
  streamMode?: 'live' | 'replay';
}

export const RecentAnomaliesCard: React.FC<RecentAnomaliesCardProps> = ({
  anomalies = [],
  isLoading = false,
  error = null,
  onRetry,
  className = '',
  streamMode = 'live',
}) => {
  const navigate = useNavigate();

  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-recent-card ${className}`}>
        <div className="sg-recent-header">
          <Skeleton width="180px" height="1.2rem" />
          <Skeleton width="100px" height="1.2rem" />
        </div>
        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <Skeleton width="100%" height="2rem" />
          <Skeleton width="100%" height="2rem" />
          <Skeleton width="100%" height="2rem" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-recent-card ${className}`}>
        <div className="sg-recent-header">
          <h3>Recent Anomalies</h3>
        </div>
        <div className="sg-recent-error">
          <p>{error}</p>
          {onRetry && (
            <button type="button" className="sg-recent-retry-btn" onClick={onRetry}>
              Retry
            </button>
          )}
        </div>
      </Card>
    );
  }

  return (
    <Card variant="glass" className={`sg-recent-card ${className}`}>
      <div className="sg-recent-header">
        <div className="sg-recent-title">
          <History size={18} className="text-accent" aria-hidden="true" />
          <h3>Recent Anomalies</h3>
        </div>
        <button
          type="button"
          className="sg-view-all-btn"
          onClick={() => navigate('/alerts')}
          aria-label="View all station anomalies on alerts console"
        >
          <span>View All Anomalies</span>
          <ArrowRight size={14} aria-hidden="true" />
        </button>
      </div>

      <div className="sg-recent-body">
        {anomalies.length === 0 ? (
          <div className="sg-recent-empty">
            <p>No anomalies recorded for this station.</p>
          </div>
        ) : (
          <div className="sg-recent-table-wrapper">
            <table className="sg-recent-table" aria-label="Recent anomalies log summary">
              <thead>
                <tr>
                  <th scope="col">Time</th>
                  <th scope="col">Severity</th>
                  <th scope="col">Type</th>
                  <th scope="col">Score</th>
                  <th scope="col">Suggested value</th>
                  <th scope="col">Root Cause</th>
                </tr>
              </thead>
              <tbody>
                {anomalies.map((anom) => {
                  let badgeSev: 'low' | 'moderate' | 'high' | 'critical' = 'low';
                  if (anom.severity === 'critical') badgeSev = 'critical';
                  else if (anom.severity === 'high') badgeSev = 'high';
                  else if (anom.severity === 'medium') badgeSev = 'moderate';

                  return (
                    <tr key={anom.anomaly_id}>
                      <td className="sg-font-mono">
                        {new Date(anom.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </td>
                      <td>
                        <StatusBadge status={badgeSev} label={anom.severity.toUpperCase()} size="sm" />
                      </td>
                      <td className="sg-anom-type">{anom.type.replace('_', ' ')}</td>
                      <td className="sg-font-mono">
                        <span className={anom.anomaly_score_pct > 75 ? 'text-critical' : 'text-warning'}>
                          {Math.round(anom.anomaly_score_pct)}%
                        </span>
                      </td>
                      <td className="sg-anom-cause">
                        {Object.entries(anom.suggested_values || {}).length > 0
                          ? Object.entries(anom.suggested_values || {}).map(([parameter, value]) =>
                            `${parameter.replace(/_/g, ' ')} ${value}`).join(' · ')
                          : 'Baseline unavailable'}
                      </td>
                      <td className="sg-anom-cause" title={anom.description}>
                        {anom.root_cause}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="sg-recent-footer">
        <span className="sg-recent-notice">
          ● {streamMode === 'replay' ? 'REPLAY MODE' : 'LIVE BACKEND'}
        </span>
      </div>
    </Card>
  );
};
