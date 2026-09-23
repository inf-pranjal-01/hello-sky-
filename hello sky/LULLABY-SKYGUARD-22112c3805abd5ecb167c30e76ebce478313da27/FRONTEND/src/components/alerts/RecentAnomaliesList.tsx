import React from 'react';
import { History, Eye, CheckCircle2, BrainCircuit } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import { ErrorState } from '../common/ErrorState';
import { RecentAnomalyItem } from '../../types';
import { formatSuggestedList, suggestedFromRecord } from '../../utils/suggestedValues';
import './RecentAnomaliesList.css';

export interface RecentAnomaliesListProps {
  anomalies: RecentAnomalyItem[];
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onSelectAnomaly: (anomaly: RecentAnomalyItem) => void;
  onOpenShap?: (anomaly: RecentAnomalyItem) => void;
  isFiltered?: boolean;
  onClearFilters?: () => void;
  className?: string;
}

export const RecentAnomaliesList: React.FC<RecentAnomaliesListProps> = ({
  anomalies = [],
  isLoading = false,
  error = null,
  onRetry,
  onSelectAnomaly,
  onOpenShap,
  isFiltered = false,
  onClearFilters,
  className = '',
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-alerts-table-card ${className}`}>
        <div className="sg-alerts-table-header">
          <Skeleton width="180px" height="1.2rem" />
          <Skeleton width="100px" height="1.2rem" />
        </div>
        <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <Skeleton width="100%" height="2.5rem" />
          <Skeleton width="100%" height="2.5rem" />
          <Skeleton width="100%" height="2.5rem" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-alerts-table-card ${className}`}>
        <ErrorState
          title="Failed to Load Anomaly History"
          message={error}
          onRetry={onRetry}
        />
      </Card>
    );
  }

  return (
    <Card variant="glass" className={`sg-alerts-table-card ${className}`}>
      <div className="sg-alerts-table-header">
        <div className="sg-alerts-table-title">
          <History size={18} className="text-accent" aria-hidden="true" />
          <h3>Station Anomaly Event History</h3>
        </div>
        <span className="sg-alerts-table-notice">
          ● LIVE BACKEND
        </span>
      </div>

      <div className="sg-alerts-table-container">
        {anomalies.length === 0 ? (
          <div className="sg-alerts-table-empty">
            {isFiltered ? (
              <EmptyState
                title="No Matching Anomalies"
                description="No recorded anomalies match your selected filters. Try broadening your criteria or reset filters."
                actionLabel="Clear Filters"
                onAction={onClearFilters}
              />
            ) : (
              <EmptyState
                icon={<CheckCircle2 size={40} className="text-optimal" />}
                title="No Recent Anomalies"
                description="This weather station currently has no logged anomalies in the recent monitoring window."
              />
            )}
          </div>
        ) : (
          <table className="sg-alerts-table" aria-label="Recent anomalies log">
            <thead>
              <tr>
                <th scope="col">Event ID</th>
                <th scope="col">Station</th>
                <th scope="col">Timestamp</th>
                <th scope="col">Severity</th>
                <th scope="col">Classification Type</th>
                <th scope="col">Anomaly Score</th>
                <th scope="col">Anomaly Indication</th>
                <th scope="col">Affected Sensor / Raw Reading</th>
                <th scope="col">Suggested Replacement</th>
                <th scope="col" style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.map((anom) => {
                let badgeSev: 'low' | 'moderate' | 'high' | 'critical' = 'low';
                if (anom.severity === 'critical') badgeSev = 'critical';
                else if (anom.severity === 'high') badgeSev = 'high';
                else if (anom.severity === 'medium') badgeSev = 'moderate';

                const formattedTime = new Date(anom.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                });
                const formattedDate = new Date(anom.timestamp).toLocaleDateString([], {
                  month: 'short',
                  day: 'numeric',
                });
                const observed = suggestedFromRecord(anom.observed_values);
                const suggested = suggestedFromRecord(anom.suggested_values);

                return (
                  <tr
                    key={anom.anomaly_id}
                    className="sg-alerts-row-clickable"
                    onClick={() => {
                      if (onOpenShap) {
                        onOpenShap(anom);
                      } else {
                        onSelectAnomaly(anom);
                      }
                    }}
                    title="Click row to open Decision X-Ray & SHAP explanation"
                  >
                    <td className="sg-alerts-table-id">
                      <span className="sg-alerts-id-link">{anom.anomaly_id}</span>
                    </td>
                    <td className="sg-alerts-table-station">
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-secondary, #94a3b8)' }}>
                        {anom.station_id}
                      </span>
                    </td>
                    <td className="sg-alerts-table-time" title={anom.timestamp}>
                      {formattedDate} {formattedTime}
                    </td>
                    <td>
                      <StatusBadge status={badgeSev} label={anom.severity.toUpperCase()} size="sm" />
                    </td>
                    <td className="sg-alerts-table-type">
                      {anom.type.replace('_', ' ')}
                    </td>
                    <td className="sg-alerts-table-score">
                      <span className={anom.anomaly_score_pct > 75 ? 'text-critical' : 'text-warning'}>
                        {Math.round(anom.anomaly_score_pct)}%
                      </span>
                    </td>
                    <td className="sg-alerts-table-cause" title={anom.description}>
                      {anom.root_cause}
                    </td>
                    <td className="sg-alerts-table-cause">
                      {anom.affected_parameters?.length
                        ? <><strong>{anom.affected_parameters.map((parameter) => parameter.replace(/_/g, ' ')).join(', ')}</strong><br />{observed.length ? formatSuggestedList(observed) : 'Raw value unavailable'}</>
                        : 'Not available for earlier incident'}
                    </td>
                    <td className="sg-alerts-table-cause">
                      {suggested.length ? formatSuggestedList(suggested) : 'Unavailable during baseline warm-up'}
                    </td>
                    <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                      {onOpenShap && (
                        <button
                          type="button"
                          className="sg-alerts-investigate-btn"
                          style={{
                            marginRight: '6px',
                            background: 'rgba(56, 189, 248, 0.12)',
                            borderColor: 'rgba(56, 189, 248, 0.35)',
                            color: '#38bdf8'
                          }}
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenShap(anom);
                          }}
                          aria-label={`View Decision X-Ray / SHAP for incident ${anom.anomaly_id}`}
                          title="Open Decision X-Ray & SHAP explanation"
                        >
                          <BrainCircuit size={13} aria-hidden="true" />
                          <span>SHAP X-Ray</span>
                        </button>
                      )}
                      <button
                        type="button"
                        className="sg-alerts-investigate-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectAnomaly(anom);
                        }}
                        aria-label={`Investigate incident ${anom.anomaly_id}`}
                        title="Open incident investigation modal"
                      >
                        <Eye size={13} aria-hidden="true" />
                        <span>Investigate</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="sg-alerts-table-footer">
        <span className="sg-alerts-table-notice">
          Showing {anomalies.length} anomaly records
        </span>
      </div>
    </Card>
  );
};
