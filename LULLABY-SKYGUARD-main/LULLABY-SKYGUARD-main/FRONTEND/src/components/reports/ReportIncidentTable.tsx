import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ClipboardList, ExternalLink } from 'lucide-react';
import { RecentAnomalyItem } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { formatSuggestedList, suggestedFromRecord } from '../../utils/suggestedValues';
import './ReportIncidentTable.css';

export interface ReportIncidentTableProps {
  incidents: RecentAnomalyItem[];
}

export const ReportIncidentTable: React.FC<ReportIncidentTableProps> = ({ incidents }) => {
  const navigate = useNavigate();

  return (
    <section className="sg-report-section" aria-labelledby="report-section-7-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <ClipboardList size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-7-heading" className="sg-report-section__title">
            7. Telemetry Incident Records & Root Cause Log
          </h2>
        </div>
        <span className="sg-report-section__badge">
          ● LIVE BACKEND
        </span>
      </div>

      {incidents.length === 0 ? (
        <p className="sg-report-incidents-empty">
          No anomaly incidents were recorded during this observation window.
        </p>
      ) : (
        <div className="sg-report-table-wrapper">
          <table
            className="sg-report-table"
            aria-label="Chronological list of detected telemetry anomalies"
          >
            <thead>
              <tr>
                <th scope="col">Incident ID</th>
                <th scope="col">Timestamp</th>
                <th scope="col">Severity</th>
                <th scope="col">Anomaly Type</th>
                <th scope="col">Anomaly Score</th>
                <th scope="col">Root Cause Analysis</th>
                <th scope="col">Affected Sensor / Raw Reading</th>
                <th scope="col">Suggested Replacement</th>
                <th scope="col" className="no-print">
                  <span className="sg-sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {incidents.map((item) => {
                const observed = suggestedFromRecord(item.observed_values);
                const suggested = suggestedFromRecord(item.suggested_values);
                return <tr key={item.anomaly_id}>
                  {/* ID */}
                  <td className="sg-font-mono text-muted">{item.anomaly_id}</td>

                  {/* Timestamp */}
                  <td className="sg-font-mono">
                    {new Date(item.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </td>

                  {/* Severity */}
                  <td>
                    <StatusBadge
                      status={
                        item.severity === 'critical'
                          ? 'critical'
                          : item.severity === 'high'
                          ? 'critical'
                          : item.severity === 'medium'
                          ? 'moderate'
                          : 'optimal'
                      }
                      label={item.severity.toUpperCase()}
                      size="sm"
                    />
                  </td>

                  {/* Type */}
                  <td className="sg-report-type-cell">
                    {item.type.replace(/_/g, ' ')}
                  </td>

                  {/* Score */}
                  <td className="sg-font-mono font-bold">
                    {item.anomaly_score_pct != null
                      ? `${item.anomaly_score_pct.toFixed(1)}%`
                      : '—'}
                  </td>

                  {/* Root Cause */}
                  <td className="sg-report-cause-cell">{item.root_cause || 'Not available'}</td>

                  <td className="sg-report-cause-cell">
                    {item.affected_parameters?.length
                      ? <><strong>{item.affected_parameters.map((parameter) => parameter.replace(/_/g, ' ')).join(', ')}</strong><br />{observed.length ? formatSuggestedList(observed) : 'Raw value unavailable'}</>
                      : 'Not available for earlier incident'}
                  </td>

                  <td className="sg-report-cause-cell">
                    {suggested.length ? formatSuggestedList(suggested) : 'Unavailable during baseline warm-up'}
                  </td>

                  {/* Action (no-print) */}
                  <td className="no-print">
                    <button
                      type="button"
                      className="sg-report-table-btn"
                      onClick={() => navigate('/alerts')}
                      aria-label={`Investigate incident ${item.anomaly_id} in Alerts`}
                    >
                      <span>Investigate</span>
                      <ExternalLink size={12} aria-hidden="true" />
                    </button>
                  </td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
