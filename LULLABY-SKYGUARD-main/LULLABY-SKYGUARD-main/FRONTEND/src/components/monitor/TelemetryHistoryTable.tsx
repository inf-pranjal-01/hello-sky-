import React from 'react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { TelemetryHistoryRecord } from '../../types';
import './TelemetryHistoryTable.css';

export interface TelemetryHistoryTableProps {
  records?: TelemetryHistoryRecord[];
  isLoading?: boolean;
  isPaused?: boolean;
  className?: string;
}

export const TelemetryHistoryTable: React.FC<TelemetryHistoryTableProps> = ({
  records = [],
  isLoading = false,
  isPaused = false,
  className = '',
}) => {
  if (isLoading && records.length === 0) {
    return (
      <Card variant="glass" className={`sg-telemetry-history-card ${className}`}>
        <div className="sg-history-header">
          <Skeleton width="180px" height="1.4rem" />
          <Skeleton width="120px" height="1.2rem" />
        </div>
        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          <Skeleton width="100%" height="2.2rem" />
          <Skeleton width="100%" height="2.2rem" />
          <Skeleton width="100%" height="2.2rem" />
          <Skeleton width="100%" height="2.2rem" />
          <Skeleton width="100%" height="2.2rem" />
        </div>
      </Card>
    );
  }

  return (
    <Card variant="glass" className={`sg-telemetry-history-card ${className}`}>
      <div className="sg-history-header">
        <div className="sg-history-title-group">
          <h3 className="sg-history-title">Recent Telemetry</h3>
          <span className="sg-history-sub">
            Continuous sequence buffer ({records.length} readings recorded)
          </span>
        </div>
        <div className="sg-history-badges">
          {isPaused ? (
            <span className="sg-history-paused-pill">BUFFER PAUSED</span>
          ) : (
            <span className="sg-history-live-pill">BUFFER ACTIVE</span>
          )}
          <span className="sg-history-notice">
            [FRONTEND ONLY — DERIVED FROM CURRENT READINGS & TRENDS]
          </span>
        </div>
      </div>

      <div className="sg-history-body">
        {records.length === 0 ? (
          <div className="sg-history-empty">
            <p>No telemetry data available.</p>
          </div>
        ) : (
          <div className="sg-history-table-container">
            <table className="sg-history-table" aria-label="Real-time sensor telemetry history stream">
              <thead>
                <tr>
                  <th scope="col">Timestamp</th>
                  <th scope="col">Temperature (°C)</th>
                  <th scope="col">Pressure (hPa)</th>
                  <th scope="col">Humidity (%)</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec, index) => {
                  const formattedTime = new Date(rec.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                  });

                  let statusBadgeType: 'optimal' | 'warning' | 'critical' | 'offline' = 'optimal';
                  if (rec.status === 'WARNING') statusBadgeType = 'warning';
                  else if (rec.status === 'CRITICAL') statusBadgeType = 'critical';
                  else if (rec.status === 'OFFLINE') statusBadgeType = 'offline';

                  return (
                    <tr
                      key={rec.id || index}
                      className={index === 0 && !isPaused ? 'sg-history-row--latest' : ''}
                    >
                      <td className="sg-font-mono sg-history-time-cell">
                        {formattedTime}
                        {index === 0 && !isPaused && (
                          <span className="sg-latest-tag" title="Most recent polled sample">
                            NEW
                          </span>
                        )}
                      </td>
                      <td className="sg-font-mono">{rec.temperature_c.toFixed(1)} °C</td>
                      <td className="sg-font-mono">{rec.pressure_hpa.toFixed(1)} hPa</td>
                      <td className="sg-font-mono">{rec.humidity_pct.toFixed(1)} %</td>
                      <td>
                        <StatusBadge status={statusBadgeType} label={rec.status} size="sm" />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="sg-history-footer">
        <span className="sg-history-limit-note">
          Displaying up to 150 live samples in local buffer memory (● LIVE BACKEND telemetry stream)
        </span>
      </div>
    </Card>
  );
};
