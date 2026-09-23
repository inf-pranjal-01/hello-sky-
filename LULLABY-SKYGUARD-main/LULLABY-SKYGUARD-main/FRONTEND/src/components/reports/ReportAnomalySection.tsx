import React from 'react';
import { AlertTriangle } from 'lucide-react';
import {
  SeverityDistribution,
  AnomalyTypeDistribution,
  AnomalySeverity,
  LatestAnomaly,
} from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import './ReportAnomalySection.css';

export interface ReportAnomalySectionProps {
  total: number;
  highestSeverity: AnomalySeverity | 'none';
  severityDistribution: SeverityDistribution;
  typeDistribution: AnomalyTypeDistribution;
  latestAnomaly: LatestAnomaly | null;
}

export const ReportAnomalySection: React.FC<ReportAnomalySectionProps> = ({
  total,
  highestSeverity,
  severityDistribution,
  typeDistribution,
  latestAnomaly,
}) => {
  const totalSev =
    severityDistribution.critical +
    severityDistribution.high +
    severityDistribution.medium +
    severityDistribution.low;

  const totalTypes =
    typeDistribution.spike +
    typeDistribution.frozen_value +
    typeDistribution.drift +
    typeDistribution.dropout +
    typeDistribution.multivariate_inconsistency;

  return (
    <section className="sg-report-section" aria-labelledby="report-section-3-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <AlertTriangle size={18} className="text-warning" aria-hidden="true" />
          <h2 id="report-section-3-heading" className="sg-report-section__title">
            3. Anomaly Detection & Incident Classification
          </h2>
        </div>
        <span className="sg-report-section__badge">
          DERIVED FROM LIVE BACKEND DATA
        </span>
      </div>

      {/* Summary KPI Banner */}
      <div className="sg-report-anomaly-kpi-row">
        <div className="sg-report-anomaly-kpi">
          <span className="sg-report-anomaly-kpi-label">Total Incidents</span>
          <span className="sg-report-anomaly-kpi-val sg-font-mono">{total}</span>
        </div>

        <div className="sg-report-anomaly-kpi">
          <span className="sg-report-anomaly-kpi-label">Highest Observed Severity</span>
          <div style={{ marginTop: '0.25rem' }}>
            <StatusBadge
              status={
                highestSeverity === 'critical'
                  ? 'critical'
                  : highestSeverity === 'high'
                  ? 'critical'
                  : highestSeverity === 'medium'
                  ? 'moderate'
                  : highestSeverity === 'low'
                  ? 'optimal'
                  : 'optimal'
              }
              label={highestSeverity.toUpperCase()}
              size="sm"
            />
          </div>
        </div>

        {latestAnomaly && (
          <div className="sg-report-anomaly-kpi">
            <span className="sg-report-anomaly-kpi-label">Latest Incident Classification</span>
            <span className="sg-report-anomaly-kpi-val text-accent">
              {latestAnomaly.type.replace(/_/g, ' ')} ({latestAnomaly.anomaly_score_pct.toFixed(1)}%)
            </span>
          </div>
        )}
        {latestAnomaly && Object.keys(latestAnomaly.suggested_values || {}).length > 0 && (
          <div className="sg-report-anomaly-kpi">
            <span className="sg-report-anomaly-kpi-label">Suggested Replacement</span>
            <span className="sg-report-anomaly-kpi-val text-accent">
              {Object.entries(latestAnomaly.suggested_values || {}).map(([parameter, value]) =>
                `${parameter.replace(/_/g, ' ')} ${value}`).join(' · ')}
            </span>
          </div>
        )}
      </div>

      {/* 2-Column Severity vs Type Distribution */}
      <div className="sg-report-distribution-grid">
        {/* Severity Distribution */}
        <div className="sg-report-dist-card">
          <h3 className="sg-report-subheading">Severity Frequency Distribution</h3>
          {totalSev === 0 ? (
            <p className="sg-report-dist-empty">No anomaly events observed in this period.</p>
          ) : (
            <div className="sg-report-dist-list">
              {[
                { label: 'Critical', count: severityDistribution.critical, color: '#ef4444' },
                { label: 'High', count: severityDistribution.high, color: '#f97316' },
                { label: 'Medium', count: severityDistribution.medium, color: '#f59e0b' },
                { label: 'Low', count: severityDistribution.low, color: '#10b981' },
              ].map((item) => {
                const pct = totalSev > 0 ? Math.round((item.count / totalSev) * 100) : 0;
                return (
                  <div key={item.label} className="sg-report-dist-row">
                    <div className="sg-report-dist-meta">
                      <span>{item.label}</span>
                      <span className="sg-font-mono font-bold">{item.count}</span>
                    </div>
                    <div className="sg-report-dist-track">
                      <div
                        className="sg-report-dist-bar"
                        style={{ width: `${pct}%`, background: item.color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Type Distribution */}
        <div className="sg-report-dist-card">
          <h3 className="sg-report-subheading">Signal Anomaly Type Breakdown</h3>
          {totalTypes === 0 ? (
            <p className="sg-report-dist-empty">No anomaly types recorded in this period.</p>
          ) : (
            <div className="sg-report-dist-list">
              {[
                { label: 'Spike', count: typeDistribution.spike },
                { label: 'Frozen Value', count: typeDistribution.frozen_value },
                { label: 'Drift', count: typeDistribution.drift },
                { label: 'Dropout', count: typeDistribution.dropout },
                {
                  label: 'Multivariate Inconsistency',
                  count: typeDistribution.multivariate_inconsistency,
                },
              ].map((item) => {
                const pct = totalTypes > 0 ? Math.round((item.count / totalTypes) * 100) : 0;
                return (
                  <div key={item.label} className="sg-report-dist-row">
                    <div className="sg-report-dist-meta">
                      <span>{item.label}</span>
                      <span className="sg-font-mono font-bold">{item.count}</span>
                    </div>
                    <div className="sg-report-dist-track">
                      <div
                        className="sg-report-dist-bar"
                        style={{ width: `${pct}%`, background: '#38bdf8' }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
