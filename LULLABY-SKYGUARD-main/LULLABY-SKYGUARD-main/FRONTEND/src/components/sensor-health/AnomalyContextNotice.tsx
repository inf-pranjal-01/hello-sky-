import React from 'react';
import { useNavigate } from 'react-router-dom';
import { HelpCircle, ArrowRight, AlertTriangle, ShieldCheck } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { LatestAnomaly } from '../../types';
import './AnomalyContextNotice.css';

export interface AnomalyContextNoticeProps {
  latestAnomaly: LatestAnomaly | null;
  className?: string;
}

export const AnomalyContextNotice: React.FC<AnomalyContextNoticeProps> = ({
  latestAnomaly,
  className = '',
}) => {
  const navigate = useNavigate();

  const hasAnomaly = !!latestAnomaly;
  let severityBadge: 'low' | 'moderate' | 'high' | 'critical' = 'low';
  if (latestAnomaly?.severity === 'critical') severityBadge = 'critical';
  else if (latestAnomaly?.severity === 'high') severityBadge = 'high';
  else if (latestAnomaly?.severity === 'medium') severityBadge = 'moderate';

  return (
    <div className={`sg-anomaly-context ${className}`} role="note" aria-label="System concept distinction">
      <div className="sg-anomaly-context__left">
        <HelpCircle size={18} className="sg-anomaly-context__icon" aria-hidden="true" />
        <div className="sg-anomaly-context__text">
          <h4 className="sg-anomaly-context__title">Understanding Sensor Health vs Anomaly Risk</h4>
          <p className="sg-anomaly-context__desc">
            <strong>Sensor Health</strong> assesses hardware integrity and physical telemetry reliability.
            <strong> Anomaly Risk</strong> evaluates whether incoming meteorological observations deviate
            from expected atmospheric patterns.
          </p>
        </div>
      </div>

      <div className="sg-anomaly-context__right">
        {hasAnomaly ? (
          <div className="sg-anomaly-context__badge-group">
            <AlertTriangle size={14} className="text-warning" aria-hidden="true" />
            <span>Active Anomaly:</span>
            <StatusBadge
              status={severityBadge}
              label={`${latestAnomaly.severity.toUpperCase()} (${Math.round(latestAnomaly.anomaly_score_pct)}%)`}
              size="sm"
            />
          </div>
        ) : (
          <div className="sg-anomaly-context__badge-group">
            <ShieldCheck size={14} className="text-optimal" aria-hidden="true" />
            <span>Anomaly Status: Nominal</span>
          </div>
        )}

        <button
          type="button"
          className="sg-anomaly-context__link-btn"
          onClick={() => navigate('/alerts')}
          aria-label="Navigate to Anomaly Alerts & Incidents console"
        >
          <span>View Alerts Console</span>
          <ArrowRight size={13} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
};
