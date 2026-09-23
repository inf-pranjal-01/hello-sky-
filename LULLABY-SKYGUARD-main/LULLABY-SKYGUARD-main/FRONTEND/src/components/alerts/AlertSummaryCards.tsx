import React from 'react';
import { AlertTriangle, ShieldCheck, Activity, Flame, Layers } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { LatestAnomaly, RecentAnomalyItem } from '../../types';
import './AlertSummaryCards.css';

export interface AlertSummaryCardsProps {
  latestAnomaly: LatestAnomaly | null;
  recentAnomalies: RecentAnomalyItem[];
  isLoading?: boolean;
}

export const AlertSummaryCards: React.FC<AlertSummaryCardsProps> = ({
  latestAnomaly,
  recentAnomalies,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="sg-alert-summary-grid" aria-label="Loading anomaly summary indicators">
        {[1, 2, 3, 4].map((idx) => (
          <Card key={idx} variant="glass" className="sg-alert-summary-card">
            <div className="sg-alert-summary-card__header">
              <Skeleton width="90px" height="0.9rem" />
              <Skeleton width="28px" height="28px" borderRadius="50%" />
            </div>
            <div className="sg-alert-summary-card__body">
              <Skeleton width="70%" height="1.8rem" />
            </div>
          </Card>
        ))}
      </div>
    );
  }

  // Determine highest severity among latest or recent
  let highestSeverity: 'critical' | 'high' | 'medium' | 'low' | 'none' = 'none';
  if (latestAnomaly) {
    highestSeverity = latestAnomaly.severity;
  } else if (recentAnomalies.length > 0) {
    const severities = recentAnomalies.map((a) => a.severity);
    if (severities.includes('critical')) highestSeverity = 'critical';
    else if (severities.includes('high')) highestSeverity = 'high';
    else if (severities.includes('medium')) highestSeverity = 'medium';
    else highestSeverity = 'low';
  }

  const hasActiveAnomaly = !!latestAnomaly;
  const scorePct = latestAnomaly ? Math.round(latestAnomaly.anomaly_score_pct) : 0;
  const totalAlerts = recentAnomalies.length;

  return (
    <div className="sg-alert-summary-grid" role="region" aria-label="Anomaly summary statistics">
      {/* 1. Active Anomaly Status */}
      <Card variant="glass" className="sg-alert-summary-card">
        <div className="sg-alert-summary-card__header">
          <span className="sg-alert-summary-card__label">Active State</span>
          <div
            className={`sg-alert-summary-card__icon ${
              hasActiveAnomaly ? 'sg-alert-summary-card__icon--critical' : 'sg-alert-summary-card__icon--optimal'
            }`}
          >
            {hasActiveAnomaly ? <AlertTriangle size={16} /> : <ShieldCheck size={16} />}
          </div>
        </div>
        <div className="sg-alert-summary-card__body">
          <div className="sg-alert-summary-card__value" style={{ fontSize: '1.25rem' }}>
            {hasActiveAnomaly ? latestAnomaly.type.replace('_', ' ').toUpperCase() : 'NOMINAL'}
          </div>
          <StatusBadge
            status={hasActiveAnomaly ? (latestAnomaly.severity === 'critical' ? 'critical' : 'high') : 'optimal'}
            label={hasActiveAnomaly ? 'TRIGGERED' : 'CLEAR'}
            size="sm"
          />
        </div>
      </Card>

      {/* 2. Highest Severity */}
      <Card variant="glass" className="sg-alert-summary-card">
        <div className="sg-alert-summary-card__header">
          <span className="sg-alert-summary-card__label">Highest Severity</span>
          <div
            className={`sg-alert-summary-card__icon ${
              highestSeverity === 'critical'
                ? 'sg-alert-summary-card__icon--critical'
                : highestSeverity === 'high' || highestSeverity === 'medium'
                ? 'sg-alert-summary-card__icon--warning'
                : 'sg-alert-summary-card__icon--optimal'
            }`}
          >
            <Flame size={16} />
          </div>
        </div>
        <div className="sg-alert-summary-card__body">
          <div className="sg-alert-summary-card__value" style={{ textTransform: 'uppercase' }}>
            {highestSeverity === 'none' ? 'NONE' : highestSeverity}
          </div>
          {highestSeverity !== 'none' && (
            <StatusBadge
              status={
                highestSeverity === 'critical'
                  ? 'critical'
                  : highestSeverity === 'high'
                  ? 'high'
                  : highestSeverity === 'medium'
                  ? 'moderate'
                  : 'low'
              }
              label={highestSeverity.toUpperCase()}
              size="sm"
            />
          )}
        </div>
      </Card>

      {/* 3. Total Recent Alerts */}
      <Card variant="glass" className="sg-alert-summary-card">
        <div className="sg-alert-summary-card__header">
          <span className="sg-alert-summary-card__label">Recent Alerts</span>
          <div className="sg-alert-summary-card__icon">
            <Layers size={16} />
          </div>
        </div>
        <div className="sg-alert-summary-card__body">
          <div className="sg-alert-summary-card__value">{totalAlerts}</div>
          <span className="sg-alert-summary-card__subtitle">events logged</span>
        </div>
      </Card>

      {/* 4. Latest Anomaly Score */}
      <Card variant="glass" className="sg-alert-summary-card">
        <div className="sg-alert-summary-card__header">
          <span className="sg-alert-summary-card__label">Anomaly Score</span>
          <div
            className={`sg-alert-summary-card__icon ${
              scorePct > 75
                ? 'sg-alert-summary-card__icon--critical'
                : scorePct > 40
                ? 'sg-alert-summary-card__icon--warning'
                : 'sg-alert-summary-card__icon--optimal'
            }`}
          >
            <Activity size={16} />
          </div>
        </div>
        <div className="sg-alert-summary-card__body">
          <div className="sg-alert-summary-card__value">
            <span
              className={
                scorePct > 75 ? 'text-critical' : scorePct > 40 ? 'text-warning' : 'text-optimal'
              }
            >
              {scorePct}%
            </span>
          </div>
          <span className="sg-alert-summary-card__subtitle">
            {hasActiveAnomaly ? 'Latest anomaly' : 'Baseline'}
          </span>
        </div>
      </Card>
    </div>
  );
};
