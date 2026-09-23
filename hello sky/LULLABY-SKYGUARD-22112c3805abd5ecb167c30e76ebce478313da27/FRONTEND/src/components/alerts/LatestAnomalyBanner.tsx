import React from 'react';
import { AlertTriangle, ShieldCheck, ArrowRight, Eye, RefreshCw, BrainCircuit } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import { Skeleton } from '../common/Skeleton';
import { LatestAnomaly } from '../../types';
import './LatestAnomalyBanner.css';

export interface LatestAnomalyBannerProps {
  latestAnomaly: LatestAnomaly | null;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onInvestigate: (anomaly: LatestAnomaly) => void;
  onOpenShap?: (anomaly: LatestAnomaly) => void;
  className?: string;
}

export const LatestAnomalyBanner: React.FC<LatestAnomalyBannerProps> = ({
  latestAnomaly,
  isLoading = false,
  error = null,
  onRetry,
  onInvestigate,
  onOpenShap,
  className = '',
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-latest-banner ${className}`}>
        <div className="sg-latest-banner__header">
          <Skeleton width="200px" height="1.4rem" />
          <Skeleton width="120px" height="1.4rem" />
        </div>
        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <Skeleton width="100%" height="2rem" />
          <Skeleton width="60%" height="1.5rem" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-latest-banner sg-latest-banner--critical ${className}`}>
        <div className="sg-latest-banner__header">
          <div className="sg-latest-banner__title-group">
            <AlertTriangle size={20} className="text-critical" />
            <h3 className="sg-latest-banner__title">Latest Anomaly Stream</h3>
          </div>
          <Button variant="outline" size="sm" onClick={onRetry} leftIcon={<RefreshCw size={14} />}>
            Retry Feed
          </Button>
        </div>
        <p style={{ color: 'var(--color-status-critical, #ef4444)', margin: '0.5rem 0 0 0' }}>
          {error}
        </p>
      </Card>
    );
  }

  if (!latestAnomaly) {
    return (
      <Card variant="glass" className={`sg-latest-banner sg-latest-banner--nominal ${className}`}>
        <div className="sg-latest-banner__header">
          <div className="sg-latest-banner__title-group">
            <ShieldCheck size={22} className="text-optimal" />
            <h3 className="sg-latest-banner__title">Live Telemetry Status: Nominal</h3>
          </div>
          <StatusBadge status="optimal" label="NO ACTIVE ANOMALY" size="md" />
        </div>
        <div className="sg-latest-banner__nominal-state">
          <div className="sg-latest-banner__nominal-text">
            <h4>All meteorological sensors operating within baseline tolerance.</h4>
            <p>
              Autonomous anomaly detection models are continuously inspecting temperature, pressure,
              and humidity streams.
            </p>
          </div>
        </div>
        <span className="sg-latest-banner__notice">
          ● LIVE BACKEND
        </span>
      </Card>
    );
  }

  const isCritical = latestAnomaly.severity === 'critical';
  const bannerClass = isCritical
    ? 'sg-latest-banner--critical'
    : latestAnomaly.severity === 'high'
    ? 'sg-latest-banner--high'
    : '';

  const formattedTime = new Date(latestAnomaly.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  const suggestedReading = Object.entries(latestAnomaly.suggested_values ?? {})
    .map(([parameter, value]) => `${parameter.replace(/_/g, ' ')}: ${value.toFixed(2)}`)
    .join(' · ');

  return (
    <Card variant="glass" className={`sg-latest-banner ${bannerClass} ${className}`}>
      <div className="sg-latest-banner__header">
        <div className="sg-latest-banner__title-group">
          <AlertTriangle size={22} className={isCritical ? 'text-critical' : 'text-warning'} />
          <h3 className="sg-latest-banner__title">
            Anomaly Detected: {latestAnomaly.type.replace('_', ' ')}
          </h3>
        </div>
        <div className="sg-latest-banner__badges">
          <StatusBadge
            status={
              latestAnomaly.severity === 'critical'
                ? 'critical'
                : latestAnomaly.severity === 'high'
                ? 'high'
                : latestAnomaly.severity === 'medium'
                ? 'moderate'
                : 'low'
            }
            label={`${latestAnomaly.severity.toUpperCase()} SEVERITY`}
            size="md"
          />
          <span className="sg-latest-banner__score">
            Evidence Strength: {Math.round(latestAnomaly.anomaly_score_pct)}%
          </span>
        </div>
      </div>

      <div className="sg-latest-banner__grid">
        <div className="sg-latest-banner__block">
          <span className="sg-latest-banner__block-label">Event Timestamp</span>
          <p className="sg-latest-banner__block-value">{formattedTime} (ID: {latestAnomaly.anomaly_id})</p>
        </div>

        <div className="sg-latest-banner__block">
          <span className="sg-latest-banner__block-label">Anomaly Indication</span>
          <p className="sg-latest-banner__block-value sg-latest-banner__root-cause">
            {latestAnomaly.root_cause}
          </p>
        </div>

        {suggestedReading && (
          <div className="sg-latest-banner__block">
            <span className="sg-latest-banner__block-label">Suggested Replacement</span>
            <p className="sg-latest-banner__block-value">{suggestedReading}</p>
          </div>
        )}

        <div className="sg-latest-banner__action" style={{ display: 'flex', gap: '0.65rem', flexWrap: 'wrap' }}>
          {onOpenShap && (
            <Button
              variant="outline"
              size="md"
              onClick={() => onOpenShap(latestAnomaly)}
              leftIcon={<BrainCircuit size={16} />}
              style={{ borderColor: 'rgba(56, 189, 248, 0.45)', color: '#38bdf8' }}
              ariaLabel="Open Decision X-Ray and SHAP explanation for this anomaly"
            >
              Decision X-Ray (SHAP)
            </Button>
          )}
          <Button
            variant="primary"
            size="md"
            onClick={() => onInvestigate(latestAnomaly)}
            leftIcon={<Eye size={16} />}
            rightIcon={<ArrowRight size={16} />}
          >
            Investigate Incident
          </Button>
        </div>
      </div>

      <span className="sg-latest-banner__notice">
        ● LIVE BACKEND
      </span>
    </Card>
  );
};
