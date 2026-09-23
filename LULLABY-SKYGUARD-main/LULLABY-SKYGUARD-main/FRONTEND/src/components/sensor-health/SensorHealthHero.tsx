import React from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon, Radio, Wrench, Activity, Zap } from 'lucide-react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { SensorHealth, RepairSensorResponse } from '../../types';
import './SensorHealthHero.css';

export interface SensorHealthHeroProps {
  health: SensorHealth | null;
  lastUpdated?: Date | null;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  onRepair?: () => Promise<void>;
  onForceRecover?: () => void;
  isRepairing?: boolean;
  repairResult?: RepairSensorResponse | null;
  repairError?: string | null;
  className?: string;
}

export const SensorHealthHero: React.FC<SensorHealthHeroProps> = ({
  health,
  lastUpdated,
  isLoading = false,
  error = null,
  onRetry,
  onRepair,
  onForceRecover,
  isRepairing = false,
  repairResult = null,
  repairError = null,
  className = '',
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-health-hero ${className}`}>
        <div className="sg-health-hero__container">
          <div className="sg-health-hero__main">
            <Skeleton width="120px" height="120px" borderRadius="50%" />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', flex: 1 }}>
              <Skeleton width="180px" height="1.5rem" />
              <Skeleton width="90%" height="2rem" />
            </div>
          </div>
          <Skeleton width="160px" height="1.5rem" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-health-hero sg-health-hero--critical ${className}`}>
        <div className="sg-health-hero__title-row">
          <AlertOctagon size={22} className="text-critical" />
          <h2 className="sg-health-hero__title">Sensor Health Monitor</h2>
        </div>
        <p style={{ color: 'var(--color-status-critical, #ef4444)', margin: '0.75rem 0' }}>
          {error}
        </p>
        {onRetry && (
          <button type="button" className="sg-btn sg-btn--outline sg-btn--sm" onClick={onRetry}>
            Retry Sensor Health Feed
          </button>
        )}
      </Card>
    );
  }

  const score = health ? health.sensor_health_pct : 0;
  const status = health ? health.sensor_health_status : 'OFFLINE';

  const isHealthy = status === 'HEALTHY';
  const isWarning = status === 'WARNING';
  const isCritical = status === 'CRITICAL';
  const isOffline = status === 'OFFLINE';

  const heroModifier = isHealthy
    ? 'sg-health-hero--healthy'
    : isWarning
    ? 'sg-health-hero--warning'
    : isCritical
    ? 'sg-health-hero--critical'
    : 'sg-health-hero--offline';

  const indicatorClass = isHealthy
    ? 'sg-health-hero__indicator--healthy'
    : isWarning
    ? 'sg-health-hero__indicator--warning'
    : isCritical
    ? 'sg-health-hero__indicator--critical'
    : 'sg-health-hero__indicator--offline';

  // Human-facing operational summary copy
  let statusCopy = 'All meteorological sensor channels operating within nominal hardware tolerance.';
  if (isWarning) {
    statusCopy = 'Potential sensor reliability concern — Telemetry variance or baseline drift detected.';
  } else if (isCritical) {
    statusCopy = 'Immediate sensor attention required — Sensor signal degraded or out of bounds.';
  } else if (isOffline) {
    statusCopy = 'No current sensor data available — Station telemetry disconnected.';
  }

  // SVG Circular Gauge calculation (radius = 50, circumference = 2 * PI * 50 = 314.159)
  const radius = 50;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const accessibleText = `Sensor health is ${score} percent. Current status is ${status}. ${statusCopy}`;

  const formattedTime = lastUpdated
    ? lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : 'Pending';

  return (
    <Card
      variant="glass"
      className={`sg-health-hero ${heroModifier} ${className}`}
      role="region"
      aria-label="Sensor health overview"
    >
      <div className="sg-health-hero__container">
        <div className="sg-health-hero__main">
          {/* Circular Progress Gauge */}
          <div
            className="sg-health-hero__meter"
            role="progressbar"
            aria-valuenow={score}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={accessibleText}
          >
            <svg className="sg-health-hero__svg" viewBox="0 0 120 120" aria-hidden="true">
              <circle className="sg-health-hero__track" cx="60" cy="60" r={radius} />
              <circle
                className={`sg-health-hero__indicator ${indicatorClass}`}
                cx="60"
                cy="60"
                r={radius}
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
              />
            </svg>
            <div className="sg-health-hero__meter-value">
              <span className="sg-health-hero__score">{score}%</span>
              <span className="sg-health-hero__score-unit">Health</span>
            </div>
          </div>

          {/* Operational Status & Description */}
          <div className="sg-health-hero__details">
            <div className="sg-health-hero__title-row">
              {isHealthy && <ShieldCheck size={22} className="text-optimal" aria-hidden="true" />}
              {isWarning && <AlertTriangle size={22} className="text-warning" aria-hidden="true" />}
              {isCritical && <AlertOctagon size={22} className="text-critical" aria-hidden="true" />}
              {isOffline && <Radio size={22} className="text-muted" aria-hidden="true" />}
              <h2 className="sg-health-hero__title">Sensor Subsystem Health</h2>
              <StatusBadge
                status={
                  isHealthy ? 'optimal' : isWarning ? 'moderate' : isCritical ? 'critical' : 'offline'
                }
                label={status}
                size="md"
              />
            </div>
            <p className="sg-health-hero__description">{statusCopy}</p>

            {/* Repair / Recovery Action & Status */}
            {(onRepair || onForceRecover) && (
              <div className="sg-health-hero__repair-section">
                <div className="sg-health-hero__repair-actions">
                  {onRepair && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={onRepair}
                      isLoading={isRepairing}
                      disabled={isRepairing || isLoading || !health}
                      leftIcon={<Wrench size={14} />}
                      ariaLabel="Confirm physical repair and start gradual recovery"
                    >
                      {isRepairing ? 'Working...' : 'Mark Repaired'}
                    </Button>
                  )}
                  {onForceRecover && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onForceRecover}
                      disabled={isRepairing || isLoading || !health}
                      leftIcon={<Zap size={14} />}
                      ariaLabel="Force clear sensor health immediately"
                    >
                      Force Recovery
                    </Button>
                  )}
                </div>
                <p className="sg-health-hero__repair-hint">
                  Mark Repaired waits for 3 clean readings. Force Recovery instantly clears a stuck sensor.
                </p>

                {repairResult && (
                  <div className="sg-health-hero__recovery-notice" role="status">
                    <div className="sg-health-hero__recovery-header">
                      <Activity size={14} className="text-warning" />
                      <strong>Recovery Active:</strong>
                      <StatusBadge
                        status={
                          repairResult.status === 'HEALTHY'
                            ? 'optimal'
                            : repairResult.status === 'WARNING'
                            ? 'moderate'
                            : 'critical'
                        }
                        label={repairResult.status}
                        size="sm"
                      />
                    </div>
                    <p className="sg-health-hero__recovery-msg">{repairResult.message}</p>
                  </div>
                )}

                {repairError && (
                  <p className="sg-health-hero__repair-error" role="alert">
                    {repairError}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Aside / Metadata */}
        <div className="sg-health-hero__aside">
          <span className="sg-health-hero__timestamp">Last Check: {formattedTime}</span>
          <span className="sg-health-hero__notice">
            ● LIVE BACKEND
          </span>
        </div>
      </div>
    </Card>
  );
};
