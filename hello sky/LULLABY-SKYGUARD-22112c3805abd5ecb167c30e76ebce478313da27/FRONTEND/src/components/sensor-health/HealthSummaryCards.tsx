import React from 'react';
import { Activity, HardDrive, Clock, Wifi } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { SensorHealth, CurrentSensorReading } from '../../types';
import './HealthSummaryCards.css';

export interface HealthSummaryCardsProps {
  health: SensorHealth | null;
  reading: CurrentSensorReading | null;
  staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE';
  isLoading?: boolean;
}

export const HealthSummaryCards: React.FC<HealthSummaryCardsProps> = ({
  health,
  reading,
  staleStatusText,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="sg-health-cards-grid" aria-label="Loading sensor health summary indicators">
        {[1, 2, 3, 4].map((idx) => (
          <Card key={idx} variant="glass" className="sg-health-summary-card">
            <div className="sg-health-summary-card__header">
              <Skeleton width="100px" height="0.9rem" />
              <Skeleton width="28px" height="28px" borderRadius="50%" />
            </div>
            <div className="sg-health-summary-card__body">
              <Skeleton width="60%" height="1.8rem" />
            </div>
          </Card>
        ))}
      </div>
    );
  }

  const scorePct = health ? health.sensor_health_pct : 0;
  const status = health ? health.sensor_health_status : 'OFFLINE';

  const isHealthy = status === 'HEALTHY';
  const isWarning = status === 'WARNING';

  const formattedTimestamp = reading?.timestamp
    ? new Date(reading.timestamp).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })
    : 'No Feed';

  return (
    <div className="sg-health-cards-grid" role="region" aria-label="Sensor health metric summary">
      {/* 1. Overall Health Score */}
      <Card variant="glass" className="sg-health-summary-card">
        <div className="sg-health-summary-card__header">
          <span className="sg-health-summary-card__label">Health Score</span>
          <div
            className={`sg-health-summary-card__icon ${
              isHealthy
                ? 'sg-health-summary-card__icon--healthy'
                : isWarning
                ? 'sg-health-summary-card__icon--warning'
                : 'sg-health-summary-card__icon--critical'
            }`}
          >
            <Activity size={16} />
          </div>
        </div>
        <div className="sg-health-summary-card__body">
          <div className="sg-health-summary-card__value">
            <span
              className={
                isHealthy ? 'text-optimal' : isWarning ? 'text-warning' : 'text-critical'
              }
            >
              {scorePct}%
            </span>
          </div>
          <span className="sg-health-summary-card__subtitle">Hardware Index</span>
        </div>
      </Card>

      {/* 2. Hardware Status */}
      <Card variant="glass" className="sg-health-summary-card">
        <div className="sg-health-summary-card__header">
          <span className="sg-health-summary-card__label">Condition</span>
          <div
            className={`sg-health-summary-card__icon ${
              isHealthy
                ? 'sg-health-summary-card__icon--healthy'
                : isWarning
                ? 'sg-health-summary-card__icon--warning'
                : 'sg-health-summary-card__icon--critical'
            }`}
          >
            <HardDrive size={16} />
          </div>
        </div>
        <div className="sg-health-summary-card__body">
          <div className="sg-health-summary-card__value" style={{ fontSize: '1.25rem' }}>
            <span className={isHealthy ? 'text-optimal' : isWarning ? 'text-warning' : 'text-critical'}>
              {status}
            </span>
          </div>
          <span className="sg-health-summary-card__subtitle">
            {isHealthy ? 'Subsystem Nominal' : isWarning ? 'Degradation Detected' : 'Offline / Fault'}
          </span>
        </div>
      </Card>

      {/* 3. Last Reading */}
      <Card variant="glass" className="sg-health-summary-card">
        <div className="sg-health-summary-card__header">
          <span className="sg-health-summary-card__label">Last Reading</span>
          <div className="sg-health-summary-card__icon">
            <Clock size={16} />
          </div>
        </div>
        <div className="sg-health-summary-card__body">
          <div className="sg-health-summary-card__value" style={{ fontSize: '1.125rem' }}>
            {formattedTimestamp}
          </div>
          <span className="sg-health-summary-card__subtitle">Telemetry Sync</span>
        </div>
      </Card>

      {/* 4. Data Freshness */}
      <Card variant="glass" className="sg-health-summary-card">
        <div className="sg-health-summary-card__header">
          <span className="sg-health-summary-card__label">Data Freshness</span>
          <div className="sg-health-summary-card__icon">
            <Wifi size={16} />
          </div>
        </div>
        <div className="sg-health-summary-card__body">
          <div className="sg-health-summary-card__value" style={{ fontSize: '1.125rem' }}>
            {staleStatusText}
          </div>
          <span className="sg-health-summary-card__subtitle">
            {staleStatusText === 'LIVE' ? 'Continuous Ingestion' : 'Polling Sync'}
          </span>
        </div>
      </Card>
    </div>
  );
};
