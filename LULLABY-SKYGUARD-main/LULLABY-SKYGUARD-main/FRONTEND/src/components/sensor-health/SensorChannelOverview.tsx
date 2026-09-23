import React from 'react';
import { Thermometer, Gauge, Droplets, CheckCircle2, Radio } from 'lucide-react';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { CurrentSensorReading, SensorHealth } from '../../types';
import './SensorChannelOverview.css';

export interface SensorChannelOverviewProps {
  reading: CurrentSensorReading | null;
  health: SensorHealth | null;
  isLoading?: boolean;
  className?: string;
}

export const SensorChannelOverview: React.FC<SensorChannelOverviewProps> = ({
  reading,
  health,
  isLoading = false,
  className = '',
}) => {
  if (isLoading) {
    return (
      <div className={`sg-channels-section ${className}`} aria-label="Loading sensor channels">
        <div className="sg-channels-section__header">
          <Skeleton width="220px" height="1.4rem" />
          <Skeleton width="180px" height="1.1rem" />
        </div>
        <div className="sg-channels-grid">
          {[1, 2, 3].map((idx) => (
            <Card key={idx} variant="glass" className="sg-channel-card">
              <Skeleton width="120px" height="1.2rem" />
              <Skeleton width="80px" height="2rem" />
              <Skeleton width="100%" height="1.2rem" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const overallStatus = health?.sensor_health_status ?? 'HEALTHY';
  const isHealthy = overallStatus === 'HEALTHY';
  const isWarning = overallStatus === 'WARNING';
  const isCritical = overallStatus === 'CRITICAL';

  let badgeType: 'optimal' | 'moderate' | 'critical' | 'offline' = 'optimal';
  if (isWarning) badgeType = 'moderate';
  else if (isCritical) badgeType = 'critical';
  else if (overallStatus === 'OFFLINE') badgeType = 'offline';

  const channels = [
    {
      id: 'temp',
      name: 'Temperature Sensor',
      tech: 'Platinum RTD (PT1000)',
      icon: <Thermometer size={18} className="text-warning" />,
      value: reading?.temperature_c.value.toFixed(1) ?? '--',
      unit: '°C',
      min: reading?.temperature_c.normal_min ?? 15,
      max: reading?.temperature_c.normal_max ?? 35,
    },
    {
      id: 'press',
      name: 'Barometric Pressure',
      tech: 'Piezoresistive Transducer',
      icon: <Gauge size={18} className="text-accent" />,
      value: reading?.pressure_hpa.value.toFixed(1) ?? '--',
      unit: 'hPa',
      min: reading?.pressure_hpa.normal_min ?? 980,
      max: reading?.pressure_hpa.normal_max ?? 1030,
    },
    {
      id: 'hum',
      name: 'Relative Humidity',
      tech: 'Capacitive Thin-Film Hygrometer',
      icon: <Droplets size={18} className="text-optimal" />,
      value: reading?.humidity_pct.value.toFixed(1) ?? '--',
      unit: '%',
      min: reading?.humidity_pct.normal_min ?? 30,
      max: reading?.humidity_pct.normal_max ?? 90,
    },
  ];

  const formattedTime = reading?.timestamp
    ? (() => {
        const d = new Date(reading.timestamp);
        if (isNaN(d.getTime())) return '--';
        const dateStr = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
        const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        return `${dateStr}, ${timeStr}`;
      })()
    : 'Live Ingestion';

  return (
    <section className={`sg-channels-section ${className}`} role="region" aria-label="Monitored sensor channels">
      <div className="sg-channels-section__header">
        <div className="sg-channels-section__title-group">
          <Radio size={19} className="text-accent" aria-hidden="true" />
          <h3 className="sg-channels-section__title">Monitored Sensor Channels</h3>
        </div>
        <span className="sg-channels-section__notice">
          ● CONTINUOUS TELEMETRY AUDIT
        </span>
      </div>

      {/* 3 Channel Cards */}
      <div className="sg-channels-grid">
        {channels.map((ch) => (
          <Card key={ch.id} variant="glass" className="sg-channel-card">
            <div className="sg-channel-card__header">
              <div className="sg-channel-card__channel-name">
                {ch.icon}
                <div>
                  <span className="sg-channel-card__title">{ch.name}</span>
                  <span className="sg-channel-card__tech">{ch.tech}</span>
                </div>
              </div>
              <StatusBadge status={badgeType} label={overallStatus} size="sm" />
            </div>

            <div className="sg-channel-card__reading">
              <span className="sg-channel-card__val">{ch.value}</span>
              <span className="sg-channel-card__unit">{ch.unit}</span>
            </div>

            <div className="sg-channel-card__range-row">
              <span className="sg-channel-card__range-label">Baseline Operating Range:</span>
              <span className="sg-channel-card__range-val">
                {ch.min} {ch.unit} – {ch.max} {ch.unit}
              </span>
            </div>

            <div className="sg-channel-card__footer">
              <CheckCircle2 size={13} className={isHealthy ? 'text-optimal' : 'text-warning'} />
              <span>{isHealthy ? 'Signal Stability: Nominal (99.8%)' : 'Channel Audited under System Health'}</span>
            </div>
          </Card>
        ))}
      </div>

      {/* Operational Table */}
      <Card variant="glass" className="sg-channels-table-card">
        <table className="sg-channels-table" aria-label="Sensor channels operational status table">
          <thead>
            <tr>
              <th scope="col">Sensor Channel & Architecture</th>
              <th scope="col">Current Reading</th>
              <th scope="col">Baseline Operating Range</th>
              <th scope="col">Operational Status</th>
              <th scope="col">Last Feed Update</th>
            </tr>
          </thead>
          <tbody>
            {channels.map((ch) => (
              <tr key={ch.id}>
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', fontWeight: 500 }}>
                    {ch.icon}
                    <div>
                      <span style={{ display: 'block', color: 'var(--color-text-primary, #f8fafc)' }}>{ch.name}</span>
                      <span style={{ display: 'block', fontSize: '0.72rem', color: 'var(--color-text-muted, #94a3b8)' }}>{ch.tech}</span>
                    </div>
                  </div>
                </td>
                <td className="sg-font-mono" style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                  {ch.value} {ch.unit}
                </td>
                <td className="sg-font-mono" style={{ color: 'var(--color-text-secondary, #cbd5e1)' }}>
                  {ch.min} – {ch.max} {ch.unit}
                </td>
                <td>
                  <StatusBadge status={badgeType} label={overallStatus} size="sm" />
                </td>
                <td className="sg-font-mono" style={{ color: 'var(--color-text-muted, #94a3b8)', fontSize: '0.78rem' }}>
                  {formattedTime}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </section>
  );
};
