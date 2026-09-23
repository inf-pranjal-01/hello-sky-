import React from 'react';
import { Activity, Cpu } from 'lucide-react';
import { SensorHealth, CurrentSensorReading } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import './ReportSensorHealthSection.css';

export interface ReportSensorHealthSectionProps {
  sensorHealth: SensorHealth | null;
  currentReading: CurrentSensorReading | null;
}

export const ReportSensorHealthSection: React.FC<ReportSensorHealthSectionProps> = ({
  sensorHealth,
  currentReading,
}) => {
  const healthPct = sensorHealth?.sensor_health_pct ?? currentReading?.sensor_health_pct ?? 100;
  const healthStatus =
    sensorHealth?.sensor_health_status ?? currentReading?.sensor_health_status ?? 'HEALTHY';

  return (
    <section className="sg-report-section" aria-labelledby="report-section-4-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <Activity size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-4-heading" className="sg-report-section__title">
            4. Sensor Hardware Condition & Reliability Index
          </h2>
        </div>
        <span className="sg-report-section__badge">
          ● LIVE BACKEND
        </span>
      </div>

      <div className="sg-report-health-layout">
        {/* Health Score Card */}
        <div className="sg-report-health-score-card">
          <div className="sg-report-health-score-header">
            <span className="sg-report-health-score-title">Subsystem Health Index</span>
            <StatusBadge
              status={
                healthStatus === 'HEALTHY'
                  ? 'optimal'
                  : healthStatus === 'WARNING'
                  ? 'moderate'
                  : healthStatus === 'CRITICAL'
                  ? 'critical'
                  : 'offline'
              }
              label={healthStatus}
              size="md"
            />
          </div>

          <div className="sg-report-health-score-body">
            <span className="sg-report-health-score-val sg-font-mono">{healthPct}%</span>
            <div className="sg-report-health-progress-track">
              <div
                className="sg-report-health-progress-bar"
                style={{
                  width: `${healthPct}%`,
                  background:
                    healthPct >= 80 ? '#10b981' : healthPct >= 50 ? '#f59e0b' : '#ef4444',
                }}
              />
            </div>
          </div>

          <p className="sg-report-health-score-desc">
            {healthPct >= 80
              ? 'Transducer components and electronic telemetry circuitry are operating within nominal specifications.'
              : healthPct >= 50
              ? 'Subsystem hardware exhibits moderate signal attenuation or calibration drift requiring monitoring.'
              : 'Severe hardware transducer degradation or transmission failure detected. Immediate maintenance recommended.'}
          </p>
        </div>

        {/* Hardware Diagnostics Scope Note */}
        <div className="sg-report-health-info-card">
          <div className="sg-report-health-info-header">
            <Cpu size={15} className="text-accent" aria-hidden="true" />
            <span className="sg-report-health-info-title">Hardware Telemetry Scope</span>
          </div>

          <p className="sg-report-health-info-text">
            <strong>Monitored Sensor Channels:</strong> Ambient Thermal Transducer (°C), Piezoresistive
            Barometer (hPa), and Capacitive Polymer Hygrometer (%).
          </p>

          <p className="sg-report-health-info-subtext">
            <em>Note: Detailed telemetry for battery reserve voltage, solar panel wattage, and RF signal strength (RSSI) are reserved for future backend API contract extensions.</em>
          </p>
        </div>
      </div>
    </section>
  );
};
