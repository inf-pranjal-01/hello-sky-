import React from 'react';
import { Building2, Shield, Activity, Radio } from 'lucide-react';
import { Station, CurrentSensorReading, SensorHealth } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { formatCoordinates } from '../../utils/geospatial';
import './ReportStationOverview.css';

export interface ReportStationOverviewProps {
  station: Station;
  currentReading: CurrentSensorReading | null;
  sensorHealth: SensorHealth | null;
}

export const ReportStationOverview: React.FC<ReportStationOverviewProps> = ({
  station,
  currentReading,
  sensorHealth,
}) => {
  const coordsFormatted = formatCoordinates(station.lat, station.lon);

  const healthPct = sensorHealth?.sensor_health_pct ?? currentReading?.sensor_health_pct ?? 100;
  const healthStatus =
    sensorHealth?.sensor_health_status ?? currentReading?.sensor_health_status ?? 'HEALTHY';
  const riskLevel = currentReading?.risk_level ?? 'low';

  return (
    <section className="sg-report-section" aria-labelledby="report-section-1-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <Building2 size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-1-heading" className="sg-report-section__title">
            1. Observatory Station Overview
          </h2>
        </div>
        <span className="sg-report-section__badge">● LIVE BACKEND</span>
      </div>

      {/* Grid of Station Metadata & Status Indices */}
      <div className="sg-report-station-grid">
        {/* Identity details */}
        <div className="sg-report-station-card">
          <h3 className="sg-report-subheading">Observatory Identification</h3>
          <div className="sg-report-meta-list">
            <div className="sg-report-meta-row">
              <span className="sg-report-meta-key">Station Name:</span>
              <span className="sg-report-meta-val font-semibold">{station.name}</span>
            </div>
            <div className="sg-report-meta-row">
              <span className="sg-report-meta-key">Station ID:</span>
              <span className="sg-report-meta-val sg-font-mono">{station.station_id}</span>
            </div>
            <div className="sg-report-meta-row">
              <span className="sg-report-meta-key">Coordinates:</span>
              <span className="sg-report-meta-val sg-font-mono">{coordsFormatted}</span>
            </div>
            {station.elevation_m != null && (
              <div className="sg-report-meta-row">
                <span className="sg-report-meta-key">Elevation:</span>
                <span className="sg-report-meta-val sg-font-mono">{station.elevation_m} m ASL</span>
              </div>
            )}
            {station.region && (
              <div className="sg-report-meta-row">
                <span className="sg-report-meta-key">Geographic Region:</span>
                <span className="sg-report-meta-val">{station.region}</span>
              </div>
            )}
          </div>
        </div>

        {/* Operational Status Summary (Distinct 3-pillar indices) */}
        <div className="sg-report-station-card">
          <h3 className="sg-report-subheading">Operational Health & Risk Indices</h3>
          <div className="sg-report-status-pills">
            {/* System Status */}
            <div className="sg-report-status-pill-item">
              <div className="sg-report-status-pill-info">
                <Shield size={14} className="text-accent" aria-hidden="true" />
                <span className="sg-report-status-pill-label">System Operational Status</span>
              </div>
              <StatusBadge
                status={
                  station.status === 'NORMAL'
                    ? 'optimal'
                    : station.status === 'WARNING'
                    ? 'moderate'
                    : station.status === 'CRITICAL'
                    ? 'critical'
                    : 'offline'
                }
                label={station.status}
                size="sm"
              />
            </div>

            {/* Sensor Hardware Health */}
            <div className="sg-report-status-pill-item">
              <div className="sg-report-status-pill-info">
                <Activity size={14} className="text-accent" aria-hidden="true" />
                <span className="sg-report-status-pill-label">
                  Hardware Sensor Health ({healthPct}%)
                </span>
              </div>
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
                size="sm"
              />
            </div>

            {/* Anomaly Risk Level */}
            <div className="sg-report-status-pill-item">
              <div className="sg-report-status-pill-info">
                <Radio size={14} className="text-accent" aria-hidden="true" />
                <span className="sg-report-status-pill-label">Active Anomaly Risk Level</span>
              </div>
              <StatusBadge
                status={
                  riskLevel === 'critical'
                    ? 'critical'
                    : riskLevel === 'high'
                    ? 'critical'
                    : riskLevel === 'medium'
                    ? 'moderate'
                    : 'optimal'
                }
                label={riskLevel.toUpperCase()}
                size="sm"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
