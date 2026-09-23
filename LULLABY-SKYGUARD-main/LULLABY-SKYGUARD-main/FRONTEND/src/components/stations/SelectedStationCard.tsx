import React from 'react';
import { useNavigate } from 'react-router-dom';
import { MapPin, Mountain, Globe, AlertTriangle, ArrowRight, Radio } from 'lucide-react';
import { Station, CurrentSensorReading, LatestAnomaly } from '../../types';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { formatCoordinates } from '../../utils/geospatial';
import './SelectedStationCard.css';

export interface SelectedStationCardProps {
  station: Station | null;
  currentReading: CurrentSensorReading | null;
  latestAnomaly: LatestAnomaly | null;
  isLoading?: boolean;
}

export const SelectedStationCard: React.FC<SelectedStationCardProps> = ({
  station,
  currentReading,
  latestAnomaly,
  isLoading = false,
}) => {
  const navigate = useNavigate();

  if (isLoading || !station) {
    return (
      <Card variant="glass" className="sg-selected-station-card">
        <div className="sg-selected-station-card__header">
          <Skeleton width="180px" height="1.4rem" />
          <Skeleton width="80px" height="1.5rem" />
        </div>
        <Skeleton width="100%" height="80px" style={{ marginTop: '1rem' }} />
      </Card>
    );
  }

  const coordsFormatted = formatCoordinates(station.lat, station.lon);

  return (
    <Card
      variant="glass"
      className="sg-selected-station-card"
      role="region"
      aria-label={`Selected station details: ${station.name}`}
    >
      <div className="sg-selected-station-card__header">
        <div className="sg-selected-station-card__identity">
          <div className="sg-selected-station-card__badge-row">
            <span className="sg-selected-station-card__selected-pill">
              <Radio size={12} className="text-accent animate-pulse" aria-hidden="true" />
              PRIMARY SELECTED OBSERVATORY
            </span>
            <span className="sg-selected-station-card__id">{station.station_id}</span>
          </div>
          <h2 className="sg-selected-station-card__name">{station.name}</h2>
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
          size="md"
        />
      </div>

      {/* Geospatial Metadata Grid */}
      <div className="sg-selected-station-card__meta-grid">
        <div className="sg-selected-station-card__meta-item">
          <MapPin size={15} className="text-accent" aria-hidden="true" />
          <span className="sg-selected-station-card__meta-label">Coordinates:</span>
          <span className="sg-selected-station-card__meta-value sg-font-mono">{coordsFormatted}</span>
        </div>

        {station.elevation_m != null && (
          <div className="sg-selected-station-card__meta-item">
            <Mountain size={15} className="text-accent" aria-hidden="true" />
            <span className="sg-selected-station-card__meta-label">Elevation:</span>
            <span className="sg-selected-station-card__meta-value sg-font-mono">
              {station.elevation_m} m ASL
            </span>
          </div>
        )}

        {station.region && (
          <div className="sg-selected-station-card__meta-item">
            <Globe size={15} className="text-accent" aria-hidden="true" />
            <span className="sg-selected-station-card__meta-label">Region:</span>
            <span className="sg-selected-station-card__meta-value">{station.region}</span>
          </div>
        )}
      </div>

      {/* Real-time Telemetry Snapshot */}
      {currentReading && (
        <div className="sg-selected-station-card__telemetry-row">
          <div className="sg-selected-station-card__telemetry-chip">
            <span className="sg-selected-station-card__chip-label">Temperature</span>
            <span className="sg-selected-station-card__chip-val sg-font-mono">
              {currentReading.temperature_c.value.toFixed(1)}°C
            </span>
            <span className="sg-selected-station-card__chip-sub">
              Nominal: {currentReading.temperature_c.normal_min}–{currentReading.temperature_c.normal_max}°C
            </span>
          </div>

          <div className="sg-selected-station-card__telemetry-chip">
            <span className="sg-selected-station-card__chip-label">Barometric Pressure</span>
            <span className="sg-selected-station-card__chip-val sg-font-mono">
              {currentReading.pressure_hpa.value.toFixed(1)} hPa
            </span>
            <span className="sg-selected-station-card__chip-sub">
              Nominal: {currentReading.pressure_hpa.normal_min}–{currentReading.pressure_hpa.normal_max} hPa
            </span>
          </div>

          <div className="sg-selected-station-card__telemetry-chip">
            <span className="sg-selected-station-card__chip-label">Relative Humidity</span>
            <span className="sg-selected-station-card__chip-val sg-font-mono">
              {currentReading.humidity_pct.value.toFixed(1)}%
            </span>
            <span className="sg-selected-station-card__chip-sub">
              Nominal: {currentReading.humidity_pct.normal_min}–{currentReading.humidity_pct.normal_max}%
            </span>
          </div>
        </div>
      )}

      {/* Latest Anomaly Context Banner */}
      {latestAnomaly && (
        <div className="sg-selected-station-card__anomaly-banner" role="alert">
          <div className="sg-selected-station-card__anomaly-left">
            <AlertTriangle size={18} className="text-warning" aria-hidden="true" />
            <div>
              <span className="sg-selected-station-card__anomaly-title">
                Active Incident Detected: {latestAnomaly.severity.toUpperCase()} ·{' '}
                {latestAnomaly.type.replace(/_/g, ' ')}
              </span>
              <p className="sg-selected-station-card__anomaly-desc">{latestAnomaly.description}</p>
            </div>
          </div>
          <button
            type="button"
            className="sg-selected-station-card__anomaly-action"
            onClick={() => navigate('/alerts')}
            aria-label="Investigate anomaly in Alerts view"
          >
            <span>Investigate</span>
            <ArrowRight size={14} aria-hidden="true" />
          </button>
        </div>
      )}
    </Card>
  );
};
