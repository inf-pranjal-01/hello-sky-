import React from 'react';
import { BarChart3, RefreshCw, MapPin } from 'lucide-react';
import { Station } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import './AnalyticsHeader.css';

export interface AnalyticsHeaderProps {
  station: Station | null | undefined;
  hours: number;
  onHoursChange: (hours: number) => void;
  staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE';
  onRefresh: () => void;
  isLoading?: boolean;
}

export const AnalyticsHeader: React.FC<AnalyticsHeaderProps> = ({
  station,
  hours,
  onHoursChange,
  staleStatusText,
  onRefresh,
  isLoading = false,
}) => {
  return (
    <header className="sg-analytics-header">
      <div className="sg-analytics-header__title-area">
        <div className="sg-analytics-header__title-row">
          <BarChart3 size={26} className="text-accent" aria-hidden="true" />
          <h1 className="sg-analytics-header__title">Meteorological Analytics & Insights</h1>
        </div>
        <p className="sg-analytics-header__subtitle">
          Historical telemetry aggregation, variance statistics, anomaly distribution, and derived
          operational intelligence.
        </p>
      </div>

      <div className="sg-analytics-header__controls">
        {station && (
          <div className="sg-analytics-header__station-badge">
            <MapPin size={14} className="text-accent" aria-hidden="true" />
            <span>{station.name} ({station.station_id})</span>
          </div>
        )}

        <StatusBadge
          status={
            staleStatusText === 'LIVE'
              ? 'optimal'
              : staleStatusText === 'DATA DELAYED'
              ? 'moderate'
              : 'critical'
          }
          label={staleStatusText}
          size="sm"
        />

        {/* Time-Range Selector */}
        <div
          className="sg-analytics-header__time-selector"
          role="group"
          aria-label="Analytics historical duration selector"
        >
          {[6, 12, 24].map((h) => (
            <button
              key={h}
              type="button"
              className={`sg-analytics-header__time-btn ${
                hours === h ? 'sg-analytics-header__time-btn--active' : ''
              }`}
              onClick={() => onHoursChange(h)}
              aria-pressed={hours === h}
            >
              {h}H
            </button>
          ))}
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          isLoading={isLoading}
          leftIcon={<RefreshCw size={14} />}
          ariaLabel="Refresh analytics data"
        >
          Refresh
        </Button>
      </div>
    </header>
  );
};
