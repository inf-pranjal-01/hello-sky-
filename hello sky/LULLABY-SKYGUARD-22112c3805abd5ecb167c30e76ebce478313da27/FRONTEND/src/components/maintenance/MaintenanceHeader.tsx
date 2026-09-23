import React from 'react';
import { Wrench, RefreshCw, MapPin } from 'lucide-react';
import { Station } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import './MaintenanceHeader.css';

export interface MaintenanceHeaderProps {
  selectedStation: Station | null;
  onRefresh: () => void;
  isLoading?: boolean;
}

export const MaintenanceHeader: React.FC<MaintenanceHeaderProps> = ({
  selectedStation,
  onRefresh,
  isLoading = false,
}) => {
  return (
    <header className="sg-maintenance-header">
      <div className="sg-maintenance-header__title-area">
        <div className="sg-maintenance-header__title-row">
          <Wrench size={26} className="text-accent" aria-hidden="true" />
          <h1 className="sg-maintenance-header__title">Maintenance Operations</h1>
        </div>
        <p className="sg-maintenance-header__subtitle">
          Incident triage and field maintenance ticket dispatch for anomalous meteorological sensor
          transducers.
        </p>
      </div>

      <div className="sg-maintenance-header__controls">
        {selectedStation && (
          <div className="sg-maintenance-header__station-badge">
            <MapPin size={14} className="text-accent" aria-hidden="true" />
            <span className="sg-maintenance-header__station-text">
              {selectedStation.name} ({selectedStation.station_id})
            </span>
            <StatusBadge
              status={
                selectedStation.status === 'NORMAL'
                  ? 'optimal'
                  : selectedStation.status === 'WARNING'
                  ? 'moderate'
                  : selectedStation.status === 'CRITICAL'
                  ? 'critical'
                  : 'offline'
              }
              label={selectedStation.status}
              size="sm"
            />
          </div>
        )}

        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          isLoading={isLoading}
          leftIcon={<RefreshCw size={14} />}
          ariaLabel="Refresh station anomalies"
        >
          Refresh Feed
        </Button>
      </div>
    </header>
  );
};
