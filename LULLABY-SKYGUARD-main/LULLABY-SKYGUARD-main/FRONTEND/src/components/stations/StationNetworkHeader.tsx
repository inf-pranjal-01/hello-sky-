import React from 'react';
import { Network, RefreshCw } from 'lucide-react';
import { Station } from '../../types';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import './StationNetworkHeader.css';

export interface StationNetworkHeaderProps {
  selectedStation: Station | null;
  onRefresh: () => void;
  isLoading?: boolean;
}

export const StationNetworkHeader: React.FC<StationNetworkHeaderProps> = ({
  selectedStation,
  onRefresh,
  isLoading = false,
}) => {
  return (
    <header className="sg-station-header">
      <div className="sg-station-header__title-area">
        <div className="sg-station-header__title-row">
          <Network size={26} className="text-accent" aria-hidden="true" />
          <h1 className="sg-station-header__title">Station Network Map</h1>
        </div>
        <p className="sg-station-header__subtitle">
          Locator for every Automatic Weather Station. Zoom the map to separate overlapping sites.
        </p>
      </div>

      <div className="sg-station-header__controls">
        {selectedStation && (
          <div className="sg-station-header__status-group">
            <span className="sg-station-header__status-label">Selected status:</span>
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

        {/* Refresh button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          isLoading={isLoading}
          leftIcon={<RefreshCw size={14} />}
          ariaLabel="Refresh station network data"
        >
          Refresh
        </Button>
      </div>
    </header>
  );
};
