import React from 'react';
import { useStationNetworkData } from '../hooks/useStationNetworkData';
import {
  StationNetworkHeader,
  SelectedStationCard,
  NetworkOverview,
} from '../components/stations';
import { EmptyState } from '../components/common/EmptyState';
import { Card } from '../components/common/Card';
import './StationsPage.css';

export const StationsPage: React.FC = () => {
  const {
    selectedStation,
    stations,
    selectStation,
    currentReading,
    latestAnomaly,
    isLoading,
    error,
    refresh,
  } = useStationNetworkData();

  if (!isLoading && !selectedStation) {
    return (
      <div className="page-container sg-stations-page" role="main" aria-label="Station network page">
        <div className="sg-stations-page__no-station">
          <EmptyState
            title="No Station Selected"
            description="Please select a meteorological station from the navigation bar to view the network map."
          />
        </div>
      </div>
    );
  }

  if (error && !isLoading) {
    return (
      <div className="page-container sg-stations-page" role="main" aria-label="Station network page">
        <StationNetworkHeader selectedStation={selectedStation} onRefresh={refresh} isLoading={false} />
        <div className="sg-stations-page__error">
          <div className="sg-stations-page__error-box" role="alert">
            <h2>Station Network Unavailable</h2>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container sg-stations-page" role="main" aria-label="Station Network Map">
      <StationNetworkHeader
        selectedStation={selectedStation}
        onRefresh={refresh}
        isLoading={isLoading}
      />

      <SelectedStationCard
        station={selectedStation}
        currentReading={currentReading}
        latestAnomaly={latestAnomaly}
        isLoading={isLoading}
      />

      <Card variant="glass" className="sg-stations-spatial-pause">
        <p>
          Spatial neighbor validation is paused. Stations in this network are more than 10 km apart,
          so distance-based cross-checks are not used operationally. The map below is a locator only.
        </p>
      </Card>

      <NetworkOverview
        stations={stations}
        selectedStation={selectedStation}
        onSelectStation={selectStation}
        isLoading={isLoading}
      />
    </div>
  );
};

export default StationsPage;
