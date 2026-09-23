import React from 'react';
import { Table, Radio, Filter } from 'lucide-react';
import { Station, NeighborStationItem } from '../../types';
import { StationFilterStatus } from '../../hooks/useStationNetworkData';
import { Card } from '../common/Card';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import { formatDistance } from '../../utils/geospatial';
import './NeighborStationTable.css';

export interface NeighborStationTableProps {
  neighbors: NeighborStationItem[];
  filteredNeighbors: NeighborStationItem[];
  statusFilter: StationFilterStatus;
  onFilterChange: (filter: StationFilterStatus) => void;
  onSelectStation: (station: Station) => void;
  isLoading?: boolean;
}

export const NeighborStationTable: React.FC<NeighborStationTableProps> = ({
  neighbors,
  filteredNeighbors,
  statusFilter,
  onFilterChange,
  onSelectStation,
  isLoading = false,
}) => {
  const filterOptions: Array<{ id: StationFilterStatus; label: string }> = [
    { id: 'ALL', label: `All (${neighbors.length})` },
    { id: 'NORMAL', label: 'Normal' },
    { id: 'WARNING', label: 'Warning' },
    { id: 'CRITICAL', label: 'Critical' },
    { id: 'OFFLINE', label: 'Offline' },
  ];

  if (isLoading) {
    return (
      <Card variant="glass" className="sg-neighbor-table-card">
        <div className="sg-neighbor-table__header">
          <Skeleton width="180px" height="1.3rem" />
          <Skeleton width="240px" height="1.8rem" />
        </div>
        <div className="sg-neighbor-table__skeleton-list">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} width="100%" height="48px" style={{ marginBottom: '0.5rem' }} />
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card
      variant="glass"
      className="sg-neighbor-table-card"
      role="region"
      aria-label="Neighboring weather stations telemetry table"
    >
      <div className="sg-neighbor-table__header">
        <div className="sg-neighbor-table__title-area">
          <div className="sg-neighbor-table__title-row">
            <Table size={18} className="text-accent" aria-hidden="true" />
            <h3 className="sg-neighbor-table__title">Neighboring AWS Telemetry & Distance</h3>
          </div>
          <span className="sg-neighbor-table__notice">
            DEMO SPATIAL COMPARISON — DISTANCES DERIVED FROM COORDINATES
          </span>
        </div>

        {/* Status Filter Tabs */}
        <div
          className="sg-neighbor-table__filter-group"
          role="tablist"
          aria-label="Filter neighboring stations by status"
        >
          <Filter size={13} className="text-muted" aria-hidden="true" />
          {filterOptions.map((opt) => (
            <button
              key={opt.id}
              type="button"
              role="tab"
              aria-selected={statusFilter === opt.id}
              className={`sg-neighbor-table__filter-btn ${
                statusFilter === opt.id ? 'sg-neighbor-table__filter-btn--active' : ''
              }`}
              onClick={() => onFilterChange(opt.id)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {filteredNeighbors.length === 0 ? (
        <EmptyState
          title="No Matching Stations"
          description={
            statusFilter !== 'ALL'
              ? `No neighboring stations match the '${statusFilter}' status filter.`
              : 'No neighboring stations are currently available in the network.'
          }
          actionLabel={statusFilter !== 'ALL' ? 'Reset Filters' : undefined}
          onAction={statusFilter !== 'ALL' ? () => onFilterChange('ALL') : undefined}
        />
      ) : (
        <div className="sg-neighbor-table__wrapper">
          <table
            className="sg-neighbor-table"
            aria-label="Neighboring weather stations ranked by proximity"
          >
            <thead>
              <tr>
                <th scope="col">Station Identifier</th>
                <th scope="col">Geodesic Distance</th>
                <th scope="col">Temperature</th>
                <th scope="col">Pressure</th>
                <th scope="col">Humidity</th>
                <th scope="col">Status</th>
                <th scope="col">
                  <span className="sg-sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredNeighbors.map(({ station, distance_km, reading }) => (
                <tr key={station.station_id} className="sg-neighbor-table__row">
                  {/* Station ID & Name */}
                  <td>
                    <div className="sg-neighbor-table__station-cell">
                      <span className="sg-neighbor-table__station-id sg-font-mono">
                        {station.station_id}
                      </span>
                      <span className="sg-neighbor-table__station-name">{station.name}</span>
                    </div>
                  </td>

                  {/* Geodesic Distance */}
                  <td className="sg-font-mono sg-neighbor-table__distance-cell">
                    {formatDistance(distance_km)}
                  </td>

                  {/* Temperature */}
                  <td className="sg-font-mono">
                    {reading ? `${reading.temperature_c.toFixed(1)}°C` : '—'}
                  </td>

                  {/* Pressure */}
                  <td className="sg-font-mono">
                    {reading ? `${reading.pressure_hpa.toFixed(1)} hPa` : '—'}
                  </td>

                  {/* Humidity */}
                  <td className="sg-font-mono">
                    {reading ? `${reading.humidity_pct.toFixed(1)}%` : '—'}
                  </td>

                  {/* Status */}
                  <td>
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
                  </td>

                  {/* Select Focus Action */}
                  <td className="sg-neighbor-table__action-cell">
                    <button
                      type="button"
                      className="sg-neighbor-table__select-btn"
                      onClick={() => onSelectStation(station)}
                      aria-label={`Select ${station.name} as primary station`}
                    >
                      <Radio size={12} aria-hidden="true" />
                      <span>Set Focus</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
};
