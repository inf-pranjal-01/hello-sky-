import React from 'react';
import { Search, RotateCcw } from 'lucide-react';
import { AnomalySeverity, AnomalyType } from '../../types';
import './AnomalyFilters.css';

export interface AnomalyFilterValues {
  severity: AnomalySeverity | 'all';
  type: AnomalyType | 'all';
  searchQuery: string;
  stationId?: string | 'all';
}

export interface AnomalyFiltersProps {
  filters: AnomalyFilterValues;
  onChange: (filters: AnomalyFilterValues) => void;
  onReset: () => void;
  totalCount: number;
  filteredCount: number;
  stations?: Array<{ station_id: string; name: string }>;
}

export const AnomalyFilters: React.FC<AnomalyFiltersProps> = ({
  filters,
  onChange,
  onReset,
  totalCount,
  filteredCount,
  stations = [],
}) => {
  const isFiltered =
    filters.severity !== 'all' ||
    filters.type !== 'all' ||
    filters.searchQuery.trim().length > 0 ||
    (filters.stationId && filters.stationId !== 'all');

  const handleStationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({
      ...filters,
      stationId: e.target.value,
    });
  };

  const handleSeverityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({
      ...filters,
      severity: e.target.value as AnomalySeverity | 'all',
    });
  };

  const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onChange({
      ...filters,
      type: e.target.value as AnomalyType | 'all',
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    onChange({
      ...filters,
      searchQuery: e.target.value,
    });
  };

  return (
    <div className="sg-anomaly-filters" role="search" aria-label="Filter anomaly records">
      <div className="sg-anomaly-filters__controls">
        {/* Station Filter */}
        {stations.length > 0 && (
          <div className="sg-anomaly-filters__group">
            <label htmlFor="sg-filter-station" className="sg-anomaly-filters__label">
              Station
            </label>
            <select
              id="sg-filter-station"
              className="sg-anomaly-filters__select"
              value={filters.stationId || 'all'}
              onChange={handleStationChange}
              aria-label="Filter by weather station"
            >
              <option value="all">All Stations (Network-wide)</option>
              {stations.map((s) => (
                <option key={s.station_id} value={s.station_id}>
                  {s.name} ({s.station_id})
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Severity Filter */}
        <div className="sg-anomaly-filters__group">
          <label htmlFor="sg-filter-severity" className="sg-anomaly-filters__label">
            Severity
          </label>
          <select
            id="sg-filter-severity"
            className="sg-anomaly-filters__select"
            value={filters.severity}
            onChange={handleSeverityChange}
            aria-label="Filter by anomaly severity"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* Anomaly Type Filter */}
        <div className="sg-anomaly-filters__group">
          <label htmlFor="sg-filter-type" className="sg-anomaly-filters__label">
            Type
          </label>
          <select
            id="sg-filter-type"
            className="sg-anomaly-filters__select"
            value={filters.type}
            onChange={handleTypeChange}
            aria-label="Filter by anomaly classification type"
          >
            <option value="all">All Types</option>
            <option value="spike">Spike</option>
            <option value="drift">Drift</option>
            <option value="frozen_value">Frozen Value</option>
            <option value="dropout">Dropout</option>
            <option value="physical_bounds">Physical Bounds</option>
            <option value="statistical_anomaly">Statistical Anomaly</option>
            <option value="sensor_fail_low">Sensor Fail Low</option>
            <option value="multivariate_inconsistency">Multivariate Inconsistency</option>
          </select>
        </div>

        {/* Text Search */}
        <div className="sg-anomaly-filters__search">
          <Search size={14} className="sg-anomaly-filters__search-icon" aria-hidden="true" />
          <input
            type="text"
            className="sg-anomaly-filters__search-input"
            placeholder="Search ID, cause or description..."
            value={filters.searchQuery}
            onChange={handleSearchChange}
            aria-label="Search anomalies by keyword"
          />
        </div>
      </div>

      <div className="sg-anomaly-filters__actions">
        <span className="sg-anomaly-filters__count" aria-live="polite">
          Showing <span className="sg-anomaly-filters__count-strong">{filteredCount}</span> of {totalCount}
        </span>

        {isFiltered && (
          <button
            type="button"
            className="sg-anomaly-filters__clear-btn"
            onClick={onReset}
            title="Reset active filters"
          >
            <RotateCcw size={12} style={{ display: 'inline', marginRight: '4px' }} />
            Reset
          </button>
        )}
      </div>
    </div>
  );
};
