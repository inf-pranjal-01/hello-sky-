import React, { useState, useRef, useEffect, useId } from 'react';
import { Radio, Search, ChevronDown, Check } from 'lucide-react';
import { useStation } from '../../context/StationContext';
import { StatusBadge } from '../common/StatusBadge';
import { Station } from '../../types';
import './StationSelector.css';

export interface StationSelectorProps {
  isCollapsed?: boolean;
}

export const StationSelector: React.FC<StationSelectorProps> = ({ isCollapsed = false }) => {
  const { stations, selectedStation, setSelectedStation, isLoading } = useStation();
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();

  // Close dropdown on outside click or Escape key
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      document.addEventListener('keydown', handleKeyDown);
      // Focus search input when popover opens
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const filteredStations = stations.filter(
    (st) =>
      st.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      st.station_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (st.region && st.region.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleSelect = (station: Station) => {
    setSelectedStation(station);
    setIsOpen(false);
    setSearchQuery('');
  };

  if (isLoading && !selectedStation) {
    return (
      <div className="sg-station-selector sg-station-selector--loading">
        <span className="sg-station-loading-text">Loading AWS stations...</span>
      </div>
    );
  }

  return (
    <div className={`sg-station-selector ${isCollapsed ? 'sg-station-selector--collapsed' : ''}`} ref={containerRef}>
      <button
        type="button"
        className="sg-station-trigger"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-controls={listboxId}
        aria-label={`Select AWS Station. Active: ${selectedStation?.name || 'None'}`}
      >
        <div className="sg-station-trigger__icon">
          <Radio size={18} className="text-accent" />
        </div>

        {!isCollapsed && (
          <>
            <div className="sg-station-trigger__info">
              <span className="sg-station-name">{selectedStation?.name || 'Select Station'}</span>
              <span className="sg-station-id">{selectedStation?.station_id || 'AWS Network'}</span>
            </div>

            {selectedStation && (
              <div className="sg-station-trigger__badge">
                <StatusBadge
                  status={selectedStation.status.toLowerCase() as any}
                  size="sm"
                  label={selectedStation.status}
                />
              </div>
            )}

            <ChevronDown size={14} className={`sg-station-arrow ${isOpen ? 'sg-station-arrow--open' : ''}`} />
          </>
        )}
      </button>

      {isOpen && (
        <div id={listboxId} role="listbox" className="sg-station-dropdown" aria-label="AWS Observatory List">
          <div className="sg-station-search-box">
            <Search size={14} className="sg-station-search-icon" aria-hidden="true" />
            <input
              ref={searchInputRef}
              type="text"
              className="sg-station-search-input"
              placeholder="Search AWS station..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              aria-label="Filter AWS stations by name or ID"
            />
          </div>

          <div className="sg-station-list">
            <div className="sg-station-notice">
              <span>● LIVE BACKEND</span>
            </div>

            {filteredStations.length === 0 ? (
              <div className="sg-station-empty">No AWS stations found.</div>
            ) : (
              filteredStations.map((st) => {
                const isSelected = selectedStation?.station_id === st.station_id;
                return (
                  <div
                    key={st.station_id}
                    role="option"
                    aria-selected={isSelected}
                    tabIndex={0}
                    className={`sg-station-item ${isSelected ? 'sg-station-item--selected' : ''}`}
                    onClick={() => handleSelect(st)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleSelect(st);
                      }
                    }}
                  >
                    <div className="sg-station-item__left">
                      <span className="sg-station-item__name">{st.name}</span>
                      <span className="sg-station-item__meta">
                        {st.station_id} • {st.region || 'Regional Telemetry'}
                      </span>
                    </div>

                    <div className="sg-station-item__right">
                      <StatusBadge status={st.status.toLowerCase() as any} size="sm" label={st.status} />
                      {isSelected && <Check size={14} className="sg-station-check" />}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
