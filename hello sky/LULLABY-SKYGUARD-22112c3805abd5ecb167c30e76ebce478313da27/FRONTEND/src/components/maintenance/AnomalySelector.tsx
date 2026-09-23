import React from 'react';
import { AlertTriangle, Clock, MapPin } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { Skeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import { RecentAnomalyItem, LatestAnomaly } from '../../types';
import './AnomalySelector.css';

type AnomalyEntry = RecentAnomalyItem | LatestAnomaly;

function severityToStatus(severity: string): 'critical' | 'high' | 'moderate' | 'low' {
  switch (severity) {
    case 'critical': return 'critical';
    case 'high':     return 'high';
    case 'medium':   return 'moderate';
    default:         return 'low';
  }
}

function formatAnomType(type: string): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export interface AnomalySelectorProps {
  anomalies: AnomalyEntry[];
  selectedAnomalyId: string | null;
  onSelect: (id: string) => void;
  isLoading: boolean;
}

export const AnomalySelector: React.FC<AnomalySelectorProps> = ({
  anomalies,
  selectedAnomalyId,
  onSelect,
  isLoading,
}) => {
  if (isLoading) {
    return (
      <div className="sg-anomaly-selector">
        <h3 className="sg-anomaly-selector__heading">
          <AlertTriangle size={15} aria-hidden="true" />
          Select Anomaly
        </h3>
        <div className="sg-anomaly-selector__list">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height="72px" borderRadius="10px" />
          ))}
        </div>
      </div>
    );
  }

  if (anomalies.length === 0) {
    return (
      <div className="sg-anomaly-selector">
        <h3 className="sg-anomaly-selector__heading">
          <AlertTriangle size={15} aria-hidden="true" />
          Select Anomaly
        </h3>
        <EmptyState
          icon={<AlertTriangle size={28} />}
          title="No recent anomalies"
          description="No recent anomalies require maintenance action at this time."
        />
      </div>
    );
  }

  return (
    <div className="sg-anomaly-selector">
      <h3 className="sg-anomaly-selector__heading">
        <AlertTriangle size={15} aria-hidden="true" />
        Select Anomaly
        <span className="sg-anomaly-selector__count">{anomalies.length}</span>
      </h3>
      <p className="sg-anomaly-selector__hint">
        Choose a detected anomaly to dispatch a maintenance ticket.
      </p>
      <ul className="sg-anomaly-selector__list" role="listbox" aria-label="Anomaly list">
        {anomalies.map((anomaly) => {
          const id = anomaly.anomaly_id;
          const isSelected = id === selectedAnomalyId;
          const formattedDate = new Date(anomaly.timestamp).toLocaleString(undefined, {
            dateStyle: 'short',
            timeStyle: 'short',
          });

          return (
            <li
              key={id}
              role="option"
              aria-selected={isSelected}
              className={`sg-anomaly-selector__item${isSelected ? ' sg-anomaly-selector__item--selected' : ''}`}
              onClick={() => onSelect(id)}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelect(id);
                }
              }}
            >
              <div className="sg-anomaly-selector__item-top">
                <span className="sg-anomaly-selector__type">{formatAnomType(anomaly.type)}</span>
                <StatusBadge
                  status={severityToStatus(anomaly.severity)}
                  label={anomaly.severity.toUpperCase()}
                  size="sm"
                />
              </div>
              <div className="sg-anomaly-selector__item-meta">
                <span className="sg-anomaly-selector__meta-item">
                  <MapPin size={11} aria-hidden="true" />
                  {anomaly.station_id}
                </span>
                <span className="sg-anomaly-selector__meta-item">
                  <Clock size={11} aria-hidden="true" />
                  {formattedDate}
                </span>
                <span className="sg-anomaly-selector__score">
                  {Math.round(anomaly.anomaly_score_pct)}% score
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
};
