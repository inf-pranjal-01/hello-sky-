import React from 'react';
import { Wrench, MapPin, AlertTriangle, FileText } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import { LatestAnomaly, RecentAnomalyItem } from '../../types';
import './TicketPreview.css';

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

export interface TicketPreviewProps {
  anomaly: AnomalyEntry;
  onCreateTicket: () => void;
  isSubmitting: boolean;
}

export const TicketPreview: React.FC<TicketPreviewProps> = ({
  anomaly,
  onCreateTicket,
  isSubmitting,
}) => {
  const formattedDate = new Date(anomaly.timestamp).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  });
  const suggestedReading = Object.entries(anomaly.suggested_values ?? {})
    .map(([parameter, value]) => `${parameter.replace(/_/g, ' ')}: ${value.toFixed(2)}`)
    .join(' · ');

  return (
    <div className="sg-ticket-preview">
      <h3 className="sg-ticket-preview__heading">
        <FileText size={15} aria-hidden="true" />
        Ticket Preview
      </h3>

      {/* Anomaly Identity */}
      <div className="sg-ticket-preview__hero">
        <div className="sg-ticket-preview__hero-left">
          <span className="sg-ticket-preview__anomaly-id">{anomaly.anomaly_id}</span>
          <span className="sg-ticket-preview__type">{formatAnomType(anomaly.type)}</span>
        </div>
        <div className="sg-ticket-preview__hero-right">
          <StatusBadge
            status={severityToStatus(anomaly.severity)}
            label={`${anomaly.severity.toUpperCase()} SEVERITY`}
            size="md"
          />
          <span className="sg-ticket-preview__score">
            {Math.round(anomaly.anomaly_score_pct)}%
          </span>
        </div>
      </div>

      {/* Detail rows */}
      <dl className="sg-ticket-preview__details">
        <div className="sg-ticket-preview__detail-row">
          <dt className="sg-ticket-preview__detail-label">
            <MapPin size={12} aria-hidden="true" />
            Station
          </dt>
          <dd className="sg-ticket-preview__detail-value sg-ticket-preview__detail-value--mono">
            {anomaly.station_id}
          </dd>
        </div>

        <div className="sg-ticket-preview__detail-row">
          <dt className="sg-ticket-preview__detail-label">
            <AlertTriangle size={12} aria-hidden="true" />
            Detected At
          </dt>
          <dd className="sg-ticket-preview__detail-value">
            {formattedDate}
          </dd>
        </div>

        <div className="sg-ticket-preview__detail-row sg-ticket-preview__detail-row--stacked">
          <dt className="sg-ticket-preview__detail-label">Root Cause</dt>
          <dd className="sg-ticket-preview__detail-value sg-ticket-preview__detail-value--block">
            {anomaly.root_cause}
          </dd>
        </div>

        <div className="sg-ticket-preview__detail-row sg-ticket-preview__detail-row--stacked">
          <dt className="sg-ticket-preview__detail-label">Description</dt>
          <dd className="sg-ticket-preview__detail-value sg-ticket-preview__detail-value--block">
            {anomaly.description}
          </dd>
        </div>

        {suggestedReading && (
          <div className="sg-ticket-preview__detail-row sg-ticket-preview__detail-row--stacked">
            <dt className="sg-ticket-preview__detail-label">Suggested Replacement</dt>
            <dd className="sg-ticket-preview__detail-value sg-ticket-preview__detail-value--block">
              {suggestedReading}
            </dd>
          </div>
        )}
      </dl>

      {/* Dispatch notice */}
      <p className="sg-ticket-preview__notice">
        <Wrench size={13} aria-hidden="true" />
        This will dispatch a maintenance ticket to the field operations queue.
      </p>

      {/* CTA */}
      <Button
        variant="primary"
        size="md"
        onClick={onCreateTicket}
        isLoading={isSubmitting}
        leftIcon={<Wrench size={15} />}
        ariaLabel="Create maintenance ticket for this anomaly"
      >
        Create Maintenance Ticket
      </Button>
    </div>
  );
};
