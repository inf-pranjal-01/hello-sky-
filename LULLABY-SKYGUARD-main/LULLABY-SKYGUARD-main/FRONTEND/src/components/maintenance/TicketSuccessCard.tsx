import React from 'react';
import { CheckCircle, ExternalLink, RotateCcw, Ticket } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../common/Button';
import { MaintenanceTicketResponse } from '../../types';
import './TicketSuccessCard.css';

export interface TicketSuccessCardProps {
  ticket: MaintenanceTicketResponse;
  onCreateAnother: () => void;
}

export const TicketSuccessCard: React.FC<TicketSuccessCardProps> = ({
  ticket,
  onCreateAnother,
}) => {
  const navigate = useNavigate();

  const formattedDate = new Date(ticket.created_at).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  });

  return (
    <div className="sg-ticket-success" role="status" aria-live="polite">
      {/* Success icon + headline */}
      <div className="sg-ticket-success__banner">
        <CheckCircle size={32} className="sg-ticket-success__icon" aria-hidden="true" />
        <div>
          <h3 className="sg-ticket-success__title">Ticket Dispatched</h3>
          <p className="sg-ticket-success__subtitle">
            The maintenance ticket has been added to the field operations queue.
          </p>
        </div>
      </div>

      {/* Ticket details */}
      <div className="sg-ticket-success__card">
        <div className="sg-ticket-success__ticket-id-row">
          <Ticket size={14} aria-hidden="true" />
          <span className="sg-ticket-success__ticket-id">{ticket.ticket_id}</span>
        </div>

        <dl className="sg-ticket-success__details">
          <div className="sg-ticket-success__detail-row">
            <dt className="sg-ticket-success__detail-label">Station</dt>
            <dd className="sg-ticket-success__detail-value sg-ticket-success__detail-value--mono">
              {ticket.station_id}
            </dd>
          </div>
          <div className="sg-ticket-success__detail-row">
            <dt className="sg-ticket-success__detail-label">Issue</dt>
            <dd className="sg-ticket-success__detail-value">{ticket.issue}</dd>
          </div>
          <div className="sg-ticket-success__detail-row">
            <dt className="sg-ticket-success__detail-label">Priority</dt>
            <dd className={`sg-ticket-success__detail-value sg-ticket-success__priority--${ticket.priority.toLowerCase()}`}>
              {ticket.priority.toUpperCase()}
            </dd>
          </div>
          <div className="sg-ticket-success__detail-row">
            <dt className="sg-ticket-success__detail-label">Created At</dt>
            <dd className="sg-ticket-success__detail-value">{formattedDate}</dd>
          </div>
        </dl>
      </div>

      {/* Navigation actions */}
      <div className="sg-ticket-success__actions">
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate('/alerts')}
          leftIcon={<ExternalLink size={14} />}
        >
          View Alerts
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate('/reports')}
          leftIcon={<ExternalLink size={14} />}
        >
          View Reports
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onCreateAnother}
          leftIcon={<RotateCcw size={14} />}
        >
          Create Another
        </Button>
      </div>
    </div>
  );
};
