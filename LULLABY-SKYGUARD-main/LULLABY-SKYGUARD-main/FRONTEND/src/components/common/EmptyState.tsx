import React from 'react';
import { Inbox } from 'lucide-react';
import { Button } from './Button';
import './EmptyState.css';

export interface EmptyStateProps {
  title?: string;
  description?: string;
  actionLabel?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Telemetry Data Found',
  description = 'There are no active records matching the current meteorological filter.',
  actionLabel,
  onAction,
  icon = <Inbox size={42} />,
}) => {
  return (
    <div className="sg-empty-state">
      <div className="sg-empty-state__icon">{icon}</div>
      <h3 className="sg-empty-state__title">{title}</h3>
      <p className="sg-empty-state__description">{description}</p>
      {actionLabel && onAction && (
        <Button variant="outline" size="sm" onClick={onAction}>
          {actionLabel}
        </Button>
      )}
    </div>
  );
};
