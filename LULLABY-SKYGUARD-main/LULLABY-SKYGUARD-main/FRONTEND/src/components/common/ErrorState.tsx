import React from 'react';
import { AlertTriangle, RefreshCw, WifiOff } from 'lucide-react';
import { Button } from './Button';
import './ErrorState.css';

export interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  isOffline?: boolean;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title,
  message = 'An unexpected error occurred while communicating with the telemetry data layer.',
  onRetry,
  isOffline = false,
}) => {
  const displayTitle = title || (isOffline ? 'System Offline' : 'Telemetry Communication Alert');
  const icon = isOffline ? <WifiOff size={40} /> : <AlertTriangle size={40} />;

  return (
    <div className={`sg-error-state ${isOffline ? 'sg-error-state--offline' : ''}`} role="alert">
      <div className="sg-error-state__icon">{icon}</div>
      <h3 className="sg-error-state__title">{displayTitle}</h3>
      <p className="sg-error-state__message">{message}</p>
      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          onClick={onRetry}
          leftIcon={<RefreshCw size={16} />}
        >
          Retry Connection
        </Button>
      )}
    </div>
  );
};
