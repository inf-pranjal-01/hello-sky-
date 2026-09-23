import React from 'react';
import './Loading.css';

export interface LoadingProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const Loading: React.FC<LoadingProps> = ({
  message = 'Loading telemetry data...',
  size = 'md',
}) => {
  return (
    <div className="sg-loading-container" role="status" aria-live="polite">
      <div className={`sg-spinner sg-spinner--${size}`} />
      {message && <p className="sg-loading-message">{message}</p>}
    </div>
  );
};
