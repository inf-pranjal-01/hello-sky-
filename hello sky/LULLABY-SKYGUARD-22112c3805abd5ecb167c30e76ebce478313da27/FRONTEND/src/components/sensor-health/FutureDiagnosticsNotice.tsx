import React from 'react';
import { Cpu } from 'lucide-react';
import './FutureDiagnosticsNotice.css';

export interface FutureDiagnosticsNoticeProps {
  className?: string;
}

export const FutureDiagnosticsNotice: React.FC<FutureDiagnosticsNoticeProps> = ({
  className = '',
}) => {
  return (
    <aside
      className={`sg-future-notice ${className}`}
      role="note"
      aria-label="Future backend capabilities notice"
    >
      <div className="sg-future-notice__left">
        <Cpu size={16} className="text-muted" aria-hidden="true" />
        <p className="sg-future-notice__text">
          Advanced Hardware Diagnostics (Battery voltage, solar power, RF signal RSSI, predictive
          degradation) — Future backend telemetry capabilities.
        </p>
      </div>
      <span className="sg-future-notice__tag">
        PLANNED CAPABILITY — NOT IN CURRENT API CONTRACT
      </span>
    </aside>
  );
};
