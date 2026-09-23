import React from 'react';
import { Clock } from 'lucide-react';
import { ReportPeriod } from '../../types';
import './ReportPeriodSelector.css';

export interface ReportPeriodSelectorProps {
  periodHours: ReportPeriod;
  onPeriodChange: (hours: ReportPeriod) => void;
  disabled?: boolean;
}

export const ReportPeriodSelector: React.FC<ReportPeriodSelectorProps> = ({
  periodHours,
  onPeriodChange,
  disabled = false,
}) => {
  const periods: ReportPeriod[] = [6, 12, 24];

  return (
    <div className="sg-report-period-container no-print">
      <div className="sg-report-period-label">
        <Clock size={14} className="text-accent" aria-hidden="true" />
        <span>Report Observation Period:</span>
      </div>

      <div
        className="sg-report-period-group"
        role="group"
        aria-label="Select report observation period"
      >
        {periods.map((h) => (
          <button
            key={h}
            type="button"
            className={`sg-report-period-btn ${
              periodHours === h ? 'sg-report-period-btn--active' : ''
            }`}
            onClick={() => onPeriodChange(h)}
            disabled={disabled}
            aria-pressed={periodHours === h}
          >
            Last {h} Hours
          </button>
        ))}
      </div>
    </div>
  );
};
