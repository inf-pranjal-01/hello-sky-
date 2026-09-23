import React from 'react';
import { SuggestedReading } from '../../utils/suggestedValues';
import './SuggestedValues.css';

export interface SuggestedValuesProps {
  items: SuggestedReading[];
  compact?: boolean;
  showHeading?: boolean;
  emptyLabel?: string;
  className?: string;
}

export const SuggestedValues: React.FC<SuggestedValuesProps> = ({
  items,
  compact = false,
  showHeading = false,
  emptyLabel,
  className = '',
}) => {
  if (!items.length) {
    if (!emptyLabel) return null;
    return <span className={`sg-suggested sg-suggested--empty ${className}`}>{emptyLabel}</span>;
  }

  return (
    <div className={`sg-suggested ${compact ? 'sg-suggested--compact' : ''} ${className}`}>
      {(!compact || showHeading) && <span className="sg-suggested__heading">Suggested replacement</span>}
      <div className="sg-suggested__chips">
        {items.map((item) => (
          <span key={item.key} className="sg-suggested__chip">
            <span className="sg-suggested__label">{item.label}</span>
            <strong className="sg-suggested__value">
              {item.value.toFixed(1)} {item.unit}
            </strong>
          </span>
        ))}
      </div>
    </div>
  );
};
