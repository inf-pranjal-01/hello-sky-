import React from 'react';
import './StatusBadge.css';

export type StatusType =
  | 'optimal'
  | 'active'
  | 'low'
  | 'warning'
  | 'moderate'
  | 'degraded'
  | 'maintenance'
  | 'high'
  | 'critical'
  | 'offline'
  | 'NORMAL'
  | 'WARNING'
  | 'CRITICAL'
  | 'OFFLINE';

export interface StatusBadgeProps {
  status: StatusType;
  label?: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
}) => {
  const displayLabel = label || status.toUpperCase();
  const normalized = status.toLowerCase();

  let categoryClass = 'sg-badge--optimal';
  if (['warning', 'moderate', 'maintenance'].includes(normalized)) {
    categoryClass = 'sg-badge--warning';
  } else if (['degraded', 'high'].includes(normalized)) {
    categoryClass = 'sg-badge--degraded';
  } else if (['critical', 'offline'].includes(normalized)) {
    categoryClass = 'sg-badge--critical';
  } else if (['normal', 'optimal', 'active', 'low'].includes(normalized)) {
    categoryClass = 'sg-badge--optimal';
  }

  return (
    <span className={`sg-badge sg-badge--${size} ${categoryClass}`}>
      <span className="sg-badge__dot" aria-hidden="true" />
      <span className="sg-badge__text">{displayLabel}</span>
    </span>
  );
};
