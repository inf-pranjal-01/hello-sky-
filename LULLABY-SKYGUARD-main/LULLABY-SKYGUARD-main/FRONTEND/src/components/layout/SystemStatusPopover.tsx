import React, { useState, useRef, useEffect } from 'react';
import { Activity, ShieldCheck, X } from 'lucide-react';
import { StatusBadge } from '../common/StatusBadge';
import { SystemStatusSummary } from '../../types';
import { Button } from '../common/Button';
import './SystemStatusPopover.css';

export interface SystemStatusPopoverProps {
  systemStatus: SystemStatusSummary | null;
}

export const SystemStatusPopover: React.FC<SystemStatusPopoverProps> = ({ systemStatus }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const statusVal = systemStatus?.overall_status || 'NORMAL';

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        triggerRef.current?.focus(); // Return focus to trigger
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  return (
    <div className="sg-status-popover-container" ref={containerRef}>
      <button
        ref={triggerRef}
        type="button"
        className="sg-status-trigger-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-label={`System Network Status: ${statusVal}. Click for summary details.`}
      >
        <span className="sg-status-label-text">Health:</span>
        <StatusBadge status={statusVal.toLowerCase() as any} label={`STATUS: ${statusVal}`} />
      </button>

      {isOpen && (
        <div className="sg-status-popover-card" role="dialog" aria-label="System Network Health Details">
          <div className="sg-status-popover__header">
            <div className="sg-status-popover__title">
              <Activity size={18} className="text-accent" />
              <h3>AWS Network Status Summary</h3>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setIsOpen(false);
                triggerRef.current?.focus();
              }}
              ariaLabel="Close status popover"
              leftIcon={<X size={16} />}
            />
          </div>

          <div className="sg-status-popover__body">
            <div className="sg-status-notice-tag">
              <span>LIVE NETWORK ROLLUP FROM STATION HEALTH</span>
            </div>

            <div className="sg-status-metric-grid">
              <div className="sg-status-metric-item">
                <span className="sg-status-metric-label">Overall State</span>
                <StatusBadge status={statusVal.toLowerCase() as any} label={statusVal} />
              </div>

              <div className="sg-status-metric-item">
                <span className="sg-status-metric-label">Active Observatories</span>
                <span className="sg-status-metric-val">
                  {systemStatus?.active_stations_count ?? '—'} / {systemStatus?.total_stations_count ?? '—'}
                </span>
              </div>

              <div className="sg-status-metric-item">
                <span className="sg-status-metric-label">Active Anomalies</span>
                <span className="sg-status-metric-val text-critical">
                  {systemStatus?.active_anomalies_count ?? '—'}
                </span>
              </div>

              <div className="sg-status-metric-item">
                <span className="sg-status-metric-label">Avg Hardware Health</span>
                <span className="sg-status-metric-val">
                  {systemStatus?.avg_sensor_health_pct ?? '—'}%
                </span>
              </div>
            </div>

            <div className="sg-status-popover__footer-info">
              <ShieldCheck size={14} className="text-optimal" />
              <span>Real-time Strategy: HTTP polling of live station health</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
