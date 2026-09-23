import React, { useState, useEffect, useMemo } from 'react';
import { TrendingUp, Table, EyeOff } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { HealthHistoryPoint } from '../../types';
import { sensorHealthService } from '../../services/sensorHealthService';
import './HealthHistoryChart.css';

export interface HealthHistoryChartProps {
  stationId: string | null | undefined;
  className?: string;
}

export const HealthHistoryChart: React.FC<HealthHistoryChartProps> = ({
  stationId,
  className = '',
}) => {
  const [hours, setHours] = useState<number>(6);
  const [history, setHistory] = useState<HealthHistoryPoint[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showTable, setShowTable] = useState<boolean>(false);

  useEffect(() => {
    if (!stationId) {
      setHistory([]);
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    sensorHealthService
      .getHealthHistory(stationId, hours)
      .then((data) => {
        if (!isMounted) return;
        setHistory(data);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err instanceof Error ? err.message : 'Unable to load health history.');
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [stationId, hours]);

  // Compute stats for accessible text summary
  const summaryText = useMemo(() => {
    if (!history || history.length === 0) {
      return 'No sensor health history is currently available.';
    }
    const scores = history.map((p) => p.health_pct);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const latest = scores[scores.length - 1];
    return `Sensor health over the last ${hours} hours ranged from ${min} percent to ${max} percent. The latest health score is ${latest} percent.`;
  }, [history, hours]);

  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-health-history-card ${className}`}>
        <div className="sg-health-history__header">
          <Skeleton width="180px" height="1.2rem" />
          <Skeleton width="120px" height="1.8rem" />
        </div>
        <Skeleton width="100%" height="220px" style={{ marginTop: '1rem' }} />
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-health-history-card ${className}`}>
        <div className="sg-health-history__header">
          <h3 className="sg-health-history__title">Sensor Health Trajectory</h3>
        </div>
        <p style={{ color: 'var(--color-status-critical, #ef4444)', margin: '1rem 0' }}>{error}</p>
      </Card>
    );
  }

  // SVG Chart Geometry
  const svgWidth = 800;
  const svgHeight = 220;
  const padding = { top: 20, right: 30, bottom: 30, left: 45 };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = svgHeight - padding.top - padding.bottom;

  const timestamps = history.map((point) => new Date(point.timestamp).getTime());
  const minTime = Math.min(...timestamps);
  const maxTime = Math.max(...timestamps);
  const coordinates = history.map((point) => {
    const timestamp = new Date(point.timestamp).getTime();
    const x = maxTime === minTime
      ? padding.left + chartWidth / 2
      : padding.left + ((timestamp - minTime) / (maxTime - minTime)) * chartWidth;
    // Y-scale: 0 to 100%
    const y = padding.top + chartHeight - (point.health_pct / 100) * chartHeight;
    return { x, y, point };
  });

  const pathD =
    coordinates.length > 0
      ? coordinates.reduce((acc, curr, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${curr.x} ${curr.y}`, '')
      : '';

  const areaD =
    coordinates.length > 0
      ? `${pathD} L ${coordinates[coordinates.length - 1].x} ${padding.top + chartHeight} L ${
          coordinates[0].x
        } ${padding.top + chartHeight} Z`
      : '';

  const latestScore = history.length > 0 ? history[history.length - 1].health_pct : 100;
  const lineColorClass =
    latestScore < 50
      ? 'sg-health-history__line--critical'
      : latestScore < 80
      ? 'sg-health-history__line--warning'
      : '';

  return (
    <Card
      variant="glass"
      className={`sg-health-history-card ${className}`}
      role="region"
      aria-label="Sensor health history trajectory"
    >
      <div className="sg-health-history__header">
        <div className="sg-health-history__title-area">
          <div className="sg-health-history__title-row">
            <TrendingUp size={18} className="text-accent" aria-hidden="true" />
            <h3 className="sg-health-history__title">Sensor Health Trajectory</h3>
          </div>
          <span className="sg-health-history__notice">
            DEMO TREND — FRONTEND DERIVED
          </span>
        </div>

        <div className="sg-health-history__controls">
          <div className="sg-health-history__range-buttons" role="group" aria-label="Time range selector">
            {[6, 12, 24].map((h) => (
              <button
                key={h}
                type="button"
                className={`sg-health-history__range-btn ${
                  hours === h ? 'sg-health-history__range-btn--active' : ''
                }`}
                onClick={() => setHours(h)}
                aria-pressed={hours === h}
              >
                {h}H
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Accessible Text Summary */}
      <p className="sg-health-history__summary-text" aria-live="polite">
        {summaryText}
      </p>

      {/* SVG Line & Area Visualization */}
      <div className="sg-health-history__svg-container">
        <svg
          className="sg-health-history__svg"
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={summaryText}
        >
          <defs>
            <linearGradient id="healthGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Gridlines & Y-Axis */}
          {[100, 75, 50, 25, 0].map((val) => {
            const y = padding.top + chartHeight - (val / 100) * chartHeight;
            return (
              <g key={val}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={svgWidth - padding.right}
                  y2={y}
                  className="sg-health-history__grid-line"
                />
                <text x={padding.left - 8} y={y + 3} textAnchor="end" className="sg-health-history__axis-label">
                  {val}%
                </text>
              </g>
            );
          })}

          {/* Area under curve */}
          {areaD && <path d={areaD} fill="url(#healthGrad)" className="sg-health-history__area" />}

          {/* Line path */}
          {pathD && <path d={pathD} className={`sg-health-history__line ${lineColorClass}`} />}

          {/* Data Points */}
          {coordinates.map((coord, idx) => {
            const timeLabel = new Date(coord.point.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            });
            return (
              <circle
                key={idx}
                cx={coord.x}
                cy={coord.y}
                r={3}
                className="sg-health-history__point"
              >
                <title>{`${timeLabel}: ${coord.point.health_pct}% (${coord.point.status})`}</title>
              </circle>
            );
          })}
        </svg>
      </div>

      {/* Screen Reader Expandable Table Toggle */}
      <div className="sg-health-history__table-toggle">
        <button
          type="button"
          className="sg-health-history__toggle-btn"
          onClick={() => setShowTable((prev) => !prev)}
          aria-expanded={showTable}
        >
          {showTable ? <EyeOff size={13} style={{ display: 'inline', marginRight: '4px' }} /> : <Table size={13} style={{ display: 'inline', marginRight: '4px' }} />}
          {showTable ? 'Hide Raw Data Table' : 'Show Raw Data Table for Screen Readers'}
        </button>
      </div>

      {showTable && (
        <div className="sg-health-history__table-wrapper">
          <table className="sg-health-history__table" aria-label="Historical sensor health data records">
            <thead>
              <tr>
                <th scope="col">Timestamp</th>
                <th scope="col">Health %</th>
                <th scope="col">Condition</th>
              </tr>
            </thead>
            <tbody>
              {history.map((pt, idx) => (
                <tr key={idx}>
                  <td>{new Date(pt.timestamp).toLocaleTimeString()}</td>
                  <td>{pt.health_pct}%</td>
                  <td>{pt.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
};
