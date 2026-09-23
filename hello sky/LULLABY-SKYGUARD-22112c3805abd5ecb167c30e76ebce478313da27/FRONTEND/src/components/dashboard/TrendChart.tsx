import React, { useState, useId, useRef } from 'react';
import { createPortal } from 'react-dom';
import { LineChart, Clock } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { TrendPoint } from '../../types';
import { windowTrendPoints } from '../../utils/chartWindow';
import { suggestedFromTrendPoint } from '../../utils/suggestedValues';
import { SuggestedValues } from '../common/SuggestedValues';
import './TrendChart.css';

export type MetricType = 'temperature' | 'pressure' | 'humidity';

export interface TrendChartProps {
  points?: TrendPoint[];
  hours?: number;
  onHoursChange?: (hours: number) => void;
  isLoading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  className?: string;
}

const METRIC_CONFIG = {
  temperature: {
    label: 'Temperature',
    unit: '°C',
    color: '#3b82f6', // Accent blue
    key: 'temperature_c' as const,
    normalMin: 18,
    normalMax: 35,
  },
  pressure: {
    label: 'Pressure',
    unit: 'hPa',
    color: '#06b6d4', // Accent cyan
    key: 'pressure_hpa' as const,
    normalMin: 990,
    normalMax: 1025,
  },
  humidity: {
    label: 'Humidity',
    unit: '%',
    color: '#10b981', // Emerald green
    key: 'humidity_pct' as const,
    normalMin: 30,
    normalMax: 80,
  },
};

/** Uses only mutually compatible Intl options (dateStyle cannot be mixed
 * with hour/minute options in several Chromium/Windows combinations). */
const formatChartTime = (timestamp: string, includeDate = false): string => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return new Intl.DateTimeFormat(undefined, includeDate
    ? { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }
    : { hour: '2-digit', minute: '2-digit' }
  ).format(date);
};

const isMetricAnomalous = (pt: TrendPoint, metric: MetricType): boolean => {
  if (!pt.is_anomaly) return false;
  const hasSuggested = {
    temperature: pt.suggested_temperature_c != null,
    pressure: pt.suggested_pressure_hpa != null,
    humidity: pt.suggested_humidity_pct != null,
  };
  if (hasSuggested.temperature || hasSuggested.pressure || hasSuggested.humidity) {
    return Boolean(hasSuggested[metric]);
  }
  const ft = (pt.fault_type || '').toLowerCase();
  if (ft.includes('temp')) return metric === 'temperature';
  if (ft.includes('press')) return metric === 'pressure';
  if (ft.includes('humid') || ft.includes('dew')) return metric === 'humidity';
  return true;
};

export const TrendChart: React.FC<TrendChartProps> = ({
  points = [],
  hours = 10,
  onHoursChange,
  isLoading = false,
  error = null,
  onRetry,
  className = '',
}) => {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('temperature');
  const [hoveredPoint, setHoveredPoint] = useState<{
    point: TrendPoint;
    x: number;
    y: number;
  } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const chartId = useId();
  const activeCfg = METRIC_CONFIG[selectedMetric];
  const { points: windowedPoints, windowStart, windowEnd } = windowTrendPoints(points, hours);

  const renderToolbar = () => (
    <div className="sg-trend-header">
      <div className="sg-trend-title-group">
        <div className="sg-trend-title">
          <LineChart size={18} className="text-accent" aria-hidden="true" />
          <h3 id={chartId}>Sensor Telemetry Trends</h3>
        </div>
        <span className="sg-trend-notice">● {hours}H WINDOW</span>
      </div>

      <div className="sg-trend-actions">
        <div className="sg-metric-tabs" role="tablist" aria-label="Select telemetry metric to graph">
          {(Object.keys(METRIC_CONFIG) as MetricType[]).map((metricKey) => {
            const cfg = METRIC_CONFIG[metricKey];
            const isSelected = selectedMetric === metricKey;
            return (
              <button
                key={metricKey}
                type="button"
                role="tab"
                aria-selected={isSelected}
                className={`sg-metric-tab ${isSelected ? 'sg-metric-tab--active' : ''}`}
                onClick={() => setSelectedMetric(metricKey)}
              >
                <span
                  className="sg-metric-tab-dot"
                  style={{ backgroundColor: cfg.color }}
                  aria-hidden="true"
                />
                <span>{cfg.label}</span>
              </button>
            );
          })}
        </div>

        {onHoursChange && (
          <div className="sg-range-select" aria-label="Time window selection">
            <Clock size={13} className="text-muted" aria-hidden="true" />
            {[10, 24, 72].map((h) => (
              <button
                key={h}
                type="button"
                className={`sg-range-btn ${hours === h ? 'sg-range-btn--active' : ''}`}
                onClick={() => onHoursChange(h)}
              >
                {h}h
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-trend-card ${className}`}>
        {renderToolbar()}
        <div style={{ padding: '2rem 0' }}>
          <Skeleton width="100%" height="240px" borderRadius="var(--radius-md)" />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="glass" className={`sg-trend-card sg-trend-card--error ${className}`}>
        {renderToolbar()}
        <div className="sg-trend-error">
          <p>{error}</p>
          {onRetry && (
            <button type="button" className="sg-trend-retry-btn" onClick={onRetry}>
              Retry Trends
            </button>
          )}
        </div>
      </Card>
    );
  }

  if (!windowedPoints || windowedPoints.length === 0) {
    return (
      <Card variant="glass" className={`sg-trend-card ${className}`}>
        {renderToolbar()}
        <div className="sg-trend-empty">
          <p>No trend data in the selected {hours}h window for this station.</p>
        </div>
      </Card>
    );
  }

  const values = windowedPoints.map((p) => p[activeCfg.key]);
  const minVal = Math.floor(Math.min(...values) - 1);
  const maxVal = Math.ceil(Math.max(...values) + 1);
  const valRange = maxVal - minVal || 1;

  const width = 800;
  const height = 240;
  const padLeft = 48;
  const padRight = 20;
  const padTop = 20;
  const padBottom = 30;

  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const span = windowEnd - windowStart || 1;
  const getX = (timestamp: string) => {
    return padLeft + ((new Date(timestamp).getTime() - windowStart) / span) * plotWidth;
  };

  const getY = (val: number) => {
    return padTop + plotHeight - ((val - minVal) / valRange) * plotHeight;
  };

  // Generate SVG path commands
  const pathD = windowedPoints.reduce((acc, point, i) => {
    const x = getX(point.timestamp);
    const y = getY(point[activeCfg.key]);
    return i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, '');

  const areaD = `${pathD} L ${getX(windowedPoints[windowedPoints.length - 1].timestamp)} ${padTop + plotHeight} L ${getX(windowedPoints[0].timestamp)} ${padTop + plotHeight} Z`;

  // Horizontal Grid Lines & Y Labels (4 steps)
  const yTicks = [0, 0.33, 0.66, 1].map((ratio) => {
    const val = minVal + ratio * valRange;
    const y = getY(val);
    return { val: Number(val.toFixed(1)), y };
  });

  // These labels make the chart's time basis visible. They are calculated
  // from the same real timestamps used for the plotted x coordinates.
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const time = windowStart + ratio * span;
    return { x: padLeft + ratio * plotWidth, label: formatChartTime(new Date(time).toISOString(), ratio === 0 || hours > 24) };
  });

  return (
    <Card variant="glass" className={`sg-trend-card ${className}`}>
      {renderToolbar()}

      {/* SVG Interactive Chart Canvas */}
      <div className="sg-chart-wrapper">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          className="sg-chart-svg"
          aria-labelledby={chartId}
          role="img"
        >
          {/* Grid lines */}
          {yTicks.map((tick, idx) => (
            <g key={idx}>
              <line
                x1={padLeft}
                y1={tick.y}
                x2={width - padRight}
                y2={tick.y}
                className="sg-chart-grid-line"
              />
              <text
                x={padLeft - 8}
                y={tick.y + 4}
                textAnchor="end"
                className="sg-chart-axis-text"
              >
                {tick.val}
              </text>
            </g>
          ))}

          {/* Real timestamp x-axis; replay is one historical hour per point. */}
          <line
            x1={padLeft}
            y1={padTop + plotHeight}
            x2={width - padRight}
            y2={padTop + plotHeight}
            className="sg-chart-grid-line"
          />
          {xTicks.map((tick, idx) => (
            <text
              key={`x-${idx}`}
              x={tick.x}
              y={height - 8}
              textAnchor={idx === 0 ? 'start' : idx === xTicks.length - 1 ? 'end' : 'middle'}
              className="sg-chart-axis-text"
            >
              {tick.label}
            </text>
          ))}

          {/* Gradient Fill under curve */}
          <defs>
            <linearGradient id={`grad-${selectedMetric}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={activeCfg.color} stopOpacity="0.25" />
              <stop offset="100%" stopColor={activeCfg.color} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          <path d={areaD} fill={`url(#grad-${selectedMetric})`} />

          {/* Line stroke */}
          <path
            d={pathD}
            fill="none"
            stroke={activeCfg.color}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Points & Interactive Tooltip Anchors */}
          {windowedPoints.map((pt, i) => {
            const cx = getX(pt.timestamp);
            const cy = getY(pt[activeCfg.key]);
            const isAnomaly = isMetricAnomalous(pt, selectedMetric);

            return (
              <circle
                key={i}
                cx={cx}
                cy={cy}
                r={isAnomaly ? 5 : 3.5}
                className={`sg-chart-point ${isAnomaly ? 'sg-chart-point--anomaly' : ''}`}
                fill={isAnomaly ? 'var(--status-critical-text)' : activeCfg.color}
                stroke="var(--bg-surface)"
                strokeWidth="1.5"
                onMouseEnter={() => setHoveredPoint({ point: pt, x: cx, y: cy })}
                onMouseLeave={() => setHoveredPoint(null)}
                tabIndex={0}
                role="button"
                aria-label={`${activeCfg.label}: ${pt[activeCfg.key]} ${activeCfg.unit} at ${new Date(
                  pt.timestamp
                ).toLocaleTimeString()}`}
              />
            );
          })}
        </svg>

        {/* This portal prevents the chart card or adjacent sections from
            clipping a tooltip at its boundary. */}
        {hoveredPoint && svgRef.current && createPortal(
          <div
            className="sg-chart-tooltip"
            style={{
              left: svgRef.current.getBoundingClientRect().left + (hoveredPoint.x / width) * svgRef.current.getBoundingClientRect().width,
              top: svgRef.current.getBoundingClientRect().top + (hoveredPoint.y / height) * svgRef.current.getBoundingClientRect().height,
            }}
          >
            <div className="sg-tooltip-time">
              {formatChartTime(hoveredPoint.point.timestamp, true)}
            </div>
            <div className="sg-tooltip-value">
              <span className="sg-tooltip-label">{activeCfg.label}:</span>
              <strong>
                {hoveredPoint.point[activeCfg.key]} {activeCfg.unit}
              </strong>
            </div>
            {typeof hoveredPoint.point.anomaly_score_pct === 'number' && (
              <div className="sg-tooltip-score">
                Risk Score: {hoveredPoint.point.anomaly_score_pct}%
              </div>
            )}
            {hoveredPoint.point.is_anomaly && (
              <>
                <div className="sg-tooltip-score">Fault: {hoveredPoint.point.fault_type?.replace(/_/g, ' ') || 'anomaly'}</div>
                {hoveredPoint.point.severity && (
                  <div className="sg-tooltip-score">Severity: {hoveredPoint.point.severity}</div>
                )}
                <SuggestedValues
                  items={suggestedFromTrendPoint(hoveredPoint.point)}
                  compact
                  showHeading
                  emptyLabel="Suggested replacement unavailable during baseline warm-up"
                />
                {hoveredPoint.point.health_status && <div className="sg-tooltip-time">Sensor: {hoveredPoint.point.health_status}</div>}
              </>
            )}
          </div>,
          document.body
        )}
      </div>

      {/* Accessible Textual Summary for Screen Readers */}
      <details className="sg-chart-accessible-summary">
        <summary>View {activeCfg.label} data points table</summary>
        <div className="sg-accessible-table-wrapper">
          <table className="sg-accessible-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>{activeCfg.label} ({activeCfg.unit})</th>
              </tr>
            </thead>
            <tbody>
              {windowedPoints.map((pt, i) => (
                <tr key={i}>
                  <td>{formatChartTime(pt.timestamp)}</td>
                  <td>{pt[activeCfg.key]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Card>
  );
};
