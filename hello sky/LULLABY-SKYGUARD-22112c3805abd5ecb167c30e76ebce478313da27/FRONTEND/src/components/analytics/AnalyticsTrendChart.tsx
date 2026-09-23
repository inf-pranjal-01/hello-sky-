import React, { useState, useMemo } from 'react';
import { LineChart, Table, EyeOff } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import { TrendsResponse, CurrentSensorReading, TrendPoint } from '../../types';
import './AnalyticsTrendChart.css';

export interface AnalyticsTrendChartProps {
  trends: TrendsResponse | null;
  currentReading: CurrentSensorReading | null;
  hours: number;
  selectedMetric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct';
  onSelectMetric: (metric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct') => void;
  isLoading?: boolean;
  className?: string;
}

const isMetricAnomalous = (pt: TrendPoint, metric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct'): boolean => {
  const isOverall = (pt.anomaly_score_pct || 0) > 75 || pt.is_anomaly === true;
  if (!isOverall) return false;
  const hasSuggested = {
    temperature_c: pt.suggested_temperature_c != null,
    pressure_hpa: pt.suggested_pressure_hpa != null,
    humidity_pct: pt.suggested_humidity_pct != null,
  };
  if (hasSuggested.temperature_c || hasSuggested.pressure_hpa || hasSuggested.humidity_pct) {
    return Boolean(hasSuggested[metric]);
  }
  const ft = (pt.fault_type || '').toLowerCase();
  if (ft.includes('temp')) return metric === 'temperature_c';
  if (ft.includes('press')) return metric === 'pressure_hpa';
  if (ft.includes('humid') || ft.includes('dew')) return metric === 'humidity_pct';
  return true;
};

export const AnalyticsTrendChart: React.FC<AnalyticsTrendChartProps> = ({
  trends,
  currentReading,
  hours,
  selectedMetric,
  onSelectMetric,
  isLoading = false,
  className = '',
}) => {
  const [showTable, setShowTable] = useState<boolean>(false);

  const points = trends?.points || [];

  // Metric configuration
  const metricConfig = useMemo(() => {
    switch (selectedMetric) {
      case 'temperature_c':
        return {
          title: 'Temperature Telemetry Trend',
          unit: '°C',
          lineClass: 'sg-analytics-chart__line--temp',
          strokeColor: '#f59e0b',
          normalMin: currentReading?.temperature_c.normal_min ?? 15,
          normalMax: currentReading?.temperature_c.normal_max ?? 35,
          valueAccessor: (p: { temperature_c: number }) => p.temperature_c,
        };
      case 'pressure_hpa':
        return {
          title: 'Barometric Pressure Telemetry Trend',
          unit: 'hPa',
          lineClass: 'sg-analytics-chart__line--press',
          strokeColor: '#38bdf8',
          normalMin: currentReading?.pressure_hpa.normal_min ?? 980,
          normalMax: currentReading?.pressure_hpa.normal_max ?? 1030,
          valueAccessor: (p: { pressure_hpa: number }) => p.pressure_hpa,
        };
      case 'humidity_pct':
      default:
        return {
          title: 'Relative Humidity Telemetry Trend',
          unit: '%',
          lineClass: 'sg-analytics-chart__line--hum',
          strokeColor: '#10b981',
          normalMin: currentReading?.humidity_pct.normal_min ?? 30,
          normalMax: currentReading?.humidity_pct.normal_max ?? 90,
          valueAccessor: (p: { humidity_pct: number }) => p.humidity_pct,
        };
    }
  }, [selectedMetric, currentReading]);

  // Descriptive text summary for accessibility
  const summaryText = useMemo(() => {
    if (points.length === 0) {
      return 'No telemetry trend records available for this observation window.';
    }
    const values = points.map(metricConfig.valueAccessor);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const latest = values[values.length - 1];
    return `${metricConfig.title} over the last ${hours} hours ranged from ${min.toFixed(1)} ${metricConfig.unit} to ${max.toFixed(1)} ${metricConfig.unit}. The latest observed value is ${latest.toFixed(1)} ${metricConfig.unit}. Reference baseline: ${metricConfig.normalMin} to ${metricConfig.normalMax} ${metricConfig.unit}.`;
  }, [points, metricConfig, hours]);

  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-analytics-chart-card ${className}`}>
        <div className="sg-analytics-chart__header">
          <Skeleton width="220px" height="1.4rem" />
          <Skeleton width="180px" height="1.8rem" />
        </div>
        <Skeleton width="100%" height="260px" style={{ marginTop: '1rem' }} />
      </Card>
    );
  }

  if (points.length === 0) {
    return (
      <Card variant="glass" className={`sg-analytics-chart-card ${className}`}>
        <EmptyState
          title="No Historical Telemetry"
          description={`No telemetry trend points were recorded for this station during the last ${hours} hours.`}
        />
      </Card>
    );
  }

  // SVG Chart Geometry
  const svgWidth = 900;
  const svgHeight = 260;
  const padding = { top: 25, right: 35, bottom: 35, left: 55 };
  const chartWidth = svgWidth - padding.left - padding.right;
  const chartHeight = svgHeight - padding.top - padding.bottom;

  const rawValues = points.map(metricConfig.valueAccessor);
  const dataMin = Math.min(...rawValues, metricConfig.normalMin);
  const dataMax = Math.max(...rawValues, metricConfig.normalMax);
  const buffer = (dataMax - dataMin) * 0.1 || 2;
  const scaleMin = Math.floor(dataMin - buffer);
  const scaleMax = Math.ceil(dataMax + buffer);
  const scaleRange = scaleMax - scaleMin || 1;

  const timestamps = points.map((point) => new Date(point.timestamp).getTime());
  const minTime = Math.min(...timestamps);
  const maxTime = Math.max(...timestamps);
  const coordinates = points.map((pt) => {
    const val = metricConfig.valueAccessor(pt);
    const timestamp = new Date(pt.timestamp).getTime();
    const x = maxTime === minTime
      ? padding.left + chartWidth / 2
      : padding.left + ((timestamp - minTime) / (maxTime - minTime)) * chartWidth;
    const y = padding.top + chartHeight - ((val - scaleMin) / scaleRange) * chartHeight;
    const isAnomaly = isMetricAnomalous(pt, selectedMetric);
    return { x, y, val, timestamp: pt.timestamp, isAnomaly, score: pt.anomaly_score_pct };
  });

  const pathD = coordinates.reduce(
    (acc, curr, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${curr.x} ${curr.y}`,
    ''
  );

  const areaD = `${pathD} L ${coordinates[coordinates.length - 1].x} ${
    padding.top + chartHeight
  } L ${coordinates[0].x} ${padding.top + chartHeight} Z`;

  // Normal Baseline Band Y Coordinates
  const normalTopY = Math.max(
    padding.top,
    padding.top + chartHeight - ((metricConfig.normalMax - scaleMin) / scaleRange) * chartHeight
  );
  const normalBottomY = Math.min(
    padding.top + chartHeight,
    padding.top + chartHeight - ((metricConfig.normalMin - scaleMin) / scaleRange) * chartHeight
  );
  const normalBandHeight = Math.max(0, normalBottomY - normalTopY);

  // Y-axis grid increments (5 levels)
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((pct) => {
    const val = scaleMin + pct * scaleRange;
    const y = padding.top + chartHeight - pct * chartHeight;
    return { val: Number(val.toFixed(1)), y };
  });

  return (
    <Card
      variant="glass"
      className={`sg-analytics-chart-card ${className}`}
      role="region"
      aria-label="Historical telemetry analytics chart"
    >
      <div className="sg-analytics-chart__header">
        <div className="sg-analytics-chart__title-area">
          <div className="sg-analytics-chart__title-row">
            <LineChart size={18} className="text-accent" aria-hidden="true" />
            <h3 className="sg-analytics-chart__title">{metricConfig.title}</h3>
          </div>
          <span className="sg-analytics-chart__notice">
            ● LIVE BACKEND
          </span>
        </div>

        {/* Metric Switching Tabs */}
        <div
          className="sg-analytics-chart__metric-tabs"
          role="tablist"
          aria-label="Telemetry metric selector"
        >
          <button
            type="button"
            role="tab"
            aria-selected={selectedMetric === 'temperature_c'}
            className={`sg-analytics-chart__metric-btn ${
              selectedMetric === 'temperature_c' ? 'sg-analytics-chart__metric-btn--active' : ''
            }`}
            onClick={() => onSelectMetric('temperature_c')}
          >
            Temperature (°C)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedMetric === 'pressure_hpa'}
            className={`sg-analytics-chart__metric-btn ${
              selectedMetric === 'pressure_hpa' ? 'sg-analytics-chart__metric-btn--active' : ''
            }`}
            onClick={() => onSelectMetric('pressure_hpa')}
          >
            Pressure (hPa)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedMetric === 'humidity_pct'}
            className={`sg-analytics-chart__metric-btn ${
              selectedMetric === 'humidity_pct' ? 'sg-analytics-chart__metric-btn--active' : ''
            }`}
            onClick={() => onSelectMetric('humidity_pct')}
          >
            Humidity (%)
          </button>
        </div>
      </div>

      {/* Accessible Text Summary */}
      <p className="sg-analytics-chart__summary-text" aria-live="polite">
        {summaryText}
      </p>

      {/* SVG Time Series Chart */}
      <div className="sg-analytics-chart__svg-wrapper">
        <svg
          className="sg-analytics-chart__svg"
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={summaryText}
        >
          <defs>
            <linearGradient id={`analyticsGrad-${selectedMetric}`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={metricConfig.strokeColor} stopOpacity="0.35" />
              <stop offset="100%" stopColor={metricConfig.strokeColor} stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Reference Normal Operating Range Band */}
          {normalBandHeight > 0 && (
            <rect
              x={padding.left}
              y={normalTopY}
              width={chartWidth}
              height={normalBandHeight}
              className="sg-analytics-chart__baseline-band"
            >
              <title>{`Nominal Baseline Range: ${metricConfig.normalMin} - ${metricConfig.normalMax} ${metricConfig.unit}`}</title>
            </rect>
          )}

          {/* Gridlines & Y-Axis */}
          {yTicks.map((tick, i) => (
            <g key={i}>
              <line
                x1={padding.left}
                y1={tick.y}
                x2={svgWidth - padding.right}
                y2={tick.y}
                className="sg-analytics-chart__grid-line"
              />
              <text
                x={padding.left - 8}
                y={tick.y + 3}
                textAnchor="end"
                className="sg-analytics-chart__axis-text"
              >
                {tick.val}
              </text>
            </g>
          ))}

          {/* Area Fill */}
          <path
            d={areaD}
            fill={`url(#analyticsGrad-${selectedMetric})`}
            className="sg-analytics-chart__area"
          />

          {/* Main Line */}
          <path d={pathD} className={`sg-analytics-chart__line ${metricConfig.lineClass}`} />

          {/* Data Points */}
          {coordinates.map((coord, idx) => {
            const timeStr = new Date(coord.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            });
            const pointTitle = `${timeStr}: ${coord.val} ${metricConfig.unit}${
              coord.isAnomaly ? ` (ANOMALY: Score ${coord.score}%)` : ''
            }`;

            return (
              <circle
                key={idx}
                cx={coord.x}
                cy={coord.y}
                r={coord.isAnomaly ? 5 : 3}
                fill={coord.isAnomaly ? '#ef4444' : '#0b1120'}
                stroke={coord.isAnomaly ? '#fff' : metricConfig.strokeColor}
                strokeWidth={coord.isAnomaly ? 2 : 1.5}
                className={`sg-analytics-chart__point ${
                  coord.isAnomaly ? 'sg-analytics-chart__point--anomaly' : ''
                }`}
              >
                <title>{pointTitle}</title>
              </circle>
            );
          })}
        </svg>
      </div>

      {/* Screen Reader Table Toggle */}
      <div className="sg-analytics-chart__table-toggle">
        <button
          type="button"
          className="sg-analytics-chart__toggle-btn"
          onClick={() => setShowTable((prev) => !prev)}
          aria-expanded={showTable}
        >
          {showTable ? (
            <EyeOff size={13} style={{ display: 'inline', marginRight: '4px' }} />
          ) : (
            <Table size={13} style={{ display: 'inline', marginRight: '4px' }} />
          )}
          {showTable ? 'Hide Raw Telemetry Table' : 'Show Raw Telemetry Table for Screen Readers'}
        </button>
      </div>

      {showTable && (
        <div className="sg-analytics-chart__table-wrapper">
          <table className="sg-analytics-chart__table" aria-label="Historical telemetry readings table">
            <thead>
              <tr>
                <th scope="col">Timestamp</th>
                <th scope="col">Observed Value ({metricConfig.unit})</th>
                <th scope="col">Status Flag</th>
              </tr>
            </thead>
            <tbody>
              {coordinates.map((coord, idx) => (
                <tr key={idx}>
                  <td>{new Date(coord.timestamp).toLocaleTimeString()}</td>
                  <td className="sg-font-mono">{coord.val}</td>
                  <td>{coord.isAnomaly ? 'ANOMALY DETECTED' : 'NORMAL'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
};
