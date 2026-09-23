import React, { useMemo } from 'react';
import { BarChart2, Thermometer, Gauge, Droplets } from 'lucide-react';
import { MetricStatistics, TrendPoint, ReportPeriod } from '../../types';
import './ReportTelemetrySection.css';

export interface ReportTelemetrySectionProps {
  temperature: MetricStatistics;
  pressure: MetricStatistics;
  humidity: MetricStatistics;
  points: TrendPoint[];
  periodHours: ReportPeriod;
  selectedMetricTab: 'temperature_c' | 'pressure_hpa' | 'humidity_pct';
  onSelectMetricTab: (metric: 'temperature_c' | 'pressure_hpa' | 'humidity_pct') => void;
}

export const ReportTelemetrySection: React.FC<ReportTelemetrySectionProps> = ({
  temperature,
  pressure,
  humidity,
  points,
  periodHours,
  selectedMetricTab,
  onSelectMetricTab,
}) => {
  // Chart geometry
  const width = 800;
  const height = 200;
  const padding = { top: 20, right: 30, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const currentMetricConfig = useMemo(() => {
    switch (selectedMetricTab) {
      case 'temperature_c':
        return {
          title: 'Temperature Telemetry (°C)',
          unit: '°C',
          strokeColor: '#f59e0b',
          accessor: (p: TrendPoint) => p.temperature_c,
          stats: temperature,
        };
      case 'pressure_hpa':
        return {
          title: 'Barometric Pressure Telemetry (hPa)',
          unit: 'hPa',
          strokeColor: '#38bdf8',
          accessor: (p: TrendPoint) => p.pressure_hpa,
          stats: pressure,
        };
      case 'humidity_pct':
      default:
        return {
          title: 'Relative Humidity Telemetry (%)',
          unit: '%',
          strokeColor: '#10b981',
          accessor: (p: TrendPoint) => p.humidity_pct,
          stats: humidity,
        };
    }
  }, [selectedMetricTab, temperature, pressure, humidity]);

  // Coordinate projections
  const chartData = useMemo(() => {
    if (points.length === 0) return { pathD: '', areaD: '', yTicks: [] };

    const values = points.map(currentMetricConfig.accessor);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const buffer = (maxVal - minVal) * 0.1 || 1;
    const scaleMin = Math.floor(minVal - buffer);
    const scaleMax = Math.ceil(maxVal + buffer);
    const scaleRange = scaleMax - scaleMin || 1;

    const timestamps = points.map((point) => new Date(point.timestamp).getTime());
    const minTime = Math.min(...timestamps);
    const maxTime = Math.max(...timestamps);
    const coords = points.map((p) => {
      const val = currentMetricConfig.accessor(p);
      const timestamp = new Date(p.timestamp).getTime();
      const x = maxTime === minTime
        ? padding.left + chartWidth / 2
        : padding.left + ((timestamp - minTime) / (maxTime - minTime)) * chartWidth;
      const y = padding.top + chartHeight - ((val - scaleMin) / scaleRange) * chartHeight;
      return { x, y, val };
    });

    const pathD = coords.reduce(
      (acc, c, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${c.x} ${c.y}`,
      ''
    );
    const areaD = `${pathD} L ${coords[coords.length - 1].x} ${padding.top + chartHeight} L ${
      coords[0].x
    } ${padding.top + chartHeight} Z`;

    const yTicks = [0, 0.5, 1].map((pct) => {
      const val = Number((scaleMin + pct * scaleRange).toFixed(1));
      const y = padding.top + chartHeight - pct * chartHeight;
      return { val, y };
    });

    return { pathD, areaD, yTicks };
  }, [points, currentMetricConfig, chartWidth, chartHeight, padding]);

  return (
    <section className="sg-report-section" aria-labelledby="report-section-2-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <BarChart2 size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-2-heading" className="sg-report-section__title">
            2. Meteorological Telemetry Summary
          </h2>
        </div>
        <span className="sg-report-section__badge">
          DERIVED FROM LIVE BACKEND DATA
        </span>
      </div>

      {/* 3-Column Stats Grid */}
      <div className="sg-report-stats-grid">
        {/* Temperature Stats */}
        <div className="sg-report-stat-card">
          <div className="sg-report-stat-header">
            <Thermometer size={16} className="text-warning" aria-hidden="true" />
            <h3 className="sg-report-stat-title">Temperature</h3>
          </div>
          <div className="sg-report-stat-table">
            <div className="sg-report-stat-row">
              <span>Mean / Average:</span>
              <span className="sg-font-mono font-bold">{temperature.average.toFixed(1)} °C</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Minimum:</span>
              <span className="sg-font-mono">{temperature.min.toFixed(1)} °C</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Maximum:</span>
              <span className="sg-font-mono">{temperature.max.toFixed(1)} °C</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Thermal Range:</span>
              <span className="sg-font-mono">{temperature.range.toFixed(1)} °C</span>
            </div>
          </div>
        </div>

        {/* Pressure Stats */}
        <div className="sg-report-stat-card">
          <div className="sg-report-stat-header">
            <Gauge size={16} className="text-accent" aria-hidden="true" />
            <h3 className="sg-report-stat-title">Barometric Pressure</h3>
          </div>
          <div className="sg-report-stat-table">
            <div className="sg-report-stat-row">
              <span>Mean / Average:</span>
              <span className="sg-font-mono font-bold">{pressure.average.toFixed(1)} hPa</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Minimum:</span>
              <span className="sg-font-mono">{pressure.min.toFixed(1)} hPa</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Maximum:</span>
              <span className="sg-font-mono">{pressure.max.toFixed(1)} hPa</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Pressure Range:</span>
              <span className="sg-font-mono">{pressure.range.toFixed(1)} hPa</span>
            </div>
          </div>
        </div>

        {/* Humidity Stats */}
        <div className="sg-report-stat-card">
          <div className="sg-report-stat-header">
            <Droplets size={16} className="text-optimal" aria-hidden="true" />
            <h3 className="sg-report-stat-title">Relative Humidity</h3>
          </div>
          <div className="sg-report-stat-table">
            <div className="sg-report-stat-row">
              <span>Mean / Average:</span>
              <span className="sg-font-mono font-bold">{humidity.average.toFixed(1)} %</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Minimum:</span>
              <span className="sg-font-mono">{humidity.min.toFixed(1)} %</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Observed Maximum:</span>
              <span className="sg-font-mono">{humidity.max.toFixed(1)} %</span>
            </div>
            <div className="sg-report-stat-row">
              <span>Humidity Range:</span>
              <span className="sg-font-mono">{humidity.range.toFixed(1)} %</span>
            </div>
          </div>
        </div>
      </div>

      {/* Telemetry Chart with Tabs */}
      <div className="sg-report-chart-container">
        <div className="sg-report-chart-header">
          <span className="sg-report-chart-title">
            {currentMetricConfig.title} ({periodHours}H Trajectory)
          </span>

          <div
            className="sg-report-chart-tabs no-print"
            role="tablist"
            aria-label="Select report telemetry metric"
          >
            <button
              type="button"
              role="tab"
              aria-selected={selectedMetricTab === 'temperature_c'}
              className={`sg-report-chart-tab ${
                selectedMetricTab === 'temperature_c' ? 'sg-report-chart-tab--active' : ''
              }`}
              onClick={() => onSelectMetricTab('temperature_c')}
            >
              Temperature
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={selectedMetricTab === 'pressure_hpa'}
              className={`sg-report-chart-tab ${
                selectedMetricTab === 'pressure_hpa' ? 'sg-report-chart-tab--active' : ''
              }`}
              onClick={() => onSelectMetricTab('pressure_hpa')}
            >
              Pressure
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={selectedMetricTab === 'humidity_pct'}
              className={`sg-report-chart-tab ${
                selectedMetricTab === 'humidity_pct' ? 'sg-report-chart-tab--active' : ''
              }`}
              onClick={() => onSelectMetricTab('humidity_pct')}
            >
              Humidity
            </button>
          </div>
        </div>

        {/* Compact SVG Line Visualization */}
        {points.length > 0 && chartData.pathD ? (
          <div className="sg-report-svg-wrapper">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="sg-report-svg"
              role="img"
              aria-label={`Telemetry graph for ${currentMetricConfig.title} over ${periodHours} hours`}
            >
              <defs>
                <linearGradient id="reportGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor={currentMetricConfig.strokeColor} stopOpacity="0.3" />
                  <stop offset="100%" stopColor={currentMetricConfig.strokeColor} stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Gridlines */}
              {chartData.yTicks.map((t, idx) => (
                <g key={idx}>
                  <line
                    x1={padding.left}
                    y1={t.y}
                    x2={width - padding.right}
                    y2={t.y}
                    stroke="rgba(255, 255, 255, 0.05)"
                    strokeWidth="1"
                  />
                  <text
                    x={padding.left - 6}
                    y={t.y + 3}
                    textAnchor="end"
                    className="sg-report-axis-text"
                  >
                    {t.val} {currentMetricConfig.unit}
                  </text>
                </g>
              ))}

              {/* Area */}
              <path d={chartData.areaD} fill="url(#reportGrad)" />

              {/* Line */}
              <path
                d={chartData.pathD}
                fill="none"
                stroke={currentMetricConfig.strokeColor}
                strokeWidth="2"
              />
            </svg>
          </div>
        ) : (
          <p className="sg-report-empty-chart">No telemetry series points available for this period.</p>
        )}
      </div>
    </section>
  );
};
