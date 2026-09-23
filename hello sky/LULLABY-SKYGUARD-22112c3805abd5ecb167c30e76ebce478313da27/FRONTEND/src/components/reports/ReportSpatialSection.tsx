import React from 'react';
import { Network, CheckCircle, AlertTriangle, HelpCircle } from 'lucide-react';
import { SpatialComparisonSummary } from '../../types';
import './ReportSpatialSection.css';

export interface ReportSpatialSectionProps {
  spatialSummary: SpatialComparisonSummary | null | undefined;
}

export const ReportSpatialSection: React.FC<ReportSpatialSectionProps> = ({ spatialSummary }) => {
  if (!spatialSummary) {
    return (
      <section className="sg-report-section" aria-labelledby="report-section-5-heading">
        <div className="sg-report-section__header">
          <div className="sg-report-section__title-group">
            <Network size={18} className="text-accent" aria-hidden="true" />
            <h2 id="report-section-5-heading" className="sg-report-section__title">
              5. Regional Spatial Consistency
            </h2>
          </div>
          <span className="sg-report-section__badge">DEMO SPATIAL COMPARISON</span>
        </div>
        <p className="sg-report-spatial-empty">
          Spatial telemetry comparison is unavailable for this reporting interval.
        </p>
      </section>
    );
  }

  const { consistencyStatus, temperature, pressure, humidity, neighborCount, multivariateSummary } =
    spatialSummary;

  const isConsistent = consistencyStatus === 'CONSISTENT';
  const isDeviation = consistencyStatus === 'DEVIATION_DETECTED';

  return (
    <section className="sg-report-section" aria-labelledby="report-section-5-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <Network size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-5-heading" className="sg-report-section__title">
            5. Regional Spatial Validation & Neighborhood Consistency
          </h2>
        </div>
        <span className="sg-report-section__badge">
          DEMO SPATIAL COMPARISON — FRONTEND DERIVED
        </span>
      </div>

      {/* Spatial Verdict Banner */}
      <div
        className={`sg-report-spatial-verdict ${
          isConsistent
            ? 'sg-report-spatial-verdict--ok'
            : isDeviation
            ? 'sg-report-spatial-verdict--dev'
            : 'sg-report-spatial-verdict--insufficient'
        }`}
      >
        <div className="sg-report-spatial-verdict-icon">
          {isConsistent && <CheckCircle size={20} className="text-optimal" aria-hidden="true" />}
          {isDeviation && <AlertTriangle size={20} className="text-warning" aria-hidden="true" />}
          {!isConsistent && !isDeviation && (
            <HelpCircle size={20} className="text-muted" aria-hidden="true" />
          )}
        </div>

        <div className="sg-report-spatial-verdict-content">
          <span className="sg-report-spatial-verdict-tag">
            SPATIAL STATUS:{' '}
            {isConsistent
              ? 'REGIONALLY CONSISTENT'
              : isDeviation
              ? 'LOCALIZED DEVIATION DETECTED'
              : 'INSUFFICIENT DATA'}
          </span>
          <p className="sg-report-spatial-verdict-text">{multivariateSummary}</p>
        </div>
      </div>

      {/* 3 Metric Delta Grid */}
      <div className="sg-report-spatial-deltas-grid">
        <div className="sg-report-spatial-delta-card">
          <span className="sg-report-spatial-delta-label">Temperature Delta vs Neighbors</span>
          <span
            className={`sg-report-spatial-delta-val sg-font-mono ${
              temperature.isSignificantDeviation ? 'text-warning' : 'text-optimal'
            }`}
          >
            {temperature.difference > 0 ? '+' : ''}
            {temperature.difference.toFixed(1)} °C
          </span>
          <span className="sg-report-spatial-delta-sub">
            Station: {temperature.selected.toFixed(1)} °C | Neighbor Mean:{' '}
            {temperature.neighborAverage.toFixed(1)} °C
          </span>
        </div>

        <div className="sg-report-spatial-delta-card">
          <span className="sg-report-spatial-delta-label">Barometric Pressure Delta</span>
          <span
            className={`sg-report-spatial-delta-val sg-font-mono ${
              pressure.isSignificantDeviation ? 'text-warning' : 'text-optimal'
            }`}
          >
            {pressure.difference > 0 ? '+' : ''}
            {pressure.difference.toFixed(1)} hPa
          </span>
          <span className="sg-report-spatial-delta-sub">
            Station: {pressure.selected.toFixed(1)} hPa | Neighbor Mean:{' '}
            {pressure.neighborAverage.toFixed(1)} hPa
          </span>
        </div>

        <div className="sg-report-spatial-delta-card">
          <span className="sg-report-spatial-delta-label">Relative Humidity Delta</span>
          <span
            className={`sg-report-spatial-delta-val sg-font-mono ${
              humidity.isSignificantDeviation ? 'text-warning' : 'text-optimal'
            }`}
          >
            {humidity.difference > 0 ? '+' : ''}
            {humidity.difference.toFixed(1)} %
          </span>
          <span className="sg-report-spatial-delta-sub">
            Station: {humidity.selected.toFixed(1)} % | Neighbor Mean:{' '}
            {humidity.neighborAverage.toFixed(1)} %
          </span>
        </div>
      </div>

      {/* Spatial Disclaimer */}
      <p className="sg-report-spatial-disclaimer">
        <em>
          Spatial comparison is evaluated against {neighborCount} neighboring observatory station
          {neighborCount > 1 ? 's' : ''} using client-side demonstration logic. Production spatial
          cross-validation will consume live streaming telemetry from neighboring AWS nodes via future
          backend services.
        </em>
      </p>
    </section>
  );
};
