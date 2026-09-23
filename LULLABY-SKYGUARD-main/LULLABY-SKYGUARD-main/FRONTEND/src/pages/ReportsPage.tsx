import React from 'react';
import { useReportData } from '../hooks/useReportData';
import {
  ReportHeader,
  ReportPeriodSelector,
  ReportStationOverview,
  ReportTelemetrySection,
  ReportAnomalySection,
  ReportSensorHealthSection,
  ReportSpatialSection,
  ReportInsightsSection,
  ReportIncidentTable,
} from '../components/reports';
import { EmptyState } from '../components/common/EmptyState';
import './ReportsPage.css';

/**
 * ReportsPage
 * 
 * Step 10 — Reports & Operational Reporting
 * 
 * Generates structured operational review documents aggregating station metadata,
 * telemetry statistics, anomaly records, sensor health condition, spatial validation,
 * and actionable recommendations.
 * 
 * Supports browser-side print and PDF generation (`window.print()`).
 * [FRONTEND ONLY — DERIVED FROM EXISTING CONTRACT ENDPOINTS]
 */
export const ReportsPage: React.FC = () => {
  const {
    selectedStation,
    periodHours,
    setPeriodHours,
    selectedMetricTab,
    setSelectedMetricTab,
    report,
    isLoading,
    isGenerating,
    error,
    generateReport,
    printReport,
    refresh,
  } = useReportData();

  // ── No station selected ──────────────────────────────────────────────────────
  if (!isLoading && !selectedStation) {
    return (
      <div className="page-container sg-reports-page" role="main" aria-label="Reports & Operational Review">
        <div className="sg-reports-page__no-station">
          <EmptyState
            title="No Station Selected"
            description="Please select a meteorological station from the navigation bar to generate an operational report."
          />
        </div>
      </div>
    );
  }

  // ── Error state ──────────────────────────────────────────────────────────────
  if (error && !isLoading) {
    return (
      <div className="page-container sg-reports-page" role="main" aria-label="Reports & Operational Review">
        <ReportHeader
          station={selectedStation}
          metadata={report?.metadata ?? null}
          onGenerateReport={generateReport}
          onPrintReport={printReport}
          onRefresh={refresh}
          isLoading={false}
        />
        <div className="sg-reports-page__error">
          <div className="sg-reports-page__error-box" role="alert">
            <h2>Report Generation Unavailable</h2>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container sg-reports-page" role="main" aria-label="Station Operational Report & Export">
      {/* ── Page & Document Header ── */}
      <ReportHeader
        station={selectedStation}
        metadata={report?.metadata ?? null}
        onGenerateReport={generateReport}
        onPrintReport={printReport}
        onRefresh={refresh}
        isGenerating={isGenerating}
        isLoading={isLoading}
      />

      {/* ── Period Selector Controls (Hidden in Print) ── */}
      <ReportPeriodSelector
        periodHours={periodHours}
        onPeriodChange={setPeriodHours}
        disabled={isLoading || isGenerating}
      />

      {/* ── Report Document Body ── */}
      {report && (
        <article className="sg-report-document" role="document" aria-label="Station Operational Report Document">
          {/* Section 1: Observatory Station Overview */}
          <ReportStationOverview
            station={report.station}
            currentReading={report.currentReading}
            sensorHealth={report.sensorHealth}
          />

          {/* Section 2: Telemetry Summary & Compact Chart */}
          <ReportTelemetrySection
            temperature={report.telemetry.temperature}
            pressure={report.telemetry.pressure}
            humidity={report.telemetry.humidity}
            points={report.telemetry.points}
            periodHours={report.metadata.periodHours}
            selectedMetricTab={selectedMetricTab}
            onSelectMetricTab={setSelectedMetricTab}
          />

          {/* Section 3: Anomaly Summary & Distributions */}
          <ReportAnomalySection
            total={report.anomalies.total}
            highestSeverity={report.anomalies.highestSeverity}
            severityDistribution={report.anomalies.severityDistribution}
            typeDistribution={report.anomalies.typeDistribution}
            latestAnomaly={report.anomalies.latestAnomaly}
          />

          {/* Section 4: Sensor Hardware Condition */}
          <ReportSensorHealthSection
            sensorHealth={report.sensorHealth}
            currentReading={report.currentReading}
          />

          {/* Section 5: Regional Spatial Validation Context */}
          <ReportSpatialSection spatialSummary={report.spatial} />

          {/* Section 6: Operational Observations & Recommendations */}
          <ReportInsightsSection
            insights={report.insights}
            recommendations={report.recommendations}
          />

          {/* Section 7: Incident Records Log */}
          <ReportIncidentTable incidents={report.anomalies.incidents} />

          {/* Section 8: Official Operational Certification & Sign-off */}
          <footer className="sg-report-certification" role="contentinfo" aria-label="Official Certification">
            <div className="sg-report-certification__header">
              <span className="sg-report-certification__badge">AUDIT VERIFICATION</span>
              <span className="sg-report-certification__classification">OFFICIAL METEOROLOGICAL RECORD</span>
            </div>
            <p className="sg-report-certification__statement">
              This automated observational data quality assurance report was computed by the SkyGuard AI
              Distributed Telemetry Engine. All anomalies, thermodynamic boundary checks, and spatial neighbor
              deviations have been compiled in accordance with WMO-No. 8 meteorological sensor standards.
            </p>
            <div className="sg-report-certification__grid">
              <div className="sg-report-certification__col">
                <span className="sg-report-certification__label">Authorized Operator Signature:</span>
                <div className="sg-report-certification__sig-line" />
                <span className="sg-report-certification__sub">Meteorological Systems Engineer</span>
              </div>
              <div className="sg-report-certification__col">
                <span className="sg-report-certification__label">Station Dispatch Action:</span>
                <div className="sg-report-certification__val">
                  {report.sensorHealth && (report.sensorHealth.sensor_health_status === 'CRITICAL' || report.sensorHealth.sensor_health_status === 'OFFLINE' || report.sensorHealth.sensor_health_status === 'WARNING')
                    ? 'ACTION REQUIRED — SENSOR CALIBRATION / FIELD MAINTENANCE'
                    : 'ROUTINE MONITORING — SYSTEM NOMINAL'}
                </div>
              </div>
              <div className="sg-report-certification__col">
                <span className="sg-report-certification__label">Verification Timestamp:</span>
                <div className="sg-report-certification__val sg-font-mono">
                  {new Date(report.metadata.generatedAt).toLocaleString()}
                </div>
              </div>
            </div>
          </footer>
        </article>
      )}
    </div>
  );
};

export default ReportsPage;
