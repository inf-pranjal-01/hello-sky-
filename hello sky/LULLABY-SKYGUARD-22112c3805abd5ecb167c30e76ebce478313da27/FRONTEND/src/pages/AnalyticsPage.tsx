import React, { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useStation } from '../context/StationContext';
import { useAnalyticsData } from '../hooks/useAnalyticsData';
import {
  AnalyticsHeader,
  AnalyticsSummaryCards,
  MetricStatisticsCard,
  AnalyticsTrendChart,
  AnomalyAnalytics,
  AnomalyTimeline,
  SensorHealthTrend,
  AnalyticsInsights,
  ExplainabilityCommandCenter,
} from '../components/analytics';
import { EmptyState } from '../components/common/EmptyState';
import './AnalyticsPage.css';

/**
 * AnalyticsPage
 *
 * Step 8 — Analytics & Insights Dashboard
 *
 * All analytics are derived frontend-side from existing telemetry/anomaly
 * feeds. No dedicated analytics backend endpoint is called.
 *
 * [FRONTEND ONLY] [DERIVED FROM LIVE BACKEND DATA]
 */
const AnalyticsPage: React.FC = () => {
  const { stations, selectedStation, setSelectedStation, isLoading: isLoadingStation } = useStation();
  const [searchParams] = useSearchParams();
  const targetStationId = searchParams.get('station_id');
  const targetAnomalyId = searchParams.get('anomaly_id');

  // Synchronise active station if deep-linked via URL query parameter
  useEffect(() => {
    if (targetStationId && stations.length > 0 && selectedStation?.station_id !== targetStationId) {
      const match = stations.find((s) => s.station_id === targetStationId);
      if (match) {
        setSelectedStation(match);
      }
    }
  }, [targetStationId, stations, selectedStation, setSelectedStation]);

  const stationId = selectedStation?.station_id ?? null;

  const {
    hours,
    setHours,
    selectedMetric,
    setSelectedMetric,
    trends,
    currentReading,
    anomalies,
    analyticsSummary,
    isLoading,
    error,
    staleStatusText,
    refresh,
  } = useAnalyticsData(stationId, 24);

  // ── No station selected ──────────────────────────────────────────────────────
  if (!isLoadingStation && !selectedStation) {
    return (
      <main className="sg-analytics-page" aria-label="Analytics page">
        <div className="sg-analytics-page__no-station">
          <EmptyState
            title="No Station Selected"
            description="Please select a weather station from the navigation bar to view analytics data."
          />
        </div>
      </main>
    );
  }

  // ── Data error ───────────────────────────────────────────────────────────────
  if (error && !isLoading) {
    return (
      <main className="sg-analytics-page" aria-label="Analytics page">
        <AnalyticsHeader
          station={selectedStation}
          hours={hours}
          onHoursChange={setHours}
          staleStatusText={staleStatusText}
          onRefresh={refresh}
          isLoading={false}
        />
        <div className="sg-analytics-page__error">
          <div className="sg-analytics-page__error-box" role="alert">
            <h2>Analytics Unavailable</h2>
            <p>{error}</p>
          </div>
        </div>
      </main>
    );
  }

  // ── Main layout ──────────────────────────────────────────────────────────────
  return (
    <main className="sg-analytics-page" aria-label="Analytics & Insights Dashboard">
      {/* ── Header ── */}
      <AnalyticsHeader
        station={selectedStation}
        hours={hours}
        onHoursChange={setHours}
        staleStatusText={staleStatusText}
        onRefresh={refresh}
        isLoading={isLoading}
      />

      {/* ── KPI Summary Cards ── */}
      <AnalyticsSummaryCards summary={analyticsSummary} isLoading={isLoading} />

      {/* ── Metric Statistics ── */}
      <MetricStatisticsCard
        temperatureStats={analyticsSummary?.temperature}
        pressureStats={analyticsSummary?.pressure}
        humidityStats={analyticsSummary?.humidity}
        currentReading={currentReading}
        isLoading={isLoading}
      />

      {/* ── Trend Chart ── */}
      <AnalyticsTrendChart
        trends={trends}
        currentReading={currentReading}
        hours={hours}
        selectedMetric={selectedMetric}
        onSelectMetric={setSelectedMetric}
        isLoading={isLoading}
      />

      <ExplainabilityCommandCenter
        anomalies={anomalies}
        isLoading={isLoading}
        initialAnomalyId={targetAnomalyId ?? undefined}
      />

      {/* ── Anomaly Distribution Cards ── */}
      <p className="sg-analytics-page__section-label" aria-hidden="true">
        Anomaly Distribution — DERIVED FROM LIVE BACKEND DATA
      </p>
      <AnomalyAnalytics analyticsSummary={analyticsSummary} isLoading={isLoading} />

      {/* ── Anomaly Timeline Table ── */}
      <AnomalyTimeline anomalies={anomalies} isLoading={isLoading} />

      {/* ── Sensor Health Trajectory (reuses HealthHistoryChart) ── */}
      <p className="sg-analytics-page__section-label" aria-hidden="true">
        Sensor Health Trend — DEMO TREND — FRONTEND DERIVED
      </p>
      <SensorHealthTrend stationId={stationId} />

      {/* ── Operational Insights ── */}
      <AnalyticsInsights
        insights={analyticsSummary?.insights ?? []}
        isLoading={isLoading}
      />
    </main>
  );
};

export default AnalyticsPage;

// Named export alias for compatibility with AppRouter's named import pattern
export { AnalyticsPage };
