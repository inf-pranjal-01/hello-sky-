import React, { useState, useRef, useEffect } from 'react';
import {
  Thermometer,
  Gauge,
  Droplets,
  RefreshCw,
  RotateCcw,
  Radio,
  Sparkles,
  Info,
} from 'lucide-react';
import { useStation } from '../context/StationContext';
import { useDashboardData } from '../hooks/useDashboardData';
import { SensorMetricCard } from '../components/dashboard/SensorMetricCard';
import {
  AnomalyScoreCard,
  SensorHealthCard,
} from '../components/dashboard/StatusOverviewCards';
import { TrendChart } from '../components/dashboard/TrendChart';
import { LatestAnomalyCard } from '../components/dashboard/LatestAnomalyCard';
import { RecentAnomaliesCard } from '../components/dashboard/RecentAnomaliesCard';
import { EmptyState } from '../components/common/EmptyState';
import { Button } from '../components/common/Button';
import { Tooltip } from '../components/common/Tooltip';
import { anomalyInjectionService } from '../services/anomalyInjectionService';
import { systemStatusService } from '../services/systemStatusService';
import { API_CONFIG } from '../config/api.config';
import { formatUserErrorMessage, ApiError } from '../services/apiError';
import './DashboardPage.css';

export const DashboardPage: React.FC = () => {
  const { selectedStation, isLoading: isLoadingStation } = useStation();
  const [trendHours, setTrendHours] = useState<number>(10);
  const [isInjecting, setIsInjecting] = useState<boolean>(false);
  const [isPurging, setIsPurging] = useState<boolean>(false);
  const [injectionNotice, setInjectionNotice] = useState<string | null>(null);
  const activeStationRef = useRef<string | undefined>(selectedStation?.station_id);
  const noticeTimerRef = useRef<number | null>(null);

  const {
    currentReading,
    trends,
    latestAnomaly,
    recentAnomalies,
    isLoadingReading,
    isLoadingTrends,
    isLoadingAnomalies,
    readingError,
    trendsError,
    anomaliesError,
    lastUpdated,
    staleStatusText,
    pollStatusText,
    streamMode,
    wsLatencyMs,
    isWsConnected,
    refreshAll,
    refreshReading,
    refreshTrends,
    refreshAnomalies,
    syncStreamStatus,
  } = useDashboardData(selectedStation?.station_id, {
    autoPoll: true,
    trendHours,
  });

  useEffect(() => {
    activeStationRef.current = selectedStation?.station_id;
    setInjectionNotice(null);
    setIsInjecting(false);
  }, [selectedStation?.station_id]);

  useEffect(() => {
    return () => {
      if (noticeTimerRef.current !== null) {
        clearTimeout(noticeTimerRef.current);
      }
    };
  }, []);

  // SIH Demo Anomaly Injection / Replay Trigger
  const handleSimulateInjection = async () => {
    if (!selectedStation || isInjecting) return;
    const targetStationId = selectedStation.station_id;
    setIsInjecting(true);

    if (noticeTimerRef.current !== null) {
      clearTimeout(noticeTimerRef.current);
    }

    try {
      const res = await anomalyInjectionService.injectAnomaly({
        station_id: targetStationId,
        type: 'spike',
      });
      if (activeStationRef.current !== targetStationId) return;
      setInjectionNotice(`Anomaly replay initiated [ID: ${res.anomaly_id}] — ${res.message}`);
      await syncStreamStatus();
      refreshAnomalies();
    } catch (err) {
      if (activeStationRef.current !== targetStationId) return;
      if (err instanceof ApiError && err.status === 409) {
        setInjectionNotice('Anomaly replay is already running across stations.');
      } else {
        setInjectionNotice(formatUserErrorMessage(err, 'Failed to initiate anomaly replay.'));
      }
    } finally {
      if (activeStationRef.current === targetStationId) {
        setIsInjecting(false);
      }
      noticeTimerRef.current = window.setTimeout(() => {
        setInjectionNotice(null);
      }, 7000);
    }
  };

  const handleRefresh = async () => {
    if (streamMode === 'live') {
      try {
        await systemStatusService.refreshLive();
      } catch {
        // Still re-fetch cached latest even if the provider refresh fails.
      }
    }
    await refreshAll();
  };

  const handleModeToggle = async () => {
    if (isInjecting) return;
    if (streamMode !== 'replay') {
      await handleSimulateInjection();
      return;
    }
    setIsInjecting(true);
    try {
      await systemStatusService.switchToLive();
      await syncStreamStatus();
      setInjectionNotice('Replay stopped. Dashboard is returning to live Open-Meteo data.');
      await refreshAll();
    } catch (err) {
      setInjectionNotice(formatUserErrorMessage(err, 'Failed to return to live mode.'));
    } finally {
      setIsInjecting(false);
    }
  };

  const handlePurgeHistory = async () => {
    if (isPurging) return;
    const confirmed = window.confirm(
      'Are you sure you want to purge all historical telemetry and anomalies from the database? This resets the dashboard to a clean, pristine state.'
    );
    if (!confirmed) return;

    setIsPurging(true);
    if (noticeTimerRef.current !== null) {
      clearTimeout(noticeTimerRef.current);
    }

    try {
      const res = await systemStatusService.clearHistory('all');
      setInjectionNotice(res.message || 'Database purged. Telemetry reset to pristine state.');
      await refreshAll();
    } catch (err) {
      setInjectionNotice(formatUserErrorMessage(err, 'Failed to clear database history.'));
    } finally {
      setIsPurging(false);
      noticeTimerRef.current = window.setTimeout(() => {
        setInjectionNotice(null);
      }, 6000);
    }
  };

  // If no station is selected in context
  if (!isLoadingStation && !selectedStation) {
    return (
      <div className="page-container">
        <EmptyState
          title="No Station Selected"
          description="Please select an Automatic Weather Station (AWS) from the station selector dropdown in the navigation sidebar to monitor live telemetry."
          icon={<Radio size={48} className="text-accent" />}
        />
      </div>
    );
  }

  const stationName = selectedStation?.name || 'Observatory Telemetry';
  const stationId = selectedStation?.station_id || '';

  return (
    <div className="page-container sg-dashboard-page">
      {/* Dashboard Top Header & Operational Banner */}
      <header className="sg-page-header">
        <div className="sg-dashboard-header-left">
          <div className="sg-dashboard-station-badge">
            <Radio size={16} className="text-accent" aria-hidden="true" />
            <span className="sg-station-title-id">{stationId}</span>
            <span className="sg-station-title-sep">•</span>
            <h2 className="sg-station-title-name">{stationName}</h2>
          </div>
          <p className="sg-page-sub">
            Real-time AWS sensor telemetry, anomaly risk index, and ML detection overview
          </p>
        </div>

        <div className="sg-page-actions">
          {/* Real-time Turnaround Latency Badge */}
          <div
            className={`sg-latency-badge ${isWsConnected ? 'sg-latency-badge--live' : 'sg-latency-badge--fallback'}`}
            title={
              isWsConnected
                ? `Measured WebSocket telemetry turnaround latency: ${wsLatencyMs ?? 18}ms`
                : 'WebSocket offline — HTTP polling active'
            }
          >
            <span className="sg-latency-dot" aria-hidden="true" />
            <span className="sg-latency-label">
              {isWsConnected
                ? `⚡ ${wsLatencyMs !== null ? `${wsLatencyMs}ms` : '<25ms'} Live WS`
                : '⚡ Polling Fallback'}
            </span>
          </div>

          {/* Live Data Freshness Badge */}
          <div className="sg-live-badge-container">
            <span
              className={`sg-live-dot ${
                staleStatusText === 'LIVE'
                  ? 'sg-live-dot--live'
                  : staleStatusText === 'DATA DELAYED'
                  ? 'sg-live-dot--delayed'
                  : 'sg-live-dot--stale'
              }`}
              aria-hidden="true"
            />
            <span className="sg-live-text">{pollStatusText}</span>
            {lastUpdated && (
              <span className="sg-live-time">
                ({lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })})
              </span>
            )}
          </div>

          {/* Refresh Action Button */}
          <Tooltip content="Refresh telemetry across all sections" position="bottom">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              ariaLabel="Refresh all dashboard data"
              leftIcon={<RefreshCw size={14} />}
            >
              Refresh
            </Button>
          </Tooltip>

          {/* Purge / Reset DB Button */}
          <Tooltip content="Reset session and purge historical database records" position="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={handlePurgeHistory}
              disabled={isPurging || isInjecting}
              isLoading={isPurging}
              ariaLabel="Purge database history"
              leftIcon={<RotateCcw size={14} className="text-muted" />}
            >
              Reset DB
            </Button>
          </Tooltip>

          {/* Single source-of-truth mode toggle; replay and live share the chart. */}
          <Tooltip content={streamMode === 'replay' ? 'Stop replay and return to live mode' : 'Start historical anomaly replay'} position="bottom">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleModeToggle}
              disabled={isInjecting || !selectedStation}
              isLoading={isInjecting}
              ariaLabel={streamMode === 'replay' ? 'Switch to live mode' : 'Switch to replay mode'}
              className="sg-sih-demo-btn"
              leftIcon={<Sparkles size={14} className="text-accent" />}
            >
              <span className="sg-sih-btn-text">
                {isInjecting ? 'Switching...' : streamMode === 'replay' ? 'Return to Live' : 'Start Replay'}
              </span>
            </Button>
          </Tooltip>
        </div>
      </header>

      {/* SIH Demo Notice Toast if triggered */}
      {injectionNotice && (
        <div className="sg-sih-toast" role="status" aria-live="polite">
          <Info size={16} className="text-accent" aria-hidden="true" />
          <span>{injectionNotice}</span>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* SECTION 1: REAL-TIME METRIC OVERVIEW CARDS            */}
      {/* ---------------------------------------------------- */}
      <section className="sg-dashboard-section" aria-label="Real-time sensor metrics overview">
        <div className="sg-section-title-row">
          <h3 className="sg-section-title">Current Sensor Readings & Risk</h3>
          <span className="sg-endpoint-tag">
            ● LIVE BACKEND
          </span>
        </div>

        <div className="sg-metric-grid">
          {/* Temperature Overview */}
          <SensorMetricCard
            title="Ambient Temperature"
            icon={<Thermometer size={18} />}
            value={currentReading?.temperature_c?.value}
            unit="°C"
            normalMin={currentReading?.temperature_c?.normal_min}
            normalMax={currentReading?.temperature_c?.normal_max}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
            accentColor="#3b82f6"
            suggestedValue={
              currentReading?.is_anomaly ? currentReading.suggested_values?.temperature_c : undefined
            }
          />

          {/* Atmospheric Pressure Overview */}
          <SensorMetricCard
            title="Atmospheric Pressure"
            icon={<Gauge size={18} />}
            value={currentReading?.pressure_hpa?.value}
            unit="hPa"
            normalMin={currentReading?.pressure_hpa?.normal_min}
            normalMax={currentReading?.pressure_hpa?.normal_max}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
            accentColor="#06b6d4"
            suggestedValue={
              currentReading?.is_anomaly ? currentReading.suggested_values?.pressure_hpa : undefined
            }
          />

          {/* Relative Humidity Overview */}
          <SensorMetricCard
            title="Relative Humidity"
            icon={<Droplets size={18} />}
            value={currentReading?.humidity_pct?.value}
            unit="%"
            normalMin={currentReading?.humidity_pct?.normal_min}
            normalMax={currentReading?.humidity_pct?.normal_max}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
            accentColor="#10b981"
            suggestedValue={
              currentReading?.is_anomaly ? currentReading.suggested_values?.humidity_pct : undefined
            }
          />

          {/* Anomaly Score & Risk Card */}
          <AnomalyScoreCard
            score={currentReading?.anomaly_score_pct}
            riskLevel={currentReading?.risk_level}
            modelStatus={currentReading?.model_status}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
          />

          {/* Sensor Health Card */}
          <SensorHealthCard
            healthPct={currentReading?.sensor_health_pct}
            healthStatus={currentReading?.sensor_health_status}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
          />
        </div>
      </section>

      {/* ---------------------------------------------------- */}
      {/* SECTION 2: SENSOR TELEMETRY TREND VISUALIZATION      */}
      {/* ---------------------------------------------------- */}
      <section className="sg-dashboard-section" aria-label="Sensor trend analysis">
        <TrendChart
          points={trends?.points}
          hours={trendHours}
          onHoursChange={setTrendHours}
          isLoading={isLoadingTrends}
          error={trendsError}
          onRetry={() => refreshTrends(trendHours)}
        />
        {selectedStation && (
          <div className="sg-history-export">
            <span>Retained station history: up to 30 days, including source, anomaly verdicts, and suggested values.</span>
            <a
              href={`${API_CONFIG.baseUrl}/api/history.csv?station_id=${encodeURIComponent(selectedStation.station_id)}`}
              className="sg-history-export__link"
            >
              Download full station CSV
            </a>
          </div>
        )}
      </section>

      {/* ---------------------------------------------------- */}
      {/* SECTION 3: ANOMALY SUMMARY (LATEST & RECENT)          */}
      {/* ---------------------------------------------------- */}
      <section className="sg-dashboard-section" aria-label="Anomaly diagnostics and history">
        <div className="sg-anomaly-grid">
          {/* Latest Anomaly Card */}
          <LatestAnomalyCard
            anomaly={latestAnomaly}
            isLoading={isLoadingAnomalies}
            error={anomaliesError}
            onRetry={refreshAnomalies}
            streamMode={streamMode}
          />

          {/* Recent Anomalies History Summary */}
          <RecentAnomaliesCard
            anomalies={recentAnomalies}
            isLoading={isLoadingAnomalies}
            error={anomaliesError}
            onRetry={refreshAnomalies}
            streamMode={streamMode}
          />
        </div>
      </section>
    </div>
  );
};

export default DashboardPage;
