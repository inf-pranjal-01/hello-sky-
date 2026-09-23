import React, { useState } from 'react';
import {
  Activity,
  Thermometer,
  Gauge,
  Droplets,
  Radio,
  Play,
  Pause,
  RefreshCw,
  HeartPulse,
} from 'lucide-react';
import { useStation } from '../context/StationContext';
import { useDashboardData } from '../hooks/useDashboardData';
import { SensorMetricCard } from '../components/dashboard/SensorMetricCard';
import { TrendChart } from '../components/dashboard/TrendChart';
import { TelemetryHistoryTable } from '../components/monitor/TelemetryHistoryTable';
import { StatusBadge } from '../components/common/StatusBadge';
import { Button } from '../components/common/Button';
import { Tooltip } from '../components/common/Tooltip';
import { EmptyState } from '../components/common/EmptyState';
import './MonitorPage.css';

export const MonitorPage: React.FC = () => {
  const { selectedStation, isLoading: isLoadingStation } = useStation();
  const [selectedHours, setSelectedHours] = useState<number>(10);

  // Single centralized polling architecture shared with dashboard
  const {
    currentReading,
    trends,
    telemetryHistory,
    isLoadingReading,
    isLoadingTrends,
    readingError,
    trendsError,
    lastUpdated,
    freshness,
    isPaused,
    togglePause,
    refreshAll,
    refreshReading,
    refreshTrends,
  } = useDashboardData(selectedStation?.station_id, {
    autoPoll: true,
    trendHours: selectedHours,
  });

  const handleHoursChange = (h: number) => {
    setSelectedHours(h);
    refreshTrends(h);
  };

  // If no station is selected in context
  if (!isLoadingStation && !selectedStation) {
    return (
      <div className="page-container">
        <EmptyState
          title="No Station Selected"
          description="Please select an Automatic Weather Station (AWS) from the station selector dropdown in the navigation sidebar to monitor high-frequency telemetry."
          icon={<Radio size={48} className="text-accent" />}
        />
      </div>
    );
  }

  const stationName = selectedStation?.name || 'Observatory Telemetry';
  const stationId = selectedStation?.station_id || 'AWS-001';
  const stationStatus = selectedStation?.status || 'NORMAL';

  // Sensor Health Status
  const healthStatus = currentReading?.sensor_health_status || 'HEALTHY';
  const healthPct = currentReading?.sensor_health_pct ?? 96;

  let healthBadgeStatus: 'optimal' | 'warning' | 'critical' | 'offline' = 'optimal';
  if (healthStatus === 'WARNING') healthBadgeStatus = 'warning';
  else if (healthStatus === 'CRITICAL') healthBadgeStatus = 'critical';
  else if (healthStatus === 'OFFLINE') healthBadgeStatus = 'offline';

  return (
    <div className="page-container sg-monitor-page">
      {/* ---------------------------------------------------- */}
      {/* 1. MONITOR PAGE HEADER & LIVE CONTROLS               */}
      {/* ---------------------------------------------------- */}
      <header className="sg-page-header">
        <div className="sg-monitor-header-left">
          <div className="sg-monitor-station-row">
            <div className="sg-monitor-title-badge">
              <Activity size={18} className="text-accent" aria-hidden="true" />
              <h2 className="sg-monitor-main-title">Real-Time Telemetry Monitor</h2>
            </div>
            <span className="sg-station-id-pill">{stationId}</span>
            <StatusBadge status={stationStatus.toLowerCase() as any} label={`STATUS: ${stationStatus}`} size="sm" />
          </div>
          <p className="sg-page-sub">
            {stationName} • Detailed multi-sensor telemetry stream & high-resolution observation
          </p>
        </div>

        <div className="sg-page-actions sg-monitor-header-actions">
          {/* Freshness Status Pill */}
          <div
            className={`sg-freshness-pill ${
              isPaused
                ? 'sg-freshness-pill--paused'
                : freshness.status === 'LIVE'
                ? 'sg-freshness-pill--live'
                : freshness.status === 'DATA DELAYED'
                ? 'sg-freshness-pill--delayed'
                : 'sg-freshness-pill--stale'
            }`}
            aria-live="polite"
          >
            <span className="sg-freshness-dot" aria-hidden="true" />
            <span className="sg-freshness-label">{freshness.label}</span>
            {lastUpdated && !isPaused && (
              <span className="sg-freshness-time">
                ({freshness.secondsAgo}s ago)
              </span>
            )}
          </div>

          {/* Pause / Resume Monitoring Toggle [FRONTEND ONLY] */}
          <Tooltip
            content={
              isPaused
                ? 'Resume frontend polling data stream'
                : 'Pause frontend polling stream (data is retained)'
            }
            position="bottom"
          >
            <Button
              variant={isPaused ? 'primary' : 'outline'}
              size="sm"
              onClick={togglePause}
              ariaLabel={isPaused ? 'Resume live monitoring' : 'Pause live monitoring'}
              className="sg-pause-btn"
              leftIcon={isPaused ? <Play size={15} /> : <Pause size={15} />}
            >
              {isPaused ? 'Resume Monitoring' : 'Pause Monitoring'}
            </Button>
          </Tooltip>

          {/* Manual Telemetry Refresh Button */}
          <Tooltip content="Trigger immediate telemetry sample fetch" position="bottom">
            <Button
              variant="outline"
              size="sm"
              onClick={refreshAll}
              disabled={isPaused}
              ariaLabel="Refresh telemetry data manual action"
              leftIcon={<RefreshCw size={14} />}
            >
              Refresh Now
            </Button>
          </Tooltip>
        </div>
      </header>

      {/* Pause Notification Banner */}
      {isPaused && (
        <div className="sg-paused-banner" role="status">
          <Pause size={16} className="text-warning" aria-hidden="true" />
          <span>
            <strong>MONITORING PAUSED (FRONTEND ONLY):</strong> Telemetry polling is halted. Existing chart history and buffered samples are preserved. Click &ldquo;Resume Monitoring&rdquo; to restart live updates.
          </span>
        </div>
      )}

      {/* ---------------------------------------------------- */}
      {/* 2. CURRENT SENSOR OVERVIEW (3 SENSORS + STATUS)      */}
      {/* ---------------------------------------------------- */}
      <section className="sg-monitor-section" aria-label="Current sensor measurements">
        <div className="sg-section-header-row">
          <h3 className="sg-section-title">Current Sensor Overview</h3>
          <span className="sg-section-tag">
            ● LIVE BACKEND
          </span>
        </div>

        <div className="sg-sensor-cards-grid">
          {/* Temperature Sensor Overview Card */}
          <SensorMetricCard
            title="Temperature"
            icon={<Thermometer size={18} />}
            value={currentReading?.temperature_c?.value}
            unit="°C"
            normalMin={currentReading?.temperature_c?.normal_min}
            normalMax={currentReading?.temperature_c?.normal_max}
            isLoading={isLoadingReading}
            error={readingError}
            onRetry={refreshReading}
            accentColor="#3b82f6"
          />

          {/* Pressure Sensor Overview Card */}
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
          />

          {/* Humidity Sensor Overview Card */}
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
          />

          {/* Sensor Hardware Health Compact Indicator */}
          <div className="sg-sensor-health-card">
            <div className="sg-health-card-header">
              <span className="sg-health-card-title">Hardware Telemetry Health</span>
              <HeartPulse size={18} className={healthStatus === 'HEALTHY' ? 'text-optimal' : 'text-critical'} aria-hidden="true" />
            </div>
            <div className="sg-health-card-body">
              <div className="sg-health-number-row">
                <span className="sg-health-number">{healthPct}%</span>
                <StatusBadge status={healthBadgeStatus} label={healthStatus} size="sm" />
              </div>
              <p className="sg-health-sub">
                Overall physical transducer condition index (● LIVE BACKEND)
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------- */}
      {/* 3. HIGH-RESOLUTION LIVE TELEMETRY CHART             */}
      {/* ---------------------------------------------------- */}
      <section className="sg-monitor-section" aria-label="Expanded live telemetry trend visualization">
        <TrendChart
          points={trends?.points}
          hours={selectedHours}
          onHoursChange={handleHoursChange}
          isLoading={isLoadingTrends}
          error={trendsError}
          onRetry={() => refreshTrends(selectedHours)}
        />
      </section>

      {/* ---------------------------------------------------- */}
      {/* 4. DETAILED RECENT TELEMETRY HISTORY TABLE           */}
      {/* ---------------------------------------------------- */}
      <section className="sg-monitor-section" aria-label="Sequential telemetry reading log">
        <TelemetryHistoryTable
          records={telemetryHistory}
          isLoading={isLoadingReading}
          isPaused={isPaused}
        />
      </section>
    </div>
  );
};

export default MonitorPage;
