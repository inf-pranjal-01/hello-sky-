import { useState, useEffect, useCallback, useRef } from 'react';
import {
  CurrentSensorReading,
  TrendsResponse,
  LatestAnomaly,
  RecentAnomalyItem,
  TelemetryHistoryRecord,
  SensorHealth,
  SystemStreamStatus,
} from '../types';
import { currentReadingService } from '../services/currentReadingService';
import { trendsService } from '../services/trendsService';
import { anomalyService } from '../services/anomalyService';
import { sensorHealthService } from '../services/sensorHealthService';
import { systemStatusService } from '../services/systemStatusService';
import { ApiError, formatUserErrorMessage } from '../services/apiError';
import { calculateFreshness, FreshnessState } from '../utils/freshness';
import { TELEMETRY_REFRESH_EVENT } from '../utils/refreshEvents';
import { API_CONFIG } from '../config/api.config';

export interface DashboardDataState {
  currentReading: CurrentSensorReading | null;
  trends: TrendsResponse | null;
  latestAnomaly: LatestAnomaly | null;
  recentAnomalies: RecentAnomalyItem[];
  telemetryHistory: TelemetryHistoryRecord[];
  sensorHealth: SensorHealth | null;
  
  // Section-isolated loading states
  isLoadingReading: boolean;
  isLoadingTrends: boolean;
  isLoadingAnomalies: boolean;
  isLoadingHealth: boolean;

  // Section-isolated error states
  readingError: string | null;
  trendsError: string | null;
  anomaliesError: string | null;
  healthError: string | null;

  // Real-time metadata & freshness
  lastUpdated: Date | null;
  isStale: boolean;
  isDelayed: boolean;
  staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE';
  freshness: FreshnessState;
  pollStatusText: string;
  streamMode: 'live' | 'replay';

  // Live WebSocket Ingestion & Latency Readout
  wsLatencyMs: number | null;
  isWsConnected: boolean;

  // Pause / Resume controls [FRONTEND ONLY]
  isPaused: boolean;
  isPreWarming: boolean;
  setIsPaused: (paused: boolean) => void;
  togglePause: () => void;

  // Manual triggers
  refreshAll: () => Promise<void>;
  refreshReading: () => Promise<void>;
  refreshTrends: (hours?: number) => Promise<void>;
  refreshAnomalies: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  syncStreamStatus: () => Promise<void>;
}

export interface UseDashboardDataOptions {
  pollingIntervalMs?: number;
  autoPoll?: boolean;
  trendHours?: number;
}

/** Merge API and push updates by their observation timestamp, never by arrival time. */
function mergeTrendPoints(base: TrendsResponse['points'], additions: TrendsResponse['points'], hours: number) {
  const byTimestamp = new Map<string, TrendsResponse['points'][number]>();
  for (const point of base) byTimestamp.set(point.timestamp, point);
  for (const point of additions) {
    const existing = byTimestamp.get(point.timestamp);
    byTimestamp.set(point.timestamp, existing ? { ...existing, ...point } : point);
  }

  const points = Array.from(byTimestamp.values())
    .filter((point) => !Number.isNaN(new Date(point.timestamp).getTime()))
    .sort((left, right) => new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime());
  const latestTimestamp = points.at(-1) ? new Date(points.at(-1)!.timestamp).getTime() : 0;
  const cutoff = latestTimestamp - hours * 60 * 60 * 1000;
  return points.filter((point) => new Date(point.timestamp).getTime() >= cutoff);
}

/**
 * useDashboardData
 * 
 * Centralized telemetry data & polling hook reused across Dashboard and Monitor:
 * - Listens to selected stationId from StationContext
 * - Manages single centralized polling loop against service boundaries
 * - Provides pause/resume frontend controls without breaking state
 * - Maintains bounded telemetry history (max 150 items) derived from live feeds
 * - Provides section-isolated loading & error boundaries
 * - Implements frontend data staleness detection and graceful failure recovery
 */
export function useDashboardData(
  stationId: string | null | undefined,
  options: UseDashboardDataOptions = {}
): DashboardDataState {
  const { pollingIntervalMs = 30 * 60 * 1000, autoPoll = true, trendHours = 10 } = options;
  const trendHoursRef = useRef(trendHours);
  trendHoursRef.current = trendHours;

  const [currentReading, setCurrentReading] = useState<CurrentSensorReading | null>(null);
  const [trends, setTrends] = useState<TrendsResponse | null>(null);
  const [latestAnomaly, setLatestAnomaly] = useState<LatestAnomaly | null>(null);
  const [recentAnomalies, setRecentAnomalies] = useState<RecentAnomalyItem[]>([]);
  const [telemetryHistory, setTelemetryHistory] = useState<TelemetryHistoryRecord[]>([]);
  const [sensorHealth, setSensorHealth] = useState<SensorHealth | null>(null);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [streamStatus, setStreamStatus] = useState<SystemStreamStatus>({
    mode: 'live', replay_step_seconds: null, live_poll_interval_seconds: 30 * 60,
  });

  const [isLoadingReading, setIsLoadingReading] = useState<boolean>(true);
  const [isLoadingTrends, setIsLoadingTrends] = useState<boolean>(true);
  const [isLoadingAnomalies, setIsLoadingAnomalies] = useState<boolean>(true);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(true);

  const [readingError, setReadingError] = useState<string | null>(null);
  const [trendsError, setTrendsError] = useState<string | null>(null);
  const [anomaliesError, setAnomaliesError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  // WebSocket Live Push & Latency Readout
  const [wsLatencyMs, setWsLatencyMs] = useState<number | null>(null);
  const [isWsConnected, setIsWsConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);

  // References to prevent overlapping polling requests and race conditions
  const activeStationIdRef = useRef(stationId);
  activeStationIdRef.current = stationId;
  const isFetchingReadingRef = useRef(false);
  const isFetchingHealthRef = useRef(false);
  const wsConnectedRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const modeTimerRef = useRef<number | null>(null);
  const priorStreamModeRef = useRef<'live' | 'replay'>(streamStatus.mode);
  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;
  const fetchAnomaliesRef = useRef<() => Promise<void>>(() => Promise.resolve());
  const fetchReadingRef = useRef<() => Promise<void>>(() => Promise.resolve());
  const fetchTrendsRef = useRef<(h?: number) => Promise<void>>(() => Promise.resolve());
  const fetchHealthRef = useRef<() => Promise<void>>(() => Promise.resolve());
  const trendsRef = useRef<TrendsResponse | null>(null);
  const trendRevisionRef = useRef(0);
  const currentReadingRef = useRef<CurrentSensorReading | null>(null);
  const latestAnomalyRef = useRef<LatestAnomaly | null>(null);
  const sensorHealthRef = useRef<SensorHealth | null>(null);

  // Real-time WebSocket connection to backend /ws/live
  useEffect(() => {
    let isUnmounted = false;

    function connectWs() {
      if (isUnmounted) return;
      try {
        const socket = new WebSocket(API_CONFIG.wsUrl);
        wsRef.current = socket;

        socket.onopen = () => {
          if (isUnmounted) {
            socket.close();
            return;
          }
          setIsWsConnected(true);
          wsConnectedRef.current = true;
          // WS reconnect means we might have missed state changes (or server abruptly restarted).
          // Force a full refresh to ensure frontend is perfectly in sync.
          fetchReadingRef.current();
          fetchHealthRef.current();
          fetchTrendsRef.current(trendHoursRef.current);
          fetchAnomaliesRef.current();

          if (pingIntervalRef.current !== null) {
            clearInterval(pingIntervalRef.current);
          }
          pingIntervalRef.current = window.setInterval(() => {
            if (socket.readyState === WebSocket.OPEN) {
              socket.send('ping');
            }
          }, 15000);
        };

        socket.onmessage = (event) => {
          if (isUnmounted) return;
          try {
            const data = JSON.parse(event.data);

            if (data.type === 'TELEMETRY_TICK') {
              if (data.station_id === activeStationIdRef.current) {
                // End-to-end turnaround latency readout (real measured delta)
                if (data.ingest_time_ms) {
                  const measuredLatency = Math.max(1, Date.now() - data.ingest_time_ms);
                  setWsLatencyMs(measuredLatency);
                }

                // Instantaneously update current reading directly from tick payload
                if (data.reading) {
                  setCurrentReading((prev) => {
                    const mappedSeverity = data.verdict?.severity ?? 'low';
                    const riskLevel: 'low' | 'medium' | 'high' | 'critical' =
                      mappedSeverity === 'critical' ? 'critical' :
                      mappedSeverity === 'high' ? 'high' :
                      mappedSeverity === 'medium' ? 'medium' : 'low';

                    return {
                      station_id: data.station_id,
                      timestamp: data.timestamp,
                      temperature_c: {
                        value: data.reading.temperature_c !== undefined ? data.reading.temperature_c : null,
                        normal_min: prev?.temperature_c?.normal_min ?? 10.0,
                        normal_max: prev?.temperature_c?.normal_max ?? 45.0,
                      },
                      pressure_hpa: {
                        value: data.reading.pressure_hpa !== undefined ? data.reading.pressure_hpa : null,
                        normal_min: prev?.pressure_hpa?.normal_min ?? 900.0,
                        normal_max: prev?.pressure_hpa?.normal_max ?? 1050.0,
                      },
                      humidity_pct: {
                        value: data.reading.humidity_pct !== undefined ? data.reading.humidity_pct : null,
                        normal_min: prev?.humidity_pct?.normal_min ?? 10.0,
                        normal_max: prev?.humidity_pct?.normal_max ?? 100.0,
                      },
                      is_anomaly: data.verdict?.is_anomaly ?? false,
                      anomaly_score_pct: data.verdict?.anomaly_score_pct ?? 0,
                      model_status: data.verdict?.model_status ?? (prev?.model_status || 'AVAILABLE'),
                      fault_type: data.verdict?.fault_type ?? null,
                      severity: data.verdict?.severity ?? null,
                      suggested_values: data.verdict?.suggested_values ?? (prev?.suggested_values || {}),
                      source: (data.mode ?? 'live') as 'live' | 'replay',
                      risk_level: riskLevel,
                      sensor_health_pct: prev?.sensor_health_pct ?? (data.verdict?.health_status === 'OFFLINE' ? 0 : 100),
                      sensor_health_status: (data.verdict?.health_status ?? (prev?.sensor_health_status || 'HEALTHY')) as 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE',
                    };
                  });
                  setReadingError(null);
                  setIsLoadingReading(false);
                  setLastUpdated(new Date());

                  // Seamlessly initialize and append to active trend chart buffer
                  setTrends((previous) => {
                    const point = {
                      timestamp: data.timestamp,
                      temperature_c: data.reading.temperature_c ?? null,
                      pressure_hpa: data.reading.pressure_hpa ?? null,
                      humidity_pct: data.reading.humidity_pct ?? null,
                      anomaly_score_pct: data.verdict?.anomaly_score_pct,
                      is_anomaly: data.verdict?.is_anomaly ?? false,
                      fault_type: data.verdict?.fault_type,
                      suggested_temperature_c: data.verdict?.suggested_values?.temperature_c,
                      suggested_pressure_hpa: data.verdict?.suggested_values?.pressure_hpa,
                      suggested_humidity_pct: data.verdict?.suggested_values?.humidity_pct,
                      health_status: data.verdict?.health_status,
                      source: data.mode,
                    };

                    const visibleHours = trendHoursRef.current;
                    if (!previous || previous.station_id !== data.station_id) {
                      return {
                        station_id: data.station_id,
                        hours: visibleHours,
                        points: [point],
                      };
                    }

                    const existingSource = previous.points.at(-1)?.source;
                    if (existingSource && point.source && existingSource !== point.source) {
                      return { ...previous, points: [point] };
                    }
                    const nextTrends = {
                      ...previous,
                      hours: visibleHours,
                      points: mergeTrendPoints(previous.points, [point], visibleHours),
                    };
                    trendRevisionRef.current += 1;
                    trendsRef.current = nextTrends;
                    return nextTrends;
                  });
                  setIsLoadingTrends(false);

                  if (data.verdict?.is_anomaly) {
                    fetchAnomaliesRef.current();
                  }
                }
              }

            } else if (data.type === 'ANOMALY_EVENT') {
              if (data.station_id === activeStationIdRef.current || !data.station_id) {
                if (data.anomaly) {
                  const anom = data.anomaly;
                  setLatestAnomaly({
                    anomaly_id: anom.anomaly_id,
                    timestamp: anom.timestamp,
                    station_id: anom.station_id,
                    anomaly_score_pct: anom.anomaly_score_pct,
                    severity: anom.severity,
                    type: anom.type,
                    root_cause: anom.root_cause || anom.type,
                    description: anom.description || `${anom.root_cause || anom.type} detected at ${anom.station_id}.`,
                    suggested_values: anom.suggested_values || {},
                    observed_values: anom.observed_values || {},
                    affected_parameters: anom.affected_parameters || [],
                    regime: anom.regime,
                    network_corroboration: anom.network_corroboration,
                    decision_basis: anom.decision_basis,
                    model_status: anom.model_status,
                  });
                  setRecentAnomalies((prevRecent) => {
                    const withoutDup = prevRecent.filter((a) => a.anomaly_id !== anom.anomaly_id);
                    const item: RecentAnomalyItem = {
                      anomaly_id: anom.anomaly_id,
                      timestamp: anom.timestamp,
                      station_id: anom.station_id,
                      anomaly_score_pct: anom.anomaly_score_pct,
                      severity: anom.severity,
                      type: anom.type,
                      root_cause: anom.root_cause || anom.type,
                      description: anom.description || `${anom.root_cause || anom.type} detected at ${anom.station_id}.`,
                      suggested_values: anom.suggested_values,
                      affected_parameters: anom.affected_parameters,
                      observed_values: anom.observed_values,
                      regime: anom.regime,
                      network_corroboration: anom.network_corroboration,
                      decision_basis: anom.decision_basis,
                      model_status: anom.model_status,
                    };
                    return [item, ...withoutDup].slice(0, 50);
                  });

                  // Retroactively mark the anomalous point in trends so it turns red immediately
                  setTrends((prev) => {
                    if (!prev || !prev.points) return prev;
                    const anomTime = new Date(anom.timestamp).getTime();
                    let matched = false;
                    const updatedPoints = prev.points.map((p) => {
                      const pTime = new Date(p.timestamp).getTime();
                      if (Math.abs(pTime - anomTime) < 60000) {
                        matched = true;
                        return {
                          ...p,
                          is_anomaly: true,
                          fault_type: anom.type || p.fault_type,
                          severity: anom.severity || p.severity,
                          anomaly_score_pct: anom.anomaly_score_pct ?? p.anomaly_score_pct,
                          suggested_temperature_c: anom.suggested_values?.temperature_c ?? p.suggested_temperature_c,
                          suggested_pressure_hpa: anom.suggested_values?.pressure_hpa ?? p.suggested_pressure_hpa,
                          suggested_humidity_pct: anom.suggested_values?.humidity_pct ?? p.suggested_humidity_pct,
                        };
                      }
                      return p;
                    });
                    if (matched) {
                      const next = { ...prev, points: updatedPoints };
                      trendsRef.current = next;
                      return next;
                    }
                    return prev;
                  });
                }
                fetchAnomaliesRef.current();
              }

            } else if (data.type === 'MODE_CHANGE') {
              setStreamStatus((prev) => ({
                ...prev,
                mode: data.mode,
                replay_step_seconds: data.mode === 'replay' ? 2 : null,
              }));

              if (data.mode === 'live') {
                // Immediately purge replay state from UI to prevent leakage when transitioning automatically
                trendsRef.current = null;
                currentReadingRef.current = null;
                latestAnomalyRef.current = null;
                sensorHealthRef.current = null;
                setCurrentReading(null);
                setTrends(null);
                setLatestAnomaly(null);
                setRecentAnomalies([]);
                setSensorHealth(null);
                setTelemetryHistory([]);
                fetchReadingRef.current();
                fetchHealthRef.current();
                fetchTrendsRef.current(trendHoursRef.current);
                fetchAnomaliesRef.current();
              }

            } else if (data.type === 'HISTORY_PURGED') {
              // DB was cleared (e.g. user clicked "Reset DB").
              // Immediately wipe all local state so the chart goes blank without
              // needing a page refresh, then re-fetch so latest live readings
              // are shown right away.
              trendsRef.current = null;
              currentReadingRef.current = null;
              latestAnomalyRef.current = null;
              sensorHealthRef.current = null;
              setCurrentReading(null);
              setTrends(null);
              setLatestAnomaly(null);
              setRecentAnomalies([]);
              setSensorHealth(null);
              setTelemetryHistory([]);
              setLastUpdated(null);
              fetchReadingRef.current();
              fetchHealthRef.current();
              fetchTrendsRef.current(trendHoursRef.current);
              fetchAnomaliesRef.current();
            }

          } catch {
            // Heartbeat or non-JSON message — silently ignore
          }
        };

        socket.onclose = () => {
          setIsWsConnected(false);
          wsConnectedRef.current = false;
          if (pingIntervalRef.current !== null) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }
          if (!isUnmounted) {
            reconnectTimeoutRef.current = window.setTimeout(connectWs, 2500);
          }
        };

        socket.onerror = () => {
          socket.close();
        };
      } catch {
        setIsWsConnected(false);
        wsConnectedRef.current = false;
        if (!isUnmounted) {
          reconnectTimeoutRef.current = window.setTimeout(connectWs, 2500);
        }
      }
    }

    connectWs();

    return () => {
      isUnmounted = true;
      if (reconnectTimeoutRef.current !== null) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pingIntervalRef.current !== null) {
        clearInterval(pingIntervalRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // 1. Fetch Current Reading
  const fetchReading = useCallback(async () => {
    if (!stationId) return;
    if (isFetchingReadingRef.current) return;
    isFetchingReadingRef.current = true;
    const targetStationId = stationId;

    if (!currentReadingRef.current) {
      setIsLoadingReading(true);
    }

    try {
      const data = await currentReadingService.getCurrentReading(targetStationId);
      if (activeStationIdRef.current !== targetStationId) {
        // Discard stale response if active station changed while awaiting network
        return;
      }
      currentReadingRef.current = data;
      setCurrentReading(data);
      setReadingError(null);
      const updateTime = new Date();
      setLastUpdated(updateTime);

      if (data.is_anomaly) {
        fetchAnomaliesRef.current();
      }

      // Immediate rendering path: add the freshly processed backend
      // reading to the timestamped graph without waiting for CSV-backed
      // history or re-fetching the whole trend range.
      setTrends((previous) => {
        const visibleHours = trendHoursRef.current;
        const point = {
          timestamp: data.timestamp,
          temperature_c: data.temperature_c.value,
          pressure_hpa: data.pressure_hpa.value,
          humidity_pct: data.humidity_pct.value,
          anomaly_score_pct: data.anomaly_score_pct,
          is_anomaly: data.is_anomaly ?? false,
          fault_type: data.fault_type,
          suggested_temperature_c: data.suggested_values?.temperature_c,
          suggested_pressure_hpa: data.suggested_values?.pressure_hpa,
          suggested_humidity_pct: data.suggested_values?.humidity_pct,
          health_status: data.sensor_health_status,
          source: data.source,
        };

        if (!previous || previous.station_id !== data.station_id) {
          return {
            station_id: data.station_id,
            hours: visibleHours,
            points: [point],
          };
        }

        const existingSource = previous.points.at(-1)?.source;
        if (existingSource && point.source && existingSource !== point.source) {
          return { ...previous, points: [point] };
        }
        const nextTrends = {
          ...previous,
          hours: visibleHours,
          points: mergeTrendPoints(previous.points, [point], visibleHours),
        };
        trendRevisionRef.current += 1;
        trendsRef.current = nextTrends;
        return nextTrends;
      });

      // Append to in-memory telemetry history (bounded to 150 entries)
      setTelemetryHistory((prev) => {
        let statusToken: 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE' = 'NORMAL';
        if (data.risk_level === 'critical' || data.sensor_health_status === 'CRITICAL') statusToken = 'CRITICAL';
        else if (data.risk_level === 'high' || data.risk_level === 'medium' || data.sensor_health_status === 'WARNING') statusToken = 'WARNING';
        else if (data.sensor_health_status === 'OFFLINE') statusToken = 'OFFLINE';

        const newRecord: TelemetryHistoryRecord = {
          id: `${data.station_id}-${data.timestamp}-${Math.random().toString(36).slice(2, 6)}`,
          timestamp: data.timestamp,
          temperature_c: data.temperature_c.value,
          pressure_hpa: data.pressure_hpa.value,
          humidity_pct: data.humidity_pct.value,
          status: statusToken,
        };

        // Avoid exact duplicate timestamp at index 0
        if (prev.length > 0 && prev[0].timestamp === data.timestamp && prev[0].temperature_c === data.temperature_c.value) {
          return prev;
        }

        return [newRecord, ...prev].slice(0, 150);
      });
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      // Retain last known valid reading while setting error indicator
      setReadingError(formatUserErrorMessage(err, 'Unable to load current sensor readings.'));
      // Uvicorn can accept the frontend request a moment before the first
      // concurrent Open-Meteo bootstrap finishes. A 404 here means "not
      // fetched yet", not a broken station; retry quickly once instead of
      // leaving the dashboard empty until the normal 30-minute live poll.
      if (err instanceof ApiError && err.status === 404) {
        window.setTimeout(() => {
          if (activeStationIdRef.current === targetStationId) fetchReading();
        }, 3000);
      }
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingReading(false);
      }
      isFetchingReadingRef.current = false;
    }
  }, [stationId]);

  fetchReadingRef.current = fetchReading;

  // 2. Fetch Trends
  const fetchTrends = useCallback(async (hours: number = trendHours) => {
    if (!stationId) return;
    const targetStationId = stationId;
    const requestRevision = trendRevisionRef.current;
    if (!trendsRef.current || trendsRef.current.station_id !== targetStationId) {
      setIsLoadingTrends(true);
    }
    try {
      const data = await trendsService.getTrends(targetStationId, hours);
      if (activeStationIdRef.current !== targetStationId) return;
      const latestLocal = trendsRef.current;
      // A WebSocket/current-reading update can arrive while this request is
      // in flight. Merge it by recorded timestamp instead of letting an older
      // HTTP response make the graph jump backwards.
      const nextTrends = latestLocal?.station_id === targetStationId && trendRevisionRef.current !== requestRevision
        ? { ...data, points: mergeTrendPoints(data.points, latestLocal.points, hours) }
        : data;
      trendsRef.current = nextTrends;
      setTrends(nextTrends);
      setTrendsError(null);

      // Pre-seed telemetry history from trend points on initial load if history is empty
      setTelemetryHistory((prev) => {
        if (prev.length > 0) return prev;
        const initialFromTrends: TelemetryHistoryRecord[] = (data.points || [])
          .slice(-25)
          .reverse()
          .map((pt, idx) => {
            const isAnomaly = (pt.anomaly_score_pct || 0) > 75;
            return {
              id: `hist-${idx}-${pt.timestamp}`,
              timestamp: pt.timestamp,
              temperature_c: pt.temperature_c,
              pressure_hpa: pt.pressure_hpa,
              humidity_pct: pt.humidity_pct,
              status: isAnomaly ? 'CRITICAL' : 'NORMAL',
            };
          });
        return initialFromTrends;
      });
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setTrendsError(formatUserErrorMessage(err, 'Unable to load sensor trends.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingTrends(false);
      }
    }
  }, [stationId, trendHours]);

  fetchTrendsRef.current = fetchTrends;

  // 3. Fetch Anomalies
  const fetchAnomalies = useCallback(async () => {
    if (!stationId) return;
    const targetStationId = stationId;
    if (!latestAnomalyRef.current) {
      setIsLoadingAnomalies(true);
    }
    try {
      const [latest, recent] = await Promise.all([
        anomalyService.getLatestAnomaly(targetStationId),
        anomalyService.getRecentAnomalies(targetStationId, 5),
      ]);
      if (activeStationIdRef.current !== targetStationId) return;
      latestAnomalyRef.current = latest;
      setLatestAnomaly(latest);
      setRecentAnomalies(recent);
      setAnomaliesError(null);
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setAnomaliesError(formatUserErrorMessage(err, 'Unable to load anomaly summary.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingAnomalies(false);
      }
    }
  }, [stationId]);

  fetchAnomaliesRef.current = fetchAnomalies;

  // 4. Fetch Sensor Health [API: GET /api/sensor-health?station_id=...]
  const fetchHealth = useCallback(async () => {
    if (!stationId) return;
    if (isFetchingHealthRef.current) return;
    isFetchingHealthRef.current = true;
    const targetStationId = stationId;

    if (!sensorHealthRef.current) {
      setIsLoadingHealth(true);
    }

    try {
      const healthData = await sensorHealthService.getSensorHealth(targetStationId);
      if (activeStationIdRef.current !== targetStationId) return;
      sensorHealthRef.current = healthData;
      setSensorHealth(healthData);
      setHealthError(null);
    } catch (err) {
      if (activeStationIdRef.current !== targetStationId) return;
      setHealthError(formatUserErrorMessage(err, 'Unable to load sensor health.'));
    } finally {
      if (activeStationIdRef.current === targetStationId) {
        setIsLoadingHealth(false);
      }
      isFetchingHealthRef.current = false;
    }
  }, [stationId]);

  fetchHealthRef.current = fetchHealth;

  const fetchStreamStatus = useCallback(async () => {
    try {
      setStreamStatus(await systemStatusService.get());
    } catch {
      // Keep the last known mode: a control-plane hiccup must not make
      // the graph discard timestamped telemetry already on screen.
    }
  }, []);

  // 5. Refresh all sections
  const refreshAll = useCallback(async () => {
    await Promise.all([fetchReading(), fetchTrends(trendHours), fetchAnomalies(), fetchHealth()]);
  }, [fetchReading, fetchTrends, fetchAnomalies, fetchHealth, trendHours]);

  // Toggle pause frontend polling
  const togglePause = useCallback(() => {
    setIsPaused((prev) => !prev);
  }, []);

  // Initialize and stationId change effect
  useEffect(() => {
    if (!stationId) {
      setCurrentReading(null);
      setTrends(null);
      setLatestAnomaly(null);
      setRecentAnomalies([]);
      setTelemetryHistory([]);
      setSensorHealth(null);
      setIsLoadingReading(false);
      setIsLoadingTrends(false);
      setIsLoadingAnomalies(false);
      setIsLoadingHealth(false);
      return;
    }

    // Immediately clear previous station's data and set loading
    trendsRef.current = null;
    currentReadingRef.current = null;
    latestAnomalyRef.current = null;
    sensorHealthRef.current = null;
    setCurrentReading(null);
    setTrends(null);
    setLatestAnomaly(null);
    setRecentAnomalies([]);
    setSensorHealth(null);
    setTelemetryHistory([]);

    setIsLoadingReading(true);
    setIsLoadingTrends(true);
    setIsLoadingAnomalies(true);
    setIsLoadingHealth(true);
    setReadingError(null);
    setTrendsError(null);
    setAnomaliesError(null);
    setHealthError(null);

    fetchReading();
    fetchTrends(trendHoursRef.current);
    fetchAnomalies();
    fetchHealth();
  }, [stationId, fetchReading, fetchTrends, fetchAnomalies, fetchHealth]);

  useEffect(() => {
    if (!stationId) return;
    fetchTrends(trendHours);
  }, [stationId, trendHours, fetchTrends]);

  // Mode is deliberately light-weight and checked each second so a
  // user-triggered replay changes data cadence promptly. Sensor data
  // itself remains 30-min live / 1-sec replay.
  useEffect(() => {
    fetchStreamStatus();
    modeTimerRef.current = window.setInterval(fetchStreamStatus, 1000);
    return () => {
      if (modeTimerRef.current !== null) clearInterval(modeTimerRef.current);
    };
  }, [fetchStreamStatus]);

  // A replay and live stream use different timelines. Clear the visual
  // timeline and fetch the new source's retained window at the transition,
  // rather than appending it to the old source's points.
  useEffect(() => {
    const previousMode = priorStreamModeRef.current;
    priorStreamModeRef.current = streamStatus.mode;
    if (!stationId || previousMode === streamStatus.mode) return;

    setTrends(null);
    setCurrentReading(null);
    setSensorHealth(null);
    setLatestAnomaly(null);
    setRecentAnomalies([]);
    setTelemetryHistory([]);
    setLastUpdated(null);
    setIsLoadingReading(true);
    setIsLoadingTrends(true);
    setIsLoadingHealth(true);
    setIsLoadingAnomalies(true);
    fetchTrends(trendHoursRef.current);
    fetchAnomalies();
    fetchReading();
    fetchHealth();
  }, [streamStatus.mode, stationId, fetchTrends, fetchReading, fetchHealth, fetchAnomalies]);

  useEffect(() => {
    const onRefresh = () => {
      if (!activeStationIdRef.current) return;
      refreshAll();
    };
    window.addEventListener(TELEMETRY_REFRESH_EVENT, onRefresh);
    return () => window.removeEventListener(TELEMETRY_REFRESH_EVENT, onRefresh);
  }, [refreshAll]);

  // Single Centralized Polling loop for telemetry
  useEffect(() => {
    if (!autoPoll || !stationId || isPaused) {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    const effectivePollingMs = streamStatus.mode === 'replay' ? 1000 : pollingIntervalMs;
    timerRef.current = window.setInterval(() => {
      if (!isPausedRef.current) {
        if (streamStatus.mode === 'live') {
          // In live mode: always poll at the configured interval.
          // WS push (TELEMETRY_TICK) only fires when the backend completes an
          // Open-Meteo fetch (~every 30 min). Between those pushes the WS is
          // silent, so skipping REST polling would mean the UI never updates.
          // pollingIntervalMs is already 30 min, so this adds zero extra load.
          fetchReading();
          fetchHealth();
        } else {
          // In replay mode: poll current reading (which continuously updates trends seamlessly)
          fetchReading();
          fetchHealth();
          if (!wsConnectedRef.current || !trendsRef.current || trendsRef.current.points.length < 2) {
            fetchTrends(trendHoursRef.current);
          }
        }
      }
    }, effectivePollingMs);

    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [autoPoll, stationId, isPaused, pollingIntervalMs, streamStatus.mode, fetchReading, fetchHealth, fetchTrends]);

  // Calculate freshness state
  // lastUpdated means the backend successfully answered a reading request;
  // do not label valid hourly Open-Meteo data stale after 60 seconds.
  const freshness = calculateFreshness(
    lastUpdated,
    isPaused,
    streamStatus.mode === 'replay' ? 2 : streamStatus.live_poll_interval_seconds
  );
  const isDelayed = freshness.status === 'DATA DELAYED';
  const isStale = freshness.status === 'DATA STALE';
  const staleStatusText: 'LIVE' | 'DATA DELAYED' | 'DATA STALE' = isStale
    ? 'DATA STALE'
    : isDelayed
    ? 'DATA DELAYED'
    : 'LIVE';
  const replaySeconds = streamStatus.replay_step_seconds ?? 2;
  const pollStatusText = streamStatus.mode === 'replay'
    ? `REPLAY · 1H / ${replaySeconds}S`
    : freshness.label;

  return {
    currentReading,
    trends,
    latestAnomaly,
    recentAnomalies,
    telemetryHistory,
    sensorHealth,
    isLoadingReading,
    isLoadingTrends,
    isLoadingAnomalies,
    isLoadingHealth,
    readingError,
    trendsError,
    anomaliesError,
    healthError,
    lastUpdated,
    isStale,
    isDelayed,
    staleStatusText,
    freshness,
    pollStatusText,
    streamMode: streamStatus.mode,
    wsLatencyMs,
    isWsConnected,
    isPaused,
    setIsPaused,
    isPreWarming: streamStatus.is_pre_warming ?? false,
    togglePause,
    refreshAll,
    refreshReading: fetchReading,
    refreshTrends: fetchTrends,
    refreshAnomalies: fetchAnomalies,
    refreshHealth: fetchHealth,
    syncStreamStatus: fetchStreamStatus,
  };
}
