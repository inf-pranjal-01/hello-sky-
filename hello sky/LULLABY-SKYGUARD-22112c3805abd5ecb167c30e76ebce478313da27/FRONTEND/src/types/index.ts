/**
 * SkyGuard AI — TypeScript Domain & API Schemas
 * Standardized data models strictly aligned with the API Contract specification.
 */

export type RiskLevel = 'low' | 'moderate' | 'medium' | 'high' | 'critical';
export type SensorHealthStatus = 'optimal' | 'warning' | 'degraded' | 'offline' | 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
export type StationOperationalStatus = 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
export type AnomalyStatus = 'detected' | 'investigating' | 'resolved' | 'dismissed';
export type SystemOverallStatus = 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE';

export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical';
export type AnomalyType =
  | 'spike'
  | 'frozen_value'
  | 'drift'
  | 'dropout'
  | 'sensor_fail_low'
  | 'multivariate_inconsistency'
  | 'physical_bounds'
  | 'statistical_anomaly'
  | (string & {});

export type NetworkCorroborationState = 'LOCALIZED' | 'REGIONAL' | 'INSUFFICIENT_CORROBORATION';

/**
 * Metric Range & Value Schema
 * Sub-object inside current-reading contract
 */
export interface MetricValueRange {
  value: number;
  normal_min: number;
  normal_max: number;
}

/**
 * Current Sensor Reading Schema
 * [API: GET /api/current-reading — INTEGRATED]
 * Matches exact backend response contract specified in Step 4.
 */
export interface CurrentSensorReading {
  station_id: string;
  timestamp: string;
  temperature_c: MetricValueRange;
  pressure_hpa: MetricValueRange;
  humidity_pct: MetricValueRange;
  anomaly_score_pct: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  sensor_health_pct: number;
  sensor_health_status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
  is_anomaly?: boolean;
  fault_type?: AnomalyType | null;
  severity?: AnomalySeverity | null;
  suggested_values?: Record<string, number>;
  source?: 'live' | 'replay';
  model_status?: string;
}

/**
 * Sensor Health Schema
 * [API: GET /api/sensor-health?station_id={station_id} — INTEGRATED]
 */
export interface SensorHealth {
  station_id: string;
  sensor_health_pct: number;
  sensor_health_status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
}

/**
 * Sensor Health History Point
 * [FRONTEND DERIVED — DEMO HISTORY]
 */
export interface HealthHistoryPoint {
  timestamp: string;
  health_pct: number;
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
}

/**
 * Trend Data Point Schema
 * [API: GET /api/trends — INTEGRATED]
 */
export interface TrendPoint {
  timestamp: string;
  temperature_c: number;
  pressure_hpa: number;
  humidity_pct: number;
  anomaly_score_pct?: number;
  is_anomaly?: boolean;
  fault_type?: AnomalyType | null;
  severity?: AnomalySeverity | null;
  suggested_temperature_c?: number | null;
  suggested_pressure_hpa?: number | null;
  suggested_humidity_pct?: number | null;
  health_status?: SensorHealthStatus;
  source?: 'live' | 'replay';
}

/**
 * Anomaly Window Schema
 * [API: GET /api/trends — INTEGRATED]
 */
export interface AnomalyWindow {
  start: string;
  end: string;
  label?: string;
}

/**
 * Trends Response Schema
 * [API: GET /api/trends — INTEGRATED]
 */
export interface TrendsResponse {
  station_id: string;
  hours: number;
  points: TrendPoint[];
  anomaly_windows?: AnomalyWindow[];
}

/** Backend control-plane state. The frontend renders it; it never infers mode. */
export interface SystemStreamStatus {
  mode: 'live' | 'replay';
  replay_step_seconds: number | null;
  live_poll_interval_seconds: number;
  is_pre_warming?: boolean;
}

/**
 * Telemetry History Record
 * [FRONTEND ONLY]
 * Derived from sequential current readings or trend points for the telemetry monitor table.
 * Maximum 100-200 entries kept in memory.
 */
export interface TelemetryHistoryRecord {
  id: string;
  timestamp: string;
  temperature_c: number;
  pressure_hpa: number;
  humidity_pct: number;
  status: 'NORMAL' | 'WARNING' | 'CRITICAL' | 'OFFLINE';
}

/**
 * Latest Anomaly Schema
 * [API: GET /api/anomalies/latest — INTEGRATED]
 */
export interface LatestAnomaly {
  anomaly_id: string;
  timestamp: string;
  station_id: string;
  anomaly_score_pct: number;
  severity: AnomalySeverity;
  type: AnomalyType;
  root_cause: string;
  description: string;
  /** Optional sensor correction hints provided by the backend model */
  suggested_values?: Record<string, number>;
  /** Parameters implicated by the detector and their original readings. */
  affected_parameters?: string[];
  observed_values?: Record<string, number | null>;
  regime?: string;
  network_corroboration?: NetworkCorroborationState;
  /** Explicit basis for the detection decision per audit §10.2 */
  decision_basis?: string;
  /** ML model availability status per audit §8.2 */
  model_status?: string;
}

/**
 * Recent Anomaly Item Schema
 * [API: GET /api/anomalies/recent — INTEGRATED]
 */
export interface RecentAnomalyItem {
  anomaly_id: string;
  timestamp: string;
  station_id: string;
  anomaly_score_pct: number;
  severity: AnomalySeverity;
  type: AnomalyType;
  root_cause: string;
  description: string;
  /** Optional sensor correction hints provided by the backend model */
  suggested_values?: Record<string, number>;
  /** Parameters implicated by the detector and their original readings. */
  affected_parameters?: string[];
  observed_values?: Record<string, number | null>;
  regime?: string;
  network_corroboration?: NetworkCorroborationState;
  /** Explicit basis for the detection decision per audit §10.2 */
  decision_basis?: string;
  /** ML model availability status per audit §8.2 */
  model_status?: string;
}

/**
 * Feature Contribution / Explainability Schema
 * [API: GET /api/explain/{anomaly_id} — INTEGRATED]
 */
export interface ExplanationFeature {
  name: string;
  impact: number;
}

export interface PeerStationReading {
  station_id: string;
  name: string;
  reading: number;
  unit: string;
}

export interface ThermodynamicContext {
  is_violation: boolean;
  law?: string;
  temperature_c?: number;
  humidity_pct?: number;
  pressure_hpa?: number;
  explanation: string;
}

export interface SpatialContext {
  cluster_id: string;
  target_station: {
    station_id: string;
    name: string;
    reading: number;
    unit: string;
  };
  peer_stations: PeerStationReading[];
  parameter_analyzed: string;
  delta: number;
  analysis_text: string;
  recommended_action: string;
  spatial_impact: string;
  thermodynamic_context?: ThermodynamicContext | null;
}

export interface AnomalyExplanation {
  anomaly_id: string;
  station_id?: string;
  timestamp?: string;
  features: ExplanationFeature[];
  /** Optional list of sensor IDs most likely responsible for the anomaly */
  likely_faulty_sensors?: string[];
  affected_parameters?: string[];
  observed_values?: Record<string, number | null>;
  suggested_values?: Record<string, number>;
  model_confidence_pct?: number | null;
  rule_confidence_pct?: number;
  anomaly_score_pct?: number;
  fault_type?: AnomalyType;
  regime?: string;
  network_corroboration?: NetworkCorroborationState;
  decision_basis?: string;
  model_status?: string;
  spatial_context?: SpatialContext;
}

/**
 * SIH Demo Anomaly Injection Schema
 * [SIH DEMO PREPARATION]
 * [API: POST /api/inject-anomaly — INTEGRATED]
 */
export interface InjectAnomalyRequest {
  station_id: string;
  type: AnomalyType;
}

export interface InjectAnomalyResponse {
  success: boolean;
  anomaly_id: string;
  message: string;
}

/**
 * Sensor Repair Request/Response Schema
 * [API: POST /api/repair-sensor — INTEGRATED]
 */
export interface RepairSensorRequest {
  station_id: string;
}

export interface RepairSensorResponse {
  success: boolean;
  station_id: string;
  status: string;
  recovery_active: boolean;
  message: string;
}

/**
 * Legacy/Simple Sensor Reading Schema
 * Retained for compatibility with existing components
 */
export interface SensorReading {
  station_id: string;
  timestamp: string;
  temperature_c: number;
  pressure_hpa: number;
  humidity_pct: number;
  anomaly_score_pct: number;
  risk_level: RiskLevel;
  sensor_health_pct: number;
  sensor_health_status: SensorHealthStatus;
}

/**
 * Station Metadata Schema
 * [API: GET /api/stations — INTEGRATED]
 * Exactly matches backend station contract specification.
 */
export interface Station {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  status: StationOperationalStatus;
  elevation_m?: number;
  region?: string;
  last_ping?: string;
}

/**
 * Meteorological Anomaly Event Schema
 */
export interface AnomalyRecord {
  anomaly_id: string;
  station_id: string;
  station_name: string;
  timestamp: string;
  anomaly_type: string;
  anomaly_score_pct: number;
  risk_level: RiskLevel;
  status: AnomalyStatus;
  description: string;
  affected_metrics: string[];
}

/**
 * Overall System Health Summary Schema
 */
export interface SystemStatusSummary {
  overall_status: SystemOverallStatus;
  active_stations_count: number;
  total_stations_count: number;
  active_anomalies_count: number;
  avg_sensor_health_pct: number;
  last_updated: string;
}

/**
 * Application Navigation Route Specification
 */
export interface AppNavigationRoute {
  path: string;
  label: string;
  iconName: string;
  badgeCount?: number;
}

/**
 * Metric Statistical Summary
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export interface MetricStatistics {
  average: number;
  min: number;
  max: number;
  range: number;
  count: number;
}

/**
 * Anomaly Severity Distribution
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export interface SeverityDistribution {
  critical: number;
  high: number;
  medium: number;
  low: number;
}

/**
 * Anomaly Type Distribution
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export interface AnomalyTypeDistribution {
  spike: number;
  frozen_value: number;
  drift: number;
  dropout: number;
  sensor_fail_low: number;
  multivariate_inconsistency: number;
  physical_bounds?: number;
  statistical_anomaly?: number;
  [key: string]: number | undefined;
}

/**
 * Analytics Summary Data Object
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export interface AnalyticsSummary {
  temperature: MetricStatistics;
  pressure: MetricStatistics;
  humidity: MetricStatistics;
  totalAnomalies: number;
  highestSeverity: AnomalySeverity | 'none';
  severityDistribution: SeverityDistribution;
  typeDistribution: AnomalyTypeDistribution;
  anomalyTimeline: RecentAnomalyItem[];
  insights: string[];
}

/**
 * Station Network Telemetry Reading
 * [FRONTEND ONLY — DEMO TELEMETRY]
 */
export interface StationNetworkReading {
  station_id: string;
  timestamp: string;
  temperature_c: number;
  pressure_hpa: number;
  humidity_pct: number;
  anomaly_score_pct?: number;
  sensor_health_pct?: number;
}

/**
 * Neighbor Station with calculated distance and telemetry
 * [FRONTEND ONLY] [DERIVED FROM STATION COORDINATES & MOCK TELEMETRY]
 */
export interface NeighborStationItem {
  station: Station;
  distance_km: number;
  reading?: StationNetworkReading;
}

/**
 * Spatial Comparison for a Single Metric
 * [FRONTEND ONLY] [DERIVED FROM MOCK NETWORK TELEMETRY]
 */
export interface MetricSpatialComparison {
  selected: number;
  neighborAverage: number;
  difference: number;
  isSignificantDeviation: boolean;
  unit: string;
}

/**
 * Spatial Consistency Status
 * [FRONTEND DEMO LOGIC] [NOT PRODUCTION ML]
 */
export type SpatialConsistencyStatus = 'CONSISTENT' | 'DEVIATION_DETECTED' | 'INSUFFICIENT_DATA';

/**
 * Spatial Demo Scenario Identifier
 * [SIH DEMO] [FRONTEND ONLY]
 */
export type SpatialDemoScenario = 'regional_consistency' | 'localized_deviation';

/**
 * Overall Spatial Comparison Summary
 * [FRONTEND ONLY] [DERIVED FROM MOCK NETWORK TELEMETRY]
 */
export interface SpatialComparisonSummary {
  selectedStationId: string;
  neighborCount: number;
  temperature: MetricSpatialComparison;
  pressure: MetricSpatialComparison;
  humidity: MetricSpatialComparison;
  consistencyStatus: SpatialConsistencyStatus;
  multivariateSummary: string;
  demoScenario: SpatialDemoScenario;
}

/**
 * Report Observation Period
 */
export type ReportPeriod = 6 | 12 | 24;

/**
 * Operational Report Metadata
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export interface ReportMetadata {
  reportId: string;
  generatedAt: string;
  periodHours: ReportPeriod;
  systemVersion: string;
}

/**
 * Station Operational Report Model
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 * Standardized data model aggregating all operational aspects for preview and print export.
 */
export interface StationOperationalReport {
  metadata: ReportMetadata;
  station: Station;
  currentReading: CurrentSensorReading | null;
  telemetry: {
    temperature: MetricStatistics;
    pressure: MetricStatistics;
    humidity: MetricStatistics;
    points: TrendPoint[];
  };
  anomalies: {
    total: number;
    highestSeverity: AnomalySeverity | 'none';
    severityDistribution: SeverityDistribution;
    typeDistribution: AnomalyTypeDistribution;
    latestAnomaly: LatestAnomaly | null;
    incidents: RecentAnomalyItem[];
  };
  sensorHealth: SensorHealth | null;
  spatial?: SpatialComparisonSummary | null;
  insights: string[];
  recommendations: string[];
}

/**
 * Maintenance Ticket Creation Request Schema
 * [API: POST /api/maintenance-ticket — INTEGRATED]
 * Strictly matches approved backend API contract.
 */
export interface MaintenanceTicketRequest {
  anomaly_id: string;
}

/**
 * Maintenance Ticket Creation Response Schema
 * [API: POST /api/maintenance-ticket — INTEGRATED]
 * Strictly matches approved backend API contract response fields.
 */
export interface MaintenanceTicketResponse {
  ticket_id: string;
  station_id: string;
  issue: string;
  priority: string;
  created_at: string;
}
