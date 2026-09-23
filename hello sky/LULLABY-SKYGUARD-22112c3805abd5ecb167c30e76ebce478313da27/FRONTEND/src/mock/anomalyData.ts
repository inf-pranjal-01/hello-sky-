import { AnomalyRecord, LatestAnomaly, RecentAnomalyItem } from '../types';

/**
 * [MOCK FIXTURES]
 * Isolated anomaly event record fixtures.
 */
export const MOCK_ANOMALIES: AnomalyRecord[] = [
  {
    anomaly_id: 'ANOM-2026-0891',
    station_id: 'ST-HYD-004',
    station_name: 'Hyderabad Central Telemetry',
    timestamp: '2026-09-08T19:42:10Z',
    anomaly_type: 'Barometric Spike & Rapid Thermal Drift',
    anomaly_score_pct: 91.3,
    risk_level: 'critical',
    status: 'detected',
    description: 'Abnormal pressure drop of 14 hPa within 10 minutes accompanied by high temp variance.',
    affected_metrics: ['pressure_hpa', 'temperature_c'],
  },
  {
    anomaly_id: 'ANOM-2026-0888',
    station_id: 'ST-MUM-002',
    station_name: 'Mumbai Coastal Radar Station',
    timestamp: '2026-09-08T18:15:00Z',
    anomaly_type: 'High Humidity Anomaly',
    anomaly_score_pct: 78.9,
    risk_level: 'high',
    status: 'investigating',
    description: 'Sustained humidity above 84% exceeding 3-sigma seasonal baseline.',
    affected_metrics: ['humidity_pct'],
  },
];

/**
 * [MOCK DATA]
 * Latest anomaly records per station matching: [API: GET /api/anomalies/latest — INTEGRATED]
 */
export const MOCK_LATEST_ANOMALIES: Record<string, LatestAnomaly | null> = {
  'ST-HYD-004': {
    anomaly_id: 'ANOM-001',
    timestamp: new Date(Date.now() - 150000).toISOString(), // 2.5 mins ago
    station_id: 'ST-HYD-004',
    anomaly_score_pct: 91,
    severity: 'critical',
    type: 'spike',
    root_cause: 'Barometric spike & rapid thermal drift',
    description: 'Sudden pressure drop of 14 hPa coupled with accelerated ambient temperature increase.',
  },
  'ST-MUM-002': {
    anomaly_id: 'ANOM-002',
    timestamp: new Date(Date.now() - 900000).toISOString(), // 15 mins ago
    station_id: 'ST-MUM-002',
    anomaly_score_pct: 78,
    severity: 'high',
    type: 'drift',
    root_cause: 'High humidity drift',
    description: 'Sustained humidity above 84% exceeding 3-sigma seasonal baseline.',
  },
  'ST-NDL-001': null, // Healthy station — returns null (204/no content)
  'ST-BLR-003': null, // Healthy station
  'ST-KOL-005': null, // Offline station
};

/**
 * [MOCK DATA]
 * Recent anomalies list matching: [API: GET /api/anomalies/recent — INTEGRATED]
 */
export const MOCK_RECENT_ANOMALIES: Record<string, RecentAnomalyItem[]> = {
  'ST-HYD-004': [
    {
      anomaly_id: 'ANOM-001',
      timestamp: new Date(Date.now() - 150000).toISOString(),
      station_id: 'ST-HYD-004',
      anomaly_score_pct: 91,
      severity: 'critical',
      type: 'spike',
      root_cause: 'Barometric spike & rapid thermal drift',
      description: 'Sudden pressure drop of 14 hPa coupled with accelerated ambient temperature increase.',
    },
    {
      anomaly_id: 'ANOM-003',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      station_id: 'ST-HYD-004',
      anomaly_score_pct: 64,
      severity: 'medium',
      type: 'drift',
      root_cause: 'Thermal sensor calibration drift',
      description: 'Gradual upward thermal bias detected across afternoon peak.',
    },
  ],
  'ST-MUM-002': [
    {
      anomaly_id: 'ANOM-002',
      timestamp: new Date(Date.now() - 900000).toISOString(),
      station_id: 'ST-MUM-002',
      anomaly_score_pct: 78,
      severity: 'high',
      type: 'drift',
      root_cause: 'High humidity drift',
      description: 'Sustained humidity above 84% exceeding 3-sigma seasonal baseline.',
    },
  ],
  'ST-NDL-001': [],
  'ST-BLR-003': [],
  'ST-KOL-005': [],
};

/**
 * [MOCK DATA]
 * Anomaly explainability / feature contributions matching: [API: GET /api/explain/{anomaly_id} — INTEGRATED]
 */
export const MOCK_ANOMALY_EXPLANATIONS: Record<string, import('../types').AnomalyExplanation> = {
  'ANOM-001': {
    anomaly_id: 'ANOM-001',
    features: [
      { name: 'Pressure Delta (10m Rate)', impact: 0.58 },
      { name: 'Temperature Variance', impact: 0.34 },
      { name: 'Diurnal Solar Profile Deviation', impact: 0.12 },
      { name: 'Humidity Gradient', impact: -0.09 },
      { name: 'Seasonal Pressure Mean', impact: -0.04 },
    ],
  },
  'ANOM-002': {
    anomaly_id: 'ANOM-002',
    features: [
      { name: 'Humidity 3-Sigma Offset', impact: 0.62 },
      { name: 'Dew Point Proximity', impact: 0.25 },
      { name: 'Atmospheric Stability Index', impact: 0.15 },
      { name: 'Barometric Trend', impact: -0.11 },
      { name: 'Ambient Temp Rate', impact: -0.06 },
    ],
  },
  'ANOM-003': {
    anomaly_id: 'ANOM-003',
    features: [
      { name: 'Thermal Sensor Calibration Bias', impact: 0.49 },
      { name: 'Solar Insolation Disparity', impact: 0.28 },
      { name: 'Neighboring AWS Correlation Gap', impact: 0.16 },
      { name: 'Pressure Rate of Change', impact: -0.14 },
    ],
  },
};

