import {
  LayoutDashboard,
  Activity,
  AlertTriangle,
  HeartPulse,
  BarChart3,
  Radio,
  FileSpreadsheet,
  Wrench,
  Settings,
  User,
  LucideIcon,
} from 'lucide-react';

export interface RouteMeta {
  path: string;
  title: string;
  shortLabel: string;
  category: 'Overview' | 'Telemetry' | 'Analytics' | 'Management' | 'User';
  icon: LucideIcon;
  badgeCount?: number;
  description: string;
}

export const ROUTE_REGISTRY: RouteMeta[] = [
  {
    path: '/dashboard',
    title: 'Operational Overview',
    shortLabel: 'Dashboard',
    category: 'Overview',
    icon: LayoutDashboard,
    description: 'Real-time summary of connected meteorological stations & anomaly alerts',
  },
  {
    path: '/monitor',
    title: 'Telemetry Monitor',
    shortLabel: 'Live Monitor',
    category: 'Telemetry',
    icon: Activity,
    description: 'High-frequency telemetry stream inspection & sensor graph view',
  },
  {
    path: '/alerts',
    title: 'Anomaly Alerts Console',
    shortLabel: 'Anomaly Alerts',
    category: 'Telemetry',
    icon: AlertTriangle,
    badgeCount: 2,
    description: 'Meteorological anomaly detection feeds & operator triage interface',
  },
  {
    path: '/sensor-health',
    title: 'Sensor Hardware Health',
    shortLabel: 'Sensor Health',
    category: 'Telemetry',
    icon: HeartPulse,
    description: 'Hardware diagnostics, battery levels, signal quality, and sensor health matrix',
  },
  {
    path: '/analytics',
    title: 'Analytics & SHAP Explainability',
    shortLabel: 'Analytics & SHAP',
    category: 'Analytics',
    icon: BarChart3,
    description: 'Machine Learning anomaly feature attribution & prediction models',
  },
  {
    path: '/stations',
    title: 'Stations Network',
    shortLabel: 'Stations',
    category: 'Management',
    icon: Radio,
    description: 'Spatial telemetry distribution, station registration, and observatory health',
  },
  {
    path: '/reports',
    title: 'Meteorological Reports',
    shortLabel: 'Reports',
    category: 'Management',
    icon: FileSpreadsheet,
    description: 'Automated environmental compliance reporting and historical data export',
  },
  {
    path: '/maintenance',
    title: 'Sensor Maintenance Log',
    shortLabel: 'Maintenance',
    category: 'Management',
    icon: Wrench,
    description: 'Calibration logs, technician dispatching, and hardware repair tracking',
  },
  {
    path: '/settings',
    title: 'System Settings',
    shortLabel: 'Settings',
    category: 'Management',
    icon: Settings,
    description: 'Telemetry polling parameters, threshold baselines, and notification webhooks',
  },
  {
    path: '/profile',
    title: 'Operator Profile',
    shortLabel: 'My Profile',
    category: 'User',
    icon: User,
    description: 'Operator identity, station access permissions, and session logs',
  },
];

export const getRouteMeta = (pathname: string): RouteMeta => {
  const matched = ROUTE_REGISTRY.find((r) => pathname.startsWith(r.path));
  if (matched) return matched;
  return {
    path: pathname,
    title: 'Telemetry View',
    shortLabel: 'Telemetry',
    category: 'Overview',
    icon: Activity,
    description: 'SkyGuard AI Telemetry Monitor',
  };
};
