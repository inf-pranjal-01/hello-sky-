import { TrendPoint } from '../types';

export interface SuggestedReading {
  key: string;
  label: string;
  value: number;
  unit: string;
}

const PARAM_META: Record<string, { label: string; unit: string }> = {
  temperature: { label: 'Temperature', unit: '°C' },
  temperature_c: { label: 'Temperature', unit: '°C' },
  pressure: { label: 'Pressure', unit: 'hPa' },
  pressure_hpa: { label: 'Pressure', unit: 'hPa' },
  humidity: { label: 'Humidity', unit: '%' },
  humidity_pct: { label: 'Humidity', unit: '%' },
  suggested_temperature_c: { label: 'Temperature', unit: '°C' },
  suggested_pressure_hpa: { label: 'Pressure', unit: 'hPa' },
  suggested_humidity_pct: { label: 'Humidity', unit: '%' },
};

function normalizeKey(rawKey: string): string {
  return rawKey.replace(/^suggested_/, '');
}

function pushReading(list: SuggestedReading[], rawKey: string, value: unknown): void {
  if (typeof value !== 'number' || Number.isNaN(value)) return;
  const key = normalizeKey(rawKey);
  const meta = PARAM_META[rawKey] || PARAM_META[key];
  if (!meta) return;
  if (list.some((item) => item.key === key)) return;
  list.push({ key, label: meta.label, value, unit: meta.unit });
}

export function suggestedFromRecord(
  values?: Record<string, number | null | undefined> | null
): SuggestedReading[] {
  if (!values) return [];
  const list: SuggestedReading[] = [];
  for (const [key, value] of Object.entries(values)) {
    pushReading(list, key, value);
  }
  return list;
}

export function suggestedFromTrendPoint(point: TrendPoint): SuggestedReading[] {
  const list: SuggestedReading[] = [];
  pushReading(list, 'temperature_c', point.suggested_temperature_c);
  pushReading(list, 'pressure_hpa', point.suggested_pressure_hpa);
  pushReading(list, 'humidity_pct', point.suggested_humidity_pct);
  return list;
}

export function formatSuggestedList(items: SuggestedReading[]): string {
  return items.map((item) => `${item.label} ${item.value.toFixed(1)} ${item.unit}`).join(' · ');
}
