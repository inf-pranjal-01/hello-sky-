export const TELEMETRY_REFRESH_EVENT = 'skyguard:refresh-telemetry';

export function requestTelemetryRefresh(): void {
  window.dispatchEvent(new Event(TELEMETRY_REFRESH_EVENT));
}
