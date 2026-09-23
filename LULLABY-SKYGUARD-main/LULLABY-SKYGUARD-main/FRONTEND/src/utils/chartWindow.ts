import { TrendPoint } from '../types';

/** Rolling chart window keyed on reading timestamps, strictly within the requested hours. */
export function windowTrendPoints(
  points: TrendPoint[],
  hours: number
): { points: TrendPoint[]; windowStart: number; windowEnd: number } {
  const spanMs = Math.max(hours, 1) * 60 * 60 * 1000;
  if (!points || points.length === 0) {
    const windowEnd = Date.now();
    return { points: [], windowStart: windowEnd - spanMs, windowEnd };
  }

  // 1. Sort points by timestamp ascending, ignoring malformed timestamps
  const sorted = [...points]
    .filter((p) => p && p.timestamp && !Number.isNaN(new Date(p.timestamp).getTime()))
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

  if (sorted.length === 0) {
    const windowEnd = Date.now();
    return { points: [], windowStart: windowEnd - spanMs, windowEnd };
  }

  // 2. Window bounds: anchored to latest point timestamp (never stretching to old history)
  const lastTime = new Date(sorted[sorted.length - 1].timestamp).getTime();
  const windowEnd = lastTime;
  const windowStart = windowEnd - spanMs;

  // 3. Strictly filter points within [windowStart, windowEnd]
  const inWindow = sorted.filter((p) => {
    const t = new Date(p.timestamp).getTime();
    return t >= windowStart && t <= windowEnd;
  });

  // 4. Deduplicate only exact observation timestamps.  Never bucket readings
  // by a time interval: two valid observations in the same interval are still
  // distinct measurements and must remain visible to an operator.
  const pointByTime = new Map<number, TrendPoint>();
  for (const point of inWindow) {
    const time = new Date(point.timestamp).getTime();
    const existing = pointByTime.get(time);
    // Prefer the richer version of a reading when the API and WebSocket both
    // report the same observation (for example, after an anomaly is confirmed).
    pointByTime.set(time, existing ? { ...existing, ...point } : point);
  }
  const deduped = Array.from(pointByTime.entries())
    .sort(([left], [right]) => left - right)
    .map(([, point]) => point);

  return {
    points: deduped,
    windowStart,
    windowEnd,
  };
}
