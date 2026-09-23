import { TrendPoint } from '../types';

/** Rolling chart window keyed on reading timestamps, not arrival order. */
export function windowTrendPoints(
  points: TrendPoint[],
  hours: number
): { points: TrendPoint[]; windowStart: number; windowEnd: number } {
  const spanMs = Math.max(hours, 1) * 60 * 60 * 1000;
  if (!points.length) {
    const windowEnd = Date.now();
    return { points: [], windowStart: windowEnd - spanMs, windowEnd };
  }

  const sorted = [...points].sort(
    (left, right) => new Date(left.timestamp).getTime() - new Date(right.timestamp).getTime()
  );
  const windowEnd = new Date(sorted[sorted.length - 1].timestamp).getTime();
  const windowStart = windowEnd - spanMs;
  return {
    points: sorted.filter((point) => new Date(point.timestamp).getTime() >= windowStart),
    windowStart,
    windowEnd,
  };
}
