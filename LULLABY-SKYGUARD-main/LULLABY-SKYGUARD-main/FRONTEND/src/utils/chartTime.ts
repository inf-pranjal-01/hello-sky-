/**
 * Formats a timestamp for SVG chart axes and tooltips.
 * Uses only mutually compatible Intl options (dateStyle cannot be mixed
 * with hour/minute options in several Chromium/Windows combinations).
 */
export const formatChartTime = (timestamp: string, includeDate = false): string => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return 'Unknown time';
  return new Intl.DateTimeFormat(undefined, includeDate
    ? { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' }
    : { hour: '2-digit', minute: '2-digit' }
  ).format(date);
};
