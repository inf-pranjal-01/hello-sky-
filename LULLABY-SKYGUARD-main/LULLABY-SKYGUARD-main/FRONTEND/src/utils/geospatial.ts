/**
 * Geospatial Utility Functions
 * 
 * Implements Haversine distance calculations and coordinate formatting.
 * [FRONTEND ONLY] [DERIVED FROM STATION COORDINATES]
 */

const EARTH_RADIUS_KM = 6371;

/**
 * Converts degrees to radians.
 */
function toRadians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}

/**
 * Validates if latitude and longitude are within standard geographical bounds.
 */
export function isValidCoordinate(lat: unknown, lon: unknown): boolean {
  if (typeof lat !== 'number' || typeof lon !== 'number') return false;
  if (isNaN(lat) || isNaN(lon)) return false;
  if (!isFinite(lat) || !isFinite(lon)) return false;
  return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

/**
 * Calculates great-circle distance between two geographical points using the Haversine formula.
 * 
 * Returns distance in kilometers, rounded to 1 decimal place.
 * Returns 0 if coordinates are identical.
 * Returns null if any coordinate is missing or invalid.
 */
export function calculateDistanceKm(
  lat1: number | null | undefined,
  lon1: number | null | undefined,
  lat2: number | null | undefined,
  lon2: number | null | undefined
): number | null {
  if (!isValidCoordinate(lat1, lon1) || !isValidCoordinate(lat2, lon2)) {
    return null;
  }

  const validLat1 = lat1 as number;
  const validLon1 = lon1 as number;
  const validLat2 = lat2 as number;
  const validLon2 = lon2 as number;

  // Exact same point
  if (validLat1 === validLat2 && validLon1 === validLon2) {
    return 0;
  }

  const dLat = toRadians(validLat2 - validLat1);
  const dLon = toRadians(validLon2 - validLon1);

  const rLat1 = toRadians(validLat1);
  const rLat2 = toRadians(validLat2);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(rLat1) * Math.cos(rLat2) * Math.sin(dLon / 2) * Math.sin(dLon / 2);

  // Clamp 'a' to [0, 1] to prevent floating point inaccuracies in Math.asin
  const clampedA = Math.max(0, Math.min(1, a));
  const c = 2 * Math.atan2(Math.sqrt(clampedA), Math.sqrt(1 - clampedA));

  const distance = EARTH_RADIUS_KM * c;

  if (isNaN(distance) || !isFinite(distance)) {
    return null;
  }

  return Math.round(distance * 10) / 10;
}

/**
 * Formats latitude and longitude with cardinal directions.
 * E.g., (28.6139, 77.2090) -> "28.6139° N, 77.2090° E"
 */
export function formatCoordinates(
  lat: number | null | undefined,
  lon: number | null | undefined
): string {
  if (!isValidCoordinate(lat, lon)) {
    return 'Coordinates Unavailable';
  }

  const latNum = lat as number;
  const lonNum = lon as number;

  const latDir = latNum >= 0 ? 'N' : 'S';
  const lonDir = lonNum >= 0 ? 'E' : 'W';

  const latFormatted = Math.abs(latNum).toFixed(4);
  const lonFormatted = Math.abs(lonNum).toFixed(4);

  return `${latFormatted}° ${latDir}, ${lonFormatted}° ${lonDir}`;
}

/**
 * Formats a distance in kilometers with unit.
 * E.g., 4.2 -> "4.2 km", 0 -> "0 km", null -> "—"
 */
export function formatDistance(distanceKm: number | null | undefined): string {
  if (distanceKm == null || isNaN(distanceKm) || !isFinite(distanceKm)) {
    return '—';
  }
  return `${distanceKm.toFixed(1)} km`;
}
