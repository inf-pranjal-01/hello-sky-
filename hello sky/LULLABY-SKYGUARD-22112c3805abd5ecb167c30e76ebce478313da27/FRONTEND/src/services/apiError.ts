/**
 * SkyGuard AI — Standard API Error Model & Formatters
 * 
 * Provides a standardized frontend error representation for network,
 * HTTP, parse, and response validation failures.
 */

export type ApiErrorCode =
  | 'NETWORK_ERROR'
  | 'TIMEOUT'
  | 'HTTP_ERROR'
  | 'PARSE_ERROR'
  | 'VALIDATION_ERROR'
  | 'MOCK_ERROR'
  | 'UNKNOWN';

export class ApiError extends Error {
  public readonly status: number | null;
  public readonly code: ApiErrorCode;
  public readonly isNetworkError: boolean;
  public readonly isTimeout: boolean;
  public readonly isValidationError: boolean;
  public readonly url?: string;

  constructor(
    message: string,
    options: {
      status?: number | null;
      code?: ApiErrorCode;
      isNetworkError?: boolean;
      isTimeout?: boolean;
      isValidationError?: boolean;
      url?: string;
      cause?: unknown;
    } = {}
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? null;
    this.code = options.code ?? 'UNKNOWN';
    this.isNetworkError = options.isNetworkError ?? false;
    this.isTimeout = options.isTimeout ?? false;
    this.isValidationError = options.isValidationError ?? false;
    this.url = options.url;

    // Preserve stack trace in V8 environments
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, ApiError);
    }
  }

  static networkError(url?: string, cause?: unknown): ApiError {
    return new ApiError('Unable to connect to the SkyGuard backend. Please check network connection.', {
      status: null,
      code: 'NETWORK_ERROR',
      isNetworkError: true,
      url,
      cause,
    });
  }

  static timeout(url?: string, timeoutMs: number = 10000): ApiError {
    return new ApiError(`Request to SkyGuard backend timed out after ${timeoutMs}ms.`, {
      status: null,
      code: 'TIMEOUT',
      isTimeout: true,
      url,
    });
  }

  static httpError(status: number, message: string, url?: string): ApiError {
    return new ApiError(message || `HTTP error ${status} received from backend.`, {
      status,
      code: 'HTTP_ERROR',
      url,
    });
  }

  static validationError(message: string, url?: string): ApiError {
    return new ApiError(message || 'SkyGuard backend returned an invalid or malformed data structure.', {
      status: null,
      code: 'VALIDATION_ERROR',
      isValidationError: true,
      url,
    });
  }

  static parseError(url?: string, cause?: unknown): ApiError {
    return new ApiError('Failed to parse backend response as JSON.', {
      status: null,
      code: 'PARSE_ERROR',
      url,
      cause,
    });
  }
}

/**
 * Format any thrown error into a clean, human-readable user-facing string.
 * Prevents raw stack traces, "fetch failed", or "undefined" from leaking into the UI.
 */
export function formatUserErrorMessage(err: unknown, fallbackMessage: string = 'An unexpected error occurred.'): string {
  if (err instanceof ApiError) {
    if (err.isNetworkError) {
      return 'Unable to connect to the SkyGuard backend. Please verify that the server is running.';
    }
    if (err.isTimeout) {
      return 'The request timed out. The backend server may be under heavy load.';
    }
    if (err.isValidationError) {
      return 'SkyGuard backend returned an unexpected data format. Please contact support.';
    }
    if (err.status === 404) {
      return 'The requested telemetry resource was not found.';
    }
    if (err.status && err.status >= 500) {
      return 'SkyGuard backend encountered an internal server error. Please retry.';
    }
    return err.message;
  }

  if (err instanceof Error) {
    if (err.name === 'AbortError') {
      return 'The request was canceled.';
    }
    if (err.message && !err.message.toLowerCase().includes('failed to fetch')) {
      return err.message;
    }
    return 'Unable to connect to the SkyGuard backend. Please check network connection.';
  }

  return fallbackMessage;
}
