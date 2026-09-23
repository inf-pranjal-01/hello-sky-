/**
 * SkyGuard AI — Centralized API Client Transport Layer
 * 
 * Provides robust native-fetch transport with timeout support, JSON parsing,
 * network failure detection, and automatic conversion to ApiError.
 * 
 * [CENTRALIZED TRANSPORT]
 * When API_CONFIG.mode === 'real', all network communication routes through this transport.
 */

import { API_CONFIG } from '../config/api.config';
import { ApiError } from './apiError';

export type BackendConnectionStatus = 'mock' | 'connected' | 'disconnected';

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
}

class ApiClient {
  private _connectionStatus: BackendConnectionStatus = API_CONFIG.mode === 'mock' ? 'mock' : 'disconnected';
  private _listeners: Set<(status: BackendConnectionStatus) => void> = new Set();

  /**
   * Current backend connectivity status.
   */
  public get connectionStatus(): BackendConnectionStatus {
    return this._connectionStatus;
  }

  /**
   * Subscribe to connection status changes.
   */
  public onStatusChange(listener: (status: BackendConnectionStatus) => void): () => void {
    this._listeners.add(listener);
    return () => {
      this._listeners.delete(listener);
    };
  }

  private setConnectionStatus(status: BackendConnectionStatus): void {
    if (this._connectionStatus !== status) {
      this._connectionStatus = status;
      this._listeners.forEach((fn) => fn(status));
    }
  }

  /**
   * Build complete URL with query parameters.
   */
  private buildUrl(path: string, params?: Record<string, string | number | boolean | undefined | null>): string {
    const base = API_CONFIG.baseUrl;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    const url = new URL(`${base}${cleanPath}`);

    if (params) {
      Object.entries(params).forEach(([key, val]) => {
        if (val !== undefined && val !== null) {
          url.searchParams.append(key, String(val));
        }
      });
    }

    return url.toString();
  }

  /**
   * Perform HTTP GET request.
   */
  public async get<T>(
    path: string,
    params?: Record<string, string | number | boolean | undefined | null>,
    options?: RequestOptions
  ): Promise<T> {
    const url = this.buildUrl(path, params);
    return this.executeRequest<T>(url, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
        ...options?.headers,
      },
    }, options);
  }

  /**
   * Perform HTTP POST request with JSON payload.
   */
  public async post<T>(
    path: string,
    body?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = this.buildUrl(path);
    return this.executeRequest<T>(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        ...options?.headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }, options);
  }

  /**
   * Core request execution pipeline.
   */
  private async executeRequest<T>(
    url: string,
    init: RequestInit,
    options?: RequestOptions
  ): Promise<T> {
    const timeoutMs = options?.timeoutMs ?? API_CONFIG.timeoutMs;
    const controller = new AbortController();
    let timeoutId: number | undefined;

    // Link caller-supplied signal if present
    if (options?.signal) {
      options.signal.addEventListener('abort', () => controller.abort());
    }

    // Set network timeout timer
    timeoutId = window.setTimeout(() => {
      controller.abort();
    }, timeoutMs);

    try {
      const response = await fetch(url, {
        ...init,
        signal: controller.signal,
      });

      window.clearTimeout(timeoutId);

      if (!response.ok) {
        let errorBodyText = '';
        try {
          const errJson = await response.json();
          const detail = errJson.detail;
          errorBodyText = typeof detail === 'string'
            ? detail
            : detail !== undefined
            ? JSON.stringify(detail)
            : typeof errJson.message === 'string'
            ? errJson.message
            : JSON.stringify(errJson);
        } catch {
          errorBodyText = await response.text().catch(() => '');
        }

        const message = errorBodyText
          ? `Backend returned error ${response.status}: ${errorBodyText}`
          : `HTTP error ${response.status} from ${url}`;

        this.setConnectionStatus('disconnected');
        throw ApiError.httpError(response.status, message, url);
      }

      // Successful HTTP response
      this.setConnectionStatus('connected');

      // Parse JSON safely
      let data: T;
      try {
        data = await response.json();
      } catch (parseErr) {
        throw ApiError.parseError(url, parseErr);
      }

      return data;
    } catch (err: unknown) {
      window.clearTimeout(timeoutId);

      if (err instanceof ApiError) {
        throw err;
      }

      // Check for abort / timeout
      if (err instanceof DOMException && err.name === 'AbortError') {
        if (options?.signal?.aborted) {
          throw new Error('Request was canceled.');
        }
        this.setConnectionStatus('disconnected');
        throw ApiError.timeout(url, timeoutMs);
      }

      // Network / connection failure
      this.setConnectionStatus('disconnected');
      throw ApiError.networkError(url, err);
    }
  }
}

export const apiClient = new ApiClient();
