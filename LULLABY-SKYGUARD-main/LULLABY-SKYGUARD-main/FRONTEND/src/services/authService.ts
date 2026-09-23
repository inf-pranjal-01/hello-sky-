import { User, LoginCredentials, AuthResponse } from '../types/auth';

/**
 * SkyGuard AI — Authentication Service Layer
 * 
 * [CLIENT-SIDE SESSION MANAGEMENT]
 * Manages operator identity and session storage for the frontend console.
 */

const SESSION_STORAGE_KEY = 'skyguard_mock_session';
const DEMO_EMAIL = 'demo@skyguard.local';
const DEMO_PASSWORD = 'demo1234';

const MOCK_USER: User = {
  id: 'usr-demo-001',
  email: 'demo@skyguard.local',
  name: 'SkyGuard Demo Operator',
  role: 'Administrator',
  stationAccess: ['ST-NDL-001', 'ST-MUM-002', 'ST-BLR-003', 'ST-HYD-004'],
  lastLogin: new Date().toISOString(),
};

export const authService = {
  /**
   * [MOCK AUTH — TEMPORARY]
   * Authenticate demo credentials with simulated network latency.
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    return new Promise((resolve) => {
      setTimeout(() => {
        const cleanEmail = credentials.email.trim().toLowerCase();
        
        if (cleanEmail === DEMO_EMAIL && credentials.password === DEMO_PASSWORD) {
          const userSession: User = {
            ...MOCK_USER,
            lastLogin: new Date().toISOString(),
          };

          // [SESSION STORAGE]
          // Store minimal session user info (NEVER passwords) based on rememberMe selection
          this.setStoredSession(userSession, !!credentials.rememberMe);

          resolve({
            success: true,
            user: userSession,
            message: 'Authentication successful.',
          });
        } else {
          resolve({
            success: false,
            message: 'Invalid demo credentials. Please use demo@skyguard.local and password demo1234.',
          });
        }
      }, 300); // 300ms artificial network delay
    });
  },

  /**
   * [SESSION MANAGEMENT]
   * Clear session state and logout user.
   */
  async logout(): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(() => {
        this.clearSession();
        resolve();
      }, 100);
    });
  },

  /**
   * [SESSION STORAGE]
   * Retrieve active user session from browser storage.
   * Checks sessionStorage first, then localStorage.
   */
  getStoredSession(): User | null {
    try {
      // 1. Check sessionStorage
      const sessionData = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (sessionData) {
        return JSON.parse(sessionData) as User;
      }
      // 2. Check localStorage
      const localData = localStorage.getItem(SESSION_STORAGE_KEY);
      if (localData) {
        return JSON.parse(localData) as User;
      }
    } catch {
      // Storage parse error fallback
      this.clearSession();
    }
    return null;
  },

  /**
   * [SESSION STORAGE]
   * Persist session data cleanly in sessionStorage or localStorage.
   */
  setStoredSession(user: User, rememberMe: boolean): void {
    const dataStr = JSON.stringify(user);
    this.clearSession(); // Clean previous entries first
    if (rememberMe) {
      localStorage.setItem(SESSION_STORAGE_KEY, dataStr);
    } else {
      sessionStorage.setItem(SESSION_STORAGE_KEY, dataStr);
    }
  },

  /**
   * Clear mock session from both sessionStorage and localStorage.
   */
  clearSession(): void {
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    localStorage.removeItem(SESSION_STORAGE_KEY);
  },

  /**
   * Helper check for authenticated state.
   */
  isAuthenticated(): boolean {
    return this.getStoredSession() !== null;
  },
};
