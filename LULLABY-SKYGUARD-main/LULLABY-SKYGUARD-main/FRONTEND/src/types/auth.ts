/**
 * SkyGuard AI — Authentication Domain Schemas & Contracts
 * 
 * [MOCK AUTH — TEMPORARY]
 * Front-end type declarations for authentication, session states, and credentials.
 */

export type UserRole = 'Administrator' | 'Meteorologist' | 'Field Technician' | 'Operator';

export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  avatarUrl?: string;
  stationAccess?: string[];
  lastLogin?: string;
}

export interface LoginCredentials {
  email: string;
  password?: string;
  rememberMe?: boolean;
}

export interface AuthResponse {
  success: boolean;
  user?: User;
  message?: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface ForgotPasswordRequest {
  email: string;
}
