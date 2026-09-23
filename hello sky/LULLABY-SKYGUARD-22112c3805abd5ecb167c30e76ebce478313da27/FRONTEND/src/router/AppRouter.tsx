import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '../context/AuthContext';
import { StationProvider } from '../context/StationContext';
import { ProtectedRoute } from './ProtectedRoute';
import { AppShell } from '../components/layout/AppShell';

import { LoginPage } from '../pages/LoginPage';
import { ForgotPasswordPage } from '../pages/ForgotPasswordPage';
import { ResetPasswordPage } from '../pages/ResetPasswordPage';

import { DashboardPage } from '../pages/DashboardPage';
import { MonitorPage } from '../pages/MonitorPage';
import { AlertsPage } from '../pages/AlertsPage';
import { SensorHealthPage } from '../pages/SensorHealthPage';
import { AnalyticsPage } from '../pages/AnalyticsPage';
import { StationsPage } from '../pages/StationsPage';
import { ReportsPage } from '../pages/ReportsPage';
import { MaintenancePage } from '../pages/MaintenancePage';
import { SettingsPage } from '../pages/SettingsPage';
import { ProfilePage } from '../pages/ProfilePage';
import { NotFoundPage } from '../pages/NotFoundPage';

export const AppRouter: React.FC = () => {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Out-of-Shell Authentication Routes */}
          <Route path="/login" element={<LoginPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/reset-password" element={<ResetPasswordPage />} />

          {/* Protected Application Routes inside AppShell */}
          <Route element={<ProtectedRoute />}>
            <Route
              element={
                <StationProvider>
                  <AppShell />
                </StationProvider>
              }
            >
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<DashboardPage />} />
              <Route path="/monitor" element={<MonitorPage />} />
              <Route path="/alerts" element={<AlertsPage />} />
              <Route path="/sensor-health" element={<SensorHealthPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/stations" element={<StationsPage />} />
              <Route path="/reports" element={<ReportsPage />} />
              <Route path="/maintenance" element={<MaintenancePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/profile" element={<ProfilePage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
};
