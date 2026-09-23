import React from 'react';
import { PlaceholderPage } from './PlaceholderPage';

export const SettingsPage: React.FC = () => (
  <PlaceholderPage
    title="System Configuration & Thresholds"
    description="Polling intervals, threshold baselines, notification webhooks, and UI theme preferences"
    category="System & Telemetry Settings"
    backendEndpointRef="/system/config"
  />
);
