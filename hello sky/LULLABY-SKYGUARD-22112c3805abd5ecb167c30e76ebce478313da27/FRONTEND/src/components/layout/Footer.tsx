import React from 'react';
import { API_CONFIG } from '../../config/api.config';
import './Footer.css';

export const Footer: React.FC = () => {
  const isMock = API_CONFIG.mode === 'mock';

  return (
    <footer className="sg-footer" aria-label="Application Footer">
      <div className="sg-footer__content">
        <div className="sg-footer__left">
          <span>SkyGuard AI v0.1.0</span>
          <span className="sg-footer__divider">•</span>
          <span className="sg-footer__notice">
            {isMock ? '[DEMO MODE — MOCK ADAPTER]' : '[REAL BACKEND MODE]'}
          </span>
          <span className="sg-footer__divider">•</span>
          <span className="sg-footer__notice">[REALTIME: HTTP POLLING (3–5s)]</span>
        </div>
        <div className="sg-footer__right">
          <span>{isMock ? 'Level 0 Backend Maturity' : `Endpoint: ${API_CONFIG.baseUrl}`}</span>
          <span className="sg-footer__divider">•</span>
          <span>FastAPI Integration Seam Ready</span>
        </div>
      </div>
    </footer>
  );
};
