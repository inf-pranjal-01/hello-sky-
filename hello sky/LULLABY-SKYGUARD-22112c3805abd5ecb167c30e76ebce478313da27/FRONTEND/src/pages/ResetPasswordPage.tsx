import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, KeyRound, ArrowLeft } from 'lucide-react';
import './ForgotPasswordPage.css';

export const ResetPasswordPage: React.FC = () => {
  return (
    <div className="sg-forgot-page">
      <div className="sg-forgot-card">
        <div className="sg-forgot-logo">
          <Shield size={36} className="text-accent" />
          <span>SkyGuard AI</span>
        </div>

        <div className="sg-forgot-header">
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.75rem', color: 'var(--accent-blue)' }}>
            <KeyRound size={40} />
          </div>
          <h2>Reset Token Specification</h2>
          <p>Password reset token validation is ready for future backend integration.</p>
        </div>

        <div className="sg-backend-notice" style={{ margin: '1rem 0 1.5rem 0' }}>
          <span>[DEMO AUTHENTICATION INTERFACE]</span>
          <p>
            The backend API contract does not currently define password reset token verification or password update endpoints.
          </p>
        </div>

        <div className="sg-forgot-footer">
          <Link to="/login" className="sg-back-link">
            <ArrowLeft size={16} />
            <span>Return to Login</span>
          </Link>
        </div>
      </div>
    </div>
  );
};
