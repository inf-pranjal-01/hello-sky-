import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Shield, Mail, ArrowLeft, Send, CheckCircle } from 'lucide-react';
import { Button } from '../components/common/Button';
import './ForgotPasswordPage.css';

export const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [emailTouched, setEmailTouched] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const validateEmail = (val: string): string | null => {
    if (!val.trim()) return 'Email is required.';
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(val.trim())) return 'Please enter a valid email address.';
    return null;
  };

  const emailError = emailTouched ? validateEmail(email) : null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setEmailTouched(true);
    const err = validateEmail(email);
    if (err) return;

    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSubmitted(true);
    }, 400);
  };

  return (
    <div className="sg-forgot-page">
      <div className="sg-forgot-card">
        <div className="sg-forgot-logo">
          <Shield size={36} className="text-accent" />
          <span>SkyGuard AI</span>
        </div>

        {!isSubmitted ? (
          <>
            <div className="sg-forgot-header">
              <h2>Reset Security Password</h2>
              <p>Enter your registered operator email to receive password recovery instructions.</p>
            </div>

            <form onSubmit={handleSubmit} noValidate className="sg-form">
              <div className="sg-form-field">
                <label htmlFor="forgot-email" className="sg-field-label">
                  Registered Operator Email
                </label>
                <div className="sg-input-group">
                  <Mail size={18} className="sg-input-left-icon" aria-hidden="true" />
                  <input
                    id="forgot-email"
                    type="email"
                    className={`sg-input ${emailError ? 'sg-input--invalid' : ''}`}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    onBlur={() => setEmailTouched(true)}
                    placeholder="demo@skyguard.local"
                    aria-invalid={!!emailError}
                    aria-describedby={emailError ? 'forgot-email-error' : undefined}
                    disabled={isSubmitting}
                    required
                  />
                </div>
                {emailError && (
                  <p id="forgot-email-error" className="sg-field-error" role="alert">
                    {emailError}
                  </p>
                )}
              </div>

              {/* Notice tag */}
              <div className="sg-backend-notice">
                <span>[DEMO AUTHENTICATION INTERFACE]</span>
                <p>Email delivery service is simulated for demonstration.</p>
              </div>

              <Button
                type="submit"
                variant="primary"
                size="lg"
                className="sg-submit-btn"
                isLoading={isSubmitting}
                rightIcon={<Send size={16} />}
              >
                Send Password Reset Request
              </Button>
            </form>
          </>
        ) : (
          <div className="sg-forgot-success">
            <CheckCircle size={48} className="sg-success-icon" />
            <h2>Reset Request Processed</h2>
            <p>
              If an active account exists for <strong>{email}</strong>, password reset instructions would be delivered.
            </p>
            <div className="sg-backend-notice" style={{ marginTop: '1rem' }}>
              <span>[DEMO AUTHENTICATION INTERFACE]</span>
              <p>This demo interface simulates the password recovery request flow.</p>
            </div>
          </div>
        )}

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
