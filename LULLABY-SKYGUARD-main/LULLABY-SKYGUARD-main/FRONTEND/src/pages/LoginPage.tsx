import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import {
  Shield,
  KeyRound,
  Mail,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Radio,
  Activity,
  ArrowRight,
  Info,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/common/Button';
import './LoginPage.css';

const DEMO_EMAIL = 'demo@skyguard.local';
const DEMO_PASSWORD = 'demo1234';

export const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isAuthenticated } = useAuth();

  // Redirect to dashboard if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, location]);

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Field interaction touched states
  const [emailTouched, setEmailTouched] = useState(false);
  const [passwordTouched, setPasswordTouched] = useState(false);

  // Error states
  const [authError, setAuthError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Validation logic
  const validateEmail = (val: string): string | null => {
    if (!val.trim()) return 'Email is required.';
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(val.trim())) return 'Please enter a valid email address.';
    return null;
  };

  const validatePassword = (val: string): string | null => {
    if (!val) return 'Password is required.';
    return null;
  };

  const emailError = emailTouched ? validateEmail(email) : null;
  const passwordError = passwordTouched ? validatePassword(password) : null;

  const handleAutoFillDemo = () => {
    setEmail(DEMO_EMAIL);
    setPassword(DEMO_PASSWORD);
    setEmailTouched(true);
    setPasswordTouched(true);
    setAuthError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setEmailTouched(true);
    setPasswordTouched(true);
    setAuthError(null);

    const eErr = validateEmail(email);
    const pErr = validatePassword(password);

    if (eErr || pErr) {
      return;
    }

    setIsSubmitting(true);

    try {
      const result = await login({
        email,
        password,
        rememberMe,
      });

      if (result.success) {
        const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';
        navigate(from, { replace: true });
      } else {
        setAuthError(result.message || 'Invalid demo credentials.');
      }
    } catch {
      setAuthError('An unexpected authentication error occurred. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="sg-auth-page">
      <div className="sg-auth-container">
        {/* Left Column — Meteorological Branding & Feature Highlights */}
        <div className="sg-auth-branding">
          <div className="sg-branding-content">
            <div className="sg-branding-logo">
              <Shield size={42} className="text-accent" />
              <span className="sg-branding-name">SkyGuard AI</span>
            </div>

            <h1 className="sg-branding-title">
              Intelligent Real-Time Anomaly Detection
            </h1>

            <p className="sg-branding-desc">
              Next-generation operational console monitoring Automatic Weather Stations (AWS). Filters environmental sensor noise, isolates thermal/barometric drifts, and alerts operators instantly.
            </p>

            <div className="sg-feature-list">
              <div className="sg-feature-item">
                <Radio size={20} className="sg-feature-icon" />
                <div>
                  <strong>AWS Telemetry Ingest</strong>
                  <p>Continuous 3–5s HTTP polling sensor data pipeline</p>
                </div>
              </div>

              <div className="sg-feature-item">
                <Activity size={20} className="sg-feature-icon" />
                <div>
                  <strong>ML Sensor Health Index</strong>
                  <p>Multi-parameter anomaly scoring & hardware degradation triage</p>
                </div>
              </div>
            </div>

            <div className="sg-branding-footer">
              <span>National Meteorological Operational Network • SIH 2026</span>
            </div>
          </div>
        </div>

        {/* Right Column — Login Card */}
        <div className="sg-auth-form-column">
          <div className="sg-auth-card">
            <div className="sg-auth-card__header">
              <h2>Welcome Back</h2>
              <p>Sign in to continue monitoring your AWS network.</p>
            </div>

            {/* Global Auth Error Alert */}
            {authError && (
              <div className="sg-auth-alert" role="alert" aria-live="assertive">
                <AlertCircle size={18} className="sg-alert-icon" />
                <span>{authError}</span>
              </div>
            )}

            <form onSubmit={handleSubmit} noValidate className="sg-form">
              {/* Email Input Field */}
              <div className="sg-form-field">
                <label htmlFor="login-email" className="sg-field-label">
                  Operator Email Address
                </label>
                <div className="sg-input-group">
                  <Mail size={18} className="sg-input-left-icon" aria-hidden="true" />
                  <input
                    id="login-email"
                    type="email"
                    className={`sg-input ${emailError ? 'sg-input--invalid' : ''}`}
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    onBlur={() => setEmailTouched(true)}
                    placeholder="name@skyguard.local"
                    aria-invalid={!!emailError}
                    aria-describedby={emailError ? 'email-error-msg' : undefined}
                    disabled={isSubmitting}
                    required
                  />
                </div>
                {emailError && (
                  <p id="email-error-msg" className="sg-field-error" role="alert">
                    {emailError}
                  </p>
                )}
              </div>

              {/* Password Input Field */}
              <div className="sg-form-field">
                <div className="sg-field-header">
                  <label htmlFor="login-password" className="sg-field-label">
                    Security Access Token / Password
                  </label>
                  <Link to="/forgot-password" className="sg-forgot-link">
                    Forgot Password?
                  </Link>
                </div>

                <div className="sg-input-group">
                  <KeyRound size={18} className="sg-input-left-icon" aria-hidden="true" />
                  <input
                    id="login-password"
                    type={showPassword ? 'text' : 'password'}
                    className={`sg-input ${passwordError ? 'sg-input--invalid' : ''}`}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onBlur={() => setPasswordTouched(true)}
                    placeholder="Enter security password"
                    aria-invalid={!!passwordError}
                    aria-describedby={passwordError ? 'password-error-msg' : undefined}
                    disabled={isSubmitting}
                    required
                  />
                  <button
                    type="button"
                    className="sg-password-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password text' : 'Show password text'}
                    title={showPassword ? 'Hide password' : 'Show password'}
                    tabIndex={0}
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
                {passwordError && (
                  <p id="password-error-msg" className="sg-field-error" role="alert">
                    {passwordError}
                  </p>
                )}
              </div>

              {/* Controls Row: Remember Me */}
              <div className="sg-controls-row">
                <label className="sg-checkbox-label">
                  <input
                    type="checkbox"
                    className="sg-checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                    disabled={isSubmitting}
                  />
                  <span>Remember me on this browser</span>
                </label>
              </div>

              {/* Submit Button */}
              <Button
                type="submit"
                variant="primary"
                size="lg"
                className="sg-submit-btn"
                isLoading={isSubmitting}
                disabled={isSubmitting}
                rightIcon={<ArrowRight size={18} />}
              >
                {isSubmitting ? 'Signing in...' : 'Sign In to Console'}
              </Button>
            </form>

            {/* Demo Environment Badge & Autofill [MOCK AUTH — TEMPORARY] */}
            <div className="sg-demo-box">
              <div className="sg-demo-badge">
                <Info size={15} />
                <span>Demo Environment — [MOCK AUTH — TEMPORARY]</span>
              </div>
              <div className="sg-demo-credentials">
                <div>
                  <strong>Email:</strong> <code>{DEMO_EMAIL}</code>
                </div>
                <div>
                  <strong>Password:</strong> <code>{DEMO_PASSWORD}</code>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleAutoFillDemo}
                className="sg-autofill-btn"
                leftIcon={<CheckCircle2 size={16} />}
              >
                Auto-fill Demo Credentials
              </Button>
            </div>

            <div className="sg-auth-card__footer">
              <p>Need access? Contact your station administrator.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
