import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Compass, Home } from 'lucide-react';
import { Button } from '../components/common/Button';

export const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="page-container" style={{ textAlign: 'center', paddingTop: '4rem' }}>
      <div style={{ color: 'var(--accent-cyan)', marginBottom: '1rem' }}>
        <Compass size={64} />
      </div>
      <h2>404 — Route Out of Horizon</h2>
      <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', marginBottom: '1.5rem' }}>
        The requested meteorological telemetry route does not exist or has been relocated.
      </p>
      <Button variant="primary" onClick={() => navigate('/dashboard')} leftIcon={<Home size={18} />}>
        Return to Dashboard Overview
      </Button>
    </div>
  );
};
