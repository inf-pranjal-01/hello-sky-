import React from 'react';
import { Card } from '../components/common/Card';
import { Button } from '../components/common/Button';
import { StatusBadge } from '../components/common/StatusBadge';
import { Construction } from 'lucide-react';

export interface PlaceholderPageProps {
  title: string;
  description: string;
  category: string;
  backendEndpointRef?: string;
}

export const PlaceholderPage: React.FC<PlaceholderPageProps> = ({
  title,
  description,
  category,
  backendEndpointRef = '[NOT IN CURRENT API CONTRACT]',
}) => {
  return (
    <div className="page-container">
      <div className="sg-page-header">
        <div>
          <h2>{title}</h2>
          <p className="sg-page-sub">{description}</p>
        </div>
        <StatusBadge status="active" label="ROUTE LOADED" />
      </div>

      <div style={{ marginTop: '1.5rem' }}>
        <Card
          title={
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Construction size={20} className="text-accent" />
              <span>{category} — Route Initialized</span>
            </div>
          }
          subtitle={`Future integration point: ${backendEndpointRef}`}
          variant="glass"
          footer={
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                [FRONTEND ONLY — DEMO ROUTE]
              </span>
              <Button variant="outline" size="sm" onClick={() => alert(`Route ${title} foundation verified.`)}>
                Verify Route Architecture
              </Button>
            </div>
          }
        >
          <div style={{ padding: '1.5rem 0', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem' }}>
              The application routing and visual shell foundation for <strong>{title}</strong> is active and fully functional.
            </p>

            <div
              style={{
                padding: '1rem',
                backgroundColor: 'rgba(11, 19, 43, 0.7)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.85rem',
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              <div>• Route Path: {window.location.pathname}</div>
              <div>• Component State: Rendered inside AppShell</div>
              <div>• Accessibility Controls: Semantic HTML5, Keyboard focus traps ready</div>
              <div>• Real-time Strategy: Prepared for HTTP Polling (3–5s)</div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
