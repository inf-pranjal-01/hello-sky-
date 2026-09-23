import React from 'react';
import { Lightbulb } from 'lucide-react';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { EmptyState } from '../common/EmptyState';
import './AnalyticsInsights.css';

export interface AnalyticsInsightsProps {
  insights: string[];
  isLoading?: boolean;
  className?: string;
}

/**
 * AnalyticsInsights
 *
 * Displays deterministically derived insight strings as a bullet list.
 * [FRONTEND ONLY] [DERIVED FROM EXISTING DATA]
 */
export const AnalyticsInsights: React.FC<AnalyticsInsightsProps> = ({
  insights,
  isLoading = false,
  className = '',
}) => {
  if (isLoading) {
    return (
      <Card variant="glass" className={`sg-analytics-insights-card ${className}`}>
        <div className="sg-analytics-insights__header">
          <Skeleton width="160px" height="1.2rem" />
        </div>
        <div className="sg-analytics-insights__list">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} width="100%" height="1rem" style={{ marginBottom: '0.75rem' }} />
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card
      variant="glass"
      className={`sg-analytics-insights-card ${className}`}
      role="region"
      aria-label="Analytics insights"
    >
      <div className="sg-analytics-insights__header">
        <div className="sg-analytics-insights__title-group">
          <Lightbulb size={17} aria-hidden="true" />
          <h3 className="sg-analytics-insights__title">Operational Insights</h3>
        </div>
        <span className="sg-analytics-insights__badge">
          [FRONTEND ONLY] [DERIVED ANALYTICS]
        </span>
      </div>

      {insights.length === 0 ? (
        <EmptyState
          title="No Insights Available"
          description="Insufficient telemetry data in this observation window to generate operational insights."
        />
      ) : (
        <ul className="sg-analytics-insights__list" aria-label="Operational insight items">
          {insights.map((insight, idx) => (
            <li key={idx} className="sg-analytics-insights__item">
              <span className="sg-analytics-insights__bullet" aria-hidden="true">›</span>
              <span className="sg-analytics-insights__text">{insight}</span>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
};
