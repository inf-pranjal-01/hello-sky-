import React from 'react';
import { Lightbulb, CheckSquare } from 'lucide-react';
import './ReportInsightsSection.css';

export interface ReportInsightsSectionProps {
  insights: string[];
  recommendations: string[];
}

export const ReportInsightsSection: React.FC<ReportInsightsSectionProps> = ({
  insights,
  recommendations,
}) => {
  return (
    <section className="sg-report-section" aria-labelledby="report-section-6-heading">
      <div className="sg-report-section__header">
        <div className="sg-report-section__title-group">
          <Lightbulb size={18} className="text-accent" aria-hidden="true" />
          <h2 id="report-section-6-heading" className="sg-report-section__title">
            6. Operational Observations & Actionable Recommendations
          </h2>
        </div>
        <span className="sg-report-section__badge">DERIVED FROM LIVE BACKEND DATA</span>
      </div>

      <div className="sg-report-insights-grid">
        {/* Operational Insights */}
        <div className="sg-report-insights-card">
          <div className="sg-report-insights-card-header">
            <Lightbulb size={15} className="text-warning" aria-hidden="true" />
            <h3 className="sg-report-subheading" style={{ margin: 0 }}>
              Telemetry Insights & Patterns
            </h3>
          </div>
          <ul className="sg-report-insights-list" aria-label="Operational insights list">
            {insights.map((insight, idx) => (
              <li key={idx} className="sg-report-insights-item">
                <span className="sg-report-insights-bullet" aria-hidden="true">›</span>
                <span className="sg-report-insights-text">{insight}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Actionable Recommendations */}
        <div className="sg-report-insights-card">
          <div className="sg-report-insights-card-header">
            <CheckSquare size={15} className="text-optimal" aria-hidden="true" />
            <h3 className="sg-report-subheading" style={{ margin: 0 }}>
              Evidence-Based Recommendations
            </h3>
          </div>
          <ul className="sg-report-insights-list" aria-label="Recommended operational actions">
            {recommendations.map((rec, idx) => (
              <li key={idx} className="sg-report-insights-item">
                <span className="sg-report-recommendation-bullet" aria-hidden="true">✓</span>
                <span className="sg-report-insights-text">{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
};
