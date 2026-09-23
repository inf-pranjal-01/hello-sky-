import React from 'react';
import { Info, Cpu } from 'lucide-react';
import { Card } from '../common/Card';
import './FutureSpatialNotice.css';

export const FutureSpatialNotice: React.FC = () => {
  return (
    <Card
      variant="glass"
      className="sg-future-spatial-card"
      role="region"
      aria-label="Future backend spatial capability notice"
    >
      <div className="sg-future-spatial__header">
        <div className="sg-future-spatial__title-group">
          <Cpu size={16} className="text-accent" aria-hidden="true" />
          <h4 className="sg-future-spatial__title">
            Future Backend Integration & Spatial ML Architecture
          </h4>
        </div>
        <span className="sg-future-spatial__badge">
          DEMO SPATIAL COMPARISON — FUTURE BACKEND PIPELINE
        </span>
      </div>

      <div className="sg-future-spatial__content">
        <Info size={18} className="text-accent flex-shrink-0" aria-hidden="true" />
        <p className="sg-future-spatial__text">
          <strong>Production Spatial Validation Roadmap:</strong> The production SkyGuard AI system
          will utilize an approved backend telemetry streaming service that delivers temporally
          aligned observations across neighboring automatic weather stations. Spatial cross-validation
          algorithms and regional variance models will be evaluated server-side to distinguish
          mesoscale atmospheric events from hardware-level transducer drift.
        </p>
      </div>
    </Card>
  );
};
