import React, { useState, useEffect } from 'react';
import { Clock, MapPin, Copy, Check, Wrench, BrainCircuit } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Modal } from '../common/Modal';
import { Button } from '../common/Button';
import { StatusBadge } from '../common/StatusBadge';
import { FeatureImpactChart } from './FeatureImpactChart';
import { LatestAnomaly, RecentAnomalyItem, ExplanationFeature } from '../../types';
import { anomalyService } from '../../services/anomalyService';
import './AnomalyDetailModal.css';

export interface AnomalyDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  anomaly: LatestAnomaly | RecentAnomalyItem | null;
}

export const AnomalyDetailModal: React.FC<AnomalyDetailModalProps> = ({
  isOpen,
  onClose,
  anomaly,
}) => {
  const navigate = useNavigate();
  const [features, setFeatures] = useState<ExplanationFeature[]>([]);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState<boolean>(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen || !anomaly) {
      setFeatures([]);
      setIsLoadingExplanation(false);
      setExplanationError(null);
      setCopied(false);
      return;
    }

    let isMounted = true;
    setIsLoadingExplanation(true);
    setExplanationError(null);

    anomalyService
      .getAnomalyExplanation(anomaly.anomaly_id)
      .then((explanation) => {
        if (!isMounted) return;
        if (explanation && explanation.features) {
          setFeatures(explanation.features);
        } else {
          setFeatures([]);
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        setExplanationError(
          err instanceof Error ? err.message : 'Unable to load feature explainability data.'
        );
      })
      .finally(() => {
        if (isMounted) setIsLoadingExplanation(false);
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen, anomaly]);

  if (!anomaly) return null;

  const suggestedReading = Object.entries(anomaly.suggested_values ?? {})
    .map(([parameter, value]) => `${parameter.replace(/_/g, ' ')}: ${value.toFixed(2)}`)
    .join(' · ');

  const handleCopySummary = () => {
    const text = [
      `SkyGuard AI — Incident Report: ${anomaly.anomaly_id}`,
      `Station: ${anomaly.station_id}`,
      `Timestamp: ${anomaly.timestamp}`,
      `Severity: ${anomaly.severity.toUpperCase()}`,
      `Type: ${anomaly.type}`,
      `Score: ${Math.round(anomaly.anomaly_score_pct)}%`,
      `Root Cause: ${anomaly.root_cause}`,
      `Description: ${anomaly.description}`,
      ...(suggestedReading ? [`Suggested replacement: ${suggestedReading}`] : []),
    ].join('\n');

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    });
  };

  const formattedDate = new Date(anomaly.timestamp).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  });

  const modalFooter = (
    <div className="sg-anomaly-modal__footer-actions">
      <div>
        {copied && (
          <span className="sg-anomaly-modal__copied-toast">
            <Check size={14} /> Summary copied to clipboard
          </span>
        )}
      </div>
      <div style={{ display: 'flex', gap: '0.75rem' }}>
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopySummary}
          leftIcon={copied ? <Check size={14} /> : <Copy size={14} />}
        >
          {copied ? 'Copied' : 'Copy Incident Report'}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            navigate(`/maintenance?anomaly_id=${anomaly.anomaly_id}`);
            onClose();
          }}
          leftIcon={<Wrench size={14} />}
        >
          Create Maintenance Ticket
        </Button>
        <Button
          variant="outline"
          size="sm"
          style={{ borderColor: 'rgba(56, 189, 248, 0.4)', color: '#38bdf8' }}
          onClick={() => {
            navigate(`/analytics?anomaly_id=${anomaly.anomaly_id}&station_id=${anomaly.station_id}`);
            onClose();
          }}
          leftIcon={<BrainCircuit size={14} />}
        >
          Decision X-Ray (SHAP)
        </Button>
        <Button variant="primary" size="sm" onClick={onClose}>
          Close Investigation
        </Button>
      </div>
    </div>
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Incident Investigation: ${anomaly.anomaly_id}`}
      size="lg"
      footer={modalFooter}
    >
      <div className="sg-anomaly-modal">
        {/* Header Hero Banner */}
        <div className="sg-anomaly-modal__hero">
          <div className="sg-anomaly-modal__hero-info">
            <span className="sg-anomaly-modal__type">{anomaly.type.replace('_', ' ')}</span>
            <div className="sg-anomaly-modal__meta-row">
              <span className="sg-anomaly-modal__meta-item">
                <MapPin size={13} /> {anomaly.station_id}
              </span>
              <span className="sg-anomaly-modal__meta-item">
                <Clock size={13} /> {formattedDate}
              </span>
            </div>
          </div>
          <div className="sg-anomaly-modal__hero-badges">
            <StatusBadge
              status={
                anomaly.severity === 'critical'
                  ? 'critical'
                  : anomaly.severity === 'high'
                  ? 'high'
                  : anomaly.severity === 'medium'
                  ? 'moderate'
                  : 'low'
              }
              label={`${anomaly.severity.toUpperCase()} SEVERITY`}
              size="md"
            />
            <span className="sg-anomaly-modal__score">
              Evidence Strength: {Math.round(anomaly.anomaly_score_pct)}%
            </span>
          </div>
        </div>

        {/* Root Cause Section */}
        <div className="sg-anomaly-modal__section">
          <h4 className="sg-anomaly-modal__section-title">Anomaly Indication</h4>
          <p className="sg-anomaly-modal__root-cause">{anomaly.root_cause}</p>
        </div>

        {/* Description Section */}
        <div className="sg-anomaly-modal__section">
          <h4 className="sg-anomaly-modal__section-title">Meteorological Context & Description</h4>
          <p className="sg-anomaly-modal__description">{anomaly.description}</p>
        </div>

        {(anomaly.regime || anomaly.network_corroboration || anomaly.decision_basis) && (
          <div className="sg-anomaly-modal__section" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', background: 'rgba(255,255,255,0.02)', padding: '1rem', borderRadius: '8px' }}>
            {anomaly.regime && (
              <div>
                <h4 className="sg-anomaly-modal__section-title" style={{ marginBottom: '0.25rem' }}>Regime</h4>
                <p className="sg-anomaly-modal__description" style={{ margin: 0 }}>{anomaly.regime.replace(/_/g, ' ')}</p>
              </div>
            )}
            {anomaly.network_corroboration && (
              <div>
                <h4 className="sg-anomaly-modal__section-title" style={{ marginBottom: '0.25rem' }}>Network Evidence</h4>
                <div>
                  <StatusBadge 
                    status={anomaly.network_corroboration === 'LOCALIZED' ? 'warning' : anomaly.network_corroboration === 'REGIONAL' ? 'moderate' : 'optimal'} 
                    label={anomaly.network_corroboration.replace(/_/g, ' ')} 
                    size="sm" 
                  />
                </div>
              </div>
            )}
            {anomaly.decision_basis && (
              <div>
                <h4 className="sg-anomaly-modal__section-title" style={{ marginBottom: '0.25rem' }}>Decision Basis</h4>
                <p className="sg-anomaly-modal__description" style={{ margin: 0, fontSize: '0.75rem', fontWeight: 600 }}>{anomaly.decision_basis.replace(/_/g, ' ')}</p>
              </div>
            )}
          </div>
        )}

        {suggestedReading && (
          <div className="sg-anomaly-modal__section">
            <h4 className="sg-anomaly-modal__section-title">Estimated Replacement (Temporal Baseline — Not a Correction)</h4>
            <p className="sg-anomaly-modal__description">{suggestedReading}</p>
          </div>
        )}

        {/* Feature Explainability Section */}
        <div className="sg-anomaly-modal__section">
          <FeatureImpactChart
            features={features}
            isLoading={isLoadingExplanation}
            error={explanationError}
          />
        </div>
      </div>
    </Modal>
  );
};
