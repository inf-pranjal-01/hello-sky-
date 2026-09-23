import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { BrainCircuit, CheckCircle2, ChevronRight, ExternalLink, Gauge, ShieldAlert, Sparkles } from 'lucide-react';
import { RecentAnomalyItem, AnomalyExplanation } from '../../types';
import { anomalyService } from '../../services/anomalyService';
import { Card } from '../common/Card';
import { Skeleton } from '../common/Skeleton';
import { SuggestedValues } from '../common/SuggestedValues';
import { suggestedFromRecord, formatSuggestedList } from '../../utils/suggestedValues';
import './ExplainabilityCommandCenter.css';

export interface ExplainabilityCommandCenterProps {
  anomalies: RecentAnomalyItem[];
  isLoading?: boolean;
  className?: string;
  initialAnomalyId?: string;
}

const displayParameter = (value: string) => {
  const lower = value.toLowerCase();
  if (lower.includes('vapor_pressure_consistency') || lower.includes('vapor pressure consistency')) {
    return 'Thermodynamic Consistency (Clausius-Clapeyron)';
  }
  if (lower.includes('vapor_pressure_deficit') || lower.includes('vapor pressure deficit')) {
    return 'Vapor Pressure Deficit (VPD)';
  }
  if (lower.includes('dewpoint_depression') || lower.includes('dew point depression')) {
    return 'Dew Point Depression';
  }
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
};

function explainFeature(name: string): string {
  const value = name.toLowerCase();
  if (value.includes('vapor_pressure_consistency') || value.includes('vapor pressure consistency')) {
    return 'Physical impossibility: Under the Clausius-Clapeyron relation, relative humidity must decrease as temperature rises. The simultaneous surge in both temperature and humidity violates water vapor conservation in ambient air.';
  }
  if (value.includes('vapor_pressure_deficit') || value.includes('vapor pressure deficit')) {
    return 'Atmospheric vapor pressure deficit (VPD) departed from equilibrium relative to ambient temperature and barometric pressure.';
  }
  if (value.includes('dewpoint') || value.includes('dew_point')) {
    return 'Spread between air temperature and calculated dew point departed from physical limits for the local air parcel.';
  }
  const sensor = value.includes('temp') ? 'temperature' : value.includes('pressure') ? 'barometric pressure' : value.includes('humidity') ? 'relative humidity' : 'weather parameter';
  if (value.includes('roc') || value.includes('change')) return `A rapid ${sensor} step change disagreed with the previous hourly observation.`;
  if (value.includes('deviation')) return `The ${sensor} reading departed significantly from this station's historical baseline.`;
  if (value.includes('rolling') || value.includes('mean') || value.includes('std')) return `The ${sensor} signal deviated from its recent stable diurnal envelope.`;
  if (value.includes('hour') || value.includes('doy')) return `The ${sensor} reading was anomalous for this time of day or seasonal period.`;
  return `The ${sensor} reading contributed significant evidence to the anomaly detection score.`;
}

export const ExplainabilityCommandCenter: React.FC<ExplainabilityCommandCenterProps> = ({
  anomalies,
  isLoading = false,
  className = '',
  initialAnomalyId,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(initialAnomalyId ?? null);
  const [explanation, setExplanation] = useState<AnomalyExplanation | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const effectiveAnomalies = useMemo(() => {
    return anomalies;
  }, [anomalies]);

  const selected = useMemo(() => {
    if (selectedId) {
      const found = effectiveAnomalies.find((anomaly) => anomaly.anomaly_id === selectedId);
      if (found) return found;
      // Deep-linked anomaly not yet in the active list: provide placeholder using explanation data
      return {
        anomaly_id: selectedId,
        station_id: explanation?.station_id || '',
        timestamp: explanation?.timestamp || '',
        type: (explanation?.fault_type || 'ANOMALY_INCIDENT') as any,
        severity: 'medium',
        anomaly_score_pct: explanation?.anomaly_score_pct ?? 75,
        root_cause: 'Sensor anomaly detected',
        description: 'Incident selected for explainability review.',
        decision_basis: explanation?.decision_basis,
      } as RecentAnomalyItem;
    }
    return effectiveAnomalies[0] ?? null;
  }, [effectiveAnomalies, selectedId, explanation]);

  // Ensure selectedId is populated once selected anomaly is resolved
  useEffect(() => {
    if (selected && selected.anomaly_id !== selectedId && !selectedId) {
      setSelectedId(selected.anomaly_id);
    }
  }, [selected, selectedId]);

  // Fetch explanation whenever selected anomaly ID changes
  useEffect(() => {
    if (!selected) {
      setExplanation(null);
      return;
    }
    let active = true;
    setLoadingExplanation(true);
    setError(null);
    anomalyService.getAnomalyExplanation(selected.anomaly_id)
      .then((value) => {
        if (active) setExplanation(value);
      })
      .catch(() => {
        if (active) setError('Explanation is not yet available for this incident.');
      })
      .finally(() => {
        if (active) setLoadingExplanation(false);
      });
    return () => { active = false; };
  }, [selected?.anomaly_id]);

  // Ensure current selected anomaly is available in the pill selectors
  const selectorList = useMemo(() => {
    if (selected && selected.timestamp && !effectiveAnomalies.some((a) => a.anomaly_id === selected.anomaly_id)) {
      return [selected, ...effectiveAnomalies];
    }
    return effectiveAnomalies;
  }, [effectiveAnomalies, selected]);

  const features = useMemo(() => {
    return [...(explanation?.features ?? [])]
      .sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
      .slice(0, 3);
  }, [explanation?.features]);

  const top = features[0];
  const implicated = explanation?.affected_parameters?.length
    ? explanation.affected_parameters
    : (selected?.affected_parameters ?? []);
  const observed = suggestedFromRecord(explanation?.observed_values ?? selected?.observed_values);
  const suggested = suggestedFromRecord(explanation?.suggested_values ?? selected?.suggested_values);
  const score = explanation?.anomaly_score_pct ?? (selected?.anomaly_score_pct ?? 0);

  // ALL HOOKS EXECUTED UNCONDITIONALLY ABOVE
  if (isLoading && effectiveAnomalies.length === 0 && !selected) {
    return (
      <Card variant="glass" className={`sg-explain-card ${className}`}>
        <Skeleton width="260px" height="1.2rem" />
        <Skeleton width="100%" height="220px" style={{ marginTop: '1rem' }} />
      </Card>
    );
  }

  if (!selected) {
    return (
      <Card variant="glass" className={`sg-explain-card ${className}`}>
        <div className="sg-explain-card__empty">
          <CheckCircle2 size={24} />
          <strong>Decision X-Ray is standing by</strong>
          <span>When an anomaly is detected for this station, SkyGuard will display the evidence behind its decision here.</span>
        </div>
      </Card>
    );
  }

  return (
    <Card variant="glass" className={`sg-explain-card ${className}`} role="region" aria-label="Plain-language anomaly explanation">
      <div className="sg-explain-card__header">
        <div>
          <span className="sg-explain-card__eyebrow"><BrainCircuit size={15} /> EXPLAINABLE AI</span>
          <h3>Decision X-Ray: why SkyGuard raised this alert</h3>
          <p>Plain language evidence from the live model and deterministic safety rules.</p>
        </div>
        <span className="sg-explain-card__score"><Gauge size={16} /> {Math.round(score)}% evidence strength</span>
      </div>

      <div className="sg-explain-card__selector-wrapper">
        <div className="sg-explain-card__selector" aria-label="Choose an anomaly to explain">
          {selectorList.slice(0, 6).map((anomaly) => {
            const d = new Date(anomaly.timestamp);
            const dateStr = !isNaN(d.getTime()) ? d.toLocaleDateString([], { month: 'short', day: 'numeric' }) : '';
            const timeStr = !isNaN(d.getTime()) ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
            const isActive = anomaly.anomaly_id === selected.anomaly_id;
            return (
              <button
                key={anomaly.anomaly_id}
                type="button"
                onClick={() => setSelectedId(anomaly.anomaly_id)}
                className={isActive ? 'is-active' : ''}
                title={`Station: ${anomaly.station_id || 'Network'} — Time: ${dateStr} ${timeStr}`}
              >
                {anomaly.station_id && <span className="sg-explain-card__pill-station">[{anomaly.station_id}]</span>}
                <span className="sg-explain-card__pill-type">{anomaly.type.replace(/_/g, ' ')}</span>
                <span className="sg-explain-card__pill-time">{dateStr} {timeStr}</span>
              </button>
            );
          })}
        </div>
        <Link to="/alerts" className="sg-explain-card__view-all-link" title="View all anomalies network-wide on the Alerts page">
          <span>All Alerts</span>
          <ExternalLink size={12} aria-hidden="true" />
        </Link>
      </div>

      {loadingExplanation ? (
        <Skeleton width="100%" height="180px" />
      ) : error ? (
        <p className="sg-explain-card__error">{error}</p>
      ) : (
        <>
          <div className="sg-explain-card__hero">
            <div className="sg-explain-card__driver">
              <Sparkles size={18} />
              <div>
                <span>Strongest evidence</span>
                <strong>{top ? displayParameter(top.name) : 'Rule evidence (no model explanation available)'}</strong>
                <p>{top ? explainFeature(top.name) : 'Deterministic rules detected a pattern that warrants investigation.'}</p>
              </div>
            </div>
            <div className="sg-explain-card__split">
              <div>
                <span>Model evidence · 60% weight</span>
                <strong>
                  {explanation?.model_confidence_pct == null
                    ? (explanation?.model_status === 'UNAVAILABLE_WARMUP' ? 'Warming up — insufficient history' : 'Unavailable')
                    : `${Math.round(explanation.model_confidence_pct)}% evidence strength`}
                </strong>
              </div>
              <div>
                <span>Rule evidence · 40% weight</span>
                <strong>{explanation?.rule_confidence_pct == null ? 'n.a.' : `${Math.round(explanation.rule_confidence_pct)}%`}</strong>
              </div>
              {(explanation?.decision_basis || selected?.decision_basis) && (
                <div style={{ gridColumn: '1/-1' }}>
                  <span>Decision basis</span>
                  <strong style={{ fontSize: '0.75rem', letterSpacing: '0.05em' }}>
                    {(explanation?.decision_basis || selected?.decision_basis || '').replace(/_/g, ' ')}
                  </strong>
                </div>
              )}
            </div>
          </div>

          <div className="sg-explain-card__body">
            <section>
              <h4><ShieldAlert size={15} /> Station Context</h4>
              {(() => {
                const isMultivariate = selected?.type === 'multivariate_inconsistency' || explanation?.fault_type === 'multivariate_inconsistency';
                const thermo = explanation?.spatial_context?.thermodynamic_context;
                const regime = explanation?.regime || selected?.regime;
                const regimeLabels: Record<string, string> = {
                  DAYTIME_WARMING: 'Normal daytime warming pattern — environmental drift is expected.',
                  NIGHTTIME_COOLING: 'Normal nighttime cooling pattern.',
                  STABLE: 'Stable conditions with low environmental variability.',
                  HIGH_HEAT: 'Extreme high heat conditions — elevated temperature envelope.',
                  HIGH_HUMIDITY: 'High humidity conditions.',
                  PRESSURE_SHIFT: 'Significant barometric pressure change — possible incoming weather front.',
                  HIGH_VOLATILITY: 'High variability conditions — detection thresholds may be less reliable.',
                  REGIME_TRANSITION: 'Environmental regime transition detected — false positives possible.',
                  THERMODYNAMIC_CONFLICT: 'Thermodynamic consistency violation: Temperature and humidity violate Clausius-Clapeyron atmospheric limits.',
                  UNKNOWN_INSUFFICIENT_DATA: 'Context unavailable — station still in warm-up period (insufficient history).',
                  UNKNOWN_CONTEXT_FAILURE: 'Context classification failed — check system logs.',
                };

                if (isMultivariate || thermo?.is_violation) {
                  return (
                    <>
                      <p className="sg-explain-card__muted">
                        State: <strong style={{ color: '#f87171' }}>THERMODYNAMIC INCONSISTENCY</strong>
                      </p>
                      <p className="sg-explain-card__muted" style={{ fontSize: '0.78rem' }}>
                        Coupled cross-sensor conflict: Reported temperature and relative humidity cannot co-occur in natural terrestrial atmosphere. Barometric pressure is nominal.
                      </p>
                    </>
                  );
                }

                return (
                  <>
                    <p className="sg-explain-card__muted">Regime: <strong>{regime ? regime.replace(/_/g, ' ') : 'Unknown'}</strong></p>
                    {regime && regimeLabels[regime] && <p className="sg-explain-card__muted" style={{ fontSize: '0.78rem' }}>{regimeLabels[regime]}</p>}
                  </>
                );
              })()}
            </section>

            <section>
              <h4><ShieldAlert size={15} /> Network Evidence</h4>
              {(() => {
                const corr = explanation?.network_corroboration || selected?.network_corroboration;
                const spatial = explanation?.spatial_context;

                if (spatial) {
                  const isMinimalSpatial = spatial.spatial_impact?.toLowerCase().includes('minimal') || spatial.spatial_impact?.toLowerCase().includes('agreement');
                  const pillClass = corr === 'REGIONAL'
                    ? 'is-regional'
                    : isMinimalSpatial
                    ? 'is-regional'
                    : 'is-localized';
                  const pillLabel = isMinimalSpatial
                    ? 'PEERS IN AGREEMENT'
                    : corr
                    ? corr.replace(/_/g, ' ')
                    : 'SPATIAL CONTEXT';
                  return (
                    <div className="sg-spatial-box">
                      <div className="sg-spatial-box__header">
                        <span className={`sg-spatial-status-pill ${pillClass}`}>
                          {pillLabel}
                        </span>
                        {spatial.spatial_impact && !spatial.spatial_impact.includes(spatial.analysis_text.slice(0, 30)) && (
                          <span className="sg-spatial-impact-badge">{spatial.spatial_impact}</span>
                        )}
                      </div>

                      <div className="sg-spatial-narrative">
                        {spatial.analysis_text.split('\n').map((line, idx) => (
                          <p key={idx} className="sg-explain-card__muted" style={{ margin: '0.2rem 0', color: idx === 0 ? '#f1f5f9' : '#cbd5e1' }}>
                            {idx === 0 ? <strong>{line}</strong> : line}
                          </p>
                        ))}
                      </div>

                      {spatial.thermodynamic_context?.explanation && (
                        <div style={{
                          margin: '0.45rem 0',
                          padding: '0.5rem 0.65rem',
                          borderRadius: '4px',
                          background: 'rgba(239, 68, 68, 0.08)',
                          borderLeft: '3px solid #ef4444',
                          fontSize: '0.76rem',
                          color: '#fca5a5',
                          lineHeight: '1.45',
                        }}>
                          <strong style={{ display: 'block', marginBottom: '0.2rem', color: '#fecaca' }}>
                            Thermodynamic Verification (Clausius-Clapeyron):
                          </strong>
                          {spatial.thermodynamic_context.explanation}
                        </div>
                      )}

                      <div className="sg-spatial-peers">
                        <div className="sg-spatial-peer is-target">
                          <span className="sg-spatial-peer__label">{spatial.target_station.name} (Target)</span>
                          <span className="sg-spatial-peer__value">
                            {spatial.target_station.reading}{spatial.target_station.unit.startsWith('°') ? spatial.target_station.unit : ` ${spatial.target_station.unit}`}
                          </span>
                        </div>
                        {spatial.peer_stations.map((peer) => (
                          <div key={peer.station_id} className="sg-spatial-peer">
                            <span className="sg-spatial-peer__label">{peer.name}</span>
                            <span className="sg-spatial-peer__value">
                              {peer.reading}{peer.unit.startsWith('°') ? peer.unit : ` ${peer.unit}`}
                            </span>
                          </div>
                        ))}
                      </div>

                      {spatial.recommended_action && (
                        <div className="sg-spatial-recommendation">
                          <CheckCircle2 size={14} style={{ color: '#38bdf8', flexShrink: 0, marginTop: '2px' }} />
                          <span><strong>Recommended action:</strong> {spatial.recommended_action}</span>
                        </div>
                      )}
                    </div>
                  );
                }

                if (!corr) return <p className="sg-explain-card__muted">Network analysis not performed (no anomaly detected).</p>;
                const labels: Record<string, string> = {
                  REGIONAL: 'Nearby stations show similar changes — regional environmental event possible. Sensor should not be blamed immediately.',
                  LOCALIZED: 'Nearby stations are within normal range — pattern appears localized. Sensor or telemetry investigation recommended.',
                  INSUFFICIENT_CORROBORATION: 'Insufficient peer data for network analysis. Cannot determine if event is localized or regional.',
                };
                return (
                  <>
                    <p className="sg-explain-card__muted"><strong>{corr.replace(/_/g, ' ')}</strong></p>
                    <p className="sg-explain-card__muted">{labels[corr] || 'Network state unknown.'}</p>
                  </>
                );
              })()}
            </section>

            <section>
              <h4><ShieldAlert size={15} /> {features.length ? 'Model evidence' : 'Rule evidence'}</h4>
              {features.length ? (
                <ol>
                  {(() => {
                    const totalImpact = features.reduce((acc, f) => acc + Math.abs(f.impact), 0) || 1;
                    return features.map((feature) => {
                      const pct = Math.min(100, Math.max(1, Math.round((Math.abs(feature.impact) / totalImpact) * 100)));
                      return (
                        <li key={feature.name}>
                          <span className={feature.impact >= 0 ? 'risk' : 'normal'}>
                            {feature.impact >= 0 ? 'Raises risk' : 'Offsets risk'}
                          </span>
                          <div>
                            <strong>{displayParameter(feature.name)}</strong>
                            <p>{explainFeature(feature.name)}</p>
                          </div>
                          <b>{pct}%</b>
                        </li>
                      );
                    });
                  })()}
                </ol>
              ) : (
                <p className="sg-explain-card__muted">Model explanation unavailable — this event was assessed by deterministic safety rules only. Rule evidence is not a probability estimate.</p>
              )}
            </section>

            <section>
              <h4><ChevronRight size={15} /> Operator-ready conclusion</h4>
              {(() => {
                const isMultivariate = selected?.type === 'multivariate_inconsistency' || explanation?.fault_type === 'multivariate_inconsistency';
                const thermo = explanation?.spatial_context?.thermodynamic_context;

                if (isMultivariate || thermo?.is_violation) {
                  return (
                    <div style={{ marginBottom: '0.5rem' }}>
                      <p className="sg-explain-card__conclusion" style={{ color: '#f8fafc', fontWeight: 600 }}>
                        Thermodynamic Violation: Coupled temperature-hygrometer failure detected.
                      </p>
                      <p className="sg-explain-card__muted" style={{ fontSize: '0.8rem', lineHeight: '1.45', margin: '0.3rem 0' }}>
                        The station recorded mutually contradictory values ({formatSuggestedList(observed)}). By the Clausius-Clapeyron relation, maintaining high relative humidity at elevated temperatures requires an unphysical water vapor concentration under standard surface barometric pressure. Surrounding regional peer stations confirm ambient conditions are nominal, isolating the fault to this station's sensor pair.
                      </p>
                    </div>
                  );
                }

                return (
                  <p className="sg-explain-card__conclusion">
                    {implicated.length
                      ? `Suggested investigation target: ${implicated.map(displayParameter).join(', ')}. This is an indication, not a confirmed diagnosis.`
                      : 'The detector found a station-level pattern that requires review.'}
                  </p>
                );
              })()}

              {observed.length > 0 && <p><strong>Observed:</strong> {formatSuggestedList(observed)}</p>}
              <SuggestedValues items={suggested} emptyLabel="Suggested replacement becomes available after the baseline warm-up." />
              <p className="sg-explain-card__action">
                <CheckCircle2 size={15} /> Suggested action: Keep raw telemetry visible; use the suggested reconstructed values for downstream meteorological pipelines while investigating.
              </p>
            </section>
          </div>
        </>
      )}
    </Card>
  );
};
