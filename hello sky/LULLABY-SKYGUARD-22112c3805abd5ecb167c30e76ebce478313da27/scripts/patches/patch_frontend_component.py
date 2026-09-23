with open('FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx', 'r') as f:
    text = f.read()

import re

# Block 1
old_body = '''      <div className="sg-explain-card__body"><section><h4><ShieldAlert size={15} /> What the model noticed</h4>{features.length ? <ol>{features.map((feature) => <li key={feature.name}><span className={feature.impact >= 0 ? 'risk' : 'normal'}>{feature.impact >= 0 ? 'Raises risk' : 'Offsets risk'}</span><div><strong>{displayParameter(feature.name)}</strong><p>{explainFeature(feature.name)}</p></div><b>{Math.round(Math.abs(feature.impact) * 100)}%</b></li>)}</ol> : <p className="sg-explain-card__muted">This event was confirmed by deterministic safety rules before a full SHAP feature vector was available.</p>}</section><section><h4><ChevronRight size={15} /> Operator-ready conclusion</h4><p className="sg-explain-card__conclusion">{implicated.length ? `${implicated.map(displayParameter).join(', ')} is the most likely affected sensor channel.` : 'The detector found a station-level pattern that requires review.'}</p>{observed.length > 0 && <p><strong>Observed:</strong> {formatSuggestedList(observed)}</p>}<SuggestedValues items={suggested} emptyLabel="Suggested replacement becomes available after the baseline warm-up." /><p className="sg-explain-card__action"><CheckCircle2 size={15} /> Keep raw telemetry visible; use the suggested reading for trusted downstream analysis while the sensor is investigated.</p></section></div>'''

new_body = '''      <div className="sg-explain-card__body">
        <section>
          <h4><ShieldAlert size={15} /> Station Context</h4>
          <p className="sg-explain-card__muted">Regime: <strong>{(explanation?.regime || selected?.regime || 'UNKNOWN').replace(/_/g, ' ')}</strong></p>
        </section>
        <section>
          <h4><ShieldAlert size={15} /> Network Evidence</h4>
          <p className="sg-explain-card__muted">Corroboration: <strong>{(explanation?.network_corroboration || selected?.network_corroboration || 'INSUFFICIENT CORROBORATION').replace(/_/g, ' ')}</strong></p>
          <p className="sg-explain-card__muted">{(explanation?.network_corroboration === 'REGIONAL' || selected?.network_corroboration === 'REGIONAL') ? 'Neighbors report similar anomalies.' : 'Anomaly appears localized to this station.'}</p>
        </section>
        <section>
          <h4><ShieldAlert size={15} /> What the model noticed</h4>
          {features.length ? <ol>{features.map((feature) => <li key={feature.name}><span className={feature.impact >= 0 ? 'risk' : 'normal'}>{feature.impact >= 0 ? 'Raises risk' : 'Offsets risk'}</span><div><strong>{displayParameter(feature.name)}</strong><p>{explainFeature(feature.name)}</p></div><b>{Math.round(Math.abs(feature.impact) * 100)}%</b></li>)}</ol> : <p className="sg-explain-card__muted">This event was confirmed by deterministic safety rules before a full SHAP feature vector was available.</p>}
        </section>
        <section>
          <h4><ChevronRight size={15} /> Operator-ready conclusion</h4>
          <p className="sg-explain-card__conclusion">{implicated.length ? `${implicated.map(displayParameter).join(', ')} is the most likely affected sensor channel.` : 'The detector found a station-level pattern that requires review.'}</p>
          {observed.length > 0 && <p><strong>Observed:</strong> {formatSuggestedList(observed)}</p>}
          <SuggestedValues items={suggested} emptyLabel="Suggested replacement becomes available after the baseline warm-up." />
          <p className="sg-explain-card__action"><CheckCircle2 size={15} /> Suggested action: Keep raw telemetry visible; use the suggested reading for downstream analysis while investigating.</p>
        </section>
      </div>'''

if old_body in text:
    text = text.replace(old_body, new_body)
    print("Patched ExplainabilityCommandCenter block 1")
else:
    print("ExplainabilityCommandCenter block 1 not found")

with open('FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx', 'w') as f:
    f.write(text)
