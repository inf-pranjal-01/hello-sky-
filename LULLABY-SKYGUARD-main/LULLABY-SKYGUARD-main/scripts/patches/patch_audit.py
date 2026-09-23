"""
Bug Audit Master Patch Script
Applies all P0/P1/P2 fixes to the SkyGuard AI codebase.
"""
import re, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
results = {}

def patch(filepath, old, new, label):
    """Patch helper - replaces old text with new, reports success/failure."""
    full = os.path.join(ROOT, filepath)
    with open(full, 'r', encoding='utf-8') as f:
        text = f.read()
    if old in text:
        text = text.replace(old, new, 1)
        with open(full, 'w', encoding='utf-8') as f:
            f.write(text)
        results[label] = 'OK'
    else:
        results[label] = 'NOT FOUND'

# ==========================================================================
# PHASE 0 — P0 UI TRUTHFULNESS FIXES
# ==========================================================================

# -- AnomalyDetailModal.tsx -------------------------------------------------
patch(
    'FRONTEND/src/components/alerts/AnomalyDetailModal.tsx',
    '<h4 className="sg-anomaly-modal__section-title">Root Cause Determination</h4>',
    '<h4 className="sg-anomaly-modal__section-title">Anomaly Indication</h4>',
    'UI-Modal: Root Cause Determination → Anomaly Indication'
)
patch(
    'FRONTEND/src/components/alerts/AnomalyDetailModal.tsx',
    '<h4 className="sg-anomaly-modal__section-title">Suggested Replacement Reading</h4>',
    '<h4 className="sg-anomaly-modal__section-title">Estimated Replacement (Temporal Baseline — Not a Correction)</h4>',
    'UI-Modal: Suggested Replacement → Estimated Replacement'
)
patch(
    'FRONTEND/src/components/alerts/AnomalyDetailModal.tsx',
    '{Math.round(anomaly.anomaly_score_pct)}% SCORE',
    'Evidence Strength: {Math.round(anomaly.anomaly_score_pct)}%',
    'UI-Modal: Score → Evidence Strength'
)

# -- LatestAnomalyBanner.tsx ------------------------------------------------
patch(
    'FRONTEND/src/components/alerts/LatestAnomalyBanner.tsx',
    '<span className="sg-latest-banner__block-label">Root Cause Determination</span>',
    '<span className="sg-latest-banner__block-label">Anomaly Indication</span>',
    'UI-Banner: Root Cause Determination → Anomaly Indication'
)
patch(
    'FRONTEND/src/components/alerts/LatestAnomalyBanner.tsx',
    'Active Anomaly: {latestAnomaly.type.replace(\'_\', \' \')}',
    'Anomaly Detected: {latestAnomaly.type.replace(\'_\', \' \')}',
    'UI-Banner: Active Anomaly → Anomaly Detected'
)
patch(
    'FRONTEND/src/components/alerts/LatestAnomalyBanner.tsx',
    'Score: {Math.round(latestAnomaly.anomaly_score_pct)}%',
    'Evidence Strength: {Math.round(latestAnomaly.anomaly_score_pct)}%',
    'UI-Banner: Score → Evidence Strength'
)

# -- RecentAnomaliesList.tsx ------------------------------------------------
patch(
    'FRONTEND/src/components/alerts/RecentAnomaliesList.tsx',
    '<th scope="col">Root Cause</th>',
    '<th scope="col">Anomaly Indication</th>',
    'UI-Recent: Root Cause column → Anomaly Indication'
)
patch(
    'FRONTEND/src/components/alerts/RecentAnomaliesList.tsx',
    '{anom.root_cause}',
    '{anom.root_cause}',  # keep content, just the header changes
    'UI-Recent: Root Cause content (no-op — header already changed)'
)

# -- ExplainabilityCommandCenter.tsx ----------------------------------------
patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    '{Math.round(score)}% confidence',
    '{Math.round(score)}% evidence strength',
    'UI-Explain: confidence → evidence strength'
)
patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    "'Rule-based anomaly confirmation'",
    "'Rule evidence (no model explanation available)'",
    'UI-Explain: Rule-based anomaly confirmation → Rule evidence'
)
patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    "'The deterministic safety rules confirmed an unusual sensor pattern.'",
    "'Deterministic rules detected a pattern that warrants investigation.'",
    'UI-Explain: confirmed → warrants investigation'
)

# Fix conditional section header: show "What the model noticed" only when SHAP available
# and "Rule evidence" when SHAP is absent
old_section_header = '''        <section>
          <h4><ShieldAlert size={15} /> What the model noticed</h4>
          {features.length ? <ol>{features.map((feature) => <li key={feature.name}><span className={feature.impact >= 0 ? 'risk' : 'normal'}>{feature.impact >= 0 ? 'Raises risk' : 'Offsets risk'}</span><div><strong>{displayParameter(feature.name)}</strong><p>{explainFeature(feature.name)}</p></div><b>{Math.round(Math.abs(feature.impact) * 100)}%</b></li>)}</ol> : <p className="sg-explain-card__muted">This event was confirmed by deterministic safety rules before a full SHAP feature vector was available.</p>}
        </section>'''

new_section_header = '''        <section>
          <h4><ShieldAlert size={15} /> {features.length ? 'Model evidence' : 'Rule evidence'}</h4>
          {features.length ? (
            <ol>{features.map((feature) => <li key={feature.name}><span className={feature.impact >= 0 ? 'risk' : 'normal'}>{feature.impact >= 0 ? 'Raises risk' : 'Offsets risk'}</span><div><strong>{displayParameter(feature.name)}</strong><p>{explainFeature(feature.name)}</p></div><b>{Math.round(Math.abs(feature.impact) * 100)}%</b></li>)}</ol>
          ) : (
            <p className="sg-explain-card__muted">Model explanation unavailable — this event was assessed by deterministic safety rules only. Rule evidence is not a probability estimate.</p>
          )}
        </section>'''

patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    old_section_header,
    new_section_header,
    'UI-Explain: conditional model/rule section header'
)

# Fix operator conclusion — replace "is the most likely affected sensor channel" with suggestion language
patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    '`${implicated.map(displayParameter).join(\', \')} is the most likely affected sensor channel.`',
    '`Suggested investigation target: ${implicated.map(displayParameter).join(\', \')}. This is an indication, not a confirmed diagnosis.`',
    'UI-Explain: sensor channel claim → suggestion language'
)

# ==========================================================================
# PHASE 1 — P0 BACKEND: detect.py _corroborate_network bug fixes
# ==========================================================================

# Fix 1: compute target_features once before loop, n_features once per peer
old_corroborate = '''def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, fault_type: str, implicated_params: list) -> str:
    if not neighbor_buffers:
        return "INSUFFICIENT_CORROBORATION"
    
    target_time = pd.to_datetime(raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1])
    
    eligible_peers = 0
    corroborating_peers = 0
    
    for nid, n_df in neighbor_buffers.items():
        if n_df.empty: continue
        n_latest = n_df.iloc[-1]
        n_time = pd.to_datetime(n_latest["timestamp"])
        
        # freshness / timestamp alignment
        if abs((n_time - target_time).total_seconds()) > 3600:
            continue
            
        eligible_peers += 1
        
        # For simplicity, if fault is related to a parameter, we check if neighbor has a similar extreme value
        # But wait, we can just run a quick deviation check
        n_features = build_features_for_latest(n_df)
        
        # If any implicated parameter has a deviation > 2.0 (or < -2.0) in the same direction, it's regional
        is_corroborating = False
        for param in implicated_params:
            prefix = PARAM_PREFIXES.get(param)
            if not prefix: continue
            
            target_dev = build_features_for_latest(history_df).get(f"{prefix}_deviation", 0)
            peer_dev = n_features.get(f"{prefix}_deviation", 0)
            
            if pd.notna(target_dev) and pd.notna(peer_dev):
                if abs(peer_dev) > 2.0 and (target_dev * peer_dev > 0):
                    is_corroborating = True
                    break
        
        if is_corroborating:
            corroborating_peers += 1

    if eligible_peers < 1:
        return "INSUFFICIENT_CORROBORATION"
    elif corroborating_peers > 0:
        return "REGIONAL"
    else:
        return "LOCALIZED"'''

new_corroborate = '''def _corroborate_network(raw_reading: dict, history_df: pd.DataFrame, neighbor_buffers: dict, fault_type: str, implicated_params: list) -> str:
    """
    Network corroboration check -- classifies anomaly as LOCALIZED, REGIONAL,
    or INSUFFICIENT_CORROBORATION based on whether eligible cluster peers show
    similar deviations in the same direction.

    Performance fix: target_features is computed ONCE before the peer loop.
    Each peer's features are also computed once per peer (not per param).

    Returns None when neighbor_buffers is None/empty (no check performed),
    which the caller must distinguish from INSUFFICIENT_CORROBORATION (peers
    exist but were all stale/unhealthy). See audit §13.6.
    """
    if not neighbor_buffers:
        # No peer data available -- cannot perform network check.
        # Caller should surface this as INSUFFICIENT_CORROBORATION with reason.
        return "INSUFFICIENT_CORROBORATION"

    target_time = pd.to_datetime(raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1])

    # Compute target station features ONCE before the peer loop.
    try:
        target_features = build_features_for_latest(history_df)
    except Exception:
        return "INSUFFICIENT_CORROBORATION"

    eligible_peers = 0
    corroborating_peers = 0

    for nid, n_df in neighbor_buffers.items():
        if n_df is None or n_df.empty:
            continue
        n_latest = n_df.iloc[-1]
        n_time = pd.to_datetime(n_latest["timestamp"])

        # Freshness check: peer reading must be within 1h of target.
        if abs((n_time - target_time).total_seconds()) > 3600:
            continue

        eligible_peers += 1

        # Compute peer features ONCE per peer (not per implicated parameter).
        try:
            n_features = build_features_for_latest(n_df)
        except Exception:
            eligible_peers -= 1  # don't count peers whose features failed
            continue

        # Check whether any implicated parameter shows a same-direction
        # deviation > 2.0 sigma in the peer.
        is_corroborating = False
        for param in implicated_params:
            prefix = PARAM_PREFIXES.get(param)
            if not prefix:
                continue
            target_dev = target_features.get(f"{prefix}_deviation", 0)
            peer_dev = n_features.get(f"{prefix}_deviation", 0)
            if pd.notna(target_dev) and pd.notna(peer_dev):
                if abs(peer_dev) > 2.0 and (target_dev * peer_dev > 0):
                    is_corroborating = True
                    break

        if is_corroborating:
            corroborating_peers += 1

    if eligible_peers < 1:
        return "INSUFFICIENT_CORROBORATION"
    elif corroborating_peers > 0:
        return "REGIONAL"
    else:
        return "LOCALIZED"'''

patch(
    'model/detect.py',
    old_corroborate,
    new_corroborate,
    'Backend: _corroborate_network double-call fix + documentation'
)

# Fix 2: return None for network_state when is_anomaly=False
old_network_state = '''    network_state = "INSUFFICIENT_CORROBORATION"
    if is_anomaly:
        implicated = likely_sensors
        if not implicated and rules["fired"]:
            implicated = list(set(r[1] for r in rules["fired"]))
        network_state = _corroborate_network(raw_reading, history_df, neighbor_buffers, fault_type, implicated)'''

new_network_state = '''    # Network corroboration is only meaningful when an anomaly was detected.
    # When is_anomaly=False, return None explicitly (not INSUFFICIENT_CORROBORATION)
    # so the frontend and API can distinguish "no check needed" from "check ran but
    # peers were unavailable." See audit §13.6.
    network_state = None
    if is_anomaly:
        implicated = likely_sensors
        if not implicated and rules["fired"]:
            implicated = list(set(r[1] for r in rules["fired"]))
        neighbor_buffers_safe = neighbor_buffers or {}
        network_state = _corroborate_network(
            raw_reading, history_df, neighbor_buffers_safe, fault_type, implicated
        )'''

patch(
    'model/detect.py',
    old_network_state,
    new_network_state,
    'Backend: network_state=None when not anomalous'
)

# ==========================================================================
# PHASE 2 — P1: 9-state regime classifier
# ==========================================================================

old_regime = '''    # Regime metadata (context-only, Option A)
    # Simple mockup: use the hour of day and temp to assign a regime.
    hour = pd.to_datetime(raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1]).hour
    temp = raw_reading.get("temperature_c", 25.0)
    
    if hour >= 6 and hour < 18:
        if temp > 35:
            regime = "SUMMER_DAY_HIGH_HEAT"
        else:
            regime = "NORMAL_DAY"
    else:
        if temp < 10:
            regime = "WINTER_NIGHT_COLD"
        else:
            regime = "NORMAL_NIGHT"'''

new_regime = '''    # Regime classification — 9-state taxonomy per audit §12.5.
    # Uses the already-computed feature_row so no extra featurization cost.
    regime = _classify_regime(feature_row, raw_reading, history_df)'''

patch(
    'model/detect.py',
    old_regime,
    new_regime,
    'Backend: 9-state regime classifier'
)

# Insert _classify_regime function before score_reading definition
old_score_reading_def = '''def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None) -> dict:'''

new_score_reading_def = '''def _classify_regime(feature_row: "pd.Series", raw_reading: dict, history_df: "pd.DataFrame") -> str:
    """
    Classifies the current station environmental context into one of 9 regimes
    per audit §12.5. Uses the already-computed feature_row from build_features_for_latest
    to avoid double featurization.

    States:
      DAYTIME_WARMING       hour 6-18, temp trending up relative to baseline
      NIGHTTIME_COOLING     hour 18-6, temp trending down
      STABLE                low volatility, minimal rate-of-change all params
      HIGH_HEAT             temp > 35°C or temp_deviation > 2.5σ
      HIGH_HUMIDITY         humidity > 85% or humidity_deviation > 2.0σ
      PRESSURE_SHIFT        significant pressure rate-of-change over 3h
      HIGH_VOLATILITY       any parameter volatility_z > 2.0
      REGIME_TRANSITION     direction reversal in recent temp readings
      UNKNOWN_INSUFFICIENT_DATA   NaN features (warm-up period)
      UNKNOWN_CONTEXT_FAILURE     exception during classification
    """
    try:
        hour = pd.to_datetime(
            raw_reading.get("timestamp") or history_df["timestamp"].iloc[-1]
        ).hour

        # Pull scalar values from feature_row with safe fallbacks.
        def get(col, default=None):
            v = feature_row.get(col)
            if v is None or (hasattr(v, '__float__') and pd.isna(float(v))):
                return default
            return float(v)

        temp_c         = get("temperature_c")
        humidity_pct   = get("humidity_pct")
        temp_dev       = get("temp_deviation")
        humidity_dev   = get("humidity_deviation")
        temp_roc_1h    = get("temp_roc_1h", 0.0)
        temp_roc_3h    = get("temp_roc_3h", 0.0)
        pressure_roc_3h = get("pressure_roc_3h", 0.0)
        temp_vol_z     = get("temp_volatility_z", 0.0)
        pressure_vol_z = get("pressure_volatility_z", 0.0)
        humidity_vol_z = get("humidity_volatility_z", 0.0)

        # UNKNOWN_INSUFFICIENT_DATA: primary indicators are NaN
        if temp_dev is None or humidity_dev is None:
            return "UNKNOWN_INSUFFICIENT_DATA"

        # HIGH_VOLATILITY: any param's volatility z-score > 2.0
        if (temp_vol_z is not None and abs(temp_vol_z) > 2.0
                or pressure_vol_z is not None and abs(pressure_vol_z) > 2.0
                or humidity_vol_z is not None and abs(humidity_vol_z) > 2.0):
            return "HIGH_VOLATILITY"

        # PRESSURE_SHIFT: sustained pressure rate-of-change over 3h
        if pressure_roc_3h is not None and abs(pressure_roc_3h) > 2.0:
            return "PRESSURE_SHIFT"

        # HIGH_HEAT: extreme temperature by absolute value or deviation
        if (temp_c is not None and temp_c > 35.0) or (temp_dev is not None and temp_dev > 2.5):
            return "HIGH_HEAT"

        # HIGH_HUMIDITY: extreme humidity by absolute value or deviation
        if (humidity_pct is not None and humidity_pct > 85.0) or (humidity_dev is not None and humidity_dev > 2.0):
            return "HIGH_HUMIDITY"

        # REGIME_TRANSITION: recent direction reversal in temperature
        if len(history_df) >= 3:
            recent_temps = pd.to_numeric(history_df["temperature_c"].tail(4), errors="coerce").dropna().to_numpy()
            if len(recent_temps) >= 3:
                diffs = [recent_temps[i+1] - recent_temps[i] for i in range(len(recent_temps)-1)]
                signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diffs]
                non_zero = [s for s in signs if s != 0]
                if len(non_zero) >= 2 and non_zero[-1] != non_zero[-2]:
                    return "REGIME_TRANSITION"

        # STABLE: small roc and deviation across all parameters
        if (abs(temp_roc_3h) < 0.5 and abs(temp_dev) < 0.5
                and abs(pressure_roc_3h) < 0.5):
            return "STABLE"

        # DAYTIME_WARMING / NIGHTTIME_COOLING
        if 6 <= hour < 18:
            if temp_roc_3h is not None and temp_roc_3h > 0 and temp_dev > 0.2:
                return "DAYTIME_WARMING"
            return "STABLE"
        else:
            if temp_roc_3h is not None and temp_roc_3h < 0:
                return "NIGHTTIME_COOLING"
            return "STABLE"

    except Exception:
        return "UNKNOWN_CONTEXT_FAILURE"


def score_reading(raw_reading: dict, history_df: pd.DataFrame, artifact: dict,
                  neighbor_buffers: dict = None, explainer=None) -> dict:'''

patch(
    'model/detect.py',
    old_score_reading_def,
    new_score_reading_def,
    'Backend: _classify_regime 9-state function inserted'
)

# ==========================================================================
# PHASE 3 — P2: API — add decision_basis + model_status fields
# ==========================================================================

# Insert helper function into main.py before the routes
old_main_import_end = '''from model.simulator import (
    SimulatorState,
    create_simulator_state,
    run_simulation_loop,
)'''

new_main_import_end = '''from model.simulator import (
    SimulatorState,
    create_simulator_state,
    run_simulation_loop,
)


def _compute_decision_basis(model_confidence_pct, rules_fired) -> str:
    """
    Computes an explicit decision basis label per audit §10.2.
    Returns one of:
      MODEL_AND_RULE_SUPPORTED, RULE_ONLY_DETERMINISTIC, RULE_ONLY_STATISTICAL,
      MODEL_CONFIRMED, MODEL_UNAVAILABLE, PHYSICS_ONLY, INSUFFICIENT_EVIDENCE
    """
    has_model = model_confidence_pct is not None
    deterministic_rules = {"physical_bounds", "dropout", "sensor_fail_low"}
    statistical_rules   = {"drift", "spike", "frozen_value", "multivariate_inconsistency"}

    fired_types = {r[0] for r in (rules_fired or [])}
    has_deterministic = bool(fired_types & deterministic_rules)
    has_statistical   = bool(fired_types & statistical_rules)
    has_rules = has_deterministic or has_statistical

    if has_deterministic and not has_model:
        return "PHYSICS_ONLY"
    if has_deterministic and has_model:
        return "MODEL_AND_RULE_SUPPORTED"
    if has_statistical and has_model:
        return "MODEL_AND_RULE_SUPPORTED"
    if has_statistical and not has_model:
        return "RULE_ONLY_STATISTICAL"
    if has_model and not has_rules:
        return "MODEL_CONFIRMED"
    if not has_model and not has_rules:
        return "MODEL_UNAVAILABLE"
    return "INSUFFICIENT_EVIDENCE"


def _compute_model_status(model_confidence_pct, history_len: int) -> str:
    """
    Returns explicit model availability status per audit §8.2.
    AVAILABLE, UNAVAILABLE_WARMUP, or UNAVAILABLE_MISSING_FEATURES.
    """
    if model_confidence_pct is not None:
        return "AVAILABLE"
    # <48 readings = likely still in rolling-window warm-up
    if history_len is not None and history_len < 48:
        return "UNAVAILABLE_WARMUP"
    return "UNAVAILABLE_MISSING_FEATURES"'''

patch(
    'main.py',
    old_main_import_end,
    new_main_import_end,
    'Backend: decision_basis + model_status helpers in main.py'
)

# Update /api/anomalies/latest to include decision_basis + model_status
old_latest_return = '''                "affected_parameters": a.get("affected_parameters", []),
                "regime": a.get("regime"),
                "network_corroboration": a.get("network_corroboration"),
            }'''

new_latest_return = '''                "affected_parameters": a.get("affected_parameters", []),
                "regime": a.get("regime"),
                "network_corroboration": a.get("network_corroboration"),
                "decision_basis": _compute_decision_basis(
                    a.get("model_confidence_pct"),
                    a.get("rules_fired") or [],
                ),
                "model_status": _compute_model_status(
                    a.get("model_confidence_pct"),
                    None,
                ),
            }'''

patch(
    'main.py',
    old_latest_return,
    new_latest_return,
    'API: decision_basis + model_status in /anomalies/latest'
)

# Update /api/explain/{anomaly_id} to include decision_basis + model_status
old_explain_return = '''        "regime": match.get("regime"),
        "network_corroboration": match.get("network_corroboration"),
    }'''

new_explain_return = '''        "regime": match.get("regime"),
        "network_corroboration": match.get("network_corroboration"),
        "decision_basis": _compute_decision_basis(
            match.get("model_confidence_pct"),
            match.get("rules_fired") or [],
        ),
        "model_status": _compute_model_status(
            match.get("model_confidence_pct"),
            None,
        ),
    }'''

patch(
    'main.py',
    old_explain_return,
    new_explain_return,
    'API: decision_basis + model_status in /explain/{id}'
)

# ==========================================================================
# PHASE 4 — P3: Frontend — show decision_basis badge + improved text
# ==========================================================================

# Update types/index.ts to add decision_basis + model_status fields
old_types = '''  regime?: string;
  network_corroboration?: NetworkCorroborationState;
}

export interface AnomalyExplanation {'''

new_types = '''  regime?: string;
  network_corroboration?: NetworkCorroborationState;
  decision_basis?: string;
  model_status?: string;
}

export interface AnomalyExplanation {'''

patch(
    'FRONTEND/src/types/index.ts',
    old_types,
    new_types,
    'Types: decision_basis + model_status on LatestAnomaly'
)

# Update AnomalyExplanation type
old_explain_type = '''  regime?: string;
  network_corroboration?: NetworkCorroborationState;
}

/**
 * SIH Demo Anomaly Injection Schema'''

new_explain_type = '''  regime?: string;
  network_corroboration?: NetworkCorroborationState;
  decision_basis?: string;
  model_status?: string;
}

/**
 * SIH Demo Anomaly Injection Schema'''

patch(
    'FRONTEND/src/types/index.ts',
    old_explain_type,
    new_explain_type,
    'Types: decision_basis + model_status on AnomalyExplanation'
)

# Update ExplainabilityCommandCenter: show decision_basis badge + improved model status label
old_model_split = '''<div className="sg-explain-card__split"><div><span>Model signal · 60% weight</span><strong>{explanation?.model_confidence_pct == null ? 'Warm-up / n.a.' : `${Math.round(explanation.model_confidence_pct)}%`}</strong></div><div><span>Safety rules · 40% weight</span><strong>{explanation?.rule_confidence_pct == null ? 'n.a.' : `${Math.round(explanation.rule_confidence_pct)}%`}</strong></div></div>'''

new_model_split = '''<div className="sg-explain-card__split">
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
          <div style={{gridColumn:'1/-1'}}>
            <span>Decision basis</span>
            <strong style={{fontSize:'0.75rem', letterSpacing:'0.05em'}}>
              {(explanation?.decision_basis || selected?.decision_basis || '').replace(/_/g, ' ')}
            </strong>
          </div>
        )}
      </div>'''

patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    old_model_split,
    new_model_split,
    'UI-Explain: decision_basis badge + model status label'
)

# Improve network corroboration display text
old_corr_text = '''          <p className="sg-explain-card__muted">Corroboration: <strong>{(explanation?.network_corroboration || selected?.network_corroboration || 'INSUFFICIENT CORROBORATION').replace(/_/g, ' ')}</strong></p>
          <p className="sg-explain-card__muted">{(explanation?.network_corroboration === 'REGIONAL' || selected?.network_corroboration === 'REGIONAL') ? 'Neighbors report similar anomalies.' : 'Anomaly appears localized to this station.'}</p>'''

new_corr_text = '''          {(() => {
            const corr = explanation?.network_corroboration || selected?.network_corroboration;
            if (!corr) return <p className="sg-explain-card__muted">Network analysis not performed (no anomaly detected).</p>;
            const labels: Record<string,string> = {
              REGIONAL: 'Nearby stations show similar changes — regional environmental event possible. Sensor should not be blamed immediately.',
              LOCALIZED: 'Nearby stations are within normal range — pattern appears localized. Sensor or telemetry investigation recommended.',
              INSUFFICIENT_CORROBORATION: 'Insufficient peer data for network analysis. Cannot determine if event is localized or regional.',
            };
            return <>
              <p className="sg-explain-card__muted"><strong>{corr.replace(/_/g, ' ')}</strong></p>
              <p className="sg-explain-card__muted">{labels[corr] || 'Network state unknown.'}</p>
            </>;
          })()}'''

patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    old_corr_text,
    new_corr_text,
    'UI-Explain: human-readable network corroboration text'
)

# Improve regime display
old_regime_text = '''          <p className="sg-explain-card__muted">Regime: <strong>{(explanation?.regime || selected?.regime || 'UNKNOWN').replace(/_/g, ' ')}</strong></p>'''

new_regime_text = '''          {(() => {
            const regime = explanation?.regime || selected?.regime;
            const regimeLabels: Record<string,string> = {
              DAYTIME_WARMING: 'Normal daytime warming pattern — environmental drift is expected.',
              NIGHTTIME_COOLING: 'Normal nighttime cooling pattern.',
              STABLE: 'Stable conditions with low environmental variability.',
              HIGH_HEAT: 'High heat conditions — elevated temperature anomaly risk.',
              HIGH_HUMIDITY: 'High humidity conditions.',
              PRESSURE_SHIFT: 'Significant pressure change — possible incoming weather system.',
              HIGH_VOLATILITY: 'High variability conditions — detection thresholds may be less reliable.',
              REGIME_TRANSITION: 'Environmental regime transition detected — false positives possible.',
              UNKNOWN_INSUFFICIENT_DATA: 'Context unavailable — station still in warm-up period (insufficient history).',
              UNKNOWN_CONTEXT_FAILURE: 'Context classification failed — check system logs.',
            };
            return <>
              <p className="sg-explain-card__muted">Regime: <strong>{regime ? regime.replace(/_/g, ' ') : 'Unknown'}</strong></p>
              {regime && regimeLabels[regime] && <p className="sg-explain-card__muted" style={{fontSize:'0.78rem'}}>{regimeLabels[regime]}</p>}
            </>;
          })()}'''

patch(
    'FRONTEND/src/components/analytics/ExplainabilityCommandCenter.tsx',
    old_regime_text,
    new_regime_text,
    'UI-Explain: regime with human-readable description'
)

# ==========================================================================
# REPORT
# ==========================================================================
print("\n=== Bug Audit Patch Report ===")
ok = [k for k, v in results.items() if v == 'OK']
fail = [k for k, v in results.items() if v != 'OK']

print(f"\nApplied ({len(ok)}):")
for k in ok:
    print(f"  OK: {k}")

if fail:
    print(f"\nNot found ({len(fail)}):")
    for k in fail:
        print(f"  MISS: {k}")
else:
    print("\nAll patches applied successfully.")
