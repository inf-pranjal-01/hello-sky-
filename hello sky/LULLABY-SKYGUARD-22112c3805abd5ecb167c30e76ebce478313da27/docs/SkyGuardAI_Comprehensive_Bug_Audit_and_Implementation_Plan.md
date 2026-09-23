# SkyGuardAI — Comprehensive Bug Audit, Drift Redesign, Validation Plan, and Implementation Checklist

> **Primary regression case:** `AWS-RAN-067` (Ranchi), 2 January 2025.
> The labelled CSV contains no injected anomaly, while the dashboard
> displays drift anomalies.  
> **Important:** This document is a plan and audit framework. Do not
> apply thresholds or architectural changes blindly. Reproduce, measure,
> test, and review every change.

------------------------------------------------------------------------

## 1. Executive Summary

SkyGuardAI currently combines telemetry validation, feature engineering,
Isolation Forest scoring, deterministic rules, fusion logic, fault
classification, explanations, network context, suggested readings, and
sensor health.

The observed symptoms suggest that errors may be propagating across
multiple layers:

- Normal sunrise warming and sunset cooling are interpreted as drift.
- A rule-driven decision may be displayed as if it were model-confirmed.
- Model warming-up/unavailable states may be confused with high
  confidence.
- Station context is shown as `UNKNOWN`.
- Network corroboration is shown as insufficient without enough
  diagnostic detail.
- Suggested readings may be simple baseline estimates rather than
  validated reconstructions.
- Health history may be frontend-derived rather than authoritative
  backend history.
- A known fault label may be assigned to an unknown statistical anomaly.
- The production model, rule engine, explanation layer, and frontend may
  not be using one persisted decision record.

The required engineering principle is:

``` text
raw telemetry
→ validation
→ causal features and baselines
→ model evidence
→ rule evidence
→ fusion
→ fault hypothesis
→ context
→ network evidence
→ health transition
→ explanation
→ persistence
→ UI
```

Every stage must be reproducible and auditable.

------------------------------------------------------------------------

## 2. Scope and Non-Negotiable Rules

### 2.1 Required guarantees

- Preserve raw telemetry permanently.
- Never overwrite raw readings with suggested or corrected values.
- Store model, feature, rule, and configuration versions.
- Store the exact evidence used for every decision.
- Prevent future-data leakage.
- Distinguish anomaly detection from fault diagnosis.
- Distinguish model evidence from rule evidence.
- Distinguish estimates from measured values.
- Distinguish anomaly risk from sensor health.
- Keep the current production model frozen during future experiments.
- Provide rollback and shadow-mode support.
- Make `UNKNOWN`, `INSUFFICIENT_EVIDENCE`, and `MODEL_UNAVAILABLE` valid
  states.

### 2.2 Definitions

| Concept           | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| Anomaly           | Observation or pattern that differs from expected behaviour      |
| Drift             | Persistent deviation from an appropriate reference process       |
| Fault type        | Hypothesis about the mechanism behind an anomaly                 |
| Model evidence    | Contribution from the ML model                                   |
| Rule evidence     | Contribution from deterministic checks                           |
| Context           | Temporal, environmental, and station conditions                  |
| Network evidence  | Agreement/disagreement from eligible peer stations               |
| Health            | Operational reliability of a sensor/channel                      |
| Suggested reading | An estimate, not automatically truth                             |
| Confidence        | Calibrated probability or explicitly qualified evidence strength |

------------------------------------------------------------------------

# 3. P0 — Reproduce the Ranchi False Positive

## 3.1 Regression fixture

Create a permanent regression fixture containing the clean Ranchi
observations:

``` text
station_id: AWS-RAN-067
location: Ranchi
primary_date: 2025-01-02
ground_truth: no injected anomaly
```

The fixture must include the night-time cooling, morning warming,
afternoon peak, evening cooling, humidity movement, and pressure
movement.

## 3.2 Required timestamps

At minimum reproduce:

``` text
2025-01-02 08:00
2025-01-02 09:00
2025-01-02 10:00
2025-01-02 11:00
2025-01-02 12:00
```

For each timestamp print and persist:

- Raw values.
- Previous values.
- Sampling interval.
- Actual ROC.
- Expected ROC.
- ROC residual.
- Rolling mean and scale.
- Normalised residual.
- Positive and negative CUSUM.
- Direction streak.
- Counter-direction count.
- Rule confidence.
- Raw model decision function.
- Transformed model score.
- Model status.
- Model contribution.
- Rule contribution.
- Fusion score.
- Threshold and bypass reason.
- Final anomaly decision.
- Fault type.
- Context status.
- Network status.
- Health transition.

The debug output must contain actual runtime numbers; illustrative
values must never be presented as production evidence.

## 3.3 Verify the deployed path

- [ ] Record the running backend commit hash.
- [ ] Record the frontend commit/build.
- [ ] Record model file and model version.
- [ ] Record configuration version.
- [ ] Verify environment variables.
- [ ] Verify API base URL.
- [ ] Verify container/image version and restart time.
- [ ] Verify the endpoint called by the dashboard.
- [ ] Compare direct API output with dashboard output.
- [ ] Confirm no stale process is running.
- [ ] Confirm no frontend-only anomaly labels are generated.
- [ ] Confirm alert and explanation screens use the same decision ID.

**Exit criterion:** the team can explain numerically why one clean
timestamp was labelled `DRIFT`.

------------------------------------------------------------------------

# 4. Drift Detection: Correct Definition

## 4.1 Drift is persistent reference disagreement

A useful conceptual definition is:

``` text
residual(t) = observed_sensor(t) - expected_reference(t)
```

The reference can combine:

- Station-specific historical behaviour.
- Same-hour and same-season behaviour.
- Recent environmental context.
- Physics-based relationships.
- Healthy neighbouring stations.
- A validated prediction model.

Drift should represent persistent bias or degradation relative to a
trustworthy reference.

## 4.2 What must not define drift

Do not define drift solely as:

``` text
temperature increased for N consecutive hours
```

Normal sunrise warming can satisfy this condition.

Do not define drift solely as:

``` text
reading outside a global range
```

Global range checks are useful for validity, not for long-term drift
diagnosis.

Do not define drift solely as:

``` text
high Isolation Forest score
```

A statistical anomaly is not automatically a drift fault.

## 4.3 Two monitoring timescales

### Fast monitoring

Use for:

- Spikes.
- Sudden drops.
- Step changes.
- Dropouts.
- Frozen values.
- Abrupt multivariate inconsistency.

Possible tools:

- Robust residual z-scores.
- Rate-of-change checks.
- Shewhart-style checks.
- Missingness and freshness rules.
- Cross-parameter consistency.

### Slow monitoring

Use for:

- Persistent positive or negative bias.
- Gradual calibration loss.
- Long-term disagreement with peers.
- Slow degradation.

Possible tools:

- EWMA residual monitoring.
- CUSUM residual monitoring.
- 7-day, 14-day, and 30-day summaries.
- Reference-station comparison.
- Calibration/reference comparisons.

The fast and slow monitors should have separate parameters and alert
semantics.

------------------------------------------------------------------------

# 5. Drift Calculation Audit

## 5.1 Unit consistency

All subtraction must use compatible units:

``` text
actual_roc (°C/hour)
-
expected_roc (°C/hour)
=
roc_residual (°C/hour)
```

Never subtract a dimensionless normalised ROC from a raw ROC in physical
units.

Correct sequence:

``` text
raw actual ROC
− raw expected ROC
= raw residual
↓
standardise residual using a validated scale
```

Test temperature, pressure, and humidity independently because their
units and dynamics differ.

## 5.2 Robust scaling

Ordinary standard deviation can be distorted by outliers. Evaluate a
robust scale such as:

``` text
MAD = median(abs(x − median(x)))
robust_scale = 1.4826 × MAD
```

Handle:

- Zero or near-zero scale.
- Too few baseline samples.
- Constant history.
- Contaminated history.
- Regime changes.
- New stations.
- Missing values.

When the scale is unreliable, emit:

``` text
INSUFFICIENT_BASELINE
```

instead of inventing a strong drift score.

## 5.3 Expected ROC provenance

Every expected ROC must include:

``` text
expected_roc_source
baseline_sample_count
baseline_start
baseline_end
baseline_quality
fallback_used
parameter
station_id
```

Possible reference hierarchy:

1.  Station + parameter + same-hour historical reference.
2.  Station + parameter + seasonal reference.
3.  Recent causal trend model.
4.  Healthy peer/reference estimate.
5.  Conservative fallback.
6.  Unknown/insufficient baseline.

## 5.4 CUSUM controls

Maintain separate positive and negative accumulators:

``` text
S+ = max(0, S+previous + scaled_residual − allowance)
S− = min(0, S−previous + scaled_residual + allowance)
```

Test:

- Parameter-specific allowance.
- Parameter-specific threshold.
- Minimum baseline length.
- Missing-data handling.
- Reset behaviour.
- Hysteresis.
- Cooldown.
- Regime transitions.
- Maximum accumulation age.
- Positive and negative drift.
- Interrupted drift.
- Drift with noise.

Direction streaks may be supporting evidence but must not be the
definition of drift.

------------------------------------------------------------------------

# 6. Baseline Architecture

Maintain distinct baselines:

| Baseline          | Purpose                          |
|-------------------|----------------------------------|
| Short rolling     | Recent local behaviour           |
| Same-hour         | Daily cycle                      |
| Seasonal          | Annual/seasonal behaviour        |
| Long-term station | Station-specific characteristics |
| Peer/reference    | Spatial comparison               |
| Physics           | Cross-parameter consistency      |
| Model residual    | Expected model error             |

### Contamination controls

Exclude or downweight readings that are:

- Confirmed anomalous.
- Imputed.
- Stale.
- Under recovery.
- From a degraded/offline sensor.
- Outside physical limits.
- Part of an unresolved persistent event.

Do not permanently exclude a station after one alert; use explicit state
and recovery policy.

### New station policy

``` text
model_status = WARMING_UP
baseline_status = INSUFFICIENT_HISTORY
drift_status = NOT_EVALUATED
```

A short history must not produce a strong drift claim.

------------------------------------------------------------------------

# 7. Drift Evaluation Plan

## 7.1 Clean-weather control benchmark

Include:

- Normal sunrise warming.
- Normal sunset cooling.
- Stable nights.
- Hot afternoons.
- Humid mornings.
- Pressure changes.
- Seasonal transitions.
- Regional weather events.
- Irregular sampling.
- Missing/stale readings.
- New-station warm-up.

Measure:

- False positives per station-day.
- False positives per 1,000 observations.
- False alarms by hour.
- False alarms at sunrise/sunset.
- Maximum clean CUSUM.
- Residual distribution.
- Average run length before a false alarm.

## 7.2 Drift injection design

The injector should simulate sensor bias:

``` text
observed_sensor =
    true_environment
    + slowly_varying_bias
    + measurement_noise
```

Test durations:

``` text
1 day
3 days
7 days
14 days
30 days
optional 60/90 days
```

Test magnitudes in both directions, for example:

``` text
±0.02 °C/day
±0.05 °C/day
±0.10 °C/day
±0.20 °C/day
```

These are test candidates, not production thresholds.

Test shapes:

- Linear.
- Piecewise-linear.
- Gradual exponential.
- Noisy.
- Interrupted.
- Partial recovery.
- Beginning at sunrise.
- Beginning at night.
- Single-parameter.
- Multivariate.
- With missing readings.

## 7.3 Required metrics

``` text
detection_rate
false_alarm_rate
time_to_detection
miss_rate
precision
recall
F1
mean_time_between_false_alarms
recovery_latency
unknown_class_rate
```

Thresholds must be selected using clean-weather false alarms and
injected-drift detection, not injected-drift recall alone.

------------------------------------------------------------------------

# 8. Model, Rules, and Fusion Audit

## 8.1 Model score semantics

Isolation Forest output is not automatically a probability.

If converted to a 0–100 scale, call it:

``` text
model_evidence_score
```

Do not display “85% confidence” unless the value has been calibrated and
validated.

## 8.2 Model status

Use explicit states:

``` text
AVAILABLE
WARMING_UP
INSUFFICIENT_HISTORY
MISSING_MODEL
FEATURE_ERROR
OUT_OF_DISTRIBUTION
DISABLED
```

The UI must never show strong model confidence when the model is
unavailable.

## 8.3 Persist the fusion calculation

Store:

``` text
model_status
model_score
rule_score
model_weight
rule_weight
model_contribution
rule_contribution
fusion_score
fusion_threshold
bypass_applied
bypass_reason
decision_basis
```

Valid decision bases may include:

``` text
MODEL_ONLY
RULE_ONLY
MODEL_AND_RULE_FUSION
MODEL_UNAVAILABLE_RULE_POLICY
INSUFFICIENT_EVIDENCE
```

If rule-only decisions are allowed:

- Label them explicitly.
- Explain why the rule is sufficient.
- Show model status.
- Store the policy version.
- Test the false-positive rate of bypasses.
- Never claim the model confirmed the decision.

------------------------------------------------------------------------

# 9. Unknown Anomaly Handling

Do not force every anomaly into a known fault label.

Support:

``` text
DRIFT
SPIKE
STEP_CHANGE
FROZEN_VALUE
FAIL_LOW
FAIL_HIGH
DROPOUT
MULTIVARIATE_INCONSISTENCY
CALIBRATION_BIAS
UNKNOWN_STATISTICAL_ANOMALY
ENVIRONMENTAL_TRANSITION
INSUFFICIENT_EVIDENCE
```

The fault helper should answer:

``` text
Which known pattern is most consistent with this anomaly?
```

It must not silently become the primary anomaly detector.

When evidence is weak:

``` text
fault_type = UNKNOWN_STATISTICAL_ANOMALY
fault_confidence = LOW
```

------------------------------------------------------------------------

# 10. Suggested Reading / Replacement Audit

## 10.1 Correct terminology

| Term                  | Meaning                                                |
|-----------------------|--------------------------------------------------------|
| Prediction            | Expected value before the current observation          |
| Imputation            | Filling a missing value                                |
| Reconstruction        | Estimating a value from surrounding/reference evidence |
| Replacement candidate | Proposed value for downstream analysis                 |
| Corrected value       | Value accepted under a documented correction policy    |

A temporal mean is not automatically the true value.

## 10.2 Candidate estimates

Evaluate independently:

1.  Temporal rolling baseline.
2.  Same-hour/same-season baseline.
3.  Recent trend estimate.
4.  Physics-constrained estimate.
5.  Healthy-peer estimate.
6.  Multivariate regression estimate.
7.  Optional reconstruction model.

Each candidate needs:

``` text
value
method
reference_window
supporting_features
uncertainty
data_quality
```

## 10.3 Agreement and uncertainty

Calculate:

``` text
candidate_spread
candidate_median
weighted_candidate_value
uncertainty
```

If candidates disagree:

``` text
replacement_status = DO_NOT_AUTOMATICALLY_REPLACE
```

If candidates agree within a validated tolerance:

``` text
replacement_status = SUGGESTED_WITH_VALIDATION
```

Always:

- Preserve raw data.
- Show the estimate label.
- Validate physical range.
- Validate rate of change.
- Validate cross-parameter relationships.
- Validate peer consistency.
- Avoid future data in live mode.
- Store provenance.

------------------------------------------------------------------------

# 11. Station Context and Regime Layer

## 11.1 Implement context before model retraining

First implement context as a separate interpretation layer. Do not
immediately feed it into the current production model.

Candidate regime labels:

``` text
STABLE_NIGHT
MORNING_WARMUP
AFTERNOON_HEAT
EVENING_COOLDOWN
HIGH_VOLATILITY
RAPID_PRESSURE_CHANGE
HUMIDITY_TRANSITION
SEASONAL_TRANSITION
UNKNOWN
```

## 11.2 Features

Temperature:

- Current value.
- Rolling mean/std.
- Short and long slopes.
- 1h/3h/6h/24h range.
- Same-hour residual.
- Seasonal residual.

Pressure:

- Current value.
- Short-term slope.
- Multi-window change.
- Volatility.
- Baseline deviation.

Humidity:

- Current value.
- Slope.
- Volatility.
- Temperature–humidity relationship.
- Persistence.

Time:

- Hour and day/night.
- Day of year.
- Cyclical hour/year.
- Sampling regularity.
- Time since last valid reading.

## 11.3 Context reliability

Store:

``` text
regime
regime_reliability
history_length
window_used
missing_feature_count
out_of_distribution_flag
fallback_used
```

Use `UNKNOWN` when history is insufficient, inputs are missing, or the
current situation is outside the known regime space.

------------------------------------------------------------------------

# 12. Spatial Clusters and Network Corroboration

## 12.1 Separate layer

Use spatial clusters for interpretation first, without changing the
current production model.

The network layer should answer:

``` text
Do nearby healthy stations show a similar change?
```

## 12.2 Peer eligibility

A peer must:

- Belong to the configured cluster.
- Use the same parameter and compatible units.
- Be fresh.
- Have acceptable health.
- Be aligned in time.
- Not be the target station.
- Not be known unreliable.
- Have sufficient data quality.

## 12.3 Required network evidence

Store:

``` text
cluster_id
eligible_peer_count
fresh_peer_count
healthy_peer_count
supporting_peer_count
contradicting_peer_count
stale_peer_count
direction_agreement
magnitude_statistics
time_alignment
network_status
```

Supported statuses:

``` text
LOCALIZED_ANOMALY_INDICATION
REGIONAL_EVENT_INDICATION
INSUFFICIENT_CORROBORATION
CONFLICTING_NETWORK_EVIDENCE
NO_ELIGIBLE_PEERS
```

These are interpretations, not definitive diagnoses.

## 12.4 Plain-language examples

### Localized

``` text
Chennai temperature increases by 11°C.
Tambaram, Ambattur, and Sriperumbudur remain normal.
```

Explanation:

> The change appears localized to the target station. Sensor and
> installation investigation is recommended. The evidence does not prove
> a hardware fault.

### Regionally corroborated

``` text
Chennai temperature decreases rapidly.
Three nearby stations show a similar directional change.
Humidity and pressure also change consistently.
```

Explanation:

> The change is supported by multiple nearby stations and may represent
> a regional environmental transition. The system should not immediately
> blame the target sensor.

------------------------------------------------------------------------

# 13. Explainability and TrueSHAP

## 13.1 Three explanation layers

### Model explanation

- Feature name.
- Feature value.
- Contribution.
- Direction.
- Rank.
- Model version.
- Explanation method.

### Rule explanation

- Rule ID.
- Rule name.
- Observed value.
- Threshold.
- Trigger direction.
- Reason.
- Severity.

### Operational interpretation

- Likely interpretation.
- Supporting evidence.
- Contradicting evidence.
- Uncertainty.
- Recommended action.

## 13.2 SHAP requirements

- Explain the exact deployed model.
- Use the same preprocessing as inference.
- Use a documented background dataset.
- Store feature order and schema.
- Validate explanation stability.
- Validate perturbation direction.
- Report when SHAP is unavailable.
- Never claim SHAP proves physical causality.

## 13.3 Fallback labels

A magnitude ranking is not SHAP. Use:

``` text
SHAP_TREE_EXPLAINER
SHAP_APPROXIMATION
FEATURE_MAGNITUDE_FALLBACK
RULE_ONLY_EXPLANATION
NO_EXPLANATION_AVAILABLE
```

The frontend must visibly distinguish these methods.

------------------------------------------------------------------------

# 14. Sensor Health and Recovery

## 14.1 Separate states

Maintain:

``` text
anomaly_risk
sensor_health
data_freshness
recovery_state
```

A sensor may be healthy while an environmental event occurs, or degraded
while the current reading appears normal.

## 14.2 Suggested state machine

``` text
HEALTHY
→ SUSPECT
→ DEGRADED
→ OFFLINE
→ RECOVERY_PENDING
→ HEALTHY
```

Every transition must be persisted with a reason.

## 14.3 Recovery rules

- Mark Repaired must not erase history.
- Force Recovery must be auditable.
- Clean readings must be evaluated using a defined streak/window.
- Recovery should be per parameter when appropriate.
- Recurring anomalies must interrupt recovery.
- The UI must show why the current state exists.
- Live health charts must come from backend history, not demo-derived
  frontend values.

## 14.4 Health event schema

``` json
{
  "station_id": "AWS-RAN-067",
  "parameter": "temperature",
  "timestamp": "2025-01-02T10:00:00",
  "previous_state": "HEALTHY",
  "new_state": "SUSPECT",
  "reason_code": "REPEATED_RESIDUAL_DEVIATION",
  "anomaly_event_id": "evt_123",
  "health_score": 72,
  "recovery_session_id": null
}
```

------------------------------------------------------------------------

# 15. PostgreSQL + TimescaleDB Persistence

## 15.1 Time-series tables

Recommended hypertables:

``` text
raw_telemetry
validated_telemetry
derived_features
anomaly_decisions
decision_evidence
sensor_health_events
suggested_readings
network_observations
```

## 15.2 Relational tables

``` text
stations
station_parameters
spatial_clusters
cluster_members
station_neighbors
model_versions
feature_schema_versions
rule_versions
recovery_sessions
incident_events
```

## 15.3 Decision record

Each decision should include:

``` text
event_id
station_id
parameter
observed_at
ingested_at
model_version
feature_schema_version
rule_version
model_status
model_score
rule_score
fusion_score
decision_basis
fault_type
fault_confidence
context_status
network_status
health_state
freshness_status
suggested_reading_id
explanation_id
```

## 15.4 Replay requirement

Given a station, timestamp, model version, and configuration version,
the system must reconstruct:

- Inputs.
- Features.
- Baselines.
- Model output.
- Rule output.
- Fusion.
- Fault hypothesis.
- Context.
- Network evidence.
- Explanation.
- Health transition.

------------------------------------------------------------------------

# 16. Implementation Order

## Phase 0 — Freeze and reproduce

- Freeze current production model.
- Record commit/config/model/data versions.
- Reproduce Ranchi false positives.
- Verify the running path.
- Produce numeric traces.

**Exit:** one false-positive decision is fully explained.

## Phase 1 — Correctness and observability

- Verify units.
- Fix confirmed calculation defects.
- Store baseline provenance.
- Store model status.
- Store fusion breakdown.
- Store decision basis.
- Persist evidence.

**Exit:** every decision is auditable without relying on logs alone.

## Phase 2 — Health consistency

- Remove frontend-derived history from live mode.
- Persist health transitions.
- Implement per-parameter state.
- Test recovery and recurrence.

**Exit:** health summary, history, alerts, and recovery screens agree.

## Phase 3 — Drift validation

- Build clean control benchmark.
- Build 1/3/7/14/30-day bias injector.
- Evaluate positive/negative/noisy/interrupted drift.
- Compare EWMA and CUSUM.
- Select thresholds using false-alarm and detection evidence.

**Exit:** validated multi-timescale drift performance.

## Phase 4 — Suggested readings

- Separate prediction, imputation, reconstruction, and correction.
- Add candidate estimates.
- Add physics and peer checks.
- Add uncertainty and candidate agreement.
- Preserve raw values.

**Exit:** every suggestion has provenance and limitations.

## Phase 5 — Context and spatial interpretation

- Add regime metadata.
- Add context reliability.
- Resolve clusters.
- Implement peer eligibility.
- Count supporting/contradicting peers.
- Add localized/regional/insufficient states.
- Add plain-language summaries.

**Exit:** the system can explain environmental versus localized
evidence.

## Phase 6 — Explanation validation

- Validate SHAP or selected method.
- Distinguish fallbacks.
- Add perturbation tests.
- Store explanation provenance.
- Ensure UI language matches evidence.

**Exit:** explanation matches the actual decision path.

------------------------------------------------------------------------

# 17. Future Season + Spatial Model Retraining

> **This must be completed separately and must not disturb the current
> production model.**

## 17.1 Model tracks

``` text
PRODUCTION_MODEL_CURRENT
EXPERIMENTAL_TEMPORAL_CONTEXT_MODEL
EXPERIMENTAL_SPATIAL_CONTEXT_MODEL
EXPERIMENTAL_COMBINED_MODEL
```

Never overwrite the production model file.

## 17.2 Future features

Temporal/seasonal:

- Hour/day/year encodings.
- Regime label and reliability.
- Rolling slopes and volatility.
- Same-hour and seasonal residuals.
- Recent environmental context.

Spatial:

- Healthy peer median.
- Peer residual.
- Peer range.
- Peer direction agreement.
- Cluster trend.
- Distance-weighted statistics.
- Peer freshness and health.

Quality:

- Missingness.
- Sampling irregularity.
- Freshness.
- Station age.
- Recovery state.
- Baseline reliability.
- Regime reliability.

## 17.3 Isolated training

Use separate experiment directories and registry entries:

``` text
models/
  production/
    current/
  experiments/
    temporal_context_001/
    spatial_context_001/
    temporal_spatial_001/
```

Record:

``` text
dataset_version
feature_schema_version
training_window
chronological_split
station_split
random_seed
hyperparameters
metrics
limitations
```

## 17.4 Train/serve consistency

Freeze and version:

- Feature names and order.
- Units.
- Missing-value policy.
- Encoding/scaling.
- Context windows.
- Peer eligibility.
- Unknown-context handling.

Use the same feature builder in training, evaluation, shadow mode, and
inference.

## 17.5 Evaluation tracks

Compare:

``` text
A. Current production model
B. Temporal-context experiment
C. Spatial-context experiment
D. Combined temporal + spatial experiment
```

Measure:

- Clean-weather false positives.
- Sunrise/sunset false positives.
- Drift detection.
- Spike detection.
- Frozen-value detection.
- Multivariate inconsistency.
- Unknown anomaly performance.
- New-station performance.
- Missing-peer behaviour.
- Explanation quality.
- Runtime and memory.
- Station-level performance.

## 17.6 Shadow mode

``` text
Production model → controls alerts
Experimental model → runs without controlling alerts
```

Store:

``` text
production_output
experimental_output
difference
```

Promotion requires:

- No unacceptable regression.
- Improved target metrics.
- All existing regression tests pass.
- Safe unknown/missing-context handling.
- Validated explanations.
- Documented limitations.
- Tested rollback.
- Explicit approval.

------------------------------------------------------------------------

# 18. Comprehensive Test Matrix

## 18.1 Data quality

- [ ] Missing temperature.
- [ ] Missing pressure.
- [ ] Missing humidity.
- [ ] Stale timestamp.
- [ ] Future timestamp.
- [ ] Duplicate timestamp.
- [ ] Out-of-order timestamp.
- [ ] Irregular interval.
- [ ] Invalid units.
- [ ] Impossible physical value.
- [ ] Constant value.
- [ ] Empty history.
- [ ] Very short history.
- [ ] Station/parameter mismatch.

## 18.2 Environmental behaviour

- [ ] Sunrise warming.
- [ ] Sunset cooling.
- [ ] Stable night.
- [ ] Hot afternoon.
- [ ] Humid morning.
- [ ] Pressure transition.
- [ ] Seasonal transition.
- [ ] Regional event.
- [ ] Localized event.
- [ ] Natural high-volatility period.

## 18.3 Fault injection

- [ ] Positive drift.
- [ ] Negative drift.
- [ ] 1-day drift.
- [ ] 7-day drift.
- [ ] 30-day drift.
- [ ] Noisy drift.
- [ ] Interrupted drift.
- [ ] Spike.
- [ ] Step change.
- [ ] Frozen value.
- [ ] Fail-low.
- [ ] Fail-high.
- [ ] Dropout.
- [ ] Multivariate inconsistency.
- [ ] Unknown anomaly.

## 18.4 Model and fusion

- [ ] Model available.
- [ ] Model warming up.
- [ ] Model unavailable.
- [ ] Feature error.
- [ ] Rule-only decision.
- [ ] Model-only decision.
- [ ] Model/rule agreement.
- [ ] Model/rule disagreement.
- [ ] Fusion boundary.
- [ ] Bypass boundary.
- [ ] Unknown helper output.
- [ ] SHAP fallback.
- [ ] Model version mismatch.

## 18.5 Network

- [ ] No peers.
- [ ] One peer.
- [ ] Supporting peers.
- [ ] Contradicting peers.
- [ ] Stale peers.
- [ ] Unhealthy peers.
- [ ] Mixed peer quality.
- [ ] Timestamp misalignment.
- [ ] Missing cluster.
- [ ] Localized event.
- [ ] Regional event.
- [ ] Conflicting evidence.

## 18.6 Health

- [ ] Healthy → suspect.
- [ ] Suspect → degraded.
- [ ] Degraded → offline.
- [ ] Offline → recovery.
- [ ] Recovery → healthy.
- [ ] Recovery interrupted.
- [ ] Force recovery.
- [ ] Mark repaired.
- [ ] Per-parameter recovery.
- [ ] Simultaneous faults.
- [ ] Backend/frontend history agreement.

------------------------------------------------------------------------

# 19. Acceptance Criteria

The implementation is complete only when:

1.  Ranchi 067's clean daily cycle does not create unjustified drift
    alerts.
2.  Every alert can be reconstructed numerically.
3.  ROC units are compatible.
4.  Expected ROC provenance is stored.
5.  Baseline quality is visible.
6.  Model status is truthful.
7.  Rule-only decisions are explicit.
8.  Model and rule evidence are separate.
9.  Unknown anomalies are supported.
10. Suggested readings have provenance and uncertainty.
11. Raw observations remain unchanged.
12. Network evidence uses fresh, eligible, healthy peers.
13. Localized and regional interpretations are distinguishable.
14. Health history is backend-authoritative.
15. Recovery transitions are persisted.
16. Explanation method is explicit.
17. SHAP and fallback explanations are distinguished.
18. Clean-weather false alarms are measured.
19. Drift is tested at multiple timescales.
20. Future context/spatial training is isolated.
21. Experimental models support shadow mode.
22. Rollback is tested.
23. Documentation matches deployed code.
24. Regression tests run before release.

------------------------------------------------------------------------

# 20. Micro Checklist — Final Nothing-Left-Behind Gate

## Reproduction

- [ ] Commit, model, configuration, and dataset versions recorded.
- [ ] Ranchi clean fixture committed.
- [ ] Five false-positive timestamps reproduced.
- [ ] Runtime path verified.
- [ ] Direct API and dashboard agree.
- [ ] Decision ID is shared across alert and explanation views.

## Mathematics

- [ ] Actual and expected ROC use matching units.
- [ ] Residual trace is tested.
- [ ] Robust scale is tested.
- [ ] Zero-scale handling exists.
- [ ] Baseline count and quality are stored.
- [ ] Contaminated readings are controlled.
- [ ] No future leakage exists.
- [ ] Positive and negative drift are tested.
- [ ] Direction streak is not the sole drift condition.
- [ ] CUSUM reset/hysteresis is tested.
- [ ] EWMA is evaluated.
- [ ] 1/3/7/14/30-day drift is tested.

## Model and fusion

- [ ] Model status is explicit.
- [ ] Score semantics are documented.
- [ ] No uncalibrated score is called probability.
- [ ] Model/rule scores are separate.
- [ ] Fusion formula is persisted.
- [ ] Threshold and bypass reason are persisted.
- [ ] Rule-only decisions are labelled.
- [ ] Unknown fault output exists.
- [ ] Model and feature versions are stored.

## Suggested readings

- [ ] Raw observation is preserved.
- [ ] Estimate is clearly labelled.
- [ ] Temporal candidate exists.
- [ ] Peer candidate exists when eligible.
- [ ] Physics checks exist.
- [ ] Candidate spread is calculated.
- [ ] Uncertainty is reported.
- [ ] Disagreement blocks automatic replacement.
- [ ] Live mode avoids future data.
- [ ] Provenance is persisted.

## Context and network

- [ ] Context is computed or explicitly unknown.
- [ ] Context reliability is stored.
- [ ] Cluster is resolved.
- [ ] Peer freshness and health are checked.
- [ ] Supporting/contradicting peers are counted.
- [ ] Time alignment is checked.
- [ ] Localized state exists.
- [ ] Regional state exists.
- [ ] Insufficient/conflicting states exist.
- [ ] Plain-language interpretation exists.

## Health

- [ ] Backend is authoritative.
- [ ] Per-parameter health exists.
- [ ] Transitions are persisted.
- [ ] Recovery sessions are persisted.
- [ ] Repair actions are auditable.
- [ ] Clean streak policy is tested.
- [ ] Recurring anomalies interrupt recovery.
- [ ] Live chart is not demo-derived.
- [ ] Summary and history agree.

## Explainability

- [ ] Model and rule explanations are separate.
- [ ] Explanation method is stored.
- [ ] SHAP is distinguished from fallback.
- [ ] Preprocessing matches inference.
- [ ] Feature order is versioned.
- [ ] Explanation stability is tested.
- [ ] Plain-language summary is generated.
- [ ] Uncertainty is shown.
- [ ] Causality is not claimed from SHAP.
- [ ] Recommended action is evidence-based.

## Future retraining

- [ ] Production model is frozen.
- [ ] Temporal experiment is separate.
- [ ] Spatial experiment is separate.
- [ ] Combined experiment is separate.
- [ ] Feature schema is versioned.
- [ ] Chronological and station leakage checks exist.
- [ ] Missing-peer behaviour is tested.
- [ ] Shadow mode exists.
- [ ] Promotion criteria exist.
- [ ] Rollback is tested.

## Release

- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Regression suite passes.
- [ ] Clean false-alarm rate is acceptable.
- [ ] Drift metrics are recorded.
- [ ] Latency is measured.
- [ ] Database migrations are tested.
- [ ] API contracts are validated.
- [ ] Dashboard labels match backend truth.
- [ ] Documentation is updated.
- [ ] Rollback plan is documented.
- [ ] Final sign-off is recorded.

------------------------------------------------------------------------

# Final Engineering Position

Do not begin by blindly increasing thresholds or extending the rolling
window.

The correct order is:

``` text
reproduce
→ trace numbers
→ verify deployed path
→ fix correctness and state inconsistencies
→ measure clean-weather false alarms
→ implement residual-based multi-timescale drift monitoring
→ validate suggested readings
→ add network corroboration
→ fix health persistence
→ validate explanations
→ train temporal/spatial context models separately in shadow mode
→ promote only after regression testing
```

Seasonal and spatial retraining may be valuable, but it is a later,
isolated experiment. The current production model must remain stable
until the experimental model demonstrates measurable improvement without
reintroducing false positives or reducing explainability.
