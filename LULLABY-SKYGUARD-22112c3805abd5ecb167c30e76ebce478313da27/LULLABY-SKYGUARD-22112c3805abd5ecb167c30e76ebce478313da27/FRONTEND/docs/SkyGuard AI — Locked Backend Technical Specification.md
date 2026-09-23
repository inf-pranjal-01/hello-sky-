# SkyGuard AI — LOCKED BACKEND TECHNICAL SPECIFICATION

## 0. PURPOSE

This document is the authoritative backend integration and implementation specification for SkyGuard AI.

The backend is an AI/ML-based anomaly detection system for a network of 20 Automatic Weather Stations (AWS).

The backend must:

- ingest real weather data from Open-Meteo;
- support historical replay mode;
- support live mode;
- process all 20 stations as synchronized simulation snapshots;
- perform per-station anomaly detection;
- perform same-cluster spatial consistency analysis;
- maintain real-time station state in memory;
- expose the existing FastAPI API contract;
- provide anomaly explanations and likely faulty sensor identification;
- support sensor repair/recovery;
- support maintenance-ticket generation;
- preserve the existing frontend API contract.

This specification is LOCKED.

Do not redesign the architecture, rename existing API fields, introduce a database, replace Isolation Forest, or invent new API endpoints unless explicitly requested.

---

# 1. TECHNOLOGY STACK

Backend technologies:

- Python
- FastAPI
- Uvicorn
- pandas
- NumPy
- scikit-learn
- IsolationForest
- SHAP / TreeExplainer
- requests
- Open-Meteo Historical Weather API
- Open-Meteo current weather API
- CSV files for historical/replay data
- in-memory state for runtime state

Database:

- NONE for v1.
- Do not introduce PostgreSQL, SQLite, MongoDB, Redis, or another database.

Model:

- scikit-learn `IsolationForest`
- trained offline
- loaded once by the backend
- inference performed for individual station readings

Explainability:

- SHAP TreeExplainer
- deterministic magnitude-based fallback when SHAP is unavailable or fails

---

# 2. PROJECT MODULES

The backend consists of:

```text
main.py
config.py
state.py
simulator.py

data_fetch.py
anomaly_injector.py

features.py
train.py
detect.py
explain.py

model_artifacts/
    isolation_forest.pkl
```

Responsibilities:

### `config.py`

Contains:

- station configuration;
- cluster information where applicable;
- normal-range constants;
- severity thresholds;
- health configuration;
- runtime constants.

### `data_fetch.py`

Responsible for historical Open-Meteo data collection.

### `anomaly_injector.py`

Responsible for synthetic fault injection into historical clean data.

### `features.py`

Responsible for offline feature engineering and feature matrix construction.

### `train.py`

Responsible for training and saving the Isolation Forest model.

Training must use clean data only.

### `detect.py`

Responsible for:

- per-reading feature construction;
- Isolation Forest inference;
- deterministic rule checks;
- anomaly score calculation;
- severity;
- spatial feature calculation;
- explanation integration.

### `explain.py`

Responsible for:

- SHAP explanation;
- fallback feature attribution;
- likely faulty raw sensor identification.

### `state.py`

Responsible for runtime state:

- station buffers;
- current station status;
- recent anomaly information;
- neighbor buffers;
- recovery state;
- health state;
- anomaly records.

### `simulator.py`

Responsible for:

- live data acquisition;
- replay data acquisition;
- synchronized 20-station snapshots;
- feeding readings into the state/detection pipeline.

### `main.py`

Responsible for:

- FastAPI application;
- API routes;
- request validation;
- returning runtime state.

---

# 3. STATION NETWORK

There are exactly 20 stations.

There are exactly 5 clusters.

Each cluster contains:

- 1 center station;
- 3 nearby neighbor stations.

Therefore:

- 5 center stations;
- 15 neighbor stations;
- 20 total stations.

The clusters are:

- CHN
- DEL
- MUM
- KOL
- BHO

Spatial comparisons MUST occur only between stations belonging to the same cluster.

Do NOT compare:

- Chennai against Delhi;
- Mumbai against Kolkata;
- Bhopal against Chennai;
- or any other cross-cluster pair.

The station metadata is defined by the existing `data_fetch.py` / configuration and must be reused rather than recreated with guessed IDs.

Known station ID examples include:

- `AWS-CHN-024`
- `AWS-CHN-101`
- `AWS-CHN-102`
- `AWS-CHN-103`
- `AWS-DEL-011`
- `AWS-DEL-101`
- `AWS-DEL-102`
- `AWS-DEL-103`
- `AWS-MUM-007`
- `AWS-MUM-101`
- `AWS-MUM-102`
- `AWS-MUM-103`

Use the actual configuration as the source of truth for all remaining station metadata.

---

# 4. RAW SENSOR VARIABLES

The application's canonical internal variable names are LOCKED.

Use:

```text
temperature_c
pressure_hpa
humidity_pct
```

Timestamp:

```text
timestamp
```

Station metadata:

```text
station_id
station_name
cluster_id
role
```

The application MUST NOT replace `pressure_hpa` with `surface_pressure`.

Open-Meteo uses:

```text
surface_pressure
```

internally in its API response.

The API boundary must map:

```text
surface_pressure → pressure_hpa
```

After that mapping, all backend application logic uses:

```text
pressure_hpa
```

This applies to:

- features;
- detection;
- state;
- replay;
- API responses;
- anomaly records.

---

# 5. HISTORICAL DATA

Historical data is retrieved from Open-Meteo.

The historical dataset contains:

```text
timestamp
temperature_c
pressure_hpa
humidity_pct
station_id
station_name
cluster_id
role
```

Historical data is clean before synthetic anomaly injection.

The backend must preserve station and cluster metadata.

The `role` field is application metadata.

It is NOT an Open-Meteo API parameter.

---

# 6. SYNTHETIC ANOMALY INJECTION

Synthetic faults are used for controlled evaluation and replay.

Supported fault types include:

```text
spike
frozen_value
drift
dropout
sensor_fail_low
multivariate_inconsistency
```

The injector modifies clean historical rows and adds ground-truth labels.

The anomaly injection system must remain separate from model training.

Training data MUST NOT contain:

```text
is_anomaly
fault_type
```

or other synthetic ground-truth labels.

Synthetic evaluation results must be described as performance against injected faults, not as proof of real-world production accuracy.

---

# 7. FEATURE ENGINEERING

The current model uses 22 features.

Feature groups include:

### Raw features

```text
temperature_c
pressure_hpa
humidity_pct
```

### Temporal features

- rolling/deviation features;
- rate-of-change features;
- temporal behavior;
- time-of-day information;
- seasonal information.

### Cross-parameter features

Features capturing inconsistency between:

- temperature;
- pressure;
- humidity.

### Spatial features

Three spatial deviation features are used.

The required spatial mechanism is:

> station deviation minus the median deviation of other stations in the same cluster at the same timestamp.

The median is intentionally used because it is robust against one abnormal neighbor.

The advanced station-own rolling spatial baseline / volatility / z-score architecture is NOT required for the current implementation.

Do not introduce it unless explicitly requested.

---

# 8. SYNCHRONIZED 20-STATION PROCESSING — REQUIRED

This is a critical backend requirement.

Each simulator timestep represents one synchronized station snapshot.

The backend MUST gather the readings for all 20 stations for the same simulation timestamp before performing the spatial comparison for that timestep.

The implementation must conceptually operate as:

1. obtain the current reading for every station;
2. construct one 20-station snapshot;
3. associate all readings with the same simulation timestamp;
4. make the complete snapshot available to the detection pipeline;
5. calculate same-cluster spatial context from that snapshot;
6. perform anomaly detection for each station;
7. update runtime state only after the snapshot has been evaluated.

Do NOT rely on the processing order of stations to construct the spatial median.

Do NOT allow:

```text
station A updated → station B sees A's new state
```

while another station still represents the previous timestep.

Spatial comparison must use the synchronized snapshot for that timestep.

---

# 9. SPATIAL MEDIAN

Spatial comparison is limited to the station's own cluster.

For each station:

1. obtain its current deviation;
2. obtain the corresponding deviations of its same-cluster neighbors;
3. calculate the median of those neighbor deviations;
4. calculate the station's spatial deviation relative to that median.

For a cluster containing four stations:

- current station;
- three neighboring stations.

The current station must not be included in its own neighbor median.

Example calculation:

```text
spatial_deviation =
station_deviation - median(neighbor_deviations)
```

This spatial context must correspond to the same synchronized simulation timestamp.

The spatial mechanism is not a cross-city comparison.

---

# 10. ISOLATION FOREST

The anomaly model is:

```python
IsolationForest
```

The model is trained offline.

Current training configuration includes:

```text
n_estimators = 100
contamination = 0.02
```

The trained model is saved as:

```text
model_artifacts/isolation_forest.pkl
```

The model is loaded once at backend startup/runtime initialization.

Do not retrain the model during normal live inference.

Do not train using synthetic anomaly labels.

Do not replace Isolation Forest with:

- LSTM;
- Transformer;
- autoencoder;
- neural network;
- XGBoost;
- federated learning;
- reinforcement learning.

unless explicitly requested.

---

# 11. ANOMALY SCORING

The current detection thresholds are:

```text
MODEL_ONLY_THRESHOLD = 90
HARD_FLOOR = 85
SOFT_FLOOR = 75
IS_ANOMALY_THRESHOLD = 75
```

These values are part of the current backend behavior.

Do not silently replace them.

The detector combines:

- Isolation Forest model score;
- deterministic physical/rule checks;
- temporal checks;
- cross-parameter checks;
- spatial context.

The final anomaly score is represented as a percentage-style score.

---

# 12. DETERMINISTIC RULES

Deterministic rules complement the Isolation Forest.

They are responsible for faults that should be caught directly rather than relying exclusively on statistical modeling.

Relevant checks include:

- physical range violations;
- frozen sensor behavior;
- drift;
- dropout;
- sensor fail-low conditions;
- other deterministic consistency checks implemented by `detect.py`.

Frozen-value persistence currently requires:

```text
3 consecutive readings
```

Do not revert this to 2.

The anomaly injector's frozen-value minimum length supports this requirement.

---

# 13. SEVERITY

Current severity mapping is:

```text
score >= 90 → critical
score >= 70 → high
score >= 55 → medium
otherwise   → low
```

These thresholds are backend behavior and must remain synchronized across the detection and API layers.

---

# 14. STATION HEALTH

Runtime station health is maintained in memory.

Current configuration includes:

```text
health window size = 10 readings
offline threshold = 5 anomalous readings within the last 10
recovery requirement = 5 consecutive clean readings
```

Station status is represented internally using the existing state model.

The API maps the internal healthy state to the API-compatible status expected by the frontend.

Do not introduce a complex multi-state health machine unless explicitly requested.

---

# 15. CAUSAL HISTORY

Station history must remain causal.

A reading identified as faulty must not contaminate the clean historical baseline used for subsequent detection.

When a station is offline/faulty:

- anomalous readings are excluded from clean-history baseline construction;
- recovery readings are evaluated separately;
- healthy readings can re-enter the clean history after recovery.

This prevents anomaly contamination of future feature calculations.

---

# 16. `state.py`

`StateManager` is the runtime source of truth.

It maintains:

- station buffers;
- current readings;
- station status;
- health;
- anomaly records;
- recovery state;
- neighbor relationships;
- explanation information.

`StationBuffer` stores recent raw readings.

The state manager must provide same-cluster neighbor context to the detector.

The detector must receive the appropriate synchronized spatial context for the current snapshot.

The existing `ExplainerCache` must be reused rather than creating a new SHAP explainer for every reading.

---

# 17. SHAP EXPLAINABILITY

When an anomaly is detected, the backend can generate an explanation.

`ExplainerCache` loads/builds the TreeExplainer once.

The explanation must identify important model features.

The backend also maps important features back to raw sensor parameters.

Possible likely faulty sensors:

```text
temperature
pressure
humidity
```

The explanation response uses:

```text
anomaly_id
features
likely_faulty_sensors
```

If SHAP fails or is unavailable, the deterministic magnitude-based fallback must remain functional.

An anomaly must not disappear merely because SHAP is unavailable.

---

# 18. LIVE MODE

Live mode uses Open-Meteo current weather data.

Open-Meteo current variables:

```text
temperature_2m
surface_pressure
relative_humidity_2m
```

The API response must be mapped internally to:

```text
temperature_c
pressure_hpa
humidity_pct
```

The live fetch interval is approximately:

```text
60 seconds
```

The simulator/UI tick is:

```text
2 seconds
```

The backend must NOT make unnecessary Open-Meteo requests every 2 seconds.

The 2-second simulation/update loop and the actual external weather fetch interval are separate concepts.

Live mode must preserve the synchronized 20-station snapshot requirement.

A live simulation timestep should use the latest available reading for each station and evaluate the complete 20-station snapshot consistently.

---

# 19. REPLAY MODE

Replay mode uses the labeled historical station files generated by the anomaly injection pipeline.

Replay must:

- iterate through the historical data;
- maintain the station timestamps;
- preserve fault labels for evaluation/debugging;
- process all 20 stations as synchronized timesteps;
- run the same detection/state pipeline used by live mode;
- not bypass anomaly detection;
- not directly mark injected faults as detected anomalies.

Ground truth labels are for evaluation.

The detector must independently determine whether the reading is anomalous.

Replay mode must reset runtime state appropriately so previous live state does not contaminate replay.

---

# 20. LIVE MODE VS REPLAY MODE

Both modes must ultimately feed the same runtime detection/state pipeline.

The difference is the source of the reading.

### Live mode

Source:

```text
Open-Meteo current API
```

### Replay mode

Source:

```text
historical labeled CSV files
```

The downstream processing must remain consistent:

```text
reading snapshot
→ feature construction
→ spatial context
→ Isolation Forest
→ deterministic rules
→ score
→ explanation if required
→ state update
```

Do not create separate incompatible detection implementations for replay and live.

---

# 21. SIMULATOR TIMING

Current simulator constants include:

```text
TICK_SECONDS = 2
LIVE_FETCH_INTERVAL_SECONDS = 60
```

The 2-second tick is the application simulation/update cadence.

The 60-second interval controls external live-weather fetching.

Do not interpret the 2-second tick as requiring an Open-Meteo request every 2 seconds.

---

# 22. API CONTRACT

The existing FastAPI API contract is LOCKED.

Do not rename endpoints.

Do not rename response fields.

Do not invent additional frontend-facing endpoints.

Current endpoints:

```text
GET  /api/stations
GET  /api/current-reading
GET  /api/trends
GET  /api/anomalies/latest
GET  /api/anomalies/recent
GET  /api/explain/{anomaly_id}
GET  /api/sensor-health
POST /api/repair-sensor
POST /api/inject-anomaly
POST /api/maintenance-ticket
```

---

# 23. `GET /api/stations`

Returns station metadata and current status.

Expected fields:

```json
{
  "station_id": "...",
  "name": "...",
  "lat": 0.0,
  "lon": 0.0,
  "status": "NORMAL"
}
```

The backend's internal healthy status may be mapped to:

```text
NORMAL
```

for API compatibility.

---

# 24. `GET /api/current-reading`

Returns the current reading for a station.

Required structure includes:

```text
timestamp

temperature_c
    value
    normal_min
    normal_max

pressure_hpa
    value
    normal_min
    normal_max

humidity_pct
    value
    normal_min
    normal_max

anomaly_score_pct
risk_level

sensor_health_pct
sensor_health_status
```

`pressure_hpa` is mandatory.

Do not return `surface_pressure` as the application's pressure field.

---

# 25. `GET /api/trends`

Returns recent trend data.

Each point contains:

```text
timestamp
temperature_c
pressure_hpa
humidity_pct
```

The response may also contain:

```text
anomaly_windows
```

The backend currently uses the station's runtime buffer.

Do not invent a new trend storage system.

---

# 26. `GET /api/anomalies/latest`

Returns the latest detected anomaly.

Required fields include:

```text
anomaly_id
timestamp
station_id
anomaly_score_pct
severity
type
root_cause
description
suggested_values
```

If no anomaly exists, the current API behavior may return HTTP 404.

---

# 27. `GET /api/anomalies/recent`

Query parameters may include:

```text
station_id
limit
```

Each anomaly record contains:

```text
anomaly_id
type
label
station_id
timestamp
score_pct
severity
suggested_values
```

Do not rename:

```text
score_pct
```

to:

```text
anomaly_score_pct
```

inside this endpoint unless the existing implementation is explicitly changed as part of a contract migration.

---

# 28. `GET /api/explain/{anomaly_id}`

Returns explanation data.

Required fields:

```text
anomaly_id
features
likely_faulty_sensors
```

Feature entries contain:

```text
name
impact
```

Positive and negative impacts are preserved.

The endpoint must use the explanation associated with the actual anomaly record.

Do not fabricate explanation data.

---

# 29. `GET /api/sensor-health`

Query:

```text
station_id
```

Response:

```json
{
  "station_id": "...",
  "health_pct": 98,
  "status": "HEALTHY"
}
```

Health must originate from runtime state.

Do not calculate a fake health percentage independently in the API layer.

---

# 30. `POST /api/repair-sensor`

Request:

```json
{
  "station_id": "..."
}
```

The endpoint must call the existing repair/recovery mechanism.

The repair operation:

- marks the station repaired;
- activates recovery;
- allows subsequent clean readings to restore health;
- does not instantly fake a fully healthy sensor without recovery logic.

The current response includes:

```text
success
station_id
status
recovery_active
message
```

---

# 31. `POST /api/inject-anomaly`

This endpoint exists for frontend/API compatibility.

Current runtime behavior is tied to replay mode.

Request:

```json
{
  "station_id": "...",
  "type": "spike"
}
```

The existing implementation may trigger replay rather than directly modifying a single live Open-Meteo reading.

Do not assume this endpoint directly mutates a live station unless that behavior is explicitly implemented.

If replay is already running, the existing conflict behavior must be preserved.

---

# 32. `POST /api/maintenance-ticket`

Request:

```json
{
  "anomaly_id": "..."
}
```

Response includes:

```text
ticket_id
station_id
issue
priority
created_at
```

The ticket must be generated from the referenced anomaly.

Do not fabricate a station unrelated to the anomaly ID.

---

# 33. API SOURCE OF TRUTH

API routes must read from runtime state and backend services.

Do not make the frontend responsible for:

- anomaly scoring;
- severity calculation;
- health calculation;
- spatial median;
- sensor fault classification;
- SHAP calculation;
- suggested values;
- station status inference.

These are backend responsibilities.

---

# 34. RUNTIME DATA FLOW

The complete runtime processing sequence is:

1. Select operating mode.
2. Obtain the current reading for every station.
3. Build the synchronized 20-station snapshot.
4. Ensure all stations correspond to the same simulation timestep.
5. Build current feature inputs.
6. Build same-cluster spatial context.
7. Score each station independently using Isolation Forest.
8. Run deterministic rules.
9. Combine model and rule results.
10. Calculate anomaly score and severity.
11. Generate explanation information for detected anomalies.
12. Update health/state.
13. Commit the completed timestep to `StateManager`.
14. Make the updated state available through FastAPI.

The API must never become the processing engine.

---

# 35. IMPORTANT SNAPSHOT SEMANTICS

A snapshot is the authoritative set of station readings for one simulation timestep.

All 20 stations must be evaluated against the same timestep.

Spatial features must never depend on accidental station processing order.

The implementation must avoid mixing:

```text
current timestep readings
```

with:

```text
previous timestep readings
```

when constructing same-timestamp spatial context.

If a live station has not received a new external weather value because Open-Meteo is fetched less frequently than the 2-second simulator tick, the most recently available reading may be reused according to the simulator's live-data policy.

The external fetch cadence and simulator cadence must remain separate.

---

# 36. STATE COMMIT SEMANTICS

Detection should be performed using the snapshot/context for the timestep before committing the new timestep into persistent station history.

This prevents one station's newly committed reading from unintentionally changing the spatial baseline used by another station in the same timestep.

After the snapshot has been evaluated:

- commit valid clean history;
- record anomalies;
- update station health;
- update current readings;
- update recovery state.

---

# 37. NO DATABASE

Do not introduce persistent database infrastructure.

Use:

```text
CSV
```

for:

- historical data;
- replay data;
- generated evaluation datasets.

Use:

```text
in-memory StateManager
```

for:

- current readings;
- rolling history;
- station health;
- anomaly state;
- runtime recovery state;
- recent anomalies.

---

# 38. TRAINING/INFERENCE SEPARATION

Training and inference are separate.

Training:

```text
clean historical CSV
→ feature matrix
→ remove warm-up/incomplete rows
→ Isolation Forest
→ model artifact
```

Inference:

```text
current/replay snapshot
→ current features
→ Isolation Forest
→ rules
→ anomaly result
```

Never retrain on the current live stream.

Never train using synthetic ground-truth anomaly labels.

---

# 39. NO FAKE DATA IN LIVE MODE

Live mode must use real Open-Meteo data.

Do not replace live readings with:

- random numbers;
- hardcoded weather;
- frontend-generated values;
- fake anomaly scores;
- fake health values.

Replay mode may use the intentionally injected historical faults.

---

# 40. PRESSURE NAMING RULE

This rule is absolute.

External Open-Meteo name:

```text
surface_pressure
```

Internal SkyGuard name:

```text
pressure_hpa
```

Correct:

```python
pressure_hpa = data["surface_pressure"]
```

Incorrect:

```python
surface_pressure = ...
```

as the canonical application variable.

All existing model/features/API/state contracts use:

```text
pressure_hpa
```

---

# 41. SPATIAL ARCHITECTURE SCOPE

Required:

- same-cluster neighbors;
- same-timestamp comparison;
- station deviation;
- median neighbor deviation;
- spatial deviation feature.

Deferred:

- station-specific rolling spatial baselines;
- spatial volatility modeling;
- spatial z-score history;
- advanced graph neural networks;
- learned spatial embeddings;
- cross-cluster spatial relationships.

Do not add deferred functionality without explicit instruction.

---

# 42. PERFORMANCE REQUIREMENTS

The backend must be lightweight enough for hackathon deployment.

20 stations is a small workload.

Isolation Forest inference must remain per-station and CPU-friendly.

SHAP explainers must be cached.

Do not perform expensive SHAP computation for every normal reading unless required by the existing implementation.

Do not make unnecessary external API calls.

Do not introduce GPU dependencies.

---

# 43. ERROR HANDLING

The backend must handle:

- Open-Meteo request failures;
- missing readings;
- invalid sensor values;
- missing replay rows;
- unavailable model artifact;
- SHAP failure;
- unknown station IDs;
- unknown anomaly IDs;
- duplicate replay requests.

A SHAP failure must not crash anomaly detection.

A single malformed station reading must not corrupt the entire runtime state.

---

# 44. CORS

FastAPI must allow the existing frontend development environment to communicate with the backend.

CORS configuration must remain compatible with the current frontend setup.

Do not add authentication or authorization infrastructure unless explicitly requested.

---

# 45. STARTUP

The backend runs using Uvicorn/FastAPI.

The model must be loaded during application/backend initialization rather than repeatedly loading:

```text
isolation_forest.pkl
```

for every request.

The simulator should run as the backend's background runtime process according to the existing implementation.

---

# 46. FRONTEND CONTRACT PRESERVATION

The backend must preserve the existing endpoint and field names.

Especially locked:

```text
temperature_c
pressure_hpa
humidity_pct
timestamp
station_id
anomaly_score_pct
score_pct
severity
risk_level
health_pct
likely_faulty_sensors
suggested_values
```

Do not silently rename fields.

Do not convert:

```text
pressure_hpa
```

to:

```text
surface_pressure
```

Do not change response structures merely for convenience.

---

# 47. IMPLEMENTATION PRIORITY

If implementation work is required, prioritize:

1. Correct synchronized 20-station snapshot processing.
2. Correct same-cluster spatial median.
3. Correct per-station anomaly detection.
4. Correct state commit semantics.
5. Correct live/replay behavior.
6. Correct API contract.
7. Correct health/recovery.
8. Correct SHAP/explanation integration.
9. Performance and cleanup.

Do not spend implementation time on:

- databases;
- advanced neural architectures;
- cross-cluster spatial modeling;
- unnecessary infrastructure;
- frontend redesign.

---

# 48. ABSOLUTE BACKEND RULES

The following rules are LOCKED:

1. There are exactly 20 stations.
2. There are exactly 5 clusters.
3. Each cluster has 1 center and 3 neighbors.
4. Spatial comparison is same-cluster only.
5. Spatial comparison uses same-timestamp readings.
6. The 20 stations must be treated as a synchronized simulation snapshot.
7. Isolation Forest remains the anomaly model.
8. `pressure_hpa` is the canonical internal pressure variable.
9. Open-Meteo `surface_pressure` is mapped to `pressure_hpa` only at the external API boundary.
10. Live Open-Meteo fetching is not performed every 2 seconds.
11. The simulator cadence is 2 seconds.
12. The live external fetch interval is approximately 60 seconds.
13. Replay uses historical labeled CSV data.
14. Synthetic anomaly labels are not used by the detector as ground truth during inference.
15. Training uses clean data only.
16. Runtime state is maintained in memory.
17. No database is required.
18. SHAP is cached.
19. SHAP failure must not break detection.
20. Faulty readings must not contaminate clean causal history.
21. Frozen detection requires 3 consecutive readings.
22. Health uses the existing 10-reading window and recovery configuration.
23. Existing FastAPI endpoints are preserved.
24. Existing API field names are preserved.
25. The backend is the source of truth for anomaly detection, health, spatial analysis, and explanations.
26. Do not invent frontend behavior.
27. Do not redesign the frontend API contract.
28. Do not introduce additional architecture unless explicitly requested.
29. Do not silently modify model thresholds.
30. Do not replace the existing ML architecture.

---

# 49. FINAL ARCHITECTURAL CONTRACT

The authoritative backend architecture is:

- Open-Meteo provides real weather data.
- Historical data is stored in CSV.
- Synthetic faults are injected only for controlled evaluation/replay.
- Live/replay data is organized into synchronized 20-station timesteps.
- Spatial features compare a station against the median deviation of its same-cluster neighbors at the same timestep.
- Isolation Forest performs statistical anomaly detection for each station.
- Deterministic rules catch physical and temporal failure modes.
- `StateManager` maintains runtime truth.
- Faulty readings do not contaminate clean causal history.
- SHAP/fallback explanation identifies important features and likely faulty sensors.
- FastAPI exposes the existing backend contract.
- No database is required.
- The system remains lightweight, deterministic enough for demonstration, and suitable for hackathon-scale deployment.

This specification must be treated as the backend source of truth unless a later explicit instruction changes a specific requirement.