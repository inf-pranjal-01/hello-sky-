# SkyGuard AI — Backend Integration Specification

## PURPOSE

SkyGuard AI is an AI-powered anomaly detection and sensor-health monitoring system for a network of 20 Automatic Weather Stations (AWS).

This document describes **only the existing backend architecture, data flow, variables, state, modes, ML pipeline, and API contract**.

Do not infer additional backend functionality from this document.

The backend is implemented using:

- Python
- FastAPI
- Uvicorn
- pandas
- NumPy
- scikit-learn
- Isolation Forest
- SHAP / TreeExplainer
- Open-Meteo API
- CSV files
- In-memory runtime state

There is **no database in v1**. The backend uses historical CSV files and in-memory state.

---

# 1. HIGH-LEVEL BACKEND ARCHITECTURE

The backend runtime flow is:

```text
Open-Meteo / Historical CSV
          |
          v
     Simulator
          |
          v
    Raw Reading
          |
          v
     StateManager
          |
          v
    Feature Engineering
          |
          v
 Isolation Forest + Rules
          |
          v
 Spatial Consistency
          |
          v
   Sensor Health Tracker
          |
          v
     SHAP Explanation
          |
          v
     In-Memory State
          |
          v
       FastAPI
          |
          v
      API responses
```

The backend's core modules are:

```text
main.py
config.py
simulator.py
state.py

model/
    features.py
    train.py
    detect.py
    explain.py

data/
    data_fetch.py
    anomaly_injector.py

model_artifacts/
    isolation_forest.pkl
```

The backend architecture intentionally separates training from serving. The Isolation Forest is trained offline and loaded for inference rather than retrained per API request.

---

# 2. STATION NETWORK

There are exactly:

- 20 stations
- 5 clusters
- 4 stations per cluster

Each cluster contains:

- 1 center station
- 3 neighboring stations

## CHN — Chennai

```text
AWS-CHN-024  Chennai
AWS-CHN-101  Tambaram
AWS-CHN-102  Ambattur
AWS-CHN-103  Sriperumbudur
```

## DEL — Delhi

```text
AWS-DEL-011  Delhi
AWS-DEL-101  Noida
AWS-DEL-102  Gurugram
AWS-DEL-103  Ghaziabad
```

## MUM — Mumbai

```text
AWS-MUM-007  Mumbai
AWS-MUM-101  Thane
AWS-MUM-102  Navi Mumbai
AWS-MUM-103  Kalyan
```

## KOL — Kolkata

```text
AWS-KOL-015  Kolkata
AWS-KOL-101  Howrah
AWS-KOL-102  Bidhannagar
AWS-KOL-103  Barrackpore
```

## BHO — Bhopal

```text
AWS-BHO-030  Bhopal
AWS-BHO-101  Sehore
AWS-BHO-102  Vidisha
AWS-BHO-103  Raisen
```

The actual metadata contains:

```text
station_id
name
lat
lon
cluster_id
role
```

The cluster relationships are specifically used for spatial consistency. Only stations in the same cluster are considered spatial neighbors.

---

# 3. RAW SENSOR VARIABLES

The canonical internal backend variable names are:

```text
temperature_c
pressure_hpa
humidity_pct
```

These names must remain unchanged throughout the backend integration.

## Temperature

```text
temperature_c
```

Unit:

```text
°C
```

## Pressure

```text
pressure_hpa
```

Unit:

```text
hPa
```

## Humidity

```text
humidity_pct
```

Unit:

```text
%
```

---

# 4. IMPORTANT PRESSURE NAMING

Open-Meteo uses:

```text
surface_pressure
```

as its external API field.

SkyGuard internally converts this to:

```text
pressure_hpa
```

The mapping occurs at the data-source boundary.

Therefore:

```text
Open-Meteo:
surface_pressure

        ↓ mapping

SkyGuard:
pressure_hpa
```

The application/backend contract uses:

```text
pressure_hpa
```

not `surface_pressure`.

The historical data fetch also maps:

```python
"pressure_hpa": hourly["surface_pressure"]
```

into the internal schema.

---

# 5. RAW DATA SCHEMA

A normal historical/raw row looks conceptually like:

```json
{
  "timestamp": "...",
  "temperature_c": 31.2,
  "pressure_hpa": 1008.4,
  "humidity_pct": 71.5,
  "station_id": "AWS-CHN-024",
  "station_name": "Chennai",
  "cluster_id": "CHN",
  "role": "center"
}
```

The injected/labeled datasets additionally contain:

```text
is_anomaly
fault_type
```

These two fields are ground-truth fields for evaluation and are **not used as training inputs**.

The anomaly injector creates these labels specifically for evaluation.

---

# 6. HISTORICAL DATA

Historical data comes from Open-Meteo's Historical Weather Archive API.

Configured period:

```text
START_DATE = "2025-01-01"
END_DATE   = "2025-03-31"
```

Hourly variables:

```text
temperature_2m
surface_pressure
relative_humidity_2m
```

These become:

```text
temperature_c
pressure_hpa
humidity_pct
```

The resulting files include:

```text
AWS-*.csv
all_stations.csv
stations_metadata.csv
```

The backend uses the combined:

```text
data/all_stations.csv
```

for model training.

---

# 7. TRAINING DATA SEPARATION

Training uses:

```text
all_stations.csv
```

which contains clean historical data.

Training must NOT use:

```text
*_labeled.csv
```

because labeled files contain injected faults.

`train.py` explicitly checks for:

```text
is_anomaly
fault_type
```

and rejects the dataset if these columns are present.

---

# 8. FEATURE ENGINEERING

`features.py` transforms raw sensor data into the model feature vector.

The model currently uses **22 features**.

Feature groups include:

## Raw values

```text
temperature_c
pressure_hpa
humidity_pct
```

## Temporal features

Examples include:

```text
rolling deviations
rate of change
recent volatility
drift-related deltas
```

## Cross-parameter features

These represent relationships between:

```text
temperature
pressure
humidity
```

They are used for multivariate consistency detection.

## Spatial features

Three spatial features exist:

```text
temperature spatial deviation
pressure spatial deviation
humidity spatial deviation
```

The spatial mechanism is:

```text
station deviation
      -
median deviation of same-cluster neighbors
```

The comparison is only against neighboring stations in the same cluster.

There is intentionally **no cross-cluster spatial comparison**.

---

# 9. SPATIAL CONSISTENCY

Spatiality is intentionally simple.

For a station:

```text
current station
      |
      v
compare with same-cluster neighbors
      |
      v
calculate neighbor median
      |
      v
calculate station deviation relative to median
```

Example:

```text
Chennai       31.0
Tambaram      30.8
Ambattur      31.2
Sriperumbudur 30.9
```

The station is spatially consistent.

If:

```text
Chennai       48.0
Tambaram      30.8
Ambattur      31.2
Sriperumbudur 30.9
```

then Chennai has a large spatial disagreement.

The backend uses the **median of neighboring stations**, not the mean.

Advanced spatial-baseline volatility/z-score machinery is intentionally deferred.

---

# 10. ISOLATION FOREST

The main ML model is:

```python
IsolationForest
```

Configuration:

```text
n_estimators = 100
contamination = 0.02
random_state = 42
n_jobs = -1
```

The trained model is saved to:

```text
model_artifacts/isolation_forest.pkl
```

The model is trained once offline.

It is loaded for serving.

It does not retrain on each reading or API request.

---

# 11. MODEL SCORE

Isolation Forest's native:

```text
decision_function()
```

has the following meaning:

```text
higher  = more normal
lower   = more anomalous
```

SkyGuard converts the model/rule result into a user-facing:

```text
anomaly_score_pct
```

where:

```text
0   = low anomaly
100 = highly anomalous
```

Higher score means higher anomaly severity.

The final score is generated by `detect.py`.

---

# 12. SEVERITY THRESHOLDS

The shared severity configuration is:

```python
SEVERITY_THRESHOLDS = {
    "critical": 90,
    "high": 70,
    "medium": 55
}
```

Therefore:

```text
90–100  critical
70–89   high
55–69   medium
0–54    low
```

The shared function is:

```python
score_to_severity(score_pct)
```

The backend is the authority for severity classification.

---

# 13. RULE-BASED DETECTION

Isolation Forest is combined with deterministic rule checks.

Rules exist because certain sensor failures are better detected deterministically than statistically.

Examples:

```text
physical bounds
frozen sensor behavior
drift
dropout
```

The rule layer operates alongside the ML model.

The final verdict therefore comes from:

```text
Isolation Forest
      +
deterministic rules
      +
temporal behavior
      +
cross-parameter behavior
      +
spatial consistency
```

---

# 14. SENSOR HEALTH

The backend maintains a sensor health state independently from the numerical anomaly score.

Internal health states:

```text
HEALTHY
WARNING
OFFLINE
```

For the `/api/stations` endpoint, internal:

```text
HEALTHY
```

is translated to:

```text
NORMAL
```

so that station status becomes:

```text
NORMAL
WARNING
OFFLINE
```

---

# 15. HEALTH CIRCUIT BREAKER

Configuration:

```python
HEALTH_WINDOW_SIZE = 10
OFFLINE_ANOMALY_COUNT_THRESHOLD = 5
```

Meaning:

The backend looks at the latest 10 verdicts.

If at least 5 of those readings are anomalous, the station can be taken:

```text
OFFLINE
```

This prevents one isolated spike from immediately taking a station offline.

---

# 16. SENSOR RECOVERY

After an operator marks a sensor repaired:

```text
recovery_active = True
```

The sensor does NOT immediately become healthy.

The backend waits for:

```text
RECOVERY_CLEAN_STREAK_REQUIRED = 5
```

consecutive clean readings.

Recovery is reading-count based rather than wall-clock based.

Flow:

```text
OFFLINE/WARNING
      |
      | mark repaired
      v
WARNING + recovery_active
      |
      v
clean reading #1
      |
      v
clean reading #2
      |
      v
clean reading #3
      |
      v
clean reading #4
      |
      v
clean reading #5
      |
      v
HEALTHY
```

An anomalous reading resets the recovery clean counter.

---

# 17. CAUSAL HISTORY EXCLUSION

This is an important backend behavior.

A station's own faulty reading should not automatically become part of its future baseline.

The backend therefore maintains a per-station history buffer.

A reading is added to the station's trusted raw history only after its current verdict has been evaluated.

Conceptually:

```text
new reading
     |
     v
score reading
     |
     v
health/verdict
     |
     +---- anomalous/offline ----> DO NOT trust for future baseline
     |
     +---- clean -----------------> add to history
```

This prevents a faulty sensor from teaching the model's rolling history that its fault is normal.

---

# 18. STATE MANAGER

`state.py` contains:

```python
StateManager
```

and:

```python
StationBuffer
```

Each station gets one `StationBuffer`.

A station buffer contains:

```text
station_id
health
_raw_rows
recovery_active
recovery_clean_count
```

The raw history is stored in a bounded deque.

The maximum history length is based on:

```text
ROLLING_WINDOW_HOURS
DRIFT_LOOKBACK_HOURS
```

plus additional slack.

---

# 19. STATE MANAGER FLOW

The central runtime method is:

```python
StateManager.ingest_reading(
    station_id,
    raw_reading,
    timestamp
)
```

This is the main entry point for scoring a live/replayed reading.

Flow:

```text
raw reading
     |
     v
retrieve station buffer
     |
     v
retrieve station history
     |
     v
append current reading temporarily
     |
     v
find same-cluster neighbor buffers
     |
     v
score_reading(...)
     |
     v
update health
     |
     v
update recovery
     |
     v
conditionally record trusted history
     |
     v
return verdict
```

`state.py` is intentionally the central integration point rather than allowing different runtime paths to call `detect.py` independently.

---

# 20. NEIGHBOR BUFFER LOOKUP

For a station:

```python
_neighbor_buffers(station_id)
```

returns raw history data for stations where:

```text
cluster_id == current station cluster_id
```

and:

```text
station_id != current station
```

Only stations with available history are included.

Thus spatial comparison is based on actual same-cluster station history.

---

# 21. EXPLAINABILITY

The backend uses:

```text
SHAP
TreeExplainer
```

through:

```python
ExplainerCache
```

The explainer is initialized once rather than recreated unnecessarily for every reading.

For anomalous readings, the backend can produce:

```text
shap_features
likely_faulty_sensors
```

These are attached to anomaly records.

The system also has a deterministic magnitude-based fallback if SHAP is unavailable or fails.

---

# 22. ANOMALY EXPLANATION DATA

An anomaly explanation is returned through:

```text
GET /api/explain/{anomaly_id}
```

Response:

```json
{
  "anomaly_id": "anom_00042",
  "features": [...],
  "likely_faulty_sensors": [...]
}
```

`features` contains the important contributing features.

`likely_faulty_sensors` maps those feature contributions back to raw sensor parameters.

---

# 23. SYNTHETIC ANOMALY INJECTION

Synthetic faults are used only to evaluate/demonstrate the system.

The injector works on clean historical data.

Injection rate:

```text
INJECTION_RATE = 0.05
```

Approximately 5% of rows are affected per faulty station.

Random seed:

```text
RANDOM_SEED = 42
```

The injector deliberately selects only a subset of stations.

Current configuration:

```text
N_FAULTY_STATIONS = 3
STATION_SELECTION_SEED = 7
```

Therefore, not all 20 stations are intentionally corrupted.

This leaves healthy neighboring stations available for meaningful spatial comparison.

---

# 24. INJECTED FAULT TYPES

The backend currently supports:

```text
spike
frozen_value
drift
dropout
sensor_fail_low
multivariate_inconsistency
```

## spike

Single reading pushed far beyond its normal statistical range.

## frozen_value

Reading stays almost constant for several consecutive rows.

Freeze length:

```text
3–6 rows approximately
```

with small noise added.

## drift

Slowly growing calibration offset.

Drift duration:

```text
20–49 rows approximately
```

## dropout

Sensor value becomes:

```text
NaN
```

## sensor_fail_low

Sensor falls toward an implausibly low failure value.

## multivariate_inconsistency

Temperature, pressure and humidity are altered together to create an inconsistent combination.

The injector explicitly models these as known AWS failure modes.

---

# 25. LIVE MODE

Live mode uses Open-Meteo current weather data.

External fields:

```text
temperature_2m
surface_pressure
relative_humidity_2m
```

Internal mapping:

```text
temperature_c
pressure_hpa
humidity_pct
```

The simulator has:

```text
TICK_SECONDS = 2
LIVE_FETCH_INTERVAL_SECONDS = 60
```

The simulator loop therefore operates at approximately 2-second UI/simulation cadence while live weather data itself is refreshed at the configured live-fetch interval.

---

# 26. LIVE MODE FLOW

```text
Open-Meteo current API
          |
          v
fetch current weather
          |
          v
map external names
          |
          v
temperature_c
pressure_hpa
humidity_pct
          |
          v
StateManager.ingest_reading()
          |
          v
feature generation
          |
          v
Isolation Forest
+
rules
+
spatial comparison
          |
          v
health update
          |
          v
optional SHAP explanation
          |
          v
state.latest
state.trend_history
state.recent_anomalies
```

The backend only feeds genuinely new live weather readings into ML/state processing rather than treating every 2-second UI tick as a new weather observation.

---

# 27. REPLAY MODE

Replay mode uses the labeled historical CSV datasets generated by:

```text
anomaly_injector.py
```

The replay contains:

```text
real historical readings
+
known injected faults
+
ground-truth labels
```

The replay drives the stations through their historical rows.

It is used for:

- Demonstration
- End-to-end anomaly detection
- Showing faults
- Showing station health changes
- Evaluating model behavior

---

# 28. REPLAY MODE FLOW

```text
*_labeled.csv
     |
     v
Simulator
     |
     v
next historical row
     |
     v
StateManager.ingest_reading()
     |
     v
feature engineering
     |
     v
Isolation Forest
+
rules
+
spatial comparison
     |
     v
health tracking
     |
     v
SHAP explanation if anomalous
     |
     v
in-memory state
     |
     v
FastAPI responses
```

Replay eventually ends and the simulator returns to live operation.

---

# 29. IMPORTANT REPLAY DETAIL

`POST /api/inject-anomaly` is currently the trigger for replay mode.

It does NOT create a new arbitrary fault on the requested station.

The request accepts:

```json
{
  "station_id": "...",
  "type": "spike"
}
```

for API contract compatibility.

However, the backend actually calls:

```python
sim.start_replay()
```

and starts the existing labeled-data replay across the stations.

The supplied:

```text
station_id
type
```

are not used to target a new individual fault.

---

# 30. IN-MEMORY RUNTIME STATE

The simulator maintains runtime state including:

```text
latest
trend_history
recent_anomalies
metadata
StateManager
mode
```

The exact internal object is:

```text
SimulatorState
```

The state is not persisted in a database.

A backend restart resets runtime state and the simulator starts again from its configured data.

---

# 31. API CONTRACT

The backend currently exposes exactly these 9 endpoints.

---

## 31.1 GET `/api/stations`

Returns all stations.

Response:

```json
[
  {
    "station_id": "AWS-CHN-024",
    "name": "Chennai",
    "lat": 13.0827,
    "lon": 80.2707,
    "status": "NORMAL"
  }
]
```

Possible `status` values:

```text
NORMAL
WARNING
OFFLINE
```

---

# 32. GET `/api/current-reading`

Request:

```text
GET /api/current-reading?station_id=AWS-CHN-024
```

Response:

```json
{
  "station_id": "AWS-CHN-024",
  "timestamp": "...",

  "temperature_c": {
    "value": 31.2,
    "normal_min": 10.0,
    "normal_max": 45.0
  },

  "pressure_hpa": {
    "value": 1008.4,
    "normal_min": 950.0,
    "normal_max": 1050.0
  },

  "humidity_pct": {
    "value": 71.5,
    "normal_min": 10.0,
    "normal_max": 100.0
  },

  "anomaly_score_pct": 23,
  "risk_level": "low",
  "sensor_health_pct": 77,
  "sensor_health_status": "HEALTHY"
}
```

Possible 404:

```text
No reading yet for {station_id}
```

---

# 33. GET `/api/trends`

Request:

```text
GET /api/trends?station_id=AWS-CHN-024&hours=6
```

Response:

```json
{
  "station_id": "AWS-CHN-024",

  "points": [
    {
      "timestamp": "...",
      "temperature_c": 31.2,
      "pressure_hpa": 1008.4,
      "humidity_pct": 71.5
    }
  ],

  "anomaly_windows": [
    {
      "start": "...",
      "end": "...",
      "label": "Anomaly Detected"
    }
  ]
}
```

The backend returns its currently buffered trend points.

The `hours` argument exists in the API contract, but the current implementation returns the available buffered history rather than assuming a fixed number of rows per hour because live and replay have different effective observation cadences.

---

# 34. GET `/api/anomalies/latest`

Request:

```text
GET /api/anomalies/latest?station_id=AWS-CHN-024
```

Response:

```json
{
  "anomaly_id": "anom_00042",
  "timestamp": "...",
  "station_id": "AWS-CHN-024",
  "anomaly_score_pct": 91,
  "severity": "critical",
  "type": "spike",
  "root_cause": "...",
  "description": "...",
  "suggested_values": {
    "temperature_c": 31.2
  }
}
```

If there is no recorded anomaly:

```text
404
```

---

# 35. GET `/api/anomalies/recent`

Request:

```text
GET /api/anomalies/recent?station_id=AWS-CHN-024&limit=5
```

Response:

```json
[
  {
    "anomaly_id": "anom_00042",
    "type": "spike",
    "label": "...",
    "station_id": "AWS-CHN-024",
    "timestamp": "...",
    "score_pct": 91,
    "severity": "critical",
    "suggested_values": {}
  }
]
```

The default limit is:

```text
5
```

---

# 36. GET `/api/explain/{anomaly_id}`

Request:

```text
GET /api/explain/anom_00042
```

Response:

```json
{
  "anomaly_id": "anom_00042",
  "features": [],
  "likely_faulty_sensors": []
}
```

If the anomaly ID does not exist:

```text
404
```

---

# 37. GET `/api/sensor-health`

Request:

```text
GET /api/sensor-health?station_id=AWS-CHN-024
```

Response:

```json
{
  "station_id": "AWS-CHN-024",
  "health_pct": 96,
  "status": "HEALTHY"
}
```

Possible status:

```text
HEALTHY
WARNING
OFFLINE
```

---

# 38. POST `/api/repair-sensor`

Request:

```json
{
  "station_id": "AWS-CHN-024"
}
```

Response:

```json
{
  "success": true,
  "station_id": "AWS-CHN-024",
  "status": "WARNING",
  "recovery_active": true,
  "message": "Sensor marked for repair recovery. Clean readings will be evaluated before returning it to HEALTHY."
}
```

Important:

Calling this endpoint does NOT immediately make the sensor healthy.

It starts the recovery process.

---

# 39. POST `/api/inject-anomaly`

Request:

```json
{
  "station_id": "AWS-CHN-024",
  "type": "spike"
}
```

Current behavior:

```text
start replay
```

Response:

```json
{
  "success": true,
  "anomaly_id": "...",
  "message": "Simulator started: replaying labeled historical data with injected faults across all stations."
}
```

If replay is already running:

```text
409
```

with:

```text
Simulator replay already running.
```

---

# 40. POST `/api/maintenance-ticket`

Request:

```json
{
  "anomaly_id": "anom_00042"
}
```

Response:

```json
{
  "ticket_id": "TCK-0001",
  "station_id": "AWS-CHN-024",
  "issue": "...",
  "priority": "high",
  "created_at": "..."
}
```

Ticket priority is currently:

```text
high
```

for:

```text
high
critical
```

severity.

Otherwise:

```text
medium
```

The ticket counter is in-memory.

---

# 41. COMPLETE API TABLE

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/stations` | All station metadata + status |
| GET | `/api/current-reading` | Current station sensor data |
| GET | `/api/trends` | Buffered station trends + anomaly windows |
| GET | `/api/anomalies/latest` | Latest anomaly for station |
| GET | `/api/anomalies/recent` | Recent anomalies for station |
| GET | `/api/explain/{anomaly_id}` | SHAP/model explanation |
| GET | `/api/sensor-health` | Sensor health |
| POST | `/api/repair-sensor` | Start repair/recovery |
| POST | `/api/inject-anomaly` | Start replay simulation |
| POST | `/api/maintenance-ticket` | Create maintenance ticket |

---

# 42. IMPORTANT: THERE IS NO DATABASE

The backend intentionally does not use:

```text
PostgreSQL
MySQL
MongoDB
SQLite
Firebase
Supabase
```

The current v1 storage model is:

```text
historical CSV files
+
in-memory runtime state
```

This is deliberate for hackathon scale.

---

# 43. CORS

FastAPI currently allows:

```python
allow_origins=["*"]
```

along with:

```text
allow_methods=["*"]
allow_headers=["*"]
```

This is currently configured for hackathon development/deployment.

---

# 44. BACKEND STARTUP

FastAPI creates the simulator state during application lifespan:

```python
app.state.sim = create_simulator_state()
```

Then starts:

```python
run_simulation_loop(app.state.sim)
```

as an asynchronous background task.

Therefore the simulator starts with the API application.

---

# 45. BACKEND SOURCE OF TRUTH

The backend is authoritative for:

```text
current readings
station status
anomaly score
severity
sensor health
anomaly type
root cause
suggested values
SHAP features
likely faulty sensors
recovery state
maintenance ticket ID
```

The frontend integration should treat the API response values as authoritative and should not recreate the ML decision logic.

---

# 46. COMPLETE RUNTIME FLOW — LIVE

```text
FastAPI startup
      |
      v
create_simulator_state()
      |
      v
load metadata + model
      |
      v
start simulation loop
      |
      v
LIVE MODE
      |
      v
fetch Open-Meteo current data
      |
      v
map:
surface_pressure
      ↓
pressure_hpa
      |
      v
StateManager.ingest_reading()
      |
      v
history + current reading
      |
      v
same-cluster neighbor histories
      |
      v
feature generation
      |
      v
Isolation Forest
      +
rule checks
      +
spatial consistency
      |
      v
anomaly verdict
      |
      v
SensorHealthTracker
      |
      v
SHAP explanation if anomalous
      |
      v
trusted history update
      |
      v
latest/trend/anomaly state
      |
      v
FastAPI API
```

---

# 47. COMPLETE RUNTIME FLOW — REPLAY

```text
POST /api/inject-anomaly
      |
      v
start_replay()
      |
      v
REPLAY MODE
      |
      v
read next row from *_labeled.csv
      |
      v
StateManager.ingest_reading()
      |
      v
feature generation
      |
      v
Isolation Forest
      +
rules
      +
spatial consistency
      |
      v
anomaly verdict
      |
      v
health state
      |
      v
SHAP explanation
      |
      v
in-memory state
      |
      v
FastAPI
      |
      v
replay continues
      |
      v
replay exhausted
      |
      v
return to LIVE MODE
```

---

# 48. COMPLETE SENSOR FAILURE FLOW

```text
normal station
      |
      v
anomalous readings
      |
      v
health tracker records verdicts
      |
      v
5+ anomalies in latest 10 readings
      |
      v
OFFLINE
      |
      v
faulty readings excluded from trusted history
      |
      v
operator calls POST /api/repair-sensor
      |
      v
recovery_active = true
      |
      v
WARNING
      |
      v
5 consecutive clean readings
      |
      v
HEALTHY
```

---

# 49. IMPORTANT IMPLEMENTATION CONSTANTS

```text
N_ESTIMATORS = 100
RANDOM_STATE = 42

SEVERITY_THRESHOLDS:
    medium   = 55
    high     = 70
    critical = 90

HEALTH_WINDOW_SIZE = 10

OFFLINE_ANOMALY_COUNT_THRESHOLD = 5

RECOVERY_CLEAN_STREAK_REQUIRED = 5

INJECTION_RATE = 0.05

STATION_SELECTION_SEED = 7

N_FAULTY_STATIONS = 3

TICK_SECONDS = 2

LIVE_FETCH_INTERVAL_SECONDS = 60
```

---

# 50. FINAL CONTRACT RULES

The current backend contract must be treated as fixed.

Canonical sensor names:

```text
temperature_c
pressure_hpa
humidity_pct
```

Canonical station identifier:

```text
station_id
```

Canonical timestamp:

```text
timestamp
```

Canonical anomaly score:

```text
anomaly_score_pct
```

Canonical station API statuses:

```text
NORMAL
WARNING
OFFLINE
```

Canonical sensor health statuses:

```text
HEALTHY
WARNING
OFFLINE
```

Canonical severity:

```text
low
medium
high
critical
```

Spatial comparison:

```text
station vs median of same-cluster neighbors
```

ML model:

```text
IsolationForest
```

Explainability:

```text
SHAP / TreeExplainer
```

Runtime state:

```text
in-memory
```

Historical data:

```text
CSV
```

Live data:

```text
Open-Meteo
```

Replay:

```text
labeled historical CSVs
```

Backend API:

```text
9 endpoints
```

There is no additional API contract beyond the endpoints documented above.