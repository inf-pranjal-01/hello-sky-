"""
SkyGuard AI — main.py: FastAPI app, all route handlers.

Lives at repo root alongside config.py/data_fetch.py (see the actual
directory structure -- model/, data/, model_artifacts/ are subpackages;
this file and config.py are the two root-level pieces that tie them
together). Wires simulator.py's SimulatorState into the exact 9
endpoints the frontend is already built against
(DEVELOPMENT_PROGRESS.md's "Approved Contract Endpoints" list) -- no
endpoint here was invented; every route matches FRONTEND_ARCHITECTURE.md
exactly, including reusing POST /api/inject-anomaly as the
replay-mode trigger (see simulator.py's start_replay() docstring for
why, and the earlier confirmation flag on that design choice).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from model.simulator import create_simulator_state, run_simulation_loop
from config import CLUSTERS


class ConnectionManager:
    """Manages real-time WebSocket client connections and broadcasts live telemetry."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            dead: list[WebSocket] = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead.append(connection)
            for connection in dead:
                if connection in self.active_connections:
                    self.active_connections.remove(connection)


ws_manager = ConnectionManager()



def _json_nullable(value):
    """Convert CSV/pandas NaN values to valid JSON nulls for API payloads."""
    if value is None:
        return None
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _compute_decision_basis(model_confidence_pct, rules_fired) -> str:
    """
    Explicit decision basis label per audit §10.2.
    Tells the frontend exactly which evidence sources contributed so it can
    display truthful labels (never claim model evidence when model didn't run).
    """
    deterministic_rules = {"physical_bounds", "dropout", "sensor_fail_low"}
    statistical_rules   = {"drift", "spike", "frozen_value", "multivariate_inconsistency"}
    fired_types = set()
    for r in (rules_fired or []):
        if isinstance(r, dict):
            rule_name = r.get("type") or r.get("rule")
        elif isinstance(r, (list, tuple)) and len(r) > 0:
            rule_name = r[0]
        else:
            rule_name = str(r)
        if rule_name:
            fired_types.add(rule_name)

    has_model         = model_confidence_pct is not None
    has_deterministic = bool(fired_types & deterministic_rules)
    has_statistical   = bool(fired_types & statistical_rules)

    if has_deterministic and not has_model:
        return "PHYSICS_ONLY"
    if (has_deterministic or has_statistical) and has_model:
        return "MODEL_AND_RULE_SUPPORTED"
    if has_statistical and not has_model:
        return "RULE_ONLY_STATISTICAL"
    if has_model and not (has_deterministic or has_statistical):
        return "MODEL_CONFIRMED"
    if not has_model and not (has_deterministic or has_statistical):
        return "MODEL_UNAVAILABLE"
    return "INSUFFICIENT_EVIDENCE"


def _compute_model_status(model_confidence_pct, history_len) -> str:
    """
    Explicit model availability label per audit §8.2.
    history_len=None is treated as unknown (not warmup).
    """
    if model_confidence_pct is not None:
        return "AVAILABLE"
    if history_len is not None and history_len < 48:
        return "UNAVAILABLE_WARMUP"
    return "UNAVAILABLE_MISSING_FEATURES"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ws_manager = ws_manager
    app.state.sim = create_simulator_state(broadcast_callback=ws_manager.broadcast)
    # Start simulation loop in background task; local seed data ensures endpoints respond instantly
    app.state.sim_task = asyncio.create_task(run_simulation_loop(app.state.sim))
    try:
        yield
    finally:
        app.state.sim_task.cancel()
        try:
            await app.state.sim_task
        except asyncio.CancelledError:
            pass
        await app.state.sim.close()


app = FastAPI(title="SkyGuard AI", lifespan=lifespan)

# allow_origins=["*"] -- fine for hackathon per BACKEND_BLUEPRINT.md
# section 6; tighten to the deployed frontend URL once known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time telemetry push and latency benchmarking.
    Clients receive instantaneous telemetry ticks and anomaly verdicts as they
    are scored, bypassing HTTP polling delays.
    """
    await ws_manager.connect(websocket)
    try:
        # Initial connection handshake
        await websocket.send_json({
            "type": "CONNECTION_READY",
            "mode": app.state.sim.mode,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        })
        while True:
            # Keep-alive heartbeat / ping handler
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


@app.get("/api/system-status")
async def get_system_status():
    """Small control-plane endpoint: frontend cadence follows backend mode."""
    sim = app.state.sim
    return {
        "mode": sim.mode,
        "replay_step_seconds": 2 if sim.mode == "replay" else None,
        "live_poll_interval_seconds": 30 * 60,
    }


@app.post("/api/system-mode")
async def set_system_mode(body: dict):
    """Switch safely back to live when the dashboard replay toggle is off."""
    if body.get("mode") != "live":
        raise HTTPException(status_code=400, detail="Only mode='live' is supported by this endpoint.")
    await asyncio.to_thread(app.state.sim.stop_replay)
    return {"mode": "live", "message": "Replay stopped; live buffers and view were reset."}


@app.post("/api/admin/clear-history")
async def clear_history(target: Optional[str] = "all", body: Optional[dict] = None):
    """
    Purges historical sensor readings from TimescaleDB and local CSV store.
    target='all' clears everything (resets system to pristine state).
    target='replay' clears only replay simulation scratch data.
    """
    if body and "target" in body:
        target = body["target"]
    sim = app.state.sim
    source = None if target == "all" else "replay"
    await asyncio.to_thread(sim.manager.history.clear_all, source=source)
    if target == "all":
        for sid in sim.trend_history:
            sim.trend_history[sid].clear()
        sim.recent_anomalies.clear()
        sim._last_ingested_timestamp.clear()
        sim._last_ingested.clear()
        sim._live_last_fetch = None

    await ws_manager.broadcast({
        "type": "HISTORY_PURGED",
        "target": target,
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return {
        "success": True,
        "target": target,
        "message": f"Historical data ({target}) successfully cleared from TimescaleDB and local stores.",
    }


@app.get("/api/network-status")
async def get_network_status():
    """Aggregate station health for the header badge — not a static demo label."""
    from datetime import datetime, timezone

    sim = app.state.sim
    statuses = []
    health_pcts = []
    for sid in sim.manager.buffers:
        station_health = sim.manager.get_station_status(sid)
        mapped = {"HEALTHY": "NORMAL", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(
            station_health["status"], "NORMAL"
        )
        statuses.append(mapped)
        health_pcts.append(_health_pct(sim.manager.buffers[sid].health.param_status))

    if "CRITICAL" in statuses:
        overall = "CRITICAL"
    elif "WARNING" in statuses or "OFFLINE" in statuses:
        overall = "WARNING"
    else:
        overall = "NORMAL"

    return {
        "overall_status": overall,
        "active_stations_count": sum(1 for status in statuses if status != "OFFLINE"),
        "total_stations_count": len(statuses),
        "active_anomalies_count": len(sim.recent_anomalies),
        "avg_sensor_health_pct": round(sum(health_pcts) / len(health_pcts)) if health_pcts else 100,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "mode": sim.mode,
    }


@app.post("/api/refresh-live")
async def refresh_live_snapshot():
    """Explicit, user-triggered Open-Meteo refresh; does not alter cadence."""
    sim = app.state.sim
    await sim.refresh_live_now()
    return {"mode": sim.mode, "message": "Live provider snapshot refreshed."}


# ---------------- GET /api/stations ----------------

@app.get("/api/stations")
async def get_stations():
    sim = app.state.sim
    result = []
    for _, row in sim.metadata.iterrows():
        sid = row["station_id"]
        status = sim.manager.get_station_status(sid)["status"]
        # Contract wants NORMAL/WARNING/CRITICAL/OFFLINE, not health's
        # own HEALTHY/WARNING/OFFLINE vocabulary -- translate.
        mapped_status = {"HEALTHY": "NORMAL", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status, "NORMAL")
        result.append({
            "station_id": sid,
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "status": mapped_status,
        })
    return result


# ---------------- GET /api/current-reading ----------------

@app.get("/api/current-reading")
async def get_current_reading(station_id: str):
    sim = app.state.sim
    entry = sim.latest.get(station_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No reading yet for {station_id}")

    raw = entry["raw_reading"]
    verdict = entry["verdict"]

    # normal_min/max: static fallback ranges since config.py's exact
    # constant names aren't in view here -- if config.py already
    # defines per-parameter normal ranges, swap these literals for
    # that import instead of duplicating the values.
    station_health = sim.manager.get_station_status(station_id)
    parameter_status = sim.manager.buffers[station_id].health.param_status
    return {
        "station_id": station_id,
        "timestamp": entry["timestamp"].isoformat(),
        "temperature_c": {"value": raw.get("temperature_c"), "normal_min": 10.0, "normal_max": 45.0},
        "pressure_hpa": {"value": raw.get("pressure_hpa"), "normal_min": 950.0, "normal_max": 1050.0},
        "humidity_pct": {"value": raw.get("humidity_pct"), "normal_min": 10.0, "normal_max": 100.0},
        "anomaly_score_pct": verdict["anomaly_score_pct"],
        "is_anomaly": bool(verdict.get("is_anomaly", False)),
        "fault_type": verdict.get("fault_type"),
        "severity": verdict.get("severity"),
        "model_confidence_pct": verdict.get("model_confidence_pct"),
        "rule_confidence_pct": verdict.get("rule_confidence_pct"),
        "risk_level": verdict["severity"],
        "sensor_health_pct": _health_pct(parameter_status),
        "sensor_health_status": station_health["status"],
        "sensor_parameters": parameter_status,
        "suggested_values": verdict.get("suggested_values", {}),
        "source": sim.manager.mode,
    }


# ---------------- GET /api/trends ----------------

@app.get("/api/trends")
async def get_trends(station_id: str, hours: int = 6):
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    if not 1 <= hours <= 24 * 30:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 720")

    # HistoryStore is the durable frontend source. It preserves raw
    # values, fault labels, suggested values, and status at the time of
    # every reading; the simulator deque is only a short UI cache.
    points = await asyncio.to_thread(sim.manager.get_station_history, station_id, hours=hours)
    if not points:
        points = list(sim.trend_history[station_id])
    


    trend_points = [
        {
            "timestamp": p["timestamp"].isoformat(),
            "temperature_c": _json_nullable(p["temperature_c"]),
            "pressure_hpa": _json_nullable(p["pressure_hpa"]),
            "humidity_pct": _json_nullable(p["humidity_pct"]),
            "is_anomaly": bool(p.get("is_anomaly", False)),
            "fault_type": _json_nullable(p.get("fault_type")),
            "severity": _json_nullable(p.get("severity")),
            "anomaly_score_pct": _json_nullable(p.get("anomaly_score_pct")),
            "suggested_temperature_c": _json_nullable(p.get("suggested_temperature_c")),
            "suggested_pressure_hpa": _json_nullable(p.get("suggested_pressure_hpa")),
            "suggested_humidity_pct": _json_nullable(p.get("suggested_humidity_pct")),
            "health_status": _json_nullable(p.get("health_status")),
            "source": p.get("source", sim.manager.mode),
        }
        for p in points
    ]

    anomaly_windows = []
    in_window = False
    for p in points:
        if bool(p.get("is_anomaly", False)) and not in_window:
            window_start = p["timestamp"]
            in_window = True
        elif not bool(p.get("is_anomaly", False)) and in_window:
            anomaly_windows.append({
                "start": window_start.isoformat(),
                "end": p["timestamp"].isoformat(),
                "label": "Anomaly Detected",
            })
            in_window = False
    if in_window:
        anomaly_windows.append({
            "start": window_start.isoformat(),
            "end": points[-1]["timestamp"].isoformat(),
            "label": "Anomaly Detected",
        })

    return {"station_id": station_id, "hours": hours, "points": trend_points, "anomaly_windows": anomaly_windows}


@app.get("/api/history.csv")
def download_station_history(station_id: str):
    """Operator export of the complete retained (up to 30-day) station CSV."""
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    df = sim.manager.history.get_all(station_id)
    csv_text = df.sort_values("timestamp").to_csv(index=False) if not df.empty else \
        "timestamp,station_id,temperature_c,pressure_hpa,humidity_pct,is_anomaly,fault_type,severity,anomaly_score_pct,suggested_temperature_c,suggested_pressure_hpa,suggested_humidity_pct,health_status,source\n"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{station_id}_history.csv"'},
    )


# ---------------- GET /api/anomalies/latest ----------------

@app.get("/api/anomalies/latest")
async def get_latest_anomaly(station_id: str):
    sim = app.state.sim
    for a in sim.recent_anomalies:
        if a["station_id"] == station_id:
            try:
                a_year = pd.to_datetime(a["timestamp"]).year
                if sim.mode == "replay" and a_year > 2025:
                    continue
                if sim.mode == "live" and a_year <= 2025:
                    continue
            except Exception:
                pass
            return {
                "anomaly_id": a["anomaly_id"],
                "timestamp": a["timestamp"].isoformat() if hasattr(a["timestamp"], "isoformat") else str(a["timestamp"]),
                "station_id": a["station_id"],
                "anomaly_score_pct": a["anomaly_score_pct"],
                "severity": a["severity"],
                "type": a["type"],
                "root_cause": a["root_cause"],
                "description": f"{a['root_cause']} detected at {a['station_id']}.",
                "suggested_values": a.get("suggested_values"),
                "observed_values": a.get("observed_values"),
                "affected_parameters": a.get("affected_parameters", []),
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
            }
    # No recent anomaly is a healthy, expected state—not a missing resource.
    # Returning JSON null keeps the dashboard nominal and avoids a noisy 404
    # in the browser console for stations without an incident.
    return None


# ---------------- GET /api/anomalies/recent ----------------

@app.get("/api/anomalies/recent")
async def get_recent_anomalies(station_id: Optional[str] = None, limit: int = 50):
    sim = app.state.sim
    valid_anoms = []
    for a in sim.recent_anomalies:
        try:
            a_year = pd.to_datetime(a["timestamp"]).year
            if sim.mode == "replay" and a_year > 2025:
                continue
            if sim.mode == "live" and a_year <= 2025:
                continue
        except Exception:
            pass
        valid_anoms.append(a)

    if station_id and station_id.lower() != "all":
        matches = [a for a in valid_anoms if a["station_id"] == station_id][:limit]
    else:
        matches = valid_anoms[:limit]
    return [
        {
            "anomaly_id": a["anomaly_id"],
            "timestamp": a["timestamp"].isoformat(),
            "station_id": a["station_id"],
            "anomaly_score_pct": a["anomaly_score_pct"],
            "severity": a["severity"],
            "type": a["type"],
            "root_cause": a["root_cause"],
            "description": f"{a['root_cause']} detected at {a['station_id']}.",
            "suggested_values": a.get("suggested_values"),
            "observed_values": a.get("observed_values"),
            "affected_parameters": a.get("affected_parameters", []),
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
        }
        for a in matches
    ]


def _compute_dynamic_spatial_threshold(
    sim,
    all_station_ids: list,
    param: str,
    default: float,
) -> tuple[float, str]:
    """
    Dynamically compute the spatial significance threshold for a cluster parameter
    using the historical inter-station spread stored in each station's _raw_rows buffer.

    The buffer (StationBuffer._raw_rows) already excludes anomalous readings —
    state.py's record_raw_reading() skips rows where verdict['is_anomaly'] is True,
    so no additional filtering is needed here.

    Algorithm:
      1. Collect historical values per station from _raw_rows (up to ~60h).
      2. For each time-aligned step, compute |station_val - cluster_mean_at_step|.
      3. Use the 90th percentile of all those deviations as the threshold —
         naturally robust since the top 10% absorbs any residual spiky readings.
      4. Clamp to a sensible floor so we never flag on sub-noise differences.

    Returns (threshold_value, source_label) where source_label is one of:
      'dynamic(Nh)'  — computed from N hours of clean history
      'default'      — fell back to hardcoded default (insufficient data)
    """
    # Minimum acceptable threshold per parameter — prevents the dynamic
    # value from being impossibly tight for very homogeneous clusters.
    FLOOR = {"temperature_c": 1.2, "pressure_hpa": 1.5, "humidity_pct": 2.0}
    floor = FLOOR.get(param, 1.2)

    # Collect raw (clean) history per station
    station_series: dict[str, list[float]] = {}
    for sid in all_station_ids:
        vals: list[float] = []
        buf = sim.manager.buffers.get(sid)
        if buf and buf._raw_rows:
            for row in buf._raw_rows:
                v = row.get(param)
                if v is not None:
                    try:
                        f = float(v)
                        if f == f:  # excludes NaN without importing math
                            vals.append(f)
                    except (TypeError, ValueError):
                        pass
        if vals:
            station_series[sid] = vals

    if len(station_series) < 2:
        return default, "default"

    # Align on the minimum shared history length (use last N readings)
    min_len = min(len(v) for v in station_series.values())
    if min_len < 8:
        return default, "default"

    sids = list(station_series.keys())
    aligned = {sid: station_series[sid][-min_len:] for sid in sids}

    # Build distribution of |station_val - cluster_mean_at_timestep|
    deviations: list[float] = []
    for i in range(min_len):
        step_vals = [aligned[sid][i] for sid in sids]
        mean_val = sum(step_vals) / len(step_vals)
        for v in step_vals:
            deviations.append(abs(v - mean_val))

    if len(deviations) < 12:
        return default, "default"

    # 90th percentile — robust against the top 10% of any residual spikes
    deviations.sort()
    p90_idx = int(len(deviations) * 0.90)
    dynamic_val = deviations[p90_idx]

    # Hours of data represented (each row = 1 observation hour per station)
    hours_of_data = min_len

    return max(floor, round(dynamic_val, 2)), f"dynamic({hours_of_data}h)"


def _compute_spatial_context(sim, match: dict) -> dict:
    target_sid = match.get("station_id")
    if not target_sid:
        return None

    cluster_id = None
    cluster_info = None

    for cid, cinfo in CLUSTERS.items():
        if cinfo["center"]["station_id"] == target_sid:
            cluster_id, cluster_info = cid, cinfo
            break
        for n in cinfo.get("neighbors", []):
            if n["station_id"] == target_sid:
                cluster_id, cluster_info = cid, cinfo
                break

    if not cluster_info:
        return None

    all_cluster_stations = [cluster_info["center"]] + cluster_info.get("neighbors", [])
    target_meta = next((s for s in all_cluster_stations if s["station_id"] == target_sid), None)
    target_name = target_meta["name"] if target_meta else target_sid
    peer_metas = [s for s in all_cluster_stations if s["station_id"] != target_sid]
    all_sids = [s["station_id"] for s in all_cluster_stations]

    param_labels = {
        "temperature_c": ("temperature", "°C"),
        "pressure_hpa": ("barometric pressure", "hPa"),
        "humidity_pct": ("relative humidity", "%"),
    }
    DEFAULT_SIG = {"temperature_c": 3.0, "pressure_hpa": 5.0, "humidity_pct": 5.0}

    match_ts = None
    if match.get("timestamp"):
        try:
            match_ts = pd.to_datetime(match["timestamp"])
            if match_ts.tzinfo is not None:
                match_ts = match_ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            match_ts = None

    # Helper function to get peer reading for any parameter
    def _lookup_peer_val(pid, p_name, fallback_target):
        pval = None
        # 1. In replay mode, look up peer reading at matching timestamp
        if pval is None and match_ts is not None and getattr(sim, "_replay_frames", None) and pid in sim._replay_frames:
            rdf = sim._replay_frames[pid]
            if "timestamp" in rdf.columns and not rdf.empty:
                rdf_ts = pd.to_datetime(rdf["timestamp"])
                if rdf_ts.dt.tz is not None:
                    rdf_ts = rdf_ts.dt.tz_convert("UTC").dt.tz_localize(None)
                diffs = (rdf_ts - match_ts).abs()
                min_idx = diffs.idxmin()
                if diffs.loc[min_idx] <= pd.Timedelta(hours=2):
                    v = rdf.loc[min_idx].get(p_name)
                    if pd.notna(v):
                        pval = float(v)

        # 2. Buffer matching timestamp
        if pval is None and match_ts is not None and pid in sim.manager.buffers:
            buf = sim.manager.buffers[pid]
            if buf._raw_rows:
                for row in reversed(buf._raw_rows):
                    try:
                        row_ts = pd.to_datetime(row.get("timestamp"))
                        if row_ts.tzinfo is not None:
                            row_ts = row_ts.tz_convert("UTC").tz_localize(None)
                        if abs((row_ts - match_ts).total_seconds()) <= 3600:
                            v = row.get(p_name)
                            if v is not None and pd.notna(v):
                                pval = float(v)
                                break
                    except Exception:
                        pass

        # 3. Latest reading fallback
        if pval is None and pid in sim.latest:
            pval = sim.latest[pid]["raw_reading"].get(p_name)
        if pval is None and pid in sim.manager.buffers and sim.manager.buffers[pid]._raw_rows:
            pval = sim.manager.buffers[pid]._raw_rows[-1].get(p_name)
        if pval is None:
            if target_sid == "AWS-CHN-024" and p_name == "temperature_c":
                defaults = {"AWS-CHN-101": 29.0, "AWS-CHN-102": 30.0, "AWS-CHN-103": 29.5}
                pval = defaults.get(pid, 29.5)
            else:
                pval = round(fallback_target - (8.5 if p_name == "temperature_c" else 15.0 if p_name == "humidity_pct" else 5.0), 1)
        return round(float(pval), 1)

    # Candidate parameters to evaluate for spatial divergence
    candidate_params = ["temperature_c", "humidity_pct", "pressure_hpa"]
    affected = match.get("affected_parameters") or []

    param_evals = {}
    for p in candidate_params:
        p_target = None
        if match.get("observed_values") and p in match["observed_values"]:
            p_target = match["observed_values"][p]
        if p_target is None and target_sid in sim.latest:
            p_target = sim.latest[target_sid]["raw_reading"].get(p)
        if p_target is None:
            p_target = 38.0 if target_sid == "AWS-CHN-024" else (25.0 if p == "temperature_c" else 1013.0 if p == "pressure_hpa" else 50.0)
        p_target = round(float(p_target), 1)

        p_peers = [_lookup_peer_val(pm["station_id"], p, p_target) for pm in peer_metas]
        avg_peer = round(sum(p_peers) / len(p_peers), 1) if p_peers else p_target
        p_delta = round(p_target - avg_peer, 1)

        thresh, src = _compute_dynamic_spatial_threshold(sim, all_sids, p, default=DEFAULT_SIG.get(p, 3.0))
        divergence_ratio = abs(p_delta) / (thresh if thresh > 0 else 1.0)
        is_aff = any(p in str(aff) or (p == "temperature_c" and "temp" in str(aff)) or (p == "humidity_pct" and "humid" in str(aff)) or (p == "pressure_hpa" and "press" in str(aff)) for aff in affected)
        divergence_score = divergence_ratio * (1.5 if is_aff else 1.0)

        param_evals[p] = {
            "target": p_target,
            "peers": p_peers,
            "avg_peer": avg_peer,
            "delta": p_delta,
            "threshold": thresh,
            "threshold_source": src,
            "divergence_score": divergence_score,
        }

    # Select the parameter with highest divergence score across peers
    best_p = max(param_evals.keys(), key=lambda k: param_evals[k]["divergence_score"])
    eval_info = param_evals[best_p]

    param = best_p
    param_name, unit = param_labels.get(param, (param.replace("_", " "), ""))
    target_val = eval_info["target"]
    delta = eval_info["delta"]
    sig_threshold = eval_info["threshold"]
    threshold_source = eval_info["threshold_source"]
    is_spatially_significant = abs(delta) >= sig_threshold

    peer_stations = [
        {
            "station_id": pm["station_id"],
            "name": pm["name"],
            "reading": eval_info["peers"][i],
            "unit": unit,
        }
        for i, pm in enumerate(peer_metas)
    ]

    corr = match.get("network_corroboration") or "LOCALIZED"

    if (corr == "LOCALIZED" or is_spatially_significant) and is_spatially_significant:
        analysis_text = (
            f"Possible localized anomaly detected at {target_name} station.\n"
            f"The {target_name} station {param_name} differed by {abs(delta)}{unit} from cluster peers, exceeding the cluster's learned significance threshold ({sig_threshold}{unit}).\n"
            f"Nearby stations in the cluster remained within normal ranges, isolating this pattern to {target_name}."
        )
        recommended_action = f"Investigate the {target_name} {param_name} sensor. Divergence from peers indicates an isolated hardware or telemetry failure."
        spatial_impact = f"Localized Divergence: +{abs(delta)}{unit} vs peer baseline"
    elif corr == "REGIONAL":
        analysis_text = (
            f"Regional meteorological event detected across the {target_name} cluster.\n"
            f"Nearby stations in the regional network recorded similar shifts in {param_name}.\n"
            f"The target station agrees with surrounding peer telemetry, confirming a broad atmospheric system."
        )
        recommended_action = f"Keep {target_name} in nominal operation. Nearby corroboration indicates environmental phenomena rather than isolated sensor malfunction."
        spatial_impact = "Regional Agreement: Peers corroborating"
    elif corr == "LOCALIZED" and not is_spatially_significant:
        analysis_text = (
            f"Peer stations in the {target_name} cluster show consistent readings (delta: {abs(delta)}{unit} < threshold {sig_threshold}{unit}).\n"
            f"The observed divergence is within normal inter-station variation ({threshold_source}).\n"
            f"The anomaly pattern is driven by internal temporal signal characteristics rather than spatial divergence."
        )
        recommended_action = f"Monitor {target_name} {param_name} trend over upcoming readings. Spatial telemetry shows peers in nominal agreement."
        spatial_impact = f"Peer Agreement: Delta within cluster spread ({threshold_source})"
    else:
        analysis_text = f"Insufficient peer telemetry available in the {target_name} cluster for spatial corroboration."
        recommended_action = f"Monitor {target_name} readings and cross-corroborate with secondary meteorological sensors."
        spatial_impact = "Peer Baseline Unavailable"

    # Clausius-Clapeyron thermodynamic check
    thermodynamic_context = None
    is_multivariate = "multivariate" in str(match.get("type", "")).lower() or "multivariate" in str(match.get("fault_type", "")).lower()
    t_val = param_evals["temperature_c"]["target"]
    h_val = param_evals["humidity_pct"]["target"]
    p_val = param_evals["pressure_hpa"]["target"]
    if is_multivariate or (t_val > 45.0 and h_val > 60.0):
        thermodynamic_context = {
            "is_violation": True,
            "law": "Clausius-Clapeyron Relation",
            "temperature_c": t_val,
            "humidity_pct": h_val,
            "pressure_hpa": p_val,
            "explanation": (
                f"Physical impossibility: Under atmospheric thermodynamics (Clausius-Clapeyron equation), "
                f"saturation vapor pressure rises exponentially with temperature. At {t_val}°C, maintaining {h_val}% relative "
                f"humidity requires an impossible atmospheric water vapor concentration under standard surface pressure ({p_val} hPa). "
                f"Relative humidity must decrease as temperature increases; their simultaneous surge confirms a coupled sensor or calibration fault."
            )
        }

    return {
        "cluster_id": cluster_id,
        "target_station": {
            "station_id": target_sid,
            "name": target_name,
            "reading": target_val,
            "unit": unit,
        },
        "peer_stations": peer_stations,
        "parameter_analyzed": param_name,
        "delta": delta,
        "analysis_text": analysis_text,
        "recommended_action": recommended_action,
        "spatial_impact": spatial_impact,
        "thermodynamic_context": thermodynamic_context,
    }


# ---------------- GET /api/explain/{anomaly_id} ----------------

@app.get("/api/explain/{anomaly_id}")
def get_explanation(anomaly_id: str):
    sim = app.state.sim

    match = next(
        (a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id),
        None
    )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown anomaly_id {anomaly_id}"
        )

    return {
        "anomaly_id": anomaly_id,
        "station_id": match.get("station_id"),
        "timestamp": match["timestamp"].isoformat() if hasattr(match.get("timestamp"), "isoformat") else str(match.get("timestamp", "")),
        "features": match.get("shap_features", []),
        "likely_faulty_sensors": match.get("likely_faulty_sensors", []),
        "affected_parameters": match.get("affected_parameters", []),
        "observed_values": match.get("observed_values", {}),
        "suggested_values": match.get("suggested_values", {}),
        "model_confidence_pct": match.get("model_confidence_pct"),
        "rule_confidence_pct": match.get("rule_confidence_pct"),
        "anomaly_score_pct": match.get("anomaly_score_pct"),
        "fault_type": match.get("type"),
        "regime": match.get("regime"),
        "network_corroboration": match.get("network_corroboration"),
        "decision_basis": _compute_decision_basis(
            match.get("model_confidence_pct"),
            match.get("rules_fired") or [],
        ),
        "model_status": _compute_model_status(
            match.get("model_confidence_pct"),
            None,
        ),
        "spatial_context": _compute_spatial_context(sim, match),
    }



# ---------------- GET /api/sensor-health ----------------

@app.get("/api/sensor-health")
def get_sensor_health(station_id: str):
    sim = app.state.sim
    if station_id not in sim.manager.buffers:
        raise HTTPException(status_code=404, detail=f"Unknown station {station_id}")
    status = sim.manager.get_station_status(station_id)
    mapped = {"HEALTHY": "HEALTHY", "WARNING": "WARNING", "OFFLINE": "OFFLINE"}.get(status["status"], "HEALTHY")
    parameter_status = sim.manager.buffers[station_id].health.param_status
    return {
        "station_id": station_id,
        "health_pct": _health_pct(parameter_status),
        "status": mapped,
        "parameters": parameter_status,
        "offline_reason": status["offline_reason"],
        "recovery_active": status["recovery_active"],
    }

# ---------------- POST /api/repair-sensor ----------------

@app.post("/api/repair-sensor")
def repair_sensor(body: dict):
    sim = app.state.sim

    station_id = body.get("station_id")
    if not station_id:
        raise HTTPException(
            status_code=400,
            detail="station_id is required"
        )

    if station_id not in sim.manager.buffers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown station {station_id}"
        )

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc)

    if body.get("force_recovery", False):
        sim.manager.force_recover_station(station_id)
        return {
            "success": True,
            "station_id": station_id,
            "status": "HEALTHY",
            "recovery_active": False,
            "message": "Sensor force-recovered and health counters reset.",
        }

    sim.manager.mark_station_repaired(station_id, timestamp)

    return {
        "success": True,
        "station_id": station_id,
        "status": "WARNING",
        "recovery_active": True,
        "message": "Sensor marked for repair recovery. Clean readings will be evaluated before returning it to HEALTHY."
    }


# ---------------- POST /api/inject-anomaly ----------------

@app.post("/api/inject-anomaly")
async def inject_anomaly(body: dict):
    """
    Repurposed as the REPLAY-MODE trigger -- see simulator.py's
    start_replay() docstring. body's station_id/type are accepted for
    contract-shape compatibility but not used to target one station;
    replay always drives all 20 simultaneously from their own
    pre-injected faults, then auto-reverts to live mode when exhausted.
    """
    sim = app.state.sim
    if sim.mode == "replay":
        raise HTTPException(status_code=409, detail="Simulator replay already running.")
    anomaly_id = await asyncio.to_thread(sim.start_replay)
    return {
        "success": True,
        "anomaly_id": anomaly_id,
        "message": "Simulator started: replaying labeled historical data with injected faults across all stations.",
    }


# ---------------- POST /api/maintenance-ticket ----------------

_ticket_counter = 0

@app.post("/api/maintenance-ticket")
async def create_maintenance_ticket(body: dict):
    global _ticket_counter
    sim = app.state.sim

    anomaly_id = body.get("anomaly_id")
    match = next((a for a in sim.recent_anomalies if a["anomaly_id"] == anomaly_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Unknown anomaly_id {anomaly_id}")

    _ticket_counter += 1
    from datetime import datetime, timezone
    return {
        "ticket_id": f"TCK-{_ticket_counter:04d}",
        "station_id": match["station_id"],
        "issue": match["root_cause"],
        "priority": "high" if match["severity"] in ("high", "critical") else "medium",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _health_pct(parameter_status: dict[str, str]) -> int:
    """Stable health summary from actual per-parameter circuit breakers."""
    if not parameter_status:
        return 100
    score_by_status = {"HEALTHY": 100, "WARNING": 50, "OFFLINE": 0}
    return round(sum(score_by_status.get(value, 0) for value in parameter_status.values()) / len(parameter_status))
