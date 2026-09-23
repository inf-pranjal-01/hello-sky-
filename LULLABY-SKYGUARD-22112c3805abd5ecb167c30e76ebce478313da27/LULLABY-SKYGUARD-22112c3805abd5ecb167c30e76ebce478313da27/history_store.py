"""
SkyGuard AI — history_store.py: persistent, long-horizon sensor history.

Supports dual-store architecture:
1. Primary: TimescaleDB (Tiger Cloud) hypertable `sensor_readings` with native time-series
   indexing, sub-millisecond range queries, and automatic 30-day retention policies.
2. Mirror & Fallback: Local append-only per-station CSV files under DATA_DIR ensuring
   100% offline resilience, instant local audits, and zero downtime if the cloud database
   is temporarily unreachable.
"""

import csv
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data" / "history"
MAX_HISTORY_DAYS = 90
TRIM_CHECK_INTERVAL = 200

RAW_PARAMS = ["temperature_c", "pressure_hpa", "humidity_pct"]

HISTORY_COLUMNS = (
    ["timestamp", "station_id"]
    + RAW_PARAMS
    + ["is_anomaly", "fault_type", "severity", "anomaly_score_pct", "decision_basis"]
    + [f"suggested_{p}" for p in RAW_PARAMS]
    + ["health_status", "source", "model_confidence_pct", "rule_confidence_pct"]
)


class HistoryStore:
    """
    Dual-store architecture: TimescaleDB hypertable primary with local CSV mirror.
    Thread-safe per station with connection pooling for database operations.
    """

    def __init__(self, base_dir: Path = DATA_DIR, max_days: int = MAX_HISTORY_DAYS):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_days = max_days
        self._locks: dict[str, threading.Lock] = {}
        self._append_counts: dict[str, int] = {}

        # TimescaleDB pool initialization
        self.use_db = False
        self._db_pool = None
        self._init_timescale_pool()

    def _init_timescale_pool(self):
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("TIMESCALE_SERVICE_URL")
        if not db_url:
            print("[HistoryStore] No DATABASE_URL/TIMESCALE_SERVICE_URL configured. Using local CSV store.")
            return

        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool

            self._db_pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=db_url)
            # Test connection
            conn = self._db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                conn.commit()
                self.use_db = True
                print("[HistoryStore] Connected to TimescaleDB (Tiger Cloud) successfully.")
            finally:
                self._db_pool.putconn(conn)
        except Exception as e:
            print(f"[HistoryStore] TimescaleDB connection failed: {e!r}. Operating in CSV fallback mode.")
            self.use_db = False
            self._db_pool = None

    @contextmanager
    def _get_db_conn(self):
        """Context manager to acquire and return pooled connections safely."""
        if not self.use_db or not self._db_pool:
            yield None
            return
        conn = None
        try:
            conn = self._db_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            print(f"[HistoryStore DB Error] {e!r}")
        finally:
            if conn and self._db_pool:
                try:
                    self._db_pool.putconn(conn)
                except Exception:
                    pass

    def _path(self, station_id: str) -> Path:
        return self.base_dir / f"{station_id}_history.csv"

    def _lock_for(self, station_id: str) -> threading.Lock:
        return self._locks.setdefault(station_id, threading.Lock())

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read history with one canonical, timezone-aware timestamp type."""
        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        if df.empty or "timestamp" not in df.columns:
            return df
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        for str_col in ["fault_type", "severity", "health_status", "source", "decision_basis"]:
            if str_col in df.columns:
                df[str_col] = df[str_col].astype(object)
        parsed = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if parsed.isna().any():
            df = df.loc[parsed.notna()].copy()
            parsed = parsed.loc[parsed.notna()]
        df["timestamp"] = parsed
        return df

    def append(self, station_id: str, timestamp, raw_reading: dict, verdict: dict, source: str):
        """
        Writes ONE row per ingested reading. Dual-writes to TimescaleDB and CSV mirror.
        """
        suggested = verdict.get("suggested_values", {}) or {}
        ts_obj = pd.Timestamp(timestamp)
        if ts_obj.tzinfo is None:
            ts_obj = ts_obj.tz_localize("UTC")
        ts_iso = ts_obj.isoformat()

        row = {
            "timestamp": ts_iso,
            "station_id": station_id,
            **{p: raw_reading.get(p) for p in RAW_PARAMS},
            "is_anomaly": bool(verdict.get("is_anomaly", False)),
            "fault_type": verdict.get("fault_type"),
            "severity": verdict.get("severity"),
            "anomaly_score_pct": verdict.get("anomaly_score_pct"),
            "decision_basis": verdict.get("decision_basis"),
            **{f"suggested_{p}": suggested.get(p) for p in RAW_PARAMS},
            "health_status": verdict.get("health_status"),
            "source": source,
            "model_confidence_pct": verdict.get("model_confidence_pct"),
            "rule_confidence_pct": verdict.get("rule_confidence_pct"),
        }

        # 1. TimescaleDB Write
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO sensor_readings (
                                    time, station_id, temperature_c, pressure_hpa, humidity_pct,
                                    is_anomaly, fault_type, severity, anomaly_score_pct,
                                    suggested_temperature_c, suggested_pressure_hpa, suggested_humidity_pct,
                                    health_status, source, decision_basis,
                                    model_confidence_pct, rule_confidence_pct
                                ) VALUES (
                                    %s, %s, %s, %s, %s,
                                    %s, %s, %s, %s,
                                    %s, %s, %s,
                                    %s, %s, %s,
                                    %s, %s
                                )
                                ON CONFLICT (station_id, time) DO UPDATE SET
                                    temperature_c = EXCLUDED.temperature_c,
                                    pressure_hpa = EXCLUDED.pressure_hpa,
                                    humidity_pct = EXCLUDED.humidity_pct,
                                    is_anomaly = EXCLUDED.is_anomaly,
                                    fault_type = EXCLUDED.fault_type,
                                    severity = EXCLUDED.severity,
                                    anomaly_score_pct = EXCLUDED.anomaly_score_pct,
                                    suggested_temperature_c = EXCLUDED.suggested_temperature_c,
                                    suggested_pressure_hpa = EXCLUDED.suggested_pressure_hpa,
                                    suggested_humidity_pct = EXCLUDED.suggested_humidity_pct,
                                    health_status = EXCLUDED.health_status,
                                    source = EXCLUDED.source,
                                    decision_basis = EXCLUDED.decision_basis,
                                    model_confidence_pct = EXCLUDED.model_confidence_pct,
                                    rule_confidence_pct = EXCLUDED.rule_confidence_pct;
                                """,
                                (
                                    ts_obj.to_pydatetime(),
                                    station_id,
                                    row["temperature_c"],
                                    row["pressure_hpa"],
                                    row["humidity_pct"],
                                    row["is_anomaly"],
                                    row["fault_type"],
                                    row["severity"],
                                    row["anomaly_score_pct"],
                                    row["suggested_temperature_c"],
                                    row["suggested_pressure_hpa"],
                                    row["suggested_humidity_pct"],
                                    row["health_status"],
                                    row["source"],
                                    row["decision_basis"],
                                    row["model_confidence_pct"],
                                    row["rule_confidence_pct"],
                                ),
                            )
                        conn.commit()
            except Exception as e:
                print(f"[HistoryStore] TimescaleDB insert error: {e!r}")

        # 2. Local CSV Mirror Write
        path = self._path(station_id)
        write_header = not path.exists()
        with self._lock_for(station_id):
            if path.exists():
                existing = pd.read_csv(path, usecols=["timestamp", "source"])
                duplicate = (
                    (existing["timestamp"].astype(str) == row["timestamp"])
                    & (existing["source"].astype(str) == source)
                )
                if duplicate.any():
                    return
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        count = self._append_counts.get(station_id, 0) + 1
        self._append_counts[station_id] = count
        if count % TRIM_CHECK_INTERVAL == 0:
            self.trim(station_id)

    def log_health_transition(self, station_id: str, timestamp, old_state: str, new_state: str, reason: str):
        ts_obj = pd.Timestamp(timestamp)
        if ts_obj.tzinfo is None:
            ts_obj = ts_obj.tz_localize("UTC")

        # Database write
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO station_health_events (time, station_id, old_state, new_state, reason)
                                VALUES (%s, %s, %s, %s, %s);
                                """,
                                (ts_obj.to_pydatetime(), station_id, old_state, new_state, reason),
                            )
                        conn.commit()
            except Exception as e:
                print(f"[HistoryStore] Failed to log health transition to DB: {e!r}")

        # CSV mirror write
        path = self.base_dir / f"{station_id}_health_events.csv"
        write_header = not path.exists()
        row = {
            "timestamp": ts_obj.isoformat(),
            "station_id": station_id,
            "old_state": old_state,
            "new_state": new_state,
            "reason": reason,
        }
        with self._lock_for(station_id):
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "station_id", "old_state", "new_state", "reason"])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    def trim(self, station_id: str):
        """
        Enforces MAX_HISTORY_DAYS independently for each source.
        For TimescaleDB, hypertable retention policy handles auto-drop of old chunks.
        For local CSVs, trims rows older than max_days.
        """
        path = self._path(station_id)
        if not path.exists():
            return
        with self._lock_for(station_id):
            df = self._read_csv(path)
            if df.empty:
                return
            if "source" not in df.columns:
                cutoff = df["timestamp"].max() - pd.Timedelta(days=self.max_days)
                trimmed = df[df["timestamp"] >= cutoff]
            else:
                retained = []
                for _, source_rows in df.groupby("source", dropna=False):
                    cutoff = source_rows["timestamp"].max() - pd.Timedelta(days=self.max_days)
                    retained.append(source_rows[source_rows["timestamp"] >= cutoff])
                trimmed = pd.concat(retained, ignore_index=True) if retained else df.iloc[0:0]
            if len(trimmed) < len(df):
                trimmed.to_csv(path, index=False)

    def mark_spike(self, station_id: str, timestamp, parameter: str, suggested_value: float, source: str):
        """Retroactively annotate the original one-reading spike in both DB and CSV."""
        ts_obj = pd.Timestamp(timestamp)
        if ts_obj.tzinfo is None:
            ts_obj = ts_obj.tz_localize("UTC")

        # Database update
        if self.use_db and parameter in RAW_PARAMS:
            try:
                col_name = f"suggested_{parameter}"
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                f"""
                                UPDATE sensor_readings
                                SET is_anomaly = TRUE, fault_type = 'spike', severity = 'medium', {col_name} = %s
                                WHERE station_id = %s AND time = %s AND source = %s;
                                """,
                                (suggested_value, station_id, ts_obj.to_pydatetime(), source),
                            )
                        conn.commit()
            except Exception as e:
                print(f"[HistoryStore] DB mark_spike error: {e!r}")

        # CSV update
        path = self._path(station_id)
        if not path.exists():
            return
        target = pd.to_datetime(timestamp, utc=True)
        with self._lock_for(station_id):
            df = self._read_csv(path)
            mask = (df["timestamp"] == target) & (df["source"] == source)
            if not mask.any():
                return
            df["is_anomaly"] = df["is_anomaly"].astype(object)
            df["fault_type"] = df["fault_type"].astype(object)
            df["severity"] = df["severity"].astype(object)
            df.loc[mask, "is_anomaly"] = True
            df.loc[mask, "fault_type"] = "spike"
            df.loc[mask, "severity"] = "medium"
            df.loc[mask, f"suggested_{parameter}"] = suggested_value
            df.to_csv(path, index=False)

    def get_recent(
        self,
        station_id: str,
        hours: float = 24,
        relative_to: str = "latest",
        source: str | None = None,
    ) -> pd.DataFrame:
        """
        Returns retained rows for one station as a DataFrame matching HISTORY_COLUMNS.
        Queries TimescaleDB hypertable first; seamlessly falls back to CSV mirror if offline.
        """
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            # 1. Determine anchor timestamp
                            if relative_to == "latest":
                                if source:
                                    cur.execute(
                                        "SELECT MAX(time) FROM sensor_readings WHERE station_id = %s AND source = %s;",
                                        (station_id, source),
                                    )
                                else:
                                    cur.execute(
                                        "SELECT MAX(time) FROM sensor_readings WHERE station_id = %s;",
                                        (station_id,),
                                    )
                                anchor = cur.fetchone()[0]
                            else:
                                anchor = datetime.now(timezone.utc)

                            if anchor is not None:
                                cutoff = anchor - pd.Timedelta(hours=hours)
                                if source:
                                    cur.execute(
                                        """
                                        SELECT
                                            time as timestamp,
                                            station_id,
                                            temperature_c,
                                            pressure_hpa,
                                            humidity_pct,
                                            is_anomaly,
                                            fault_type,
                                            severity,
                                            anomaly_score_pct,
                                            decision_basis,
                                            suggested_temperature_c,
                                            suggested_pressure_hpa,
                                            suggested_humidity_pct,
                                            health_status,
                                            source,
                                            model_confidence_pct,
                                            rule_confidence_pct
                                        FROM sensor_readings
                                        WHERE station_id = %s AND source = %s AND time >= %s
                                        ORDER BY time ASC;
                                        """,
                                        (station_id, source, cutoff),
                                    )
                                else:
                                    cur.execute(
                                        """
                                        SELECT
                                            time as timestamp,
                                            station_id,
                                            temperature_c,
                                            pressure_hpa,
                                            humidity_pct,
                                            is_anomaly,
                                            fault_type,
                                            severity,
                                            anomaly_score_pct,
                                            decision_basis,
                                            suggested_temperature_c,
                                            suggested_pressure_hpa,
                                            suggested_humidity_pct,
                                            health_status,
                                            source,
                                            model_confidence_pct,
                                            rule_confidence_pct
                                        FROM sensor_readings
                                        WHERE station_id = %s AND time >= %s
                                        ORDER BY time ASC;
                                        """,
                                        (station_id, cutoff),
                                    )
                                rows = cur.fetchall()
                                if rows:
                                    colnames = [desc[0] for desc in cur.description]
                                    df = pd.DataFrame(rows, columns=colnames)
                                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                                    return df
            except Exception as e:
                print(f"[HistoryStore] DB get_recent query failed: {e!r}. Using CSV mirror.")

        # Fallback to CSV
        path = self._path(station_id)
        if not path.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        df = self._read_csv(path)
        if source is not None and "source" in df.columns:
            df = df[df["source"] == source]
        if df.empty:
            return df
        anchor = df["timestamp"].max() if relative_to == "latest" else pd.Timestamp.now(tz="UTC")
        cutoff = anchor - pd.Timedelta(hours=hours)
        return df[df["timestamp"] >= cutoff].reset_index(drop=True)

    def get_all(self, station_id: str) -> pd.DataFrame:
        """Full retained window for one station."""
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                SELECT
                                    time as timestamp,
                                    station_id,
                                    temperature_c,
                                    pressure_hpa,
                                    humidity_pct,
                                    is_anomaly,
                                    fault_type,
                                    severity,
                                    anomaly_score_pct,
                                    decision_basis,
                                    suggested_temperature_c,
                                    suggested_pressure_hpa,
                                    suggested_humidity_pct,
                                    health_status,
                                    source,
                                    model_confidence_pct,
                                    rule_confidence_pct
                                FROM sensor_readings
                                WHERE station_id = %s
                                ORDER BY time ASC;
                                """,
                                (station_id,),
                            )
                            rows = cur.fetchall()
                            if rows:
                                colnames = [desc[0] for desc in cur.description]
                                df = pd.DataFrame(rows, columns=colnames)
                                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                                return df
            except Exception as e:
                print(f"[HistoryStore] DB get_all failed: {e!r}. Using CSV fallback.")

        path = self._path(station_id)
        if not path.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return self._read_csv(path)

    def clear_source(self, station_id: str, source: str, sync_db: bool = True):
        """Purges every row tagged with source in DB and CSV."""
        if sync_db and self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "DELETE FROM sensor_readings WHERE station_id = %s AND source = %s;",
                                (station_id, source),
                            )
                        conn.commit()
            except Exception as e:
                print(f"[HistoryStore] DB clear_source error: {e!r}")

        path = self._path(station_id)
        if not path.exists():
            return
        with self._lock_for(station_id):
            df = self._read_csv(path)
            if df.empty or "source" not in df.columns:
                return
            remaining = df[df["source"] != source]
            if len(remaining) == len(df):
                return
            if remaining.empty:
                path.unlink()
            else:
                remaining.to_csv(path, index=False)

    def clear_all(self, source: str = None):
        """Purges source rows or truncates the store across all stations."""
        if self.use_db:
            try:
                with self._get_db_conn() as conn:
                    if conn:
                        with conn.cursor() as cur:
                            if source is None:
                                cur.execute("TRUNCATE TABLE sensor_readings;")
                            else:
                                cur.execute("DELETE FROM sensor_readings WHERE source = %s;", (source,))
                        conn.commit()
            except Exception as e:
                print(f"[HistoryStore] DB clear_all error: {e!r}")

        for path in self.base_dir.glob("*_history.csv"):
            station_id = path.stem[: -len("_history")] if path.stem.endswith("_history") else path.stem
            if source is None:
                with self._lock_for(station_id):
                    path.unlink(missing_ok=True)
            else:
                self.clear_source(station_id, source, sync_db=False)
