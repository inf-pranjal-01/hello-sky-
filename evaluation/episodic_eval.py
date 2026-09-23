"""
evaluation/episodic_eval.py
===========================
Latency-Aware Episodic Evaluation for SkyGuard AI offline benchmarking.

IMPORTANT — READ BEFORE MODIFYING
----------------------------------
This module changes ONLY how offline metrics are calculated.
It does NOT modify:
  - the Isolation Forest model or any model logic
  - the rule engine or any thresholds
  - the feature engineering pipeline
  - raw prediction arrays
  - anomaly flags written to the dashboard

The raw predictions produced by the detector are consumed here
read-only and are never altered.

METHODOLOGY & EXACT CREDIT RULES
--------------------------------
1. Ground-truth fault rows are grouped into *episodes* — maximal
   contiguous sequences of anomalous readings for the same station
   and fault_type. Two episodes of the same type on the same station
   are always kept separate if there is any normal reading between them.

2. Detection vs Attribution:
   - Detection: An episode is considered DETECTED if the model produces
     at least one positive anomaly prediction (prediction == True)
     inside that ground-truth fault episode.
   - Attribution: Evaluated separately. An episode's attribution is correct
     if the predicted fault type matches the ground-truth fault type.
     Wrong attribution does NOT turn a detected episode into a missed detection.

3. Latency Credit:
   - APPLIES ONLY TO `frozen` AND `drift` (LATENCY_CREDIT_FAULT_TYPES).
   - For detected frozen/drift episodes:
       * Pre-first-detection rows receive lag credit -> counted as TP*
       * Rows at or after the first detection evaluate actual model predictions
         (pred == True -> TP*, pred == False -> FN*)
   - For completely missed frozen/drift episodes:
       * All rows remain FN* (no latency credit).
   - For all other fault types (spike, dropout, bias, noise, unstructured_anomaly, etc.):
       * NO latency credit is applied.
       * All rows evaluate strictly as raw point-wise predictions
         (pred == True -> TP*, pred == False -> FN*).

4. Precision & False Positives:
   - Standard point-wise precision is UNCHANGED: Precision = TP / (TP + FP).
   - Latency credit NEVER forgives or alters false positives.
   - F1* = 2 * Precision * Recall* / (Precision + Recall*).

5. Invariants Strictly Preserved:
   - TP* + FN* == total ground-truth faulty readings
   - FP is byte/value identical to raw FP
   - Raw model predictions are NEVER modified
   - Credit NEVER crosses episode boundaries or recovery boundaries
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# Explicit non-negotiable definition: latency credit applies ONLY to these types
LATENCY_CREDIT_FAULT_TYPES: Set[str] = {"frozen", "frozen_value", "drift"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FaultEpisode:
    """One contiguous block of ground-truth anomalous readings."""
    station_id: str
    fault_type: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    row_indices: List[int]          # positional indices into the eval dataframe

    # Detection status
    detected: bool = False
    first_detection_idx: Optional[int] = None       # index in df of first positive prediction
    first_detection_ts: Optional[pd.Timestamp] = None
    latency_hours: Optional[float] = None           # None if missed or no timestamps

    # Attribution status
    attribution_correct: bool = False
    predicted_fault_type_at_first_detection: Optional[str] = None

    # Latency-aware row counts (calculated by evaluate_episodes)
    tp_star: int = 0
    fn_star: int = 0
    raw_tp: int = 0
    raw_fn: int = 0

    @property
    def length_rows(self) -> int:
        return len(self.row_indices)

    @property
    def allows_latency_credit(self) -> bool:
        """Latency credit applies ONLY to frozen and drift fault types."""
        ft = self.fault_type.lower()
        return ft in LATENCY_CREDIT_FAULT_TYPES or ft.startswith("frozen") or ft == "drift"


@dataclass
class EpisodicResult:
    """Full evaluation result containing Point-wise, Latency-Aware, and Event metrics."""
    episodes: List[FaultEpisode] = field(default_factory=list)

    # Raw point-wise baseline metrics (unmodified)
    raw_tp: int = 0
    raw_fp: int = 0
    raw_fn: int = 0
    raw_precision: float = 0.0
    raw_recall: float = 0.0
    raw_f1: float = 0.0

    # ---- Latency-Aware Metrics (Recall*, F1*, TP*, FN*) ----
    @property
    def tp_star(self) -> int:
        """Latency-adjusted TP* (raw TP + pre-first-detection lag rows for frozen/drift)."""
        return sum(e.tp_star for e in self.episodes)

    @property
    def fn_star(self) -> int:
        """Latency-adjusted FN*."""
        return sum(e.fn_star for e in self.episodes)

    @property
    def latency_aware_recall(self) -> float:
        """Recall* = TP* / (TP* + FN*)."""
        denom = self.tp_star + self.fn_star
        if denom == 0:
            return float("nan")
        return self.tp_star / denom

    @property
    def latency_aware_f1(self) -> float:
        """F1* = 2 * Precision * Recall* / (Precision + Recall*). Uses standard precision."""
        p = self.raw_precision
        r = self.latency_aware_recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    # ---- Event-Level Metrics ----
    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def detected_episodes(self) -> int:
        return sum(1 for e in self.episodes if e.detected)

    @property
    def missed_episodes(self) -> int:
        return sum(1 for e in self.episodes if not e.detected)

    @property
    def episode_detection_rate(self) -> float:
        """Episode Detection Rate = detected episodes / total episodes."""
        if self.total_episodes == 0:
            return float("nan")
        return self.detected_episodes / self.total_episodes

    # ---- Latency ----
    @property
    def mean_latency_hours(self) -> Optional[float]:
        lats = [e.latency_hours for e in self.episodes
                if e.detected and e.latency_hours is not None]
        return float(np.mean(lats)) if lats else None

    @property
    def median_latency_hours(self) -> Optional[float]:
        lats = [e.latency_hours for e in self.episodes
                if e.detected and e.latency_hours is not None]
        return float(np.median(lats)) if lats else None

    # ---- Attribution Breakdown ----
    @property
    def correctly_attributed_episodes(self) -> int:
        return sum(1 for e in self.episodes if e.detected and e.attribution_correct)

    @property
    def wrongly_attributed_episodes(self) -> int:
        return sum(1 for e in self.episodes if e.detected and not e.attribution_correct and e.predicted_fault_type_at_first_detection not in (None, "none", "unstructured_anomaly", "UNKNOWN_STATISTICAL_ANOMALY"))

    @property
    def unattributed_episodes(self) -> int:
        return sum(1 for e in self.episodes if e.detected and not e.attribution_correct and e.predicted_fault_type_at_first_detection in (None, "none", "unstructured_anomaly", "UNKNOWN_STATISTICAL_ANOMALY"))

    def by_fault_type(self) -> Dict[str, dict]:
        """Per-fault-type breakdown of episode counts, point-wise rows, and latency-aware rows."""
        out: Dict[str, dict] = {}
        for ep in self.episodes:
            ft = ep.fault_type
            if ft not in out:
                out[ft] = {
                    "total_ep": 0, "detected_ep": 0, "missed_ep": 0,
                    "gt_rows": 0, "raw_tp": 0, "raw_fn": 0,
                    "tp_star": 0, "fn_star": 0,
                    "allows_lag_credit": ep.allows_latency_credit,
                }
            out[ft]["total_ep"] += 1
            out[ft]["gt_rows"] += ep.length_rows
            out[ft]["raw_tp"] += ep.raw_tp
            out[ft]["raw_fn"] += ep.raw_fn
            out[ft]["tp_star"] += ep.tp_star
            out[ft]["fn_star"] += ep.fn_star
            if ep.detected:
                out[ft]["detected_ep"] += 1
            else:
                out[ft]["missed_ep"] += 1
        return out


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_gt_episodes(
    df: pd.DataFrame,
    gt_col: str = "is_anomaly",
    ft_col: str = "fault_type",
    station_col: str = "station_id",
    ts_col: str = "timestamp",
) -> List[FaultEpisode]:
    """
    Group consecutive ground-truth anomalous rows into FaultEpisodes.

    Rules
    -----
    * Episodes are scoped by (station_id, fault_type).
    * Any normal reading (gt=False) between two anomalous readings of
      the same type on the same station BREAKS the episode — they are
      two separate episodes.
    * df is expected to be sorted by (station_id, timestamp).
    """
    episodes: List[FaultEpisode] = []

    gt_arr  = df[gt_col].fillna(False).astype(bool).to_numpy()
    ft_arr  = df[ft_col].fillna("none").to_numpy()
    sid_arr = df[station_col].to_numpy()
    ts_arr  = pd.to_datetime(df[ts_col]).to_numpy()

    n = len(df)
    i = 0
    while i < n:
        if not gt_arr[i]:
            i += 1
            continue

        # Start of a new episode
        sid0 = sid_arr[i]
        ft0  = ft_arr[i]
        run  = [i]
        j    = i + 1

        while j < n and gt_arr[j] and sid_arr[j] == sid0 and ft_arr[j] == ft0:
            run.append(j)
            j += 1

        episodes.append(FaultEpisode(
            station_id=str(sid0),
            fault_type=str(ft0),
            start_ts=pd.Timestamp(ts_arr[run[0]]),
            end_ts=pd.Timestamp(ts_arr[run[-1]]),
            row_indices=run,
        ))
        i = j   # advance past this episode

    return episodes


def evaluate_episodes(
    episodes: List[FaultEpisode],
    pred_arr: np.ndarray,
    pred_ft_arr: np.ndarray,
    ts_arr: Optional[np.ndarray] = None,
) -> None:
    """
    For each episode, evaluate detection, latency, attribution, and latency-aware row credit.

    CRITICAL INVARIANTS
    -------------------
    * pred_arr is consumed READ-ONLY — never modified.
    * Detection is positive if ANY prediction inside the episode is True.
      Fault-type match is NOT required for detection.
    * Attribution is evaluated separately.
    * Latency credit applies ONLY to frozen and drift:
        - Before first detection: lag credit (TP*)
        - At or after first detection: actual model predictions
    * For all other fault types: strictly actual model predictions (NO lag credit).
    """
    for ep in episodes:
        # 1. Raw point-wise counts for this episode
        ep.raw_tp = sum(1 for idx in ep.row_indices if pred_arr[idx])
        ep.raw_fn = ep.length_rows - ep.raw_tp

        # 2. Find first positive anomaly detection inside episode
        first_det_local_idx = None
        for local_pos, idx in enumerate(ep.row_indices):
            if pred_arr[idx]:
                first_det_local_idx = local_pos
                ep.first_detection_idx = idx
                if ts_arr is not None:
                    ep.first_detection_ts = pd.Timestamp(ts_arr[idx])
                    delta = ep.first_detection_ts - ep.start_ts
                    ep.latency_hours = max(0.0, delta.total_seconds() / 3600.0)
                ep.predicted_fault_type_at_first_detection = str(pred_ft_arr[idx]) if pred_ft_arr is not None else None
                # Check attribution
                ep_ft_norm = ep.fault_type.lower()
                pred_ft_norm = str(pred_ft_arr[idx]).lower() if pred_ft_arr is not None else ""
                ep.attribution_correct = (
                    pred_ft_norm == ep_ft_norm or
                    (ep_ft_norm.startswith("frozen") and pred_ft_norm.startswith("frozen"))
                )
                break

        ep.detected = (first_det_local_idx is not None)

        # 3. Calculate latency-aware TP* and FN*
        if ep.allows_latency_credit:
            if ep.detected:
                assert first_det_local_idx is not None
                # Pre-first-detection rows: lag credit (TP*)
                pre_det_count = first_det_local_idx
                # Post/at-first-detection rows: actual model predictions
                post_det_tps = sum(1 for idx in ep.row_indices[first_det_local_idx:] if pred_arr[idx])
                post_det_fns = sum(1 for idx in ep.row_indices[first_det_local_idx:] if not pred_arr[idx])

                ep.tp_star = pre_det_count + post_det_tps
                ep.fn_star = post_det_fns
            else:
                # Completely missed frozen/drift episode: NO latency credit
                ep.tp_star = 0
                ep.fn_star = ep.length_rows
        else:
            # All other fault types (spike, dropout, bias, noise, unstructured_anomaly, etc.):
            # NO latency credit — strictly actual model predictions
            ep.tp_star = ep.raw_tp
            ep.fn_star = ep.raw_fn

        # Strict sanity invariant per episode
        assert ep.tp_star + ep.fn_star == ep.length_rows, (
            f"Episode invariant violation: tp_star ({ep.tp_star}) + fn_star ({ep.fn_star}) "
            f"!= length_rows ({ep.length_rows}) for {ep.fault_type} episode at {ep.station_id}"
        )


def compute_episodic_result(
    df: pd.DataFrame,
    pred_arr: np.ndarray,
    pred_ft_arr: np.ndarray,
    gt_col: str = "is_anomaly",
    ft_col: str = "fault_type",
    station_col: str = "station_id",
    ts_col: str = "timestamp",
    point_tp: Optional[int] = None,
    point_fp: Optional[int] = None,
    point_fn: Optional[int] = None,
    point_prec: Optional[float] = None,
    point_rec: Optional[float] = None,
    point_f1: Optional[float] = None,
) -> EpisodicResult:
    """
    Full pipeline: compute raw metrics -> extract episodes -> evaluate latency credit -> return EpisodicResult.

    Raw predictions (pred_arr, pred_ft_arr) are NEVER modified.
    """
    df_sorted = df.sort_values([station_col, ts_col]).reset_index(drop=True)
    original_index = df.sort_values([station_col, ts_col]).index
    pred_sorted    = pred_arr[original_index] if hasattr(pred_arr, '__getitem__') else pred_arr
    pred_ft_sorted = pred_ft_arr[original_index] if hasattr(pred_ft_arr, '__getitem__') else pred_ft_arr
    ts_sorted      = pd.to_datetime(df_sorted[ts_col]).to_numpy()
    gt_sorted      = df_sorted[gt_col].fillna(False).astype(bool).to_numpy()

    # Raw point-wise metrics
    if point_tp is None or point_fp is None or point_fn is None:
        raw_tp = int((pred_sorted & gt_sorted).sum())
        raw_fp = int((pred_sorted & ~gt_sorted).sum())
        raw_fn = int((~pred_sorted & gt_sorted).sum())
        raw_prec = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else float("nan")
        raw_rec  = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else float("nan")
        raw_f1_val = 2 * raw_prec * raw_rec / (raw_prec + raw_rec) if (raw_prec + raw_rec) > 0 else float("nan")
    else:
        raw_tp = point_tp
        raw_fp = point_fp
        raw_fn = point_fn
        raw_prec = point_prec if point_prec is not None else (raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0.0)
        raw_rec  = point_rec if point_rec is not None else (raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0.0)
        raw_f1_val = point_f1 if point_f1 is not None else (2 * raw_prec * raw_rec / (raw_prec + raw_rec) if (raw_prec + raw_rec) > 0 else 0.0)

    episodes = extract_gt_episodes(df_sorted, gt_col, ft_col, station_col, ts_col)
    evaluate_episodes(episodes, pred_sorted, pred_ft_sorted, ts_sorted)

    result = EpisodicResult(
        episodes=episodes,
        raw_tp=raw_tp,
        raw_fp=raw_fp,
        raw_fn=raw_fn,
        raw_precision=raw_prec,
        raw_recall=raw_rec,
        raw_f1=raw_f1_val,
    )

    # Sanity invariant check
    total_gt_rows = int(gt_sorted.sum())
    assert result.tp_star + result.fn_star == total_gt_rows, (
        f"Invariant violation: tp_star ({result.tp_star}) + fn_star ({result.fn_star}) "
        f"!= total_gt_rows ({total_gt_rows})"
    )

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

EPISODIC_FOOTNOTE = (
    "*Latency-Aware Recall applies detection-lag credit only to persistent `frozen` "
    "and `drift` fault episodes. For a detected frozen/drift episode, faulty readings "
    "occurring before the first valid detection receive lag credit; readings at or after "
    "the first detection are evaluated using the actual model predictions. Other fault types "
    "receive no latency credit. Completely missed episodes receive no lag credit. Raw model "
    "predictions and standard point-wise precision remain unchanged."
)


def print_episodic_report(
    result: EpisodicResult,
    label: str = "",
    point_tp: Optional[int] = None,
    point_fp: Optional[int] = None,
    point_fn: Optional[int] = None,
    point_prec: Optional[float] = None,
    point_rec: Optional[float] = None,
    point_f1: Optional[float] = None,
) -> None:
    """
    Print the complete offline evaluation report strictly matching the required structure.
    """
    sep = "=" * 80
    def _fmt(v, pct=False, dp=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        if pct:
            return f"{v:.{dp}%}"
        return f"{v:.{dp}f}"

    raw_p = point_prec if point_prec is not None else result.raw_precision
    raw_r = point_rec if point_rec is not None else result.raw_recall
    raw_f = point_f1 if point_f1 is not None else result.raw_f1
    raw_t = point_tp if point_tp is not None else result.raw_tp
    raw_fp_val = point_fp if point_fp is not None else result.raw_fp
    raw_fn_val = point_fn if point_fn is not None else result.raw_fn

    rec_star = result.latency_aware_recall
    f1_star = result.latency_aware_f1
    t_star = result.tp_star
    fn_star_val = result.fn_star
    ep_rate = result.episode_detection_rate

    print(f"\n{sep}")
    print("POINT-WISE DETECTION")
    print(sep)
    print(f"Precision: {_fmt(raw_p, pct=True)}")
    print(f"Recall:    {_fmt(raw_r, pct=True)}")
    print(f"F1:        {_fmt(raw_f, dp=3)}")
    print(f"TP:        {raw_t}")
    print(f"FP:        {raw_fp_val}")
    print(f"FN:        {raw_fn_val}")

    print(f"\n{sep}")
    print("LATENCY-AWARE DETECTION*")
    print(sep)
    print(f"Precision: {_fmt(raw_p, pct=True)}   <-- SAME POINT-WISE PRECISION")
    print(f"Recall*:   {_fmt(rec_star, pct=True)}")
    print(f"F1*:       {_fmt(f1_star, dp=3)}")
    print(f"TP*:       {t_star}   (+{t_star - raw_t} lag credit rows from frozen/drift)")
    print(f"FP:        {raw_fp_val}")
    print(f"FN*:       {fn_star_val}")

    print(f"\n{sep}")
    print("FAULT EVENT DETECTION")
    print(sep)
    print(f"Total fault episodes:   {result.total_episodes}")
    print(f"Detected episodes:      {result.detected_episodes}")
    print(f"Missed episodes:        {result.missed_episodes}")
    print(f"Episode Detection Rate: {_fmt(ep_rate, pct=True)}")
    mean_lat = result.mean_latency_hours
    med_lat  = result.median_latency_hours
    print(f"Mean detection latency:   {(_fmt(mean_lat) + ' h') if mean_lat is not None else 'N/A'}")
    print(f"Median detection latency: {(_fmt(med_lat) + ' h') if med_lat is not None else 'N/A'}")

    print(f"\n{sep}")
    print("FAULT ATTRIBUTION")
    print(sep)
    print(f"Correctly attributed detections: {result.correctly_attributed_episodes}")
    print(f"Wrongly attributed detections:   {result.wrongly_attributed_episodes}")
    print(f"Unattributed detections:         {result.unattributed_episodes}")

    print(f"\n{sep}")
    print("PER-FAULT-TYPE BREAKDOWN")
    print(sep)
    print(f"  {'Fault Type':<26} {'Lag Credit?':<12} {'Total Ep':>9} {'Caught':>8} {'Ep.Rate':>10} {'GT Rows':>9} {'Raw TP':>8} {'TP*':>8}")
    print(f"  {'-'*26} {'-'*12} {'-'*9} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*8}")
    by_ft = result.by_fault_type()
    for ft, counts in sorted(by_ft.items()):
        ep_rec_ft = counts["detected_ep"] / counts["total_ep"] if counts["total_ep"] > 0 else float("nan")
        lag_flag = "YES" if counts["allows_lag_credit"] else "NO"
        print(f"  {ft:<26} {lag_flag:<12} {counts['total_ep']:>9} {counts['detected_ep']:>8} "
              f"{_fmt(ep_rec_ft, pct=True, dp=1):>10} {counts['gt_rows']:>9} {counts['raw_tp']:>8} {counts['tp_star']:>8}")

    print(f"\n{EPISODIC_FOOTNOTE}")
    print(sep + "\n")
