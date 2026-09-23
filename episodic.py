def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _row_metrics(frame: pd.DataFrame) -> dict:
    truth = frame["is_anomaly_gt"].fillna(False).astype(bool)
    predicted = frame["is_anomaly_pred"].fillna(False).astype(bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall)}


def _episodes(frame: pd.DataFrame, positive_col: str, type_col: str, label: str) -> list[dict]:
    """Build episodes from contiguous positive rows, scoped by station and fault type."""
    episodes = []
    for station, group in frame.groupby("station_id", dropna=False):
        group = group.sort_values("timestamp")
        active = group[positive_col].fillna(False).astype(bool)
        starts = active & ~active.shift(1, fill_value=False)
        ends = active & ~active.shift(-1, fill_value=False)
        for start_idx, end_idx in zip(group.index[starts], group.index[ends]):
            episodes.append({"station_id": station, "fault_type": frame.at[start_idx, type_col],
                             "start": frame.at[start_idx, "timestamp"],
                             "end": frame.at[end_idx, "timestamp"], "label": label})
    return episodes


def episodic_metrics(frame: pd.DataFrame, fault_type: str) -> dict:
    """Overlap-match predicted and labeled frozen/drift episodes per station.

    Current replay labels do not retain the injected parameter when two
    independent faults overlap. Matching is therefore station+fault-type
    scoped until the labeled format carries parameter-level event IDs.
    """
    subset = frame.copy()
    subset["gt_episode"] = subset["fault_type_gt"].eq(fault_type)
    subset["pred_episode"] = subset["is_anomaly_pred"].fillna(False).astype(bool) & subset["fault_type_pred"].eq(fault_type)
    gt = _episodes(subset, "gt_episode", "fault_type_gt", "gt")
    pred = _episodes(subset, "pred_episode", "fault_type_pred", "pred")
    matched_gt, matched_pred = set(), set()
    for pi, pe in enumerate(pred):
        for gi, ge in enumerate(gt):
            if (pe["station_id"] == ge["station_id"] and
                    max(pe["start"], ge["start"]) <= min(pe["end"], ge["end"])):
                matched_pred.add(pi)
                matched_gt.add(gi)
    tp, fp, fn = len(matched_gt), len(pred) - len(matched_pred), len(gt) - len(matched_gt)
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "gt_episodes": len(gt), "pred_episodes": len(pred),
            "precision": precision, "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall)}

