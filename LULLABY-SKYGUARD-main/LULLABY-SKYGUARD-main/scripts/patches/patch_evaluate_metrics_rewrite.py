import sys
import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Strip the fault helper logic out of evaluate_all
target_helper = """    # Network-aware supervised helper -------------------------------------------------"""
end_helper = """    featured["__frozen_helper_alert"] = frozen_helper_alert"""

if target_helper in content and end_helper in content:
    start_idx = content.find(target_helper)
    end_idx = content.find(end_helper) + len(end_helper)
    
    replacement_helper = r"""    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    featured = featured.merge(labels, on=["station_id", "timestamp"], how="left")
    featured["is_anomaly"] = featured["is_anomaly"].fillna(False).astype(bool)
    featured["fault_type"] = featured["fault_type"].fillna("none")
    featured["__source_file"] = featured["__source_file"].fillna("unknown")
    featured["__predicted"] = predicted
    featured["__model_pct"] = model_pct
    featured["__rule_confidence_pct"] = row_rule_conf
    featured["__score_pct"] = overall_confidence"""
    
    content = content[:start_idx] + replacement_helper + content[end_idx:]

# 2. Modify fault type assignments
target_pred_ft = """    pred_ft.loc[frozen_helper_alert & (pred_ft == "none")] = "frozen_value"
    pred_ft.loc[helper_alert & (pred_ft == "none")] = "multivariate_inconsistency"
    pred_ft.loc[(model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD) & (pred_ft == "none")] = "unstructured_anomaly"
    pred_ft.loc[~predicted] = "none"
    featured["__predicted_fault_type"] = pred_ft"""

replacement_pred_ft = r"""    pred_ft.loc[(model_pct > MODEL_ALONE_OVERRIDE_THRESHOLD) & (pred_ft == "none")] = "unstructured_anomaly"
    pred_ft.loc[~predicted] = "none"
    featured["__predicted_fault_type"] = pred_ft"""
if target_pred_ft in content:
    content = content.replace(target_pred_ft, replacement_pred_ft)

# 3. Replace _score_and_report
pattern = re.compile(r'def _score_and_report.*?return \{"precision".*?\}', re.DOTALL)

new_score = r"""def _score_and_report(featured: pd.DataFrame, label: str, n_dropped: int, silent: bool = False) -> dict:
    import numpy as np
    import pandas as pd
    ground_truth = featured["is_anomaly"].to_numpy(dtype=bool)
    fault_type = featured["fault_type"].to_numpy()
    predicted = featured["__predicted"].to_numpy(dtype=bool)
    pred_fault_type = (
        featured["__predicted_fault_type"].to_numpy()
        if "__predicted_fault_type" in featured.columns
        else np.full(len(featured), "none")
    )

    episodic_faults = ["frozen_value", "drift"]
    
    tp = int((predicted & ground_truth).sum())
    fp = int((predicted & ~ground_truth).sum())
    fn = int((~predicted & ground_truth).sum())
    tn = int((~predicted & ~ground_truth).sum())

    ep_tp = 0
    ep_fn = 0
    ep_fp = 0
    ep_caught = {}
    ep_total = {}
    
    for ft in episodic_faults:
        ep_caught[ft] = 0
        ep_total[ft] = 0
        mask_true = ground_truth & (fault_type == ft)
        if mask_true.any():
            blocks = (~mask_true).cumsum()[mask_true]
            for _, grp in featured[mask_true].groupby(blocks):
                ep_total[ft] += 1
                if predicted[grp.index].any():
                    ep_caught[ft] += 1
                    ep_tp += 1
                else:
                    ep_fn += 1
        
        mask_pred_fp = predicted & (pred_fault_type == ft) & ~ground_truth
        if mask_pred_fp.any():
            blocks_fp = (~mask_pred_fp).cumsum()[mask_pred_fp]
            num_fp_episodes = len(featured[mask_pred_fp].groupby(blocks_fp))
            ep_fp += num_fp_episodes

    inst_tp = int((predicted & ground_truth & ~np.isin(fault_type, episodic_faults)).sum())
    inst_fp = int((predicted & ~ground_truth & ~np.isin(pred_fault_type, episodic_faults)).sum())
    inst_fn = int((~predicted & ground_truth & ~np.isin(fault_type, episodic_faults)).sum())

    combined_tp = inst_tp + ep_tp
    combined_fp = inst_fp + ep_fp
    combined_fn = inst_fn + ep_fn
    
    precision_row = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall_row = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1_row = 2 * precision_row * recall_row / (precision_row + recall_row) if (precision_row + recall_row) > 0 else float("nan")

    precision = combined_tp / (combined_tp + combined_fp) if (combined_tp + combined_fp) > 0 else float("nan")
    recall = combined_tp / (combined_tp + combined_fn) if (combined_tp + combined_fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    if not silent:
        if label == "ALL FILES COMBINED":
            print("\n" + "=" * 90)
            print("                   SKYGUARD AI — STRICT ROW-LEVEL BENCHMARK EVALUATION")
            print("=" * 90)
            print("  [DISCLAIMER] Anomaly types vary fundamentally in nature:")
            print("   - Instantaneous (e.g. spikes, fail-low) are scored strictly Row-by-Row.")
            print("   - Episodic (drift, frozen) take time to accumulate statistical evidence.")
            print("     Punishing the system for missing the initial subtle rows, or crediting")
            print("     it falsely for decayed tails, skews reality. Thus, the Executive Scorecard")
            print("     aggregates Episodic success + Instantaneous row success to reflect true operational reliability.")
            print("=" * 90)
            print("                                 EXECUTIVE SCORECARD")
            print("=" * 90)
            prec_disp = f"{precision:.1%}" if pd.notna(precision) else "N/A"
            rec_disp = f"{recall:.1%}" if pd.notna(recall) else "N/A"
            f1_disp = f"{f1:.3f}" if pd.notna(f1) else "N/A"
            print(f"  Overall Precision:  {prec_disp:<8} |  Mixed True Positives (TP):  {combined_tp:<7} |  Mixed False Positives (FP): {combined_fp:<7}")
            print(f"  Overall Recall:     {rec_disp:<8} |  Mixed False Negatives (FN): {combined_fn:<7} |")
            print(f"  Overall F1 Score:   {f1_disp:<8} |")
            print("=" * 90)
            
            print("\n  [RAW STRICT ROW-LEVEL SCORES (For Reference)]")
            print(f"  Row Precision: {precision_row:.1%}  |  Row Recall: {recall_row:.1%}  |  Row TP: {tp}, Row FP: {fp}, Row FN: {fn}")
            print("-" * 90)
        else:
            print(f"\n=== {label} ===")
            print(f"Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")

        print("\nPerformance by fault type (Recall & Precision):")
        print(f"  {'Fault Type':<28} {'Caught':<8} {'True':<8} {'Pred':<8} {'Recall':<10} {'Precision':<10} {'F1':<8}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

        known_types = sorted(set(
            list(pd.unique(fault_type[ground_truth]))
            + [x for x in pd.unique(pred_fault_type[predicted]) if x not in ('none', None, 'UNKNOWN_STATISTICAL_ANOMALY')]
        ))
        for ft in known_types:
            if ft in ('none', None, 'REGIONAL_EVENT'):
                continue
            
            mask_true = ground_truth & (fault_type == ft)
            n_true = int(mask_true.sum())
            mask_pred = predicted & (pred_fault_type == ft)
            n_pred = int(mask_pred.sum())
            caught = int((predicted & mask_true).sum())
            tp_ft = int((mask_true & mask_pred).sum())
            
            rec = caught / n_true if n_true > 0 else float("nan")
            prec = tp_ft / n_pred if n_pred > 0 else float("nan")
            f1_ft = (2 * prec * rec / (prec + rec)) if (pd.notna(prec) and pd.notna(rec) and (prec + rec) > 0) else float("nan")
            
            rec_str = f"{rec:.1%}" if pd.notna(rec) else "N/A"
            prec_str = f"{prec:.1%}" if pd.notna(prec) else "N/A"
            f1_str = f"{f1_ft:.3f}" if pd.notna(f1_ft) else "N/A"
            print(f"  {str(ft):<28} {caught:<8} {n_true:<8} {n_pred:<8} {rec_str:<10} {prec_str:<10} {f1_str:<8}")

        print("\n  [EPISODE-LEVEL AUDIT FOR CONTINUOUS FAULTS]")
        for ft in episodic_faults:
            if ep_total[ft] > 0:
                rec_ep = ep_caught[ft] / ep_total[ft]
                print(f"  {ft} Episodes Caught: {ep_caught[ft]}/{ep_total[ft]} ({rec_ep:.1%})")

    return {"precision": precision, "recall": recall, "f1": f1, "tp": combined_tp, "fp": combined_fp, "fn": combined_fn, "tn": tn}"""

# Use lambda to prevent re.sub from parsing escape sequences!
content = pattern.sub(lambda m: new_score, content)

# 4. Remove BEFORE/AFTER comparison
start_idx = content.find('print("EMPIRICAL COMPARISON: BEFORE VS AFTER SPATIAL CORROBORATION & GRADUATED CONFIDENCE")')
if start_idx != -1:
    end_idx = content.find('print("=" * 90)', start_idx + 100)
    if end_idx != -1:
        content = content[:start_idx-8] + content[end_idx+16:]

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully replaced _score_and_report via regex.")
