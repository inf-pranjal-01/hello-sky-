import sys
import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the "Performance by fault type" block to use Episodic metrics for episodic faults
target_block_start = '        print("\\nPerformance by fault type (Recall & Precision):")'
target_block_end = '        print("\\n  [EPISODE-LEVEL AUDIT FOR CONTINUOUS FAULTS]")'

if target_block_start in content and target_block_end in content:
    s_idx = content.find(target_block_start)
    e_idx = content.find(target_block_end)
    
    new_block = r"""        print("\nPerformance by fault type (Recall & Precision):")
        print("  [NOTE] Metrics for 'drift' and 'frozen_value' are calculated EPISODICALLY.")
        print("         Because these faults accumulate over time, measuring them row-by-row")
        print("         skews reality by penalizing the system for initial subtle rows.")
        print(f"  {'Fault Type':<28} {'Caught':<8} {'True':<8} {'Pred':<8} {'Recall':<10} {'Precision':<10} {'F1':<8}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

        known_types = sorted(set(
            list(pd.unique(fault_type[ground_truth]))
            + [x for x in pd.unique(pred_fault_type[predicted]) if x not in ('none', None, 'UNKNOWN_STATISTICAL_ANOMALY')]
        ))
        
        # Calculate episodic FPs per fault type to use in precision
        ep_fp_by_type = {}
        for ft in episodic_faults:
            ep_fp_by_type[ft] = 0
            mask_pred_fp = predicted & (pred_fault_type == ft) & ~ground_truth
            if mask_pred_fp.any():
                blocks_fp = (~mask_pred_fp).cumsum()[mask_pred_fp]
                ep_fp_by_type[ft] = len(featured[mask_pred_fp].groupby(blocks_fp))
                
        for ft in known_types:
            if ft in ('none', None, 'REGIONAL_EVENT'):
                continue
                
            if ft in episodic_faults:
                # Use Episodic metrics
                caught = ep_caught.get(ft, 0)
                n_true = ep_total.get(ft, 0)
                fp_count = ep_fp_by_type.get(ft, 0)
                n_pred = caught + fp_count
                
                rec = caught / n_true if n_true > 0 else float("nan")
                prec = caught / n_pred if n_pred > 0 else float("nan")
                f1_ft = (2 * prec * rec / (prec + rec)) if (pd.notna(prec) and pd.notna(rec) and (prec + rec) > 0) else float("nan")
            else:
                # Use Row-level metrics
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
            # Append an asterisk to indicate episodic
            ft_display = ft + "*" if ft in episodic_faults else ft
            print(f"  {str(ft_display):<28} {caught:<8} {n_true:<8} {n_pred:<8} {rec_str:<10} {prec_str:<10} {f1_str:<8}")
"""
    
    content = content[:s_idx] + new_block + content[e_idx:]

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully patched evaluate table.")
