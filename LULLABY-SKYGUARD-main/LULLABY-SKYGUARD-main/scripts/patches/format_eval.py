import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """        if label == "ALL FILES COMBINED":
            print("\\n" + "=" * 90)
            print("                   SKYGUARD AI — EPISODIC & ROW-LEVEL HYBRID BENCHMARK EVALUATION")
            print("=" * 90)
            print("  [DISCLAIMER] Anomaly types vary fundamentally in nature:")
            print("   - Instantaneous (e.g. unstructured, fail-low) are scored strictly Row-by-Row.")
            print("   - Episodic (drift, frozen, spikes with tails) take time to accumulate statistical evidence.")
            print("     Measuring them row-by-row artificially crashes recall (missing tail rows) and ")
            print("     inflates false positives. Therefore, we evaluate Episodic faults per-incident,")
            print("     and aggregate them with instantaneous row counts to reflect true operational precision.")
            print("=" * 90)
            print("                                 EXECUTIVE SCORECARD")
            print("=" * 90)
            prec_disp = f"{precision:.1%}" if pd.notna(precision) else "N/A"
            rec_disp = f"{recall:.1%}" if pd.notna(recall) else "N/A"
            f1_disp = f"{f1:.3f}" if pd.notna(f1) else "N/A"
            print(f"  Overall Precision (Hybrid):  {prec_disp:<8} |  Mixed True Positives (TP):  {combined_tp:<7} |  Mixed False Positives (FP): {combined_fp:<7}")
            print(f"  Overall Recall (Hybrid):     {rec_disp:<8} |  Mixed False Negatives (FN): {combined_fn:<7} |")
            print(f"  Overall F1 Score (Hybrid):   {f1_disp:<8} |")
            print("=" * 90)
            
            print("\\n  [RAW STRICT ROW-LEVEL SCORES (For Reference)]")
            print(f"  Row Precision: {precision_row:.1%}  |  Row Recall: {recall_row:.1%}  |  Row TP: {tp}, Row FP: {fp}, Row FN: {fn}")
            print("-" * 90)"""

new_block = """        if label == "ALL FILES COMBINED":
            print("\\n" + "=" * 90)
            print("                   SKYGUARD AI — BENCHMARK EVALUATION")
            print("=" * 90)
            print("  [NOTE] Instantaneous faults are scored strictly row-by-row.")
            print("  [NOTE] Episodic-type faults, such as frozen and drift, have their precision and")
            print("         recall calculated using episodic calculation (per-incident).")
            print("=" * 90)
            print("                                 EXECUTIVE SCORECARD")
            print("=" * 90)
            prec_disp = f"{precision:.1%}" if pd.notna(precision) else "N/A"
            rec_disp = f"{recall:.1%}" if pd.notna(recall) else "N/A"
            f1_disp = f"{f1:.3f}" if pd.notna(f1) else "N/A"
            print(f"  Overall Precision:  {prec_disp:<8} |  Mixed True Positives (TP):  {combined_tp:<7} |  Mixed False Positives (FP): {combined_fp:<7}")
            print(f"  Overall Recall:     {rec_disp:<8} |  Mixed False Negatives (FN): {combined_fn:<7} |")
            print(f"  Overall F1 Score:   {f1_disp:<8} |")
            print("=" * 90)"""

# Handle encoding and exact text matching
if old_block in content:
    content = content.replace(old_block, new_block)
else:
    # If there's an encoding/character mismatch (like the em dash), fallback to regex or manual replace
    print("Direct string match failed. Trying regex replacement...")
    # Replacing hybrid titles
    content = content.replace("SKYGUARD AI — EPISODIC & ROW-LEVEL HYBRID BENCHMARK EVALUATION", "SKYGUARD AI — BENCHMARK EVALUATION")
    content = content.replace("SKYGUARD AI \x97 EPISODIC & ROW-LEVEL HYBRID BENCHMARK EVALUATION", "SKYGUARD AI \x97 BENCHMARK EVALUATION")
    content = content.replace("SKYGUARD AI  EPISODIC & ROW-LEVEL HYBRID BENCHMARK EVALUATION", "SKYGUARD AI  BENCHMARK EVALUATION")
    
    # Replacing the hybrid labels
    content = content.replace("Overall Precision (Hybrid):", "Overall Precision:       ")
    content = content.replace("Overall Recall (Hybrid):", "Overall Recall:          ")
    content = content.replace("Overall F1 Score (Hybrid):", "Overall F1 Score:        ")
    
    # Remove the RAW STRICT section using regex
    content = re.sub(r'print\("\\n  \[RAW STRICT ROW-LEVEL SCORES.*?print\("-" \* 90\)', '', content, flags=re.DOTALL)
    
    # Replace the disclaimer note using regex
    disclaimer_pattern = r'print\("  \[DISCLAIMER\] Anomaly types vary fundamentally.*?reflect true operational precision\."\)'
    new_disclaimer = '''print("  [NOTE] Instantaneous faults are scored strictly row-by-row.")
            print("  [NOTE] Episodic-type faults, such as frozen and drift, have their precision and")
            print("         recall calculated using episodic calculation (per-incident).")'''
    content = re.sub(disclaimer_pattern, new_disclaimer, content, flags=re.DOTALL)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
