with open('evaluation/fast_offline_eval.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_table_header = """        print("\\nPerformance by fault type (Recall & Precision):")
        print(f"  {'Fault Type':<28} {'Caught':<8} {'True':<8} {'Pred':<8} {'Recall':<10} {'Precision':<10} {'F1':<8}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")"""

new_table_header = """        print("\\nPerformance by fault type (Row-Level):")
        print(f"  {'Fault Type':<28} | {'Total True':<10} {'Total Pred':<10} | {'Caught(Any)':<12} {'Det.Recall':<10} | {'Strict TP':<10} {'Attr.Rec':<10} {'Attr.Prec':<10} {'Attr.F1':<8}")
        print(f"  {'-'*28}-+-{'-'*10}-{'-'*10}-+-{'-'*12}-{'-'*10}-+-{'-'*10}-{'-'*10}-{'-'*10}-{'-'*8}")"""

text = text.replace(old_table_header, new_table_header)

old_table_row = """            rec_str = f"{rec:.1%}" if pd.notna(rec) else "N/A"
            prec_str = f"{prec:.1%}" if pd.notna(prec) else "N/A"
            f1_str = f"{f1_ft:.3f}" if pd.notna(f1_ft) else "N/A"
            
            print(f"  {ft:<28} {caught:<8} {n_true:<8} {n_pred:<8} {rec_str:<10} {prec_str:<10} {f1_str:<8}")"""

new_table_row = """            det_rec = caught / n_true if n_true > 0 else float("nan")
            attr_rec = tp_ft / n_true if n_true > 0 else float("nan")
            attr_prec = tp_ft / n_pred if n_pred > 0 else float("nan")
            attr_f1 = (2 * attr_prec * attr_rec / (attr_prec + attr_rec)) if (pd.notna(attr_prec) and pd.notna(attr_rec) and (attr_prec + attr_rec) > 0) else float("nan")

            det_rec_str = f"{det_rec:.1%}" if pd.notna(det_rec) else "N/A"
            attr_rec_str = f"{attr_rec:.1%}" if pd.notna(attr_rec) else "N/A"
            attr_prec_str = f"{attr_prec:.1%}" if pd.notna(attr_prec) else "N/A"
            attr_f1_str = f"{attr_f1:.3f}" if pd.notna(attr_f1) else "N/A"
            
            print(f"  {ft:<28} | {n_true:<10} {n_pred:<10} | {caught:<12} {det_rec_str:<10} | {tp_ft:<10} {attr_rec_str:<10} {attr_prec_str:<10} {attr_f1_str:<8}")"""

text = text.replace(old_table_row, new_table_row)

with open('evaluation/fast_offline_eval.py', 'w', encoding='utf-8') as f:
    f.write(text)
