import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

metric_code = '''
    report = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "total_rows_evaluated": len(df_full),
        "test_set_rows": len(res_df),
        "latency_seconds": elapsed,
        "overall_recall": recall,
        "overall_precision": precision,
        "overall_f1": f1,
        "by_fault_type": {}
    }
    
    print("\nPerformance by fault type (Row-Level):")
    faults = res_df[res_df["fault_type_gt"] != "none"]["fault_type_gt"].unique()
    for ft in sorted(faults):
        if pd.isna(ft): continue
        mask_gt = (res_df["fault_type_gt"] == ft)
        t = mask_gt.sum()
        c = (mask_gt & pred).sum()
        
        mask_pred_ft = (res_df["fault_type_pred"] == ft)
        p_class = mask_pred_ft.sum()
        c_class = (mask_pred_ft & mask_gt).sum()
        
        rec = c / t if t else 0
        prec = c_class / p_class if p_class else 0
        f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
        
        report["by_fault_type"][ft] = {
            "true": int(t),
            "caught": int(c),
            "predicted": int(p_class),
            "recall": rec,
            "precision": prec,
            "f1": f1_class
        }
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%}) | Precision: {prec:.1%} | F1: {f1_class:.3f}")
        
    res_df.to_csv(PER_SENSOR_LOG_PATH, index=False)
    print(f"\\nRow-level log saved to {PER_SENSOR_LOG_PATH}")
    
    import json
    report_json_path = DATA_DIR / "evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)
        
    report_md_path = DATA_DIR / "evaluation_report.md"
    with open(report_md_path, "w") as f:
        f.write(f"# SkyGuard AI Evaluation Report\\n\\n")
        f.write(f"**Generated:** {report['timestamp']}\\n")
        f.write(f"**Latency:** {report['latency_seconds']:.1f}s\\n\\n")
        f.write(f"## Overall Metrics\\n")
        f.write(f"- Recall: {report['overall_recall']:.1%}\\n")
        f.write(f"- Precision: {report['overall_precision']:.1%}\\n")
        f.write(f"- F1: {report['overall_f1']:.3f}\\n\\n")
        
        f.write(f"## By Fault Type\\n")
        f.write(f"| Fault Type | Recall | Precision | F1 |\\n")
        f.write(f"|---|---|---|---|\\n")
        for ft, mets in report["by_fault_type"].items():
            f.write(f"| {ft} | {mets['recall']:.1%} | {mets['precision']:.1%} | {mets['f1']:.3f} |\\n")
            
    print(f"Reports saved to {report_json_path} and {report_md_path}")
'''

# We need to replace the existing loop over faults
old_loop = '''    print("\\nPerformance by fault type (Row-Level):")
    faults = res_df[res_df["fault_type_gt"] != "none"]["fault_type_gt"].unique()
    for ft in sorted(faults):
        mask_gt = (res_df["fault_type_gt"] == ft)
        t = mask_gt.sum()
        c = (mask_gt & pred).sum()
        
        mask_pred_ft = (res_df["fault_type_pred"] == ft)
        p_class = mask_pred_ft.sum()
        c_class = (mask_pred_ft & mask_gt).sum()
        
        rec = c / t if t else 0
        prec = c_class / p_class if p_class else 0
        f1_class = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0
        print(f"  {ft:<28} Caught: {c}/{t} ({rec:.1%}) | Precision: {prec:.1%} | F1: {f1_class:.3f}")
        
    res_df.to_csv(PER_SENSOR_LOG_PATH, index=False)
    print(f"\\nRow-level log saved to {PER_SENSOR_LOG_PATH}")'''

content = content.replace(old_loop, metric_code)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
