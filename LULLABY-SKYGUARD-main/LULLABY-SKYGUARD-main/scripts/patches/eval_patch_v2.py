import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove spike from episodic_faults
content = content.replace(
    'episodic_faults = ["frozen_value", "drift", "spike", "multivariate_inconsistency"]',
    'episodic_faults = ["frozen_value", "drift", "multivariate_inconsistency"]'
)

# 2. Extract the entire Network-aware supervised helper block
helper_block_regex = re.compile(
    r'(# Network-aware supervised helper -------------------------------------------------.*?)(?=has_rule_ft = )',
    re.DOTALL
)
match = helper_block_regex.search(content)
if not match:
    print("Could not find helper block!")
    exit(1)

helper_block = match.group(1)

# Remove it from its original location
content = content.replace(helper_block, '')

# We will rewrite the helper block to provide bounded confidence and inject it BEFORE overall_confidence
new_helper_logic = """    # Network-aware supervised helper -------------------------------------------------
    helper_path = ARTIFACTS_PATH.parent / "fault_helper.pkl"
    if helper_path.exists():
        print(f"\\n(Loading cached fault_helper artifact from {helper_path}...)")
        helper_artifact = joblib.load(helper_path)
        if isinstance(helper_artifact, dict):
            helper_model = helper_artifact.get("helper_model")
            helper_columns = helper_artifact.get("helper_columns")
            frozen_helpers = helper_artifact.get("frozen_helpers", {})
        else:
            helper_model, helper_columns = helper_artifact[:2]
            frozen_helpers = helper_artifact[2] if len(helper_artifact) > 2 else {}
    else:
        print(f"\\n(Fitting network fault_helper across sparse replays with seeds {HELPER_TRAINING_SEEDS}...)")
        helper_held_out_stations = set(labels.loc[labels["is_anomaly"], "station_id"])
        helper_training = make_sparse_training_replays(helper_held_out_stations, HELPER_TRAINING_SEEDS)
        helper_model, helper_columns = fit_fault_helper(helper_training)
        frozen_helpers = fit_frozen_channel_helpers(helper_training)
        joblib.dump({
            "helper_model": helper_model,
            "helper_columns": helper_columns,
            "frozen_helpers": frozen_helpers,
        }, helper_path)
        print(f"(Cached trained fault helper models to {helper_path})")

    helper_scored = predict_faults(helper_model, helper_columns, df_full, HELPER_ALERT_THRESHOLD)
    helper_scored = score_frozen_channels(
        helper_scored, frozen_helpers, FROZEN_HELPER_ALERT_THRESHOLD,
    )
    
    featured["timestamp"] = pd.to_datetime(featured["timestamp"]).dt.tz_localize(None)
    helper_scored["timestamp"] = pd.to_datetime(helper_scored["timestamp"]).dt.tz_localize(None)

    helper_prob_lookup = helper_scored.set_index(["station_id", "timestamp"])["helper_probability"]
    helper_prob = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(helper_prob_lookup).fillna(0.0).to_numpy(dtype=float)
    
    frozen_prob_lookup = helper_scored.set_index(["station_id", "timestamp"])["frozen_helper_probability"] if "frozen_helper_probability" in helper_scored.columns else helper_scored.set_index(["station_id", "timestamp"])["frozen_helper_alert"].astype(float)
    frozen_prob = pd.MultiIndex.from_frame(featured[["station_id", "timestamp"]]).map(frozen_prob_lookup).fillna(0.0).to_numpy(dtype=float)

    # Bounded confidence contribution inside the fusion logic!
    fh_conf = np.minimum(89.9, helper_prob * 100.0)
    helper_wins = (fh_conf > row_rule_conf) & (fh_conf > 10.0)
    row_rule_conf = np.maximum(row_rule_conf, fh_conf)
    row_fault_type = np.where(helper_wins, "network_helper", row_fault_type)
    
    frz_conf = np.minimum(89.9, frozen_prob * 100.0)
    frz_wins = (frz_conf > row_rule_conf) & (frz_conf > 10.0)
    row_rule_conf = np.maximum(row_rule_conf, frz_conf)
    row_fault_type = np.where(frz_wins, "frozen_value", row_fault_type)
    
"""

# Insert new helper logic before `model_pct = vectorized_model_scores`
content = content.replace(
    'model_pct = vectorized_model_scores(featured, artifact)',
    new_helper_logic + '\n    model_pct = vectorized_model_scores(featured, artifact)'
)

# Fix the predicted fault type assignment since helper_alert is gone
old_pred_ft = """    pred_ft.loc[frozen_helper_alert & (pred_ft == "none")] = "frozen_value"
    pred_ft.loc[helper_alert & (pred_ft == "none")] = "multivariate_inconsistency\""""

content = content.replace(old_pred_ft, "")

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
