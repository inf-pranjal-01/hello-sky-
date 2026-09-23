import re

with open('model/features.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    df = df.sort_values("timestamp").reset_index(drop=True)
    for col, prefix in RULE_ONLY_PREFIXES:
        df[f"{prefix}_consec_diff"] = df[col].diff(1).abs()
        df[f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"] = df[col] - df[col].shift(DRIFT_LOOKBACK_HOURS)

        # Open-Meteo reports these values to one decimal place. Matching
        # only an integer part makes normal stable pressure look frozen;
        # matching the reported precision preserves the intended
        # persistence shape while eliminating that aliasing false positive.
        floor_vals = np.floor(df[col])
        df[f"{prefix}_floor_frozen_match"] = (
            (floor_vals == floor_vals.shift(1)) & (floor_vals == floor_vals.shift(2))
        )
        run_id = floor_vals.ne(floor_vals.shift()).cumsum()
        df[f"{prefix}_frozen_streak"] = floor_vals.groupby(run_id).cumcount() + 1
    return df"""

new_block = """    from config import FROZEN_CONSECUTIVE_REQUIRED, FROZEN_CONSECUTIVE_REQUIRED_PRESSURE
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col, prefix in RULE_ONLY_PREFIXES:
        df[f"{prefix}_consec_diff"] = df[col].diff(1).abs()
        df[f"{prefix}_{DRIFT_LOOKBACK_HOURS}h_delta"] = df[col] - df[col].shift(DRIFT_LOOKBACK_HOURS)

        # Use round(1) to match Open-Meteo precision
        floor_vals = df[col].round(1)
        
        req = FROZEN_CONSECUTIVE_REQUIRED_PRESSURE if prefix == "pressure" else FROZEN_CONSECUTIVE_REQUIRED
        
        run_id = floor_vals.ne(floor_vals.shift()).cumsum()
        streak = floor_vals.groupby(run_id).cumcount() + 1
        df[f"{prefix}_frozen_streak"] = streak
        
        # Frozen match is dynamic based on config
        df[f"{prefix}_floor_frozen_match"] = streak >= req
    return df"""

content = content.replace(old_block, new_block)

with open('model/features.py', 'w', encoding='utf-8') as f:
    f.write(content)
