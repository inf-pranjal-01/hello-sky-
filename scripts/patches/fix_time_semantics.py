import re

with open('model/features.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the ROC and slope logic in add_temporal_features
old_logic = """        df[f"{prefix}_roc_1h"] = values.diff(ROC_SHORT_HOURS)
        df[f"{prefix}_roc_3h"] = values.diff(ROC_LONG_HOURS)
        
        # Robust slopes
        df[f"{prefix}_slope_6h"] = (values - values.shift(6)) / 6.0
        df[f"{prefix}_slope_24h"] = (values - values.shift(24)) / 24.0"""

new_logic = """        # True time-based rate of change (ROC) using dt_hours
        safe_dt = df["dt_hours"].replace(0, np.nan)
        # Note: If intervals are typically 1h, this matches old logic but with proper units.
        # If intervals are 15m, this converts the 15m change into a per-hour rate, keeping threshold semantics correct.
        roc_per_hour = values.diff() / safe_dt
        df[f"{prefix}_roc_1h"] = roc_per_hour
        
        # For longer lookbacks (3h, 6h, 24h), we should ideally interpolate or resample.
        # But to avoid massive rewrite of the pipeline, if we assume roughly 1-hour intervals for history buffers,
        # we can use shift(N) divided by actual time difference to that shift:
        dt_3h = (df.index.to_series() - df.index.to_series().shift(ROC_LONG_HOURS)).dt.total_seconds() / 3600.0
        df[f"{prefix}_roc_3h"] = (values - values.shift(ROC_LONG_HOURS)) / dt_3h.replace(0, np.nan)
        
        # Robust slopes
        dt_6h = (df.index.to_series() - df.index.to_series().shift(6)).dt.total_seconds() / 3600.0
        df[f"{prefix}_slope_6h"] = (values - values.shift(6)) / dt_6h.replace(0, np.nan)
        
        dt_24h = (df.index.to_series() - df.index.to_series().shift(24)).dt.total_seconds() / 3600.0
        df[f"{prefix}_slope_24h"] = (values - values.shift(24)) / dt_24h.replace(0, np.nan)"""

content = content.replace(old_logic, new_logic)

with open('model/features.py', 'w', encoding='utf-8') as f:
    f.write(content)
