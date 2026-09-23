import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update signature and logic of _compute_dynamic_spatial_threshold
old_def = """def _compute_dynamic_spatial_threshold(
    sim,
    all_station_ids: list,
    param: str,
    default: float,
) -> tuple[float, str]:"""

new_def = """def _compute_dynamic_spatial_threshold(
    sim,
    all_station_ids: list,
    param: str,
    default: float,
    target_roc: float = None,
) -> tuple[float, str]:"""

content = content.replace(old_def, new_def)

old_return = """    # Hours of data represented (each row = 1 observation hour per station)
    hours_of_data = min_len

    return max(floor, round(dynamic_val, 2)), f"dynamic({hours_of_data}h)"
"""

new_return = """    # Hours of data represented (each row = 1 observation hour per station)
    hours_of_data = min_len
    
    threshold = max(floor, round(dynamic_val, 2))
    
    # Gradual vs instant onset adjustment
    # If the target station's rate-of-change is low relative to the spatial threshold,
    # it means the divergence arrived gradually over multiple readings (consistent with real
    # localized weather ramping in), rather than instantly (sensor spike).
    if target_roc is not None and abs(target_roc) < (threshold * 0.5):
        threshold = threshold * 3.0
        return round(threshold, 2), f"dynamic({hours_of_data}h, gradual)"

    return threshold, f"dynamic({hours_of_data}h)"
"""

content = content.replace(old_return, new_return)

# 2. Update _compute_spatial_context to pass target_roc
old_call = """        thresh, src = _compute_dynamic_spatial_threshold(sim, all_sids, p, default=DEFAULT_SIG.get(p, 3.0))"""

new_call = """        # Look up the 1-hour rate of change to check for gradual onset
        roc_param = f"{p.replace('_c', '').replace('_hpa', '').replace('_pct', '')}_roc_1h"
        target_roc = _lookup_peer_val(target_sid, roc_param, None)
        
        thresh, src = _compute_dynamic_spatial_threshold(sim, all_sids, p, default=DEFAULT_SIG.get(p, 3.0), target_roc=target_roc)"""

content = content.replace(old_call, new_call)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(content)
