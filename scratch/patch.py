import re
import pandas as pd

with open('model/detect.py', 'r') as f:
    text = f.read()

# We need to find the old spike logic.
pattern = r'    confirmed_spikes = _confirmed_spikes\(.*?\}\)\n'
old_spike_block = re.search(pattern, text, flags=re.DOTALL)
if old_spike_block:
    new_spike_block = '''    from model.spike_tracker import init_spike_state, step_spike_state, SPIKE_WINDOW_HOURS
    
    window_size = SPIKE_WINDOW_HOURS + 1
    if len(featured_buffer) > 0:
        replay_rows = featured_buffer.iloc[-window_size:] if len(featured_buffer) > window_size else featured_buffer
        for param, prefix in PARAM_PREFIXES.items():
            spike_thresh = get_threshold(thresholds, "spike", prefix, station_id)
            state = init_spike_state()
            final_conf = 0.0
            final_status = "IDLE"
            final_reason = ""
            for i in range(len(replay_rows)):
                row = replay_rows.iloc[i]
                val = row.get(param)
                dev = row.get(f"{prefix}_deviation")
                conf, status, reason = step_spike_state(
                    val, dev, spike_thresh, SPIKE_DEVIATION_MULTIPLIER, state, graduated_confidence_spike
                )
                if i == len(replay_rows) - 1:
                    final_conf = conf
                    final_status = status
                    final_reason = reason
            
            if final_conf > 0:
                val = replay_rows.iloc[-1].get(param)
                fired.append({
                    "type": "spike",
                    "parameter": param,
                    "confidence": final_conf,
                    "observed_value": float(val) if pd.notna(val) else None,
                    "threshold": f">{spike_thresh * SPIKE_DEVIATION_MULTIPLIER:.1f}",
                    "reason": final_reason,
                    "basis": "provisional" if final_status == "PROVISIONAL" else "confirmed"
                })
'''
    text = text.replace(old_spike_block.group(0), new_spike_block)
    
    # Remove confirmed_spikes from return dictionaries
    text = text.replace('        "confirmed_spikes": confirmed_spikes,\n', '')
    text = text.replace('        "confirmed_spikes": rules["confirmed_spikes"],\n', '')
    
    with open('model/detect.py', 'w') as f:
        f.write(text)
    print('Success')
else:
    print('Failed to find pattern')
