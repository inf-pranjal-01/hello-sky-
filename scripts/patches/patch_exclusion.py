import re

with open('model/state.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_record = '''    def record_raw_reading(self, raw_reading: dict, timestamp, verdict: dict):
        """
        CAUSAL EXCLUSION -- now PER-READING, not just per-sensor-OFFLINE.
        See this file's module docstring, fix #3, for the bug this
        closes (a lone spike that never tips the sensor OFFLINE was
        previously not excluded from anything and quietly degraded the
        rolling baseline for ~48h afterward).

        A row is excluded from the detection buffer if ANY of:
          - the station is currently OFFLINE (should_include_in_baseline()
            -- the original whole-sensor gate, kept for the sustained-
            fault case, where you want the WHOLE offline window
            excluded, not just individually-anomalous readings within it)
          - THIS reading's own verdict was is_anomaly=True (new --
            catches an isolated spike/dropout/etc. that doesn't push
            the sensor OFFLINE on its own)
          - a repair is in progress (recovery_active) -- don't let
            not-yet-trusted post-repair readings seed the baseline
            either

        KNOWN REMAINING LIMITATION: whole-ROW exclusion, not
        per-parameter -- see module docstring fix #3's closing note.
        """
        if not self.health.should_include_in_baseline():
            return
        if verdict.get("is_anomaly"):
            return
        if self.recovery_active:
            return

        row = dict(raw_reading, station_id=self.station_id, timestamp=timestamp)
        self._raw_rows.append(row)'''

new_record = '''    def record_raw_reading(self, raw_reading: dict, timestamp, verdict: dict):
        """
        CAUSAL EXCLUSION -- PER-PARAMETER (P1 Fix).
        Only drops the specific telemetry channels that are anomalous or OFFLINE,
        allowing healthy channels on the same station to continue building
        their baseline organically.
        """
        if self.recovery_active:
            return

        row = dict(raw_reading, station_id=self.station_id, timestamp=timestamp)
        
        # 1. Identify which params are currently OFFLINE (sustained fault)
        offline_params = set()
        for p, status in self.health.param_status.items():
            if status == "OFFLINE":
                offline_params.add(p)
                
        # 2. Identify which params fired a rule THIS tick
        tick_anomalous_params = set()
        if verdict.get("is_anomaly"):
            # Rules might implicate specific params
            for rule in verdict.get("rules_fired", []):
                param = rule.get("parameter")
                if param:
                    tick_anomalous_params.add(param)
            
            # If no specific rule fired (e.g. pure unstructured ML anomaly),
            # we rely on SHAP likely_faulty_sensors
            if not tick_anomalous_params:
                for sensor in verdict.get("likely_faulty_sensors", []):
                    tick_anomalous_params.add(sensor)
                    
            # If still nothing, it's a completely multivariate/unstructured anomaly 
            # with no attribution. We drop all params to be safe.
            if not tick_anomalous_params and verdict.get("fault_type") in ["unstructured_anomaly", "multivariate_inconsistency"]:
                tick_anomalous_params = {"temperature_c", "pressure_hpa", "humidity_pct"}
                
        # 3. Apply np.nan to excluded params
        import numpy as np
        excluded_params = offline_params.union(tick_anomalous_params)
        for p in excluded_params:
            if p in row:
                row[p] = np.nan
                
        # Only append if at least one parameter is still valid
        # Actually, append it regardless so we have the timestamp for ffill!
        self._raw_rows.append(row)'''

content = content.replace(old_record, new_record)

with open('model/state.py', 'w', encoding='utf-8') as f:
    f.write(content)
