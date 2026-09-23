import re

with open('model/state.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add caching to StationBuffer
init_target = """        self.recovery_active: bool = False
        self.recovery_clean_count: int = 0"""

init_replace = """        self.recovery_active: bool = False
        self.recovery_clean_count: int = 0
        self._cached_df = None
        self._cache_dirty = True"""

content = content.replace(init_target, init_replace)

raw_target = """    def raw_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._raw_rows))"""

raw_replace = """    def raw_history_df(self) -> pd.DataFrame:
        if self._cache_dirty:
            self._cached_df = pd.DataFrame(list(self._raw_rows))
            self._cache_dirty = False
        return self._cached_df"""

content = content.replace(raw_target, raw_replace)

# Invalidate cache when _raw_rows changes
record_target = """        self._raw_rows.append(filtered_row)"""
record_replace = """        self._raw_rows.append(filtered_row)
        self._cache_dirty = True"""
content = content.replace(record_target, record_replace)

reset_target = """        self._raw_rows.clear()"""
reset_replace = """        self._raw_rows.clear()
        self._cache_dirty = True"""
content = content.replace(reset_target, reset_replace)

spike_target = """            # rolling baselines are computed.
            buf._raw_rows = deque(
                (row for row in buf._raw_rows if pd.Timestamp(row["timestamp"]) != spike["timestamp"]),
                maxlen=RAW_HISTORY_MAXLEN_HOURS,
            )"""
spike_replace = """            # rolling baselines are computed.
            buf._raw_rows = deque(
                (row for row in buf._raw_rows if pd.Timestamp(row["timestamp"]) != spike["timestamp"]),
                maxlen=RAW_HISTORY_MAXLEN_HOURS,
            )
            buf._cache_dirty = True"""
content = content.replace(spike_target, spike_replace)

with open('model/state.py', 'w', encoding='utf-8') as f:
    f.write(content)
