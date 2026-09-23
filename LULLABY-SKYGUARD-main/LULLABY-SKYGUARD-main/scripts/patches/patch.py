with open('model/state.py', 'r') as f:
    text = f.read()

old_text = '''        self.history = history_store or HistoryStore()

        # Starts in live mode.'''

new_text = '''        self.history = history_store or HistoryStore()

        self.neighbor_map = {}
        for sid in metadata["station_id"]:
            cluster = metadata[metadata["station_id"] == sid]["cluster_id"].iloc[0]
            self.neighbor_map[sid] = metadata[(metadata["cluster_id"] == cluster) & (metadata["station_id"] != sid)]["station_id"].tolist()

        # Starts in live mode.'''

if old_text in text:
    text = text.replace(old_text, new_text)
    print("Found and replaced block 1")
else:
    print("Block 1 not found")

old_text2 = '''        history_df_with_current = (
            pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
            if not history_df.empty else pd.DataFrame([current_row])
        )

        verdict = score_reading(
            raw_reading,
            history_df_with_current,
            self.artifact,
            explainer=self.explainer,
        )'''

new_text2 = '''        history_df_with_current = (
            pd.concat([history_df, pd.DataFrame([current_row])], ignore_index=True)
            if not history_df.empty else pd.DataFrame([current_row])
        )

        neighbor_buffers = {
            nid: self.buffers[nid].raw_history_df()
            for nid in self.neighbor_map.get(station_id, [])
        }

        verdict = score_reading(
            raw_reading,
            history_df_with_current,
            self.artifact,
            neighbor_buffers=neighbor_buffers,
            explainer=self.explainer,
        )'''

if old_text2 in text:
    text = text.replace(old_text2, new_text2)
    print("Found and replaced block 2")
else:
    print("Block 2 not found")

with open('model/state.py', 'w') as f:
    f.write(text)
