import re

with open('model/state.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace import
content = content.replace(
    'from model.detect import score_reading, SensorHealthTracker, PARAMS',
    'from model.engine import DecisionEngine\nfrom model.detect import SensorHealthTracker, PARAMS'
)

# Replace call
content = content.replace(
    """        verdict = score_reading(
            raw_reading,
            history_df_with_current,
            self.artifact,
            neighbor_buffers=neighbor_buffers,
            explainer=self.explainer,
        )""",
    """        verdict = DecisionEngine.decide(
            raw_reading,
            history_df_with_current,
            neighbor_buffers,
            self.artifact,
            state=self.explainer,
        )"""
)

with open('model/state.py', 'w', encoding='utf-8') as f:
    f.write(content)
