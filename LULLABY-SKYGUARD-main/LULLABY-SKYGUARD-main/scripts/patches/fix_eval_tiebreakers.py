import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace mv_hit assignment
old_mv = """                if mv_hit:
                    evidence.append(
                        (
                            "multivariate_inconsistency",
                            RULE_BASE_CONFIDENCE[
                                "multivariate_confirmed"
                                if mv_confirmed
                                else "multivariate_single"
                            ],
                        )
                    )"""
new_mv = """                if mv_hit:
                    evidence.append(
                        (
                            "multivariate_inconsistency",
                            93.0 if mv_confirmed else 45.0,
                        )
                    )"""
content = content.replace(old_mv, new_mv)

# Replace frozen assignment
old_frozen = """                if frozen:
                    evidence.append(
                        (
                            "frozen_value",
                            RULE_BASE_CONFIDENCE["frozen_value"],
                        )
                    )"""
new_frozen = """                if frozen:
                    evidence.append(
                        (
                            "frozen_value",
                            96.0,
                        )
                    )"""
content = content.replace(old_frozen, new_frozen)

# Replace drift assignment
old_drift = """                if drift:
                    evidence.append(
                        (
                            "drift",
                            RULE_BASE_CONFIDENCE["drift"],
                        )
                    )"""
new_drift = """                if drift:
                    evidence.append(
                        (
                            "drift",
                            97.0,
                        )
                    )"""
content = content.replace(old_drift, new_drift)

# Replace spike assignment
old_spike = """                if spike:
                    evidence.append(
                        (
                            "spike",
                            RULE_BASE_CONFIDENCE["spike"],
                        )
                    )"""
new_spike = """                if spike:
                    evidence.append(
                        (
                            "spike",
                            94.0,
                        )
                    )"""
content = content.replace(old_spike, new_spike)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
