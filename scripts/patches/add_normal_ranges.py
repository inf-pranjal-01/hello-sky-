import re

with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

func = """
def get_station_normal_ranges(station_id: str) -> dict:
    return {
        "temperature_c": {"normal_min": 5.0, "normal_max": 45.0},
        "pressure_hpa": {"normal_min": 950.0, "normal_max": 1050.0},
        "humidity_pct": {"normal_min": 10.0, "normal_max": 95.0}
    }
"""
if "def get_station_normal_ranges" not in content:
    content += func

with open('config.py', 'w', encoding='utf-8') as f:
    f.write(content)
