import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.linalg as la

from config import CLUSTERS

STATION_TO_CLUSTER = {}
CLUSTER_TO_STATIONS = {}
for cid, cinfo in CLUSTERS.items():
    center = cinfo["center"]["station_id"]
    neighbors = [n["station_id"] for n in cinfo["neighbors"]]
    stns = [center] + neighbors
    CLUSTER_TO_STATIONS[cid] = stns
    for sid in stns:
        STATION_TO_CLUSTER[sid] = cid

print("Loaded clusters and stations successfully. Total stations:", len(STATION_TO_CLUSTER))
