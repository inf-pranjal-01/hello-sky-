import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from model.features import FEATURE_COLUMNS
print(FEATURE_COLUMNS)
