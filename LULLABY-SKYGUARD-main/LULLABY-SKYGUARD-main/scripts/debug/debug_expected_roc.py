import sys
sys.path.insert(0, '.')
from model.seasonal_baseline import get_expected_roc
for h in range(16, 22):
    print(f'Hour {h}: expected_roc={get_expected_roc("AWS-RAN-067", "temp", h):.2f}')
