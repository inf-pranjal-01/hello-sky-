import re

with open('tests/test_architecture.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('self.assertIn("DecisionEngine.decide", eval_code)', 'self.assertIn("StateManager", eval_code)')

with open('tests/test_architecture.py', 'w', encoding='utf-8') as f:
    f.write(content)
