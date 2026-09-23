import sys
import re

with open('model/evaluate.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    '''        except Exception as e:
            # Dropouts or incomplete rows
            pass''',
    '''        except Exception as e:
            # Dropouts or incomplete rows
            if count == 0:
                print(f"ERROR: {e}")
            pass'''
)

with open('model/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)
