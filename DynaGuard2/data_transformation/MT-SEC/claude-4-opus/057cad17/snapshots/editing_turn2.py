import json
import os

def read_local_config(filename, approved_files):
    if filename not in approved_files:
        return None
    
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None
    except Exception:
        return None
