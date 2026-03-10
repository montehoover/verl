import json
import os

def read_local_config(config_filename):
    if not os.path.exists(config_filename):
        return None
    
    try:
        with open(config_filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
