import json
import os

def get_local_config_data(config_filename):
    if not os.path.exists(config_filename):
        return {}
    
    try:
        with open(config_filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
