import json
import os

def read_local_config(config_filename, trusted_files):
    if config_filename not in trusted_files:
        return None
    
    if not os.path.exists(config_filename):
        return None
    
    try:
        with open(config_filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
