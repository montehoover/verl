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

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def handle_config_file(filename, sig):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        return None
    
    if TRUSTED_CONFIG_SIGNATURES[filename] != sig:
        return None
    
    try:
        return fetch_config_data(filename)
    except IOError:
        return None
