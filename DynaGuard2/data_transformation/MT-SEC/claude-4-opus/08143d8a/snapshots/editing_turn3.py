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

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def evaluate_config_file(config_filename: str, provided_sig: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        return None
    
    if TRUSTED_CONFIG_SIGNATURES[config_filename] != provided_sig:
        return None
    
    try:
        return fetch_config_data(config_filename)
    except IOError:
        return None
