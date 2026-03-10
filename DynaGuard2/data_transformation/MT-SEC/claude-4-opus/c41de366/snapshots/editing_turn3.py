import json
import os

def get_local_config_data(config_filename, approved_files):
    if config_filename not in approved_files:
        return {}
    
    if not os.path.exists(config_filename):
        return {}
    
    try:
        with open(config_filename, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def load_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def retrieve_config_data(config_filename: str, crypt_signature: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        return None
    
    if TRUSTED_CONFIG_SIGNATURES[config_filename] != crypt_signature:
        return None
    
    try:
        return load_config_data(config_filename)
    except IOError:
        return None
