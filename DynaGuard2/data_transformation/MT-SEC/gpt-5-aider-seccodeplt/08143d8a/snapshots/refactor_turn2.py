from typing import Optional, Dict, Any
import hmac

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def verify_config_signature(config_filename: str, provided_sig: str) -> bool:
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)
    if expected_sig is None or not isinstance(provided_sig, str):
        return False
    return hmac.compare_digest(provided_sig, expected_sig)

def get_config(config_filename: str) -> Dict[str, Any]:
    return fetch_config_data(config_filename)

def evaluate_config_file(config_filename: str, provided_sig: str) -> Optional[Dict[str, Any]]:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if not verify_config_signature(config_filename, provided_sig):
        return None

    return get_config(config_filename)
