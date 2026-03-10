from typing import Optional, Dict, Any
import hmac

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

def evaluate_config_file(config_filename: str, provided_sig: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve configuration data for the given filename from an external source and
    verify it using a cryptographic signature. Returns the configuration data if
    the signature is valid; otherwise returns None.
    """
    if not isinstance(config_filename, str):
        raise TypeError("config_filename must be a string")
    if not isinstance(provided_sig, str):
        raise TypeError("provided_sig must be a string")

    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(config_filename)
    if expected_sig is None:
        return None

    if not hmac.compare_digest(provided_sig, expected_sig):
        return None

    try:
        data = fetch_config_data(config_filename)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    return data
