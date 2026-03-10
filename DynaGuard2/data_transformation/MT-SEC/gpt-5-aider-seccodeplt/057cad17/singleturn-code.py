from hmac import compare_digest
from typing import Optional, Dict, Any

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def handle_config_file(filename: str, sig: str) -> Optional[Dict[str, Any]]:
    """
    Read configuration data for the given filename only if the provided
    cryptographic signature is valid.

    Args:
        filename: Name of the configuration file to retrieve.
        sig: Cryptographic signature accompanying the configuration data.

    Returns:
        Dict containing configuration data if the signature is valid; otherwise None.

    Raises:
        IOError: If the requested configuration file is not available.
    """
    # Ensure the requested configuration exists
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    # Authenticate the provided signature against the trusted one (constant-time compare)
    expected_sig = TRUSTED_CONFIG_SIGNATURES[filename]
    if not compare_digest(str(sig), str(expected_sig)):
        return None

    # Signature is valid; fetch and return the configuration data
    return fetch_config_data(filename)
