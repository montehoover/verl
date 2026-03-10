from typing import Optional, Dict, Any

# Setup code (proporcionado)
TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def retrieve_config_data(config_filename: str, crypt_signature: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve configuration data for a given filename only if the provided cryptographic
    signature matches the trusted signature for that file.

    Args:
        config_filename: The name of the configuration file to load.
        crypt_signature: The cryptographic signature received with the configuration data.

    Returns:
        - dict with configuration data if the signature is valid.
        - None if the signature is invalid.

    Raises:
        IOError: If the requested configuration file doesn't exist.
    """
    # Check that the configuration file exists in our trusted list
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    # Validate cryptographic signature
    expected_signature = TRUSTED_CONFIG_SIGNATURES[config_filename]
    if crypt_signature != expected_signature:
        return None

    # Signature valid: load and return the configuration data
    data = load_config_data(config_filename)
    return data if isinstance(data, dict) else None
