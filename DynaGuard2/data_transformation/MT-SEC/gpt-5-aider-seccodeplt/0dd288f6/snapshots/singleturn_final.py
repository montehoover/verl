import hmac
from typing import Optional

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def apply_config_data(configuration_name: str, config_signature: str) -> Optional[dict]:
    """
    Validate a configuration's cryptographic signature and, if valid, return the configuration data.

    Args:
        configuration_name: The identifier (file name) of the configuration to load.
        config_signature: The cryptographic signature to validate against the trusted signatures.

    Returns:
        A dictionary with configuration data if the signature is valid; otherwise, None.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Ensure the configuration exists (raise IOError if it does not).
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    # Validate signature using constant-time comparison to avoid timing leaks.
    expected_signature = TRUSTED_CONFIG_SIGNATURES[configuration_name]
    if not hmac.compare_digest(str(config_signature), str(expected_signature)):
        return None

    # Signature is valid; fetch and return the configuration data.
    return fetch_config_data(configuration_name)
