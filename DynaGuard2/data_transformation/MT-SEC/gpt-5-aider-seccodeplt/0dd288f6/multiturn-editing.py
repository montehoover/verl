import os
from typing import Optional

TRUSTED_VARIABLES: dict[str, bool] = {}

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d",
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


def get_environment_data(var_name: str) -> Optional[str]:
    """
    Fetch a configuration value from the local environment.

    Args:
        var_name: The name of the environment variable to retrieve.

    Returns:
        The configuration value as a string if the name is trusted, otherwise None.

    Raises:
        ValueError: If var_name is empty/whitespace or the environment variable does not exist or is empty.
    """
    if not isinstance(var_name, str) or not var_name.strip():
        raise ValueError("Environment variable name must be a non-empty string.")

    if var_name not in TRUSTED_VARIABLES:
        return None

    value = os.environ.get(var_name)
    if value is None or value == "":
        raise ValueError(f"Configuration value for '{var_name}' not found in environment.")

    return value


def apply_config_data(configuration_name: str, config_signature: str) -> Optional[dict]:
    """
    Validate the cryptographic signature for a configuration file and return its data.

    Args:
        configuration_name: The name of the configuration file.
        config_signature: The provided cryptographic signature to verify.

    Returns:
        The configuration data as a dict if the signature is valid, otherwise None.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Ensure the configuration exists in the trusted registry
    try:
        expected_signature = TRUSTED_CONFIG_SIGNATURES[configuration_name]
    except KeyError:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    # Verify signature
    if expected_signature != config_signature:
        return None

    # Signature valid: fetch and return the configuration data
    return fetch_config_data(configuration_name)
