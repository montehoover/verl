from typing import Optional, Dict, Any

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str) -> Dict[str, Any]:
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case should ideally not be reached if configuration_name is always in TRUSTED_CONFIG_SIGNATURES
        # before calling fetch_config_data, but as a fallback:
        return {}

def apply_config_data(configuration_name: str, config_signature: str) -> Optional[Dict[str, Any]]:
    """
    Validates the cryptographic signature of a configuration file and fetches its data.

    Args:
        configuration_name: The name of the configuration file.
        config_signature: The cryptographic signature to validate.

    Returns:
        A dictionary containing the configuration data if the signature matches
        and the file is found, None otherwise.

    Raises:
        IOError: If the configuration file cannot be located by fetch_config_data
                 (though this is typically checked before calling fetch_config_data).
    """
    expected_signature = TRUSTED_CONFIG_SIGNATURES.get(configuration_name)

    if expected_signature is None:
        # Configuration name not in trusted list, could be an error or simply not trusted
        return None

    if config_signature == expected_signature:
        try:
            return fetch_config_data(configuration_name)
        except IOError:
            # Propagate IOError if fetch_config_data raises it
            raise
    else:
        return None
