from typing import Optional, Dict, Any

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str) -> Dict[str, Any]:
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
    Retrieve configuration data for the given filename if the cryptographic signature is valid.

    Args:
        config_filename: The name of the configuration file to load.
        crypt_signature: The cryptographic signature to validate.

    Returns:
        A dictionary with configuration data if the signature is valid; otherwise, None.

    Raises:
        IOError: If the requested configuration file doesn't exist.
    """
    # Check that the configuration file is known; if not, raise IOError.
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    # Validate the cryptographic signature before proceeding to use the configuration data.
    expected_signature = TRUSTED_CONFIG_SIGNATURES[config_filename]
    if crypt_signature != expected_signature:
        return None

    # Signature is valid; safely load and return the configuration data.
    return load_config_data(config_filename)
