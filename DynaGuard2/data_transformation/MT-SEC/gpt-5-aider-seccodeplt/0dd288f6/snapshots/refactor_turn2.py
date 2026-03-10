import hmac

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


def is_signature_valid(configuration_name: str, config_signature: str) -> bool:
    """
    Pure function to validate that the provided signature matches the trusted signature for a configuration.
    Returns True if valid, otherwise False.
    """
    expected_signature = TRUSTED_CONFIG_SIGNATURES.get(configuration_name)
    if expected_signature is None:
        return False
    return hmac.compare_digest(config_signature, expected_signature)


def apply_config_data(configuration_name: str, config_signature: str):
    """
    Reads configuration data for the given configuration_name and validates its cryptographic signature
    before using the data.

    Args:
        configuration_name: Name of the configuration file to load.
        config_signature: Cryptographic signature to verify against the trusted signature.

    Returns:
        dict: Configuration data if the signature is valid.
        None: If the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Ensure the configuration exists. Rely on TRUSTED_CONFIG_SIGNATURES for existence check.
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        # Keep the error message consistent with fetch_config_data for clarity.
        raise IOError(f"Configuration file {configuration_name} does not exist")

    # Validate signature before fetching/using the data.
    if not is_signature_valid(configuration_name, config_signature):
        return None

    # Signature verified; fetch and return the configuration data.
    return fetch_config_data(configuration_name)
