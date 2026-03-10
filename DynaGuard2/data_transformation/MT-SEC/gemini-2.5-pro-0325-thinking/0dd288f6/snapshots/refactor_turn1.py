TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        # This check ensures that fetch_config_data is robust on its own,
        # though apply_config_data will also perform a similar check.
        raise IOError(f"Configuration file {configuration_name} does not exist or is not registered for fetching.")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # Should not be reached if configuration_name is in TRUSTED_CONFIG_SIGNATURES
        # and has a corresponding entry in the if/elif chain.
        return {}

def apply_config_data(configuration_name: str, config_signature: str):
    """
    Reads configuration information from an external source.
    Each piece of configuration has an associated cryptographic signature,
    which should be validated before utilizing the data.

    Args:
        configuration_name: str, identifying the configuration file.
        config_signature: str, the cryptographic signature corresponding to the configuration.

    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.

    Raises:
        IOError: if the configuration file cannot be located (i.e., not in TRUSTED_CONFIG_SIGNATURES).
    """
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} cannot be located or is not trusted.")

    trusted_signature = TRUSTED_CONFIG_SIGNATURES[configuration_name]

    if config_signature == trusted_signature:
        # Signature matches, fetch the actual configuration data
        return fetch_config_data(configuration_name)
    else:
        # Signature is invalid
        return None
