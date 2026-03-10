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


def apply_config_data(configuration_name: str, config_signature: str):
    """
    Reads configuration information and validates its cryptographic signature.
    
    Args:
        configuration_name: str, identifying the configuration file
        config_signature: str, the cryptographic signature corresponding to the configuration
    
    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.
    
    Raises:
        IOError: if the configuration file cannot be located
    """
    # Check if configuration exists (this will raise IOError if not found)
    config_data = fetch_config_data(configuration_name)
    
    # Validate the signature
    if configuration_name in TRUSTED_CONFIG_SIGNATURES:
        trusted_signature = TRUSTED_CONFIG_SIGNATURES[configuration_name]
        if config_signature == trusted_signature:
            return config_data
        else:
            return None
    else:
        # This shouldn't happen as fetch_config_data already checks this
        raise IOError(f"Configuration file {configuration_name} does not exist")
