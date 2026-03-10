TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def process_config_data(config_name: str, signature: str):
    """
    Loads configuration data from an external source and verifies its cryptographic signature.
    
    Args:
        config_name: str - the name of the configuration file to load
        signature: str - the cryptographic signature that accompanies the configuration data
    
    Returns:
        The configuration data as a dictionary if the signature is valid, otherwise None
    
    Raises:
        IOError: if the configuration file cannot be found
    """
    # This will raise IOError if config_name is not in TRUSTED_CONFIG_SIGNATURES
    config_data = fetch_config_data(config_name)
    
    # Verify the signature
    if config_name in TRUSTED_CONFIG_SIGNATURES and TRUSTED_CONFIG_SIGNATURES[config_name] == signature:
        return config_data
    else:
        return None
