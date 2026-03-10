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
    Load configuration data and verify its cryptographic signature.

    Args:
        config_name: The name of the configuration file to load.
        signature: The cryptographic signature accompanying the configuration data.

    Returns:
        dict: The configuration data if the signature is valid.
        None: If the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be found.
    """
    # Ensure the configuration exists
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    # Verify signature
    trusted_signature = TRUSTED_CONFIG_SIGNATURES.get(config_name)
    if signature != trusted_signature:
        return None

    # Fetch and return configuration data
    return fetch_config_data(config_name)
