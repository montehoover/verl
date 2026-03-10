TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def evaluate_config_file(config_filename: str, provided_sig: str):
    """
    Reads configuration data from an external source and validates its signature.
    
    Args:
        config_filename: str - The name of the configuration file to retrieve
        provided_sig: str - The cryptographic signature provided with the configuration data
    
    Returns:
        dict - Configuration data if signature is valid, None otherwise
    
    Raises:
        IOError - If the requested configuration file is not available
    """
    # Check if the configuration file exists in our trusted signatures
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")
    
    # Validate the provided signature against the trusted signature
    trusted_sig = TRUSTED_CONFIG_SIGNATURES[config_filename]
    if provided_sig == trusted_sig:
        # Signature is valid, fetch and return the configuration data
        return fetch_config_data(config_filename)
    else:
        # Signature is invalid
        return None
