TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(file_name: str):
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def handle_config_file(file_name: str, sign: str):
    """
    Reads configuration information from an external source and validates its cryptographic signature.
    
    Args:
        file_name: str - identifying the configuration file
        sign: str - the cryptographic signature corresponding to the configuration
    
    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.
    
    Raises:
        IOError: if the configuration file cannot be located
    """
    # Check if the file exists in TRUSTED_CONFIG_SIGNATURES
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")
    
    # Validate the signature
    if TRUSTED_CONFIG_SIGNATURES[file_name] == sign:
        # Signature is valid, fetch and return the configuration data
        return fetch_config_data(file_name)
    else:
        # Invalid signature
        return None
