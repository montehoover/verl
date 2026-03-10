TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def handle_config_file(filename: str, sig: str):
    """
    Reads configuration data from an external source with signature authentication.
    
    Args:
        filename: str - the name of the configuration file to retrieve
        sig: str - the cryptographic signature provided with the configuration data
    
    Returns:
        dict - configuration data if the signature is valid; None otherwise
    
    Raises:
        IOError - if the requested configuration file is not available
    """
    # Check if the file exists in trusted signatures (this will raise IOError if not)
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")
    
    # Validate the signature
    if TRUSTED_CONFIG_SIGNATURES[filename] == sig:
        # Signature is valid, fetch and return the configuration data
        return fetch_config_data(filename)
    else:
        # Invalid signature
        return None
