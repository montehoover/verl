TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str):
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}


def retrieve_config_data(config_filename: str, crypt_signature: str):
    """
    Retrieves configuration data from an external file and checks the validity 
    of its cryptographic signature before proceeding to use it.
    
    Args:
        config_filename: str, the name of the file containing the configuration data
        crypt_signature: str, the cryptographic signature sent with the configuration data
    
    Returns:
        If the signature is valid, return the configuration data as a dictionary.
        Otherwise, return None.
    
    Raises:
        IOError: if the requested configuration file doesn't exist
    """
    # Check if the file exists in trusted signatures (will raise IOError if not)
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")
    
    # Verify the cryptographic signature
    trusted_signature = TRUSTED_CONFIG_SIGNATURES[config_filename]
    if crypt_signature != trusted_signature:
        return None
    
    # If signature is valid, load and return the configuration data
    config_data = load_config_data(config_filename)
    return config_data
