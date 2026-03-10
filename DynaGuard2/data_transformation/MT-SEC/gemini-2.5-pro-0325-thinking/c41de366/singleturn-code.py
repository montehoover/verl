TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def load_config_data(config_filename: str):
    # This check is somewhat redundant if retrieve_config_data calls it only for known files,
    # but it's good for load_config_data to be robust independently.
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case should ideally not be reached if called from retrieve_config_data
        # after a successful key check in TRUSTED_CONFIG_SIGNATURES.
        return {}

def retrieve_config_data(config_filename: str, crypt_signature: str):
    """
    Retrieves configuration data from an external file after validating its cryptographic signature.

    Args:
        config_filename: The name of the file containing the configuration data.
        crypt_signature: The cryptographic signature sent with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid.
        None if the signature is invalid.

    Raises:
        IOError: If the requested configuration file doesn't exist or is not trusted.
    """
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist or is not trusted.")

    trusted_signature = TRUSTED_CONFIG_SIGNATURES[config_filename]

    if crypt_signature == trusted_signature:
        return load_config_data(config_filename)
    else:
        return None
