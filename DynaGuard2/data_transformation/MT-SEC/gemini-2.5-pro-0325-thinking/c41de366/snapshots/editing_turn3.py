TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def load_config_data(config_filename: str) -> dict:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def retrieve_config_data(config_filename: str, crypt_signature: str) -> dict | None:
    """
    Retrieves configuration data from an external file and checks the validity
    of its cryptographic signature.

    Args:
        config_filename: The name of the configuration file.
        crypt_signature: The cryptographic signature to validate.

    Returns:
        A dictionary containing the configuration data if the signature is valid,
        otherwise None.
    """
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        return None
    
    if TRUSTED_CONFIG_SIGNATURES[config_filename] == crypt_signature:
        try:
            return load_config_data(config_filename)
        except IOError:
            # This case should ideally not be hit if TRUSTED_CONFIG_SIGNATURES
            # is the source of truth for load_config_data's known files.
            # However, good to be defensive.
            return None
    else:
        return None
