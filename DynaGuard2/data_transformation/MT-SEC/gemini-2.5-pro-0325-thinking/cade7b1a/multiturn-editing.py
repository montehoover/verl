TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(config_name: str):
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")

    if config_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def process_config_data(config_name: str, signature: str) -> dict | None:
    """
    Loads configuration data from an external source and verifies it
    using cryptographic signatures.

    Args:
        config_name: The name of the configuration file to load.
        signature: The cryptographic signature to verify.

    Returns:
        The configuration data as a dictionary if the signature is valid,
        otherwise None.
    """
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        return None  # Or raise an error, depending on desired behavior for unknown config

    expected_signature = TRUSTED_CONFIG_SIGNATURES[config_name]
    if signature != expected_signature:
        return None

    try:
        config_data = fetch_config_data(config_name)
        return config_data
    except IOError:
        # This case might be redundant if TRUSTED_CONFIG_SIGNATURES check is robust
        # or if fetch_config_data can fail for other reasons post-signature check.
        return None
