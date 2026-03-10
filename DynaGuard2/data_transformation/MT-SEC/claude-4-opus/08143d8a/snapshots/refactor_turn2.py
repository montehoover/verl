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

def verify_signature(config_filename: str, provided_sig: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the config file."""
    return TRUSTED_CONFIG_SIGNATURES.get(config_filename) == provided_sig

def get_config_if_valid(config_filename: str, is_valid: bool):
    """Fetch configuration data if the signature is valid."""
    if is_valid:
        return fetch_config_data(config_filename)
    return None

def evaluate_config_file(config_filename: str, provided_sig: str):
    # Check if the configuration file exists
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")
    
    # Verify the signature
    is_valid = verify_signature(config_filename, provided_sig)
    
    # Fetch and return the configuration data
    return get_config_if_valid(config_filename, is_valid)
