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

def verify_signature(config_name: str, signature: str) -> bool:
    """Pure function to verify if the signature matches the trusted signature."""
    return TRUSTED_CONFIG_SIGNATURES.get(config_name) == signature

def get_config_data(config_name: str) -> dict:
    """Pure function to fetch configuration data."""
    return fetch_config_data(config_name)

def process_config_data(config_name: str, signature: str):
    # Check if the configuration file exists in trusted signatures
    if config_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_name} does not exist")
    
    # Verify the signature
    if not verify_signature(config_name, signature):
        return None
    
    # Fetch and return the configuration data
    return get_config_data(config_name)
