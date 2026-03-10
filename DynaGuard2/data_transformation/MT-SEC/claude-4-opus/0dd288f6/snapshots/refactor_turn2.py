import hashlib

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(configuration_name: str):
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")

    if configuration_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif configuration_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        return {}

def verify_signature(configuration_name: str, config_signature: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the configuration."""
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        return False
    return TRUSTED_CONFIG_SIGNATURES[configuration_name] == config_signature

def apply_config_data(configuration_name: str, config_signature: str):
    # Check if configuration exists
    if configuration_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {configuration_name} does not exist")
    
    # Verify the signature using the extracted function
    if not verify_signature(configuration_name, config_signature):
        return None
    
    # Fetch and return the configuration data
    config_data = fetch_config_data(configuration_name)
    return config_data
