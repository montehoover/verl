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
    # Check if the file exists in TRUSTED_CONFIG_SIGNATURES
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")
    
    # Verify the cryptographic signature
    if TRUSTED_CONFIG_SIGNATURES[config_filename] == crypt_signature:
        # Load and return the configuration data
        return load_config_data(config_filename)
    else:
        # Invalid signature
        return None
