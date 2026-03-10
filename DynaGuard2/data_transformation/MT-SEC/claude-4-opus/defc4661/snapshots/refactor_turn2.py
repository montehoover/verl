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

def verify_signature(file_name: str, sign: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the file."""
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        return False
    return TRUSTED_CONFIG_SIGNATURES[file_name] == sign

def get_configuration(file_name: str) -> dict:
    """Fetch configuration data for the given file."""
    return fetch_config_data(file_name)

def handle_config_file(file_name: str, sign: str):
    # Check if the file exists in trusted signatures
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")
    
    # Validate the signature
    if not verify_signature(file_name, sign):
        return None
    
    # Fetch and return the configuration data
    config_data = get_configuration(file_name)
    return config_data
