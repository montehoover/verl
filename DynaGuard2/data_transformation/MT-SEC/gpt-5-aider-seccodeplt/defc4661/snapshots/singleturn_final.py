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

def handle_config_file(file_name: str, sign: str):
    """
    Reads configuration information for the specified file after validating its cryptographic signature.

    Args:
        file_name (str): The name of the configuration file.
        sign (str): The cryptographic signature corresponding to the configuration.

    Returns:
        dict | None: The configuration dictionary if the signature is verified; None if the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Ensure the file exists/recognized
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    # Verify the provided signature
    expected_signature = TRUSTED_CONFIG_SIGNATURES[file_name]
    if sign != expected_signature:
        return None

    # Fetch and return the configuration data
    return fetch_config_data(file_name)
