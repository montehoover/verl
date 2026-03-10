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
        # This case implies TRUSTED_CONFIG_SIGNATURES might have keys
        # not explicitly handled by the if/elif chain above.
        return {}

def _verify_signature(file_name: str, sign: str, trusted_signatures: dict) -> bool:
    """
    Verifies the provided signature against the trusted signature for the file.

    Args:
        file_name: The name of the configuration file.
        sign: The cryptographic signature to verify.
        trusted_signatures: A dictionary of trusted file names and their signatures.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected_signature = trusted_signatures.get(file_name)
    return sign == expected_signature

def handle_config_file(file_name: str, sign: str):
    """
    Reads configuration information from an external source.
    Validates the cryptographic signature before utilizing the data.

    Args:
        file_name: str, identifying the configuration file.
        sign: str, the cryptographic signature corresponding to the configuration.

    Returns:
        A dictionary representing the configuration if the signature is verified,
        returns None if it's invalid.

    Raises:
        IOError: if the configuration file cannot be located.
    """
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} cannot be located")

    if _verify_signature(file_name, sign, TRUSTED_CONFIG_SIGNATURES):
        # Signature is valid, fetch the configuration data
        config_data = fetch_config_data(file_name)
        return config_data
    else:
        # Signature is invalid
        return None
