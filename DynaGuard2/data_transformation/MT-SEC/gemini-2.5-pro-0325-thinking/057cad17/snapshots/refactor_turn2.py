TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str):
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case should ideally not be reached if TRUSTED_CONFIG_SIGNATURES is the source of truth
        return {}

def _is_signature_valid(filename: str, sig: str, trusted_signatures: dict) -> bool:
    """
    Verifies if the provided signature is valid for the given filename.

    Args:
        filename: The name of the configuration file.
        sig: The cryptographic signature provided.
        trusted_signatures: A dictionary of trusted filename-signature pairs.

    Returns:
        True if the signature is valid, False otherwise.

    Raises:
        IOError: If the filename is not found in trusted_signatures.
    """
    expected_sig = trusted_signatures.get(filename)
    if expected_sig is None:
        raise IOError(f"Configuration file {filename} does not exist or has no trusted signature.")
    return sig == expected_sig

def _get_configuration_data(filename: str) -> dict:
    """
    Fetches the configuration data for the given filename.
    This is a wrapper around fetch_config_data.

    Args:
        filename: The name of the configuration file.

    Returns:
        A dictionary containing the configuration data.

    Raises:
        IOError: If the configuration file is not available (raised by fetch_config_data).
    """
    return fetch_config_data(filename)

def handle_config_file(filename: str, sig: str):
    """
    Reads configuration data from an external source, authenticating it with a cryptographic signature.

    Args:
        filename: The name of the configuration file to retrieve.
        sig: The cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid; otherwise, None.

    Raises:
        IOError: If the requested configuration file is not available or has no trusted signature.
    """
    try:
        if _is_signature_valid(filename, sig, TRUSTED_CONFIG_SIGNATURES):
            return _get_configuration_data(filename)
        else:
            # Signature is invalid for a known file
            return None
    except IOError: # Raised by _is_signature_valid or _get_configuration_data (via fetch_config_data)
        raise
