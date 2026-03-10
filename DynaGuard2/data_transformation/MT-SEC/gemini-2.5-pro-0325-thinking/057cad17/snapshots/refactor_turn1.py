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

def handle_config_file(filename: str, sig: str):
    """
    Reads configuration data from an external source, authenticating it with a cryptographic signature.

    Args:
        filename: The name of the configuration file to retrieve.
        sig: The cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid; otherwise, None.

    Raises:
        IOError: If the requested configuration file is not available (raised by fetch_config_data).
    """
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(filename)

    if expected_sig is None:
        # This will ensure IOError is raised if the file is not supposed to exist,
        # even before calling fetch_config_data, or rely on fetch_config_data to do it.
        # Calling fetch_config_data directly handles the IOError as per its definition.
        # For clarity and to ensure the IOError is for non-availability as per spec:
        raise IOError(f"Configuration file {filename} does not exist or has no trusted signature.")

    if sig == expected_sig:
        return fetch_config_data(filename)
    else:
        return None
