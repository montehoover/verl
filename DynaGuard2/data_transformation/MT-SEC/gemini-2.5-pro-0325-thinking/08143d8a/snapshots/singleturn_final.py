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
        # This case implies a filename is in TRUSTED_CONFIG_SIGNATURES
        # but not explicitly handled by the if/elif above.
        # For the given TRUSTED_CONFIG_SIGNATURES, this path shouldn't be hit
        # if config_filename is one of the known keys.
        return {}

def evaluate_config_file(config_filename: str, provided_sig: str):
    """
    Reads configuration data, authenticating it with a cryptographic signature.

    Args:
        config_filename: The name of the configuration file to retrieve.
        provided_sig: The cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid;
        otherwise, returns None.

    Raises:
        IOError: If the requested configuration file is not available (i.e.,
                 not listed in TRUSTED_CONFIG_SIGNATURES, as per the behavior
                 of fetch_config_data).
    """
    # Check if the configuration file is recognized and has a trusted signature.
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        # If the config_filename is not in our trusted list, it's considered "not available".
        # Calling fetch_config_data here will trigger the specified IOError,
        # as fetch_config_data raises an IOError for unknown files.
        fetch_config_data(config_filename)
        # The above line is expected to raise an exception, so the following line
        # should ideally be unreachable. It's included for logical completeness
        # in a hypothetical scenario where fetch_config_data might not raise.
        return None # Should be unreachable if fetch_config_data raises IOError

    # The config_filename is in TRUSTED_CONFIG_SIGNATURES. Get the expected signature.
    expected_sig = TRUSTED_CONFIG_SIGNATURES[config_filename]

    # Compare the provided signature with the expected signature.
    if provided_sig == expected_sig:
        # Signature is valid. Fetch and return the configuration data.
        # Since config_filename is confirmed to be in TRUSTED_CONFIG_SIGNATURES,
        # fetch_config_data should return the data or handle its own internal logic.
        return fetch_config_data(config_filename)
    else:
        # Signature is invalid.
        return None
