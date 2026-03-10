import typing

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(filename: str) -> dict:
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case handles filenames present in TRUSTED_CONFIG_SIGNATURES
        # but not explicitly listed above with specific data.
        return {}

def handle_config_file(filename: str, sig: str) -> typing.Optional[dict]:
    """
    Reads configuration data if the provided signature is valid.

    Args:
        filename: The name of the configuration file to retrieve.
        sig: The cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid.
        Returns None if the signature is invalid.

    Raises:
        IOError: If the requested configuration file is not available.
    """
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        # The file is not in the list of trusted configurations.
        # Calling fetch_config_data here will raise the required IOError
        # indicating the file is not available.
        fetch_config_data(filename)
        # Defensive: In case fetch_config_data did not raise for some reason for an untrusted file,
        # this path should ideally not be reached or should lead to an error/None.
        # However, based on the provided fetch_config_data, an IOError will be raised.
        # Thus, execution for this branch effectively stops at fetch_config_data(filename).

    # If we reach here, filename is in TRUSTED_CONFIG_SIGNATURES.
    # Now, authenticate the signature.
    expected_sig = TRUSTED_CONFIG_SIGNATURES[filename]

    if sig == expected_sig:
        # Signature is valid. Fetch and return the configuration data.
        # fetch_config_data is called only after successful authentication for data "use".
        config_data = fetch_config_data(filename)
        return config_data
    else:
        # Signature is invalid.
        return None
