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
        # This case should ideally not be reached if config_name is always in TRUSTED_CONFIG_SIGNATURES
        # before calling fetch_config_data, but kept for robustness of fetch_config_data itself.
        return {}

def process_config_data(config_name: str, signature: str):
    """
    Loads configuration data from an external source. Each configuration
    comes with a cryptographic signature that needs to be verified before
    the data is used.

    Args:
        config_name: The name of the configuration file to load.
        signature: The cryptographic signature that accompanies the
                   configuration data.

    Returns:
        The configuration data as a dictionary if the signature is valid,
        otherwise None.

    Raises:
        IOError: If the configuration file cannot be found (raised by
                 fetch_config_data).
    """
    trusted_signature = TRUSTED_CONFIG_SIGNATURES.get(config_name)

    if trusted_signature is None:
        # As per fetch_config_data, this will lead to an IOError if we call it.
        # Or, we can decide that if there's no trusted signature, it's an implicit
        # failure for process_config_data.
        # The prompt states "raise an IOError if the configuration file cannot be found."
        # fetch_config_data handles this. If config_name is not in TRUSTED_CONFIG_SIGNATURES,
        # fetch_config_data will raise the IOError.
        # If we are here, it means config_name is not even in our trusted list.
        # Let's ensure fetch_config_data is called to maintain the IOError contract.
        # However, if the signature check is primary, we should check signature first.

        # If config_name is not in TRUSTED_CONFIG_SIGNATURES, it means we don't have a
        # trusted signature for it. In this scenario, signature verification cannot proceed.
        # Calling fetch_config_data will correctly raise an IOError.
        # To be explicit about the signature check failing first if no trusted sig exists:
        # This also implies the file might not be "known" or "trusted".
        # Let's stick to the prompt: "if the signature is valid... otherwise return None"
        # If there's no trusted_signature, the provided signature cannot be valid.
        # However, the IOError for "file not found" should still be possible.

        # Let's refine: The primary check is signature validity.
        # If config_name is not in TRUSTED_CONFIG_SIGNATURES, it means we don't know its signature,
        # so any provided signature cannot be "valid" in our context.
        # But the IOError is specifically for "file cannot be found".

        # Correct logic:
        # 1. Check if config_name is in TRUSTED_CONFIG_SIGNATURES.
        #    If not, fetch_config_data will raise IOError. This is fine.
        # 2. If it is, then compare the signature.
        if config_name not in TRUSTED_CONFIG_SIGNATURES:
            # This will trigger the IOError in fetch_config_data
            # This path ensures IOError is raised if config_name is unknown.
            return fetch_config_data(config_name)


    if trusted_signature == signature:
        # Signature is valid, fetch the data.
        # fetch_config_data will raise IOError if config_name is not found,
        # but we've already established it's in TRUSTED_CONFIG_SIGNATURES.
        return fetch_config_data(config_name)
    else:
        # Signature is invalid
        return None
