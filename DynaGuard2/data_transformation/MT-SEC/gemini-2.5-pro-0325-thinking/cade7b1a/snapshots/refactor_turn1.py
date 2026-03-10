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
                 fetch_config_data if config_name is not in
                 TRUSTED_CONFIG_SIGNATURES).
    """
    trusted_signature = TRUSTED_CONFIG_SIGNATURES.get(config_name)

    if trusted_signature is None:
        # This case will also be caught by fetch_config_data raising an IOError
        # if we were to call it, but checking here first avoids calling
        # fetch_config_data unnecessarily if the config_name isn't even known.
        # The IOError will be raised by fetch_config_data if it's called.
        # If the requirement is to raise IOError *before* signature check if config_name is bad,
        # then fetch_config_data should be called first, or this check should raise.
        # Given the current fetch_config_data, it will raise if config_name is not in TRUSTED_CONFIG_SIGNATURES.
        # Let's ensure the signature check is primary for known configs.
        # If config_name is not in TRUSTED_CONFIG_SIGNATURES, fetch_config_data will raise IOError.
        # If it is, then we check the signature.
        pass # Allow fetch_config_data to handle the IOError for unknown config_name

    if trusted_signature == signature:
        # Signature matches, fetch the data.
        # fetch_config_data will raise IOError if config_name is not in its list,
        # which is TRUSTED_CONFIG_SIGNATURES.
        return fetch_config_data(config_name)
    else:
        # Signature does not match or config_name not in TRUSTED_CONFIG_SIGNATURES
        # If config_name was valid but signature was wrong, return None.
        # If config_name was invalid, fetch_config_data would raise IOError if called.
        # To ensure None is returned for bad signature for a *known* config:
        if config_name in TRUSTED_CONFIG_SIGNATURES and trusted_signature != signature:
            return None
        # If config_name is not in TRUSTED_CONFIG_SIGNATURES, fetch_config_data will handle the IOError.
        # This path implies either signature mismatch for a known config (handled above)
        # or config_name is not in TRUSTED_CONFIG_SIGNATURES.
        # To strictly adhere to "return None if signature is invalid", and "raise IOError if file not found":
        # 1. Check if config_name is in TRUSTED_CONFIG_SIGNATURES. If not, fetch_config_data will raise IOError.
        # 2. If it is, then check signature.
        
        # Revised logic for clarity:
        if config_name not in TRUSTED_CONFIG_SIGNATURES:
            # This will cause fetch_config_data to raise IOError as per its definition
            return fetch_config_data(config_name)

        if TRUSTED_CONFIG_SIGNATURES[config_name] == signature:
            return fetch_config_data(config_name)
        else:
            # Config name is known, but signature is invalid
            return None
