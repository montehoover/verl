from typing import Dict, Optional

TRUSTED_CONFIG_SIGNATURES = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def fetch_config_data(filename: str) -> Dict:
    if filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {filename} does not exist")

    if filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case should ideally not be reached if filename is in TRUSTED_CONFIG_SIGNATURES
        # and the above conditions cover all keys in TRUSTED_CONFIG_SIGNATURES.
        # However, to be safe and handle potential inconsistencies:
        return {}

def handle_config_file(filename: str, sig: str) -> Optional[Dict]:
    """
    Reads configuration data from an external source and verifies it using cryptographic signatures.

    Args:
        filename: The name of the configuration file to retrieve.
        sig: The cryptographic signature.

    Returns:
        The configuration data if the signature is valid and None otherwise.
    """
    expected_sig = TRUSTED_CONFIG_SIGNATURES.get(filename)
    if expected_sig is None:
        # Filename not found in trusted signatures
        return None

    if sig == expected_sig:
        try:
            config_data = fetch_config_data(filename)
            return config_data
        except IOError:
            # fetch_config_data might raise IOError if filename is not found,
            # though this should be caught by the initial check.
            # This also handles other potential IOErrors from fetch_config_data.
            return None
    else:
        # Signature mismatch
        return None
