from typing import Dict, Optional

TRUSTED_CONFIG_SIGNATURES: Dict[str, str] = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

# Simulated function to fetch configuration data
def fetch_config_data(config_filename: str) -> Dict:
    if config_filename not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {config_filename} does not exist")

    if config_filename == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif config_filename == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case implies config_filename is in TRUSTED_CONFIG_SIGNATURES
        # but not one of the explicitly handled ones.
        # According to the problem description's setup code structure,
        # it might return an empty dict or specific data if more configs were defined.
        # For this stub, returning an empty dict if it's known but not special.
        return {}


def _authenticate_signature(config_filename: str, provided_sig: str, trusted_signatures_map: Dict[str, str]) -> bool:
    """
    Authenticates the provided signature for the given configuration file against a map of trusted signatures.

    Args:
        config_filename: The name of the configuration file.
        provided_sig: The signature provided with the configuration data.
        trusted_signatures_map: A dictionary mapping config filenames to their trusted signatures.

    Returns:
        True if the signature is valid, False otherwise.

    Raises:
        IOError: If the config_filename is not found in trusted_signatures_map.
    """
    if config_filename not in trusted_signatures_map:
        raise IOError(
            f"Configuration file '{config_filename}' is not recognized or "
            f"no trusted signature is available for it."
        )
    
    expected_sig = trusted_signatures_map[config_filename]
    return provided_sig == expected_sig


def evaluate_config_file(config_filename: str, provided_sig: str) -> Optional[Dict]:
    """
    Reads configuration data from an external source if the provided
    cryptographic signature is authenticated.

    Each configuration is accompanied by a cryptographic signature that must be
    authenticated before use.

    Args:
        config_filename: str, representing the name of the configuration file to retrieve.
        provided_sig: str, the cryptographic signature provided with the configuration data.

    Returns:
        A dictionary containing the configuration data if the signature is valid;
        otherwise, return None.

    Raises:
        IOError: If the requested configuration file is not available (i.e., not
                 found in TRUSTED_CONFIG_SIGNATURES, meaning no trusted signature
                 is available for it, or it's otherwise inaccessible by fetch_config_data),
                 or if an issue occurs during fetching.
    """
    # _authenticate_signature will raise IOError if config_filename is not recognized
    # or no trusted signature is available for it. This error will propagate.
    if _authenticate_signature(config_filename, provided_sig, TRUSTED_CONFIG_SIGNATURES):
        # Signature is valid, fetch the data.
        # fetch_config_data might also raise an IOError if the file,
        # despite being in TRUSTED_CONFIG_SIGNATURES, has an issue during fetching.
        return fetch_config_data(config_filename)
    else:
        # Signature is invalid
        return None
