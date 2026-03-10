import os
from typing import Optional

# Simulated function to fetch configuration data
def fetch_config_data(file_name: str) -> dict:
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file {file_name} does not exist")

    if file_name == "db_config.json":
        return {"database": "mysql", "user": "admin", "password": "secret"}
    elif file_name == "app_config.yaml":
        return {"app_name": "AwesomeApp", "version": "1.2.3"}
    else:
        # This case implies that TRUSTED_CONFIG_SIGNATURES might contain keys
        # other than "db_config.json" or "app_config.yaml", or this is a fallback.
        return {}

# Global dictionary for trusted configuration file signatures
TRUSTED_CONFIG_SIGNATURES: dict = {
    "db_config.json": "afae23c5b1a7d5218cabc4e62357378e",
    "app_config.yaml": "9c6b74265e0e0409e4ada1aa3bdc973d"
}

def handle_config_file(file_name: str, sign: str) -> Optional[dict]:
    """
    Checks if the file is listed, verifies its cryptographic signature,
    and returns the configuration if valid.

    Args:
        file_name: The name of the configuration file.
        sign: The cryptographic signature to verify.

    Returns:
        A dictionary containing the configuration if the signature matches,
        None otherwise.

    Raises:
        IOError: If the configuration file is not a trusted source or cannot be located.
    """
    if file_name not in TRUSTED_CONFIG_SIGNATURES:
        raise IOError(f"Configuration file '{file_name}' is not a trusted source or cannot be located.")

    expected_sign = TRUSTED_CONFIG_SIGNATURES[file_name]
    if sign == expected_sign:
        # Signature matches, fetch and return the data.
        # fetch_config_data will be called, which also checks if the file is in TRUSTED_CONFIG_SIGNATURES.
        return fetch_config_data(file_name)
    else:
        # Signatures do not match.
        return None
