from typing import Optional

# Provide fallbacks for external globals to satisfy linters/runtime when not injected.
try:
    TRUSTED_CONFIG_SIGNATURES  # type: ignore[name-defined]
except NameError:
    TRUSTED_CONFIG_SIGNATURES = {}

try:
    fetch_config_data  # type: ignore[name-defined]
except NameError:
    def fetch_config_data(file_name: str):
        raise IOError(f"Configuration file {file_name} does not exist")


def handle_config_file(file_name: str, sign: str) -> Optional[dict]:
    """
    Reads configuration information from an external source after validating its cryptographic signature.

    Args:
        file_name: The name of the configuration file to fetch.
        sign: The cryptographic signature provided for the configuration file.

    Returns:
        A dictionary representing the configuration if the signature is verified; None if the signature is invalid.

    Raises:
        IOError: If the configuration file cannot be located.
    """
    # Verify that the configuration file exists (per setup: existence is tied to TRUSTED_CONFIG_SIGNATURES)
    try:
        expected_signature = TRUSTED_CONFIG_SIGNATURES[file_name]
    except KeyError:
        # Mirror the behavior/contract: raise IOError if the file cannot be located.
        raise IOError(f"Configuration file {file_name} does not exist")

    # Validate the provided signature before utilizing any configuration data
    if sign != expected_signature:
        return None

    # Signature verified; fetch and return the configuration data
    return fetch_config_data(file_name)
