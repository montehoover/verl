from typing import Optional, Mapping, Callable


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


def _verify_signature(file_name: str, sign: str, trusted_signatures: Mapping[str, str]) -> bool:
    """
    Pure function: Validates that the provided signature matches the trusted signature for the file.
    Raises IOError if the file is not known.
    """
    if file_name not in trusted_signatures:
        raise IOError(f"Configuration file {file_name} does not exist")
    return sign == trusted_signatures[file_name]


def _fetch_configuration(file_name: str, fetcher: Callable[[str], dict]) -> dict:
    """
    Pure function: Returns configuration data using the supplied fetcher callable.
    """
    return fetcher(file_name)


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
    # Verify signature against trusted signatures (raises IOError if file is unknown)
    if not _verify_signature(file_name, sign, TRUSTED_CONFIG_SIGNATURES):
        return None

    # Fetch and return the configuration data
    return _fetch_configuration(file_name, fetch_config_data)
