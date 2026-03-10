from typing import Dict, Any, Optional


def handle_config_file(file_name: str, sign: str) -> Optional[Dict[str, Any]]:
    """
    Verify the provided configuration file against a trusted signature list and return its data.

    Parameters:
        file_name (str): Name of the configuration file.
        sign (str): Provided cryptographic signature to verify.

    Returns:
        dict | None: Configuration data if the signature matches; otherwise None.

    Raises:
        TypeError: If file_name or sign is not a string.
        IOError: If the configuration file cannot be located.
    """
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")
    if not isinstance(sign, str):
        raise TypeError("sign must be a string")

    trusted = globals().get("TRUSTED_CONFIG_SIGNATURES")
    if not isinstance(trusted, dict) or file_name not in trusted:
        raise IOError(f"Configuration file {file_name} does not exist")

    expected_signature = trusted.get(file_name)
    if expected_signature != sign:
        return None

    fetch_fn = globals().get("fetch_config_data")
    if not callable(fetch_fn):
        raise IOError(f"Configuration file {file_name} does not exist")

    config = fetch_fn(file_name)  # type: ignore[misc]
    return config if isinstance(config, dict) else None
