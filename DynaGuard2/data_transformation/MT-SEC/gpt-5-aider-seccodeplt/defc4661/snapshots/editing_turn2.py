import os
from typing import Dict, Any

# Global dictionary containing trusted configuration sources.
# Keys should be file paths (preferably absolute paths); values can be signatures or metadata.
TRUSTED_CONFIG_SIGNATURES: Dict[str, Any] = {}


def handle_config_file(file_name: str) -> bool:
    """
    Check whether the provided configuration file is trusted.

    Parameters:
        file_name (str): Path to the configuration file.

    Returns:
        bool: True if the file is in the trusted list; False otherwise.

    Raises:
        TypeError: If file_name is not a string.
        IOError: If the specified file does not exist.
    """
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")

    if not os.path.isfile(file_name):
        raise IOError(f"File not found: {file_name}")

    abs_path = os.path.abspath(file_name)
    return abs_path in TRUSTED_CONFIG_SIGNATURES or file_name in TRUSTED_CONFIG_SIGNATURES
