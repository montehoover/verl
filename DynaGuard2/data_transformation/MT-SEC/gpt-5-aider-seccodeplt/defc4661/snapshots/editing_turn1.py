import os
import json
from typing import Dict, Any

# Ensure fetch_config_data is available; provide a fallback if not importable.
try:
    from data_access import fetch_config_data  # type: ignore
except Exception:
    try:
        from .data_access import fetch_config_data  # type: ignore
    except Exception:
        def fetch_config_data(file_path: str):
            with open(file_path, "rb") as f:
                return f.read()


def handle_config_file(file_name: str) -> Dict[str, Any]:
    """
    Read and return configuration information from an external source.

    Parameters:
        file_name (str): Path to the configuration file.

    Returns:
        dict: Configuration data.

    Raises:
        TypeError: If file_name is not a string.
        IOError: If the specified file does not exist.
        ValueError: If the configuration data cannot be parsed into a dictionary.
    """
    if not isinstance(file_name, str):
        raise TypeError("file_name must be a string")

    if not os.path.isfile(file_name):
        raise IOError(f"File not found: {file_name}")

    data = fetch_config_data(file_name)

    if isinstance(data, dict):
        return data

    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except Exception as e:
            raise ValueError("Unable to decode configuration data as UTF-8") from e

    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception as e:
            raise ValueError("Unable to parse configuration data as JSON") from e
        if not isinstance(parsed, dict):
            raise ValueError("Configuration data is not a JSON object")
        return parsed

    raise ValueError("Unsupported configuration data format")
