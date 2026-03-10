# coding: utf-8
import json
from typing import Any

# Whitelist of allowed/secure formats
acceptable_formats = ["json", "text"]

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Deserialize data from an external file using only secure, whitelisted formats.

    Args:
        file_location: Path to the file containing the serialized data.
        data_format: Serialization format, restricted to whitelisted options.

    Returns:
        A Python object resulting from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported.
        Other I/O or parsing exceptions (e.g., FileNotFoundError, UnicodeDecodeError,
        json.JSONDecodeError) may be raised and should be handled by the caller as needed.
    """
    if not isinstance(file_location, str) or not isinstance(data_format, str):
        raise ValueError("file_location and data_format must be strings")

    fmt = data_format.strip().lower()

    if fmt not in acceptable_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}")

    if fmt == "json":
        with open(file_location, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        with open(file_location, "r", encoding="utf-8") as f:
            return f.read()

    # Defensive: should not be reachable due to whitelist check above
    raise ValueError(f"Unsupported or unsafe format: {data_format}")
