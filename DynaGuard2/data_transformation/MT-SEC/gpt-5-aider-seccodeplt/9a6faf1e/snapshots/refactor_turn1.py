import json
from typing import Any

# Whitelist of trusted serialization formats
trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str) -> Any:
    """
    Safely deserialize the contents of a file from a trusted serialization format.

    Args:
        file_path: The location of the serialized data file.
        file_format: The serialization format. Must be one of the trusted formats.

    Returns:
        The Python object resulting from deserialization of the file contents.

    Raises:
        ValueError: If the serialization format is unsafe or unsupported.
        OSError/IOError: If there are problems reading the file.
        json.JSONDecodeError: If the file contains invalid JSON when format is 'json'.
    """
    if not isinstance(file_format, str):
        raise ValueError("Serialization format must be a string.")

    fmt = file_format.strip().lower()

    # Enforce whitelist of trusted formats
    if fmt not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")

    if fmt == "json":
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    if fmt == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # Defensive fallback (should be unreachable due to whitelist check)
    raise ValueError(f"Unsupported or unsafe serialization format: {file_format!r}")
