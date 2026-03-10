import json
from typing import Any

# Whitelist of acceptable data formats
trusted_formats = ["json", "text"]

def read_serialized_file(file_path: str, file_format: str) -> Any:
    """
    Read serialized data from a file in a trusted format and return the deserialized object.

    Args:
        file_path: The path to the serialized data file.
        file_format: The format of the file ('json', 'text', 'pickle').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or unsafe.
        OSError: If file I/O fails.
        json.JSONDecodeError: If JSON decoding fails when format is 'json'.
    """
    fmt = (file_format or "").strip().lower()

    if fmt not in trusted_formats:
        raise ValueError(f"Unsupported or unsafe format: {file_format!r}")

    if fmt == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if fmt == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(f"Unsupported or unsafe format: {file_format!r}")
