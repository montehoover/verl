import json
from typing import Any

# Whitelist of accepted formats (secure deserialization)
allowed_serialized_formats = ["json", "text"]


def load_serialized_data(filepath: str, data_format: str) -> Any:
    """
    Securely load serialized data from a file, restricted to trusted formats.

    Args:
        filepath: Path to the serialized file.
        data_format: The format of the serialized data. Must be one of the allowed formats.

    Returns:
        The deserialized Python object extracted from the file content.

    Raises:
        ValueError: If the requested format is untrusted or prohibited.
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If 'json' format is selected and the file contains invalid JSON.
        OSError: If there is an error opening or reading the file.
    """
    if not isinstance(data_format, str):
        raise ValueError("Format must be a string.")

    fmt = data_format.strip().lower()

    # Enforce whitelist
    if fmt not in allowed_serialized_formats:
        raise ValueError(
            f"Untrusted or unsupported data format: {data_format!r}. "
            f"Allowed formats: {', '.join(allowed_serialized_formats)}"
        )

    if fmt == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # This point should be unreachable due to whitelist check above
    raise ValueError("Internal error: unsupported format reached.")
