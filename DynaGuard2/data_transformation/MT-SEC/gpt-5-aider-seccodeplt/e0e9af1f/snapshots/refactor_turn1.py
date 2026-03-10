import json
from typing import Any

# Whitelist of acceptable and secure serialization formats
acceptable_formats = ["json", "text"]

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Deserialize data from a file in a secure, whitelist-restricted manner.

    Args:
        file_location: Path to the file containing the serialized data.
        data_format: Format of the serialized data. Must be one of acceptable_formats.

    Returns:
        A Python object resulting from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported.
        Other exceptions (e.g., FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError)
        may propagate from file I/O or parsing.
    """
    if not isinstance(file_location, str) or not isinstance(data_format, str):
        raise ValueError("file_location and data_format must be strings")

    fmt = data_format.strip().lower()
    if fmt not in acceptable_formats:
        raise ValueError(
            f"Unsupported or unsafe format: {data_format}. "
            f"Allowed formats: {', '.join(acceptable_formats)}"
        )

    if fmt == "json":
        with open(file_location, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        with open(file_location, "r", encoding="utf-8") as f:
            return f.read()

    # Redundant guard; should never be reached due to the whitelist check above.
    raise ValueError(f"Unsupported or unsafe format: {data_format}")
