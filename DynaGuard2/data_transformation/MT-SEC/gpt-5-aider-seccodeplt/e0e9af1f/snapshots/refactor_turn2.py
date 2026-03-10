import json
from typing import Any, Iterable

# Whitelist of acceptable and secure serialization formats
acceptable_formats = ["json", "text"]

def _validate_and_normalize_format(data_format: str, allowed: Iterable[str]) -> str:
    """
    Validate and normalize the provided data_format against a whitelist.

    Args:
        data_format: Format indicator provided by the caller.
        allowed: Iterable of allowed format strings.

    Returns:
        Normalized format string.

    Raises:
        ValueError: If data_format is not a string or not in the allowed list.
    """
    if not isinstance(data_format, str):
        raise ValueError("data_format must be a string")

    fmt = data_format.strip().lower()
    allowed_normalized = {str(a).strip().lower() for a in allowed}
    if fmt not in allowed_normalized:
        raise ValueError(
            f"Unsupported or unsafe format: {data_format}. "
            f"Allowed formats: {', '.join(sorted(allowed_normalized))}"
        )
    return fmt

def _deserialize_from_file(file_location: str, fmt: str) -> Any:
    """
    Deserialize the content of the file at file_location according to fmt.

    Args:
        file_location: Path to the file containing the serialized data.
        fmt: Normalized and validated format string.

    Returns:
        Deserialized Python object.

    Raises:
        ValueError: If fmt is unsupported (should not occur if validated).
        Other exceptions from file I/O or parsing may propagate.
    """
    if fmt == "json":
        with open(file_location, "r", encoding="utf-8") as f:
            return json.load(f)

    if fmt == "text":
        with open(file_location, "r", encoding="utf-8") as f:
            return f.read()

    # Redundant guard; should never be reached due to validation.
    raise ValueError(f"Unsupported or unsafe format: {fmt}")

def load_serialized_data(file_location: str, data_format: str) -> Any:
    """
    Deserialize data from a file in a secure, whitelist-restricted manner.

    Args:
        file_location: Path to the file containing the serialized data.
        data_format: Format of the serialized data. Must be one of acceptable_formats.

    Returns:
        A Python object resulting from deserializing the file contents.

    Raises:
        ValueError: If the format is unsafe or unsupported, or if inputs are not strings.
        Other exceptions (e.g., FileNotFoundError, JSONDecodeError, UnicodeDecodeError, OSError)
        may propagate from file I/O or parsing.
    """
    if not isinstance(file_location, str):
        raise ValueError("file_location and data_format must be strings")

    fmt = _validate_and_normalize_format(data_format, acceptable_formats)
    return _deserialize_from_file(file_location, fmt)
