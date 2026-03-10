import json
from typing import Any, Iterable

# Predefined list of formats considered safe for deserialization
allowed_formats = ["json", "html"]


def _normalize_format(format_type: str) -> str:
    """
    Normalize a format string (trim and lowercase).
    """
    return format_type.strip().lower()


def _validate_format(format_type: str, allowed: Iterable[str]) -> str:
    """
    Validate that the provided format_type is among allowed formats.

    Returns:
        The normalized format string.

    Raises:
        ValueError: If the format is unsupported or insecure.
    """
    normalized = _normalize_format(format_type)
    if normalized not in allowed:
        raise ValueError(f"Unsupported or insecure format: {format_type}")
    return normalized


def _deserialize(content_bytes: bytes, fmt: str) -> Any:
    """
    Deserialize bytes according to the specified, already-validated format.

    Args:
        content_bytes: The serialized data as bytes.
        fmt: A validated, normalized format (e.g., 'json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is not recognized (should not happen if validated).
        UnicodeDecodeError / json.JSONDecodeError: For invalid encoding/JSON data.
    """
    if fmt == "json":
        text = content_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "html":
        return content_bytes.decode("utf-8")

    # Redundant guard; should not be reached due to validation
    raise ValueError(f"Unsupported or insecure format: {fmt}")


def parse_serialized_content(content: bytes, format_type: str) -> Any:
    """
    Deserialize content from a trusted format into a Python object.

    Args:
        content: bytes - The serialized data to be deserialized.
        format_type: str - The format of the serialized data ('json', 'html').

    Returns:
        The Python object restored from the serialized data.

    Raises:
        ValueError: If the format is unsupported or considered insecure.
        TypeError: If content is not bytes-like or format_type is not a string.
        UnicodeDecodeError / json.JSONDecodeError: For invalid encoding/JSON data.
    """
    if not isinstance(content, (bytes, bytearray, memoryview)):
        raise TypeError("content must be a bytes-like object")
    if not isinstance(format_type, str):
        raise TypeError("format_type must be a string")

    fmt = _validate_format(format_type, allowed_formats)

    # Normalize to bytes instance for consistent handling
    content_bytes = bytes(content)

    return _deserialize(content_bytes, fmt)
