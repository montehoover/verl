import json
from typing import Any

# Predefined list of formats considered safe for deserialization
allowed_formats = ["json", "html"]


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

    fmt = format_type.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

    # Normalize to bytes instance for consistent handling
    content_bytes = bytes(content)

    if fmt == "json":
        # Decode as UTF-8 and parse JSON safely
        text = content_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "html":
        # For HTML, return the decoded string as-is (no execution)
        return content_bytes.decode("utf-8")

    # Redundant guard; should not be reached due to allowed_formats check above
    raise ValueError(f"Unsupported or insecure format: {format_type}")
