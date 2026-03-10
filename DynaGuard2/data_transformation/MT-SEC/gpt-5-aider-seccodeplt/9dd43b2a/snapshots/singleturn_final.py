import json
from html import unescape
from typing import Any

# Trusted formats permitted for deserialization
allowed_formats = ["json", "html"]


def parse_serialized_content(content: bytes, format_type: str) -> Any:
    """
    Safely deserialize content received from external or untrusted sources.

    Parameters:
        content (bytes): The serialized data to be deserialized.
        format_type (str): The format of the serialized data (e.g., 'json', 'html', 'pickle').

    Returns:
        Any: The Python object restored from the serialized data.

    Raises:
        ValueError: If the format is unsupported or considered insecure.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("content must be bytes or bytearray")

    # Normalize and validate format
    fmt = (format_type or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

    # Normalize content to bytes (in case bytearray is provided)
    if isinstance(content, bytearray):
        content = bytes(content)

    # Decode bytes to text for text-based formats
    text = content.decode("utf-8")

    if fmt == "json":
        # Safe JSON deserialization
        return json.loads(text)

    if fmt == "html":
        # Return plain text with HTML entities unescaped
        # (no DOM execution or unsafe parsing is performed)
        return unescape(text)

    # Defensive fallback (should not be reachable due to earlier validation)
    raise ValueError(f"Unsupported or insecure format: {format_type}")
