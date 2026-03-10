import json
from html import unescape
from typing import Any

# Predefined list of allowed formats deemed safe for deserialization
safe_formats = ["json", "html"]


def restore_object_from_stream(byte_data: bytes, serialization_type: str) -> Any:
    """
    Safely restore a Python object from a serialized byte stream.

    Args:
        byte_data: bytes - The serialized stream of data.
        serialization_type: str - The format of the serialized data ('json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If an unsupported or unsafe format is requested.
        UnicodeDecodeError / json.JSONDecodeError: If decoding/parsing fails.
    """
    # Basic validation and normalization
    if not isinstance(serialization_type, str):
        raise ValueError("serialization_type must be a string specifying a supported safe format")

    fmt = serialization_type.strip().lower()
    if fmt not in {s.lower() for s in safe_formats}:
        raise ValueError(f"Unsupported or unsafe serialization format: {serialization_type!r}")

    # Ensure we are working with a bytes-like object
    data_bytes = bytes(byte_data)

    if fmt == "json":
        text = data_bytes.decode("utf-8", errors="strict")
        return json.loads(text)

    if fmt == "html":
        text = data_bytes.decode("utf-8", errors="strict")
        # Return a plain string with HTML entities unescaped
        return unescape(text)

    # Defensive fallback; should not be reachable due to earlier check
    raise ValueError(f"Unsupported or unsafe serialization format: {serialization_type!r}")
