import json
from typing import Any

# Only allow handling of safe, predefined formats
allowed_formats = ["json", "html"]


def deserialize_stream(serialized_data: bytes, stream_format: str) -> Any:
    """
    Deserialize a given byte stream back into a Python object using a restricted set of safe formats.

    Args:
        serialized_data: bytes - The serialized data that needs to be deserialized.
        stream_format: str - The deserialization format ('json' or 'html').

    Returns:
        The corresponding Python object reconstructed from the serialized content.

    Raises:
        ValueError: If the specified format is unsupported or considered insecure,
                    or if the content cannot be decoded/parsed for the given format.
        TypeError: If serialized_data is not a bytes-like object.
    """
    if not isinstance(serialized_data, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_data must be a bytes-like object")

    if not isinstance(stream_format, str):
        raise ValueError("Unsupported or insecure format")

    fmt = stream_format.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")

    try:
        if fmt == "json":
            text = bytes(serialized_data).decode("utf-8-sig")
            return json.loads(text)
        elif fmt == "html":
            # For HTML, safely decode and return the raw string without executing or evaluating content.
            return bytes(serialized_data).decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Failed to decode input as UTF-8") from e
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON data") from e
