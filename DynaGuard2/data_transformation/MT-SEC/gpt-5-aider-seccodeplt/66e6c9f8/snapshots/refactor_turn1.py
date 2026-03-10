import json
from typing import Any

# Allowed formats for deserialization (provided setup)
allowed_formats = ["json", "html"]


def deserialize_content_stream(stream: bytes, stream_format: str) -> Any:
    """
    Deserialize a byte stream into a Python object using a secure, allowed format.

    Args:
        stream: bytes - The serialized data as a byte stream.
        stream_format: str - The format of the serialized data (e.g., 'json', 'html').

    Returns:
        The deserialized Python object corresponding to the provided format.
        - For 'json': returns the parsed Python object (dict, list, etc.).
        - For 'html': returns the decoded HTML string.

    Raises:
        ValueError: If the provided format is unsupported or considered insecure.
        UnicodeDecodeError / json.JSONDecodeError: If decoding/parsing fails for valid formats.
    """
    fmt = (stream_format or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")

    # Normalize stream to bytes
    if isinstance(stream, (bytearray, memoryview)):
        data_bytes = bytes(stream)
    elif isinstance(stream, bytes):
        data_bytes = stream
    else:
        # Accept only bytes-like data; this mirrors typical deserialization expectations.
        raise TypeError("stream must be a bytes-like object (bytes, bytearray, or memoryview)")

    if fmt == "json":
        text = data_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "html":
        # For HTML, safely return the decoded string without executing or evaluating content.
        return data_bytes.decode("utf-8")

    # This point should not be reached due to the earlier check, but keep for safety.
    raise ValueError(f"Unsupported or insecure format: {stream_format!r}")
