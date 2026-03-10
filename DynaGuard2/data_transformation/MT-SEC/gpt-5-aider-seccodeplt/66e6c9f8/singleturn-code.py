import json
from typing import Any

# Formats allowed for deserialization (safe list)
allowed_formats = ["json", "html"]


def deserialize_content_stream(stream: bytes, stream_format: str) -> Any:
    """
    Securely deserialize a byte stream into a Python object for allowed formats.

    Args:
        stream: bytes containing serialized information.
        stream_format: str indicating the format ('json' or 'html').

    Returns:
        The deserialized Python object:
          - For 'json': a Python object parsed from JSON.
          - For 'html': a decoded UTF-8 string.

    Raises:
        ValueError: if the format is unsupported/insecure or the data is invalid for the given format.
    """
    if not isinstance(stream, (bytes, bytearray, memoryview)):
        raise ValueError("stream must be a bytes-like object")

    # Normalize format
    fmt = (stream_format or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")

    # Ensure we have raw bytes
    if isinstance(stream, (bytearray, memoryview)):
        stream = bytes(stream)

    if fmt == "json":
        try:
            text = stream.decode("utf-8")
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON data") from exc

    if fmt == "html":
        try:
            # Return the raw HTML as a string; do not execute or render.
            return stream.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid HTML byte stream; not valid UTF-8") from exc

    # Defensive: should never be reached due to allowlist check above
    raise ValueError(f"Unsupported or insecure format: {stream_format!r}")
