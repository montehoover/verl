import json
from typing import Any

# Allowed formats for deserialization (provided setup)
allowed_formats = ["json", "html"]


def _validate_stream_format(stream_format: str) -> str:
    """
    Validate and normalize the provided stream format.

    Args:
        stream_format: str - The input format string.

    Returns:
        A normalized, validated format string.

    Raises:
        ValueError: If the format is unsupported or considered insecure.
    """
    fmt = (stream_format or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")
    return fmt


def _ensure_bytes(stream: bytes) -> bytes:
    """
    Ensure the input is a bytes-like object and normalize it to bytes.

    Args:
        stream: The serialized data (bytes-like).

    Returns:
        A bytes object.

    Raises:
        TypeError: If stream is not bytes-like.
    """
    if isinstance(stream, (bytearray, memoryview)):
        return bytes(stream)
    if isinstance(stream, bytes):
        return stream
    raise TypeError("stream must be a bytes-like object (bytes, bytearray, or memoryview)")


def _deserialize_bytes_by_format(data_bytes: bytes, fmt: str) -> Any:
    """
    Deserialize bytes according to a validated format.

    Args:
        data_bytes: The input data as bytes.
        fmt: A validated format string (e.g., 'json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If an unexpected/unsupported format is encountered.
        UnicodeDecodeError / json.JSONDecodeError: If decoding/parsing fails.
    """
    if fmt == "json":
        text = data_bytes.decode("utf-8")
        return json.loads(text)
    if fmt == "html":
        return data_bytes.decode("utf-8")
    # Should not happen if validation is correct; kept for safety.
    raise ValueError(f"Unsupported or insecure format: {fmt!r}")


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
        TypeError: If the stream is not bytes-like.
        UnicodeDecodeError / json.JSONDecodeError: If decoding/parsing fails for valid formats.
    """
    fmt = _validate_stream_format(stream_format)
    data_bytes = _ensure_bytes(stream)
    return _deserialize_bytes_by_format(data_bytes, fmt)
