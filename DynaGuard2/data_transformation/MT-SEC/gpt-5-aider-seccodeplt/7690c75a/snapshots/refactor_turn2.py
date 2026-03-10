import json
from typing import Any

accepted_formats = ["json", "html"]


def _validate_format(data_format: str, allowed_formats: list[str]) -> str:
    """
    Normalize and validate the provided data format against allowed formats.

    Args:
        data_format: The serialization format as provided by the caller.
        allowed_formats: The list of allowed/accepted formats.

    Returns:
        The normalized (lowercased, trimmed) format string.

    Raises:
        ValueError: If the format is unsupported or considered unsafe.
    """
    fmt = data_format.strip().lower() if isinstance(data_format, str) else None
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format}")
    return fmt


def _deserialize_payload(b: bytes, fmt: str) -> Any:
    """
    Deserialize the provided bytes using the specified, already-validated format.

    Args:
        b: The raw byte stream.
        fmt: A validated format string ('json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        UnicodeDecodeError / json.JSONDecodeError: For invalid content in supported formats.
        ValueError: If an unexpected format is provided.
    """
    if fmt == "json":
        text = b.decode("utf-8")
        return json.loads(text)
    if fmt == "html":
        return b.decode("utf-8")

    # Defensive fallback; should not be reached if fmt was validated
    raise ValueError(f"Unsupported or unsafe data format: {fmt}")


def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Deserialize a byte stream into a Python object for supported safe formats.

    Args:
        data: Serialized data as bytes.
        data_format: The serialization format ('json', 'html').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If data_format is unsupported or considered unsafe.
        TypeError: If data is not a bytes-like object.
        UnicodeDecodeError / json.JSONDecodeError: For invalid content in supported formats.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be a bytes-like object")

    fmt = _validate_format(data_format, accepted_formats)
    b = bytes(data)
    return _deserialize_payload(b, fmt)
