import json
from typing import Any

# Only allow handling of safe, predefined formats
allowed_formats = ["json", "html"]


def _coerce_byteslike(data: Any) -> bytes:
    """
    Ensure input is bytes-like and return a bytes object; otherwise raise TypeError.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("serialized_data must be a bytes-like object")
    return bytes(data)


def _normalize_and_validate_format(stream_format: Any) -> str:
    """
    Normalize the provided format string and validate it against allowed formats.
    Raises ValueError for unsupported or insecure formats.
    """
    if not isinstance(stream_format, str):
        raise ValueError("Unsupported or insecure format")
    fmt = stream_format.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format!r}")
    return fmt


def _decode_utf8_sig(raw: bytes) -> str:
    """
    Decode bytes using UTF-8 with BOM handling; map decode errors to ValueError.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Failed to decode input as UTF-8") from e


def _deserialize_by_format(raw: bytes, fmt: str) -> Any:
    """
    Perform the format-specific deserialization, assuming fmt is validated.
    """
    if fmt == "json":
        text = _decode_utf8_sig(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e
    elif fmt == "html":
        # For HTML, safely decode and return the raw string without executing or evaluating content.
        return _decode_utf8_sig(raw)
    # Defensive fallback; should be unreachable due to prior validation.
    raise ValueError(f"Unsupported or insecure format: {fmt!r}")


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
    raw = _coerce_byteslike(serialized_data)
    fmt = _normalize_and_validate_format(stream_format)
    return _deserialize_by_format(raw, fmt)
