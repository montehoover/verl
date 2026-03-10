from typing import Any, Callable, Dict
import json

# Predefined list of allowed formats deemed safe for deserialization
safe_formats = ["json", "html"]


def _decode_utf8_or_raise(byte_data: bytes, context: str) -> str:
    """
    Decode bytes as UTF-8, raising ValueError with a context-specific message on failure.
    """
    try:
        return byte_data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 in {context} byte stream") from e


def _deserialize_json(byte_data: bytes) -> Any:
    """
    Pure function to deserialize JSON bytes into a Python object.
    """
    text = _decode_utf8_or_raise(byte_data, "JSON")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON data") from e


def _deserialize_html(byte_data: bytes) -> str:
    """
    Pure function to 'deserialize' HTML bytes; returns the decoded string.
    """
    return _decode_utf8_or_raise(byte_data, "HTML")


_DESERIALIZERS: Dict[str, Callable[[bytes], Any]] = {
    "json": _deserialize_json,
    "html": _deserialize_html,
}


def restore_object_from_stream(byte_data: bytes, serialization_type: str) -> Any:
    """
    Deserialize a byte stream into a Python object for known safe formats.

    Args:
        byte_data: The serialized stream of data as bytes.
        serialization_type: The format of the serialized data (e.g., 'json', 'html').

    Returns:
        The Python object reconstructed from the byte stream.

    Raises:
        ValueError: If the serialization_type is unsupported or unsafe, or if
                    the data cannot be decoded/parsed for the given format.
    """
    fmt = (serialization_type or "").strip().lower()
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe serialization format: {serialization_type!r}")

    deserializer = _DESERIALIZERS.get(fmt)
    if deserializer is None:
        raise ValueError(f"No deserializer implemented for format: {serialization_type!r}")

    return deserializer(byte_data)
