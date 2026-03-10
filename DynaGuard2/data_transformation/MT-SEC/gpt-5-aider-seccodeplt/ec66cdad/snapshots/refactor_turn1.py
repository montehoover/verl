from typing import Any
import json

# Predefined list of allowed formats deemed safe for deserialization
safe_formats = ["json", "html"]


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

    if fmt == "json":
        try:
            text = byte_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid UTF-8 in JSON byte stream") from e
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "html":
        try:
            return byte_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid UTF-8 in HTML byte stream") from e

    # This point should not be reachable due to the safe_formats check above.
    raise ValueError(f"Unsupported or unsafe serialization format: {serialization_type!r}")
