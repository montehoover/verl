approved_formats = ["json", "xml"]


def json_to_object(data: dict) -> int:
    """
    Return the number of top-level keys in the given dictionary.

    Raises:
        TypeError: If data is not a dictionary.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    return len(data)


def bytes_to_obj(data_bytes: bytes, format_name: str) -> object:
    """
    Deserialize a Python object from raw bytes using the specified format.

    Supported formats are limited to those in approved_formats.

    Args:
        data_bytes: The serialized data as bytes.
        format_name: The format name ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        TypeError: If data_bytes is not bytes or format_name is not a string.
        ValueError: If the format is unsupported/dangerous or data cannot be deserialized.
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be bytes-like")
    if not isinstance(format_name, str):
        raise TypeError("format_name must be a string")

    fmt = format_name.strip().lower()
    if fmt not in approved_formats:
        raise ValueError(f"Unsupported or dangerous format: {format_name}")

    if fmt == "json":
        import json
        try:
            text = bytes(data_bytes).decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ValueError("Invalid JSON bytes encoding") from e
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError("Invalid JSON data") from e

    if fmt == "xml":
        import xml.etree.ElementTree as ET
        # Try parsing bytes directly; ElementTree can handle bytes with encoding declaration.
        try:
            return ET.fromstring(bytes(data_bytes))
        except Exception:
            # Fallback: attempt UTF-8 decoding with BOM handling then parse string
            try:
                text = bytes(data_bytes).decode("utf-8-sig")
                return ET.fromstring(text)
            except Exception as e:
                raise ValueError("Invalid XML data") from e

    # Should not reach here because of approved_formats check
    raise ValueError(f"Unsupported or dangerous format: {format_name}")
