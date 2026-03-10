import json
import configparser
from typing import Any, Dict, List

# Controlled list of formats considered safe for deserialization operations.
trusted_formats = ["json", "ini"]


def _normalize_and_validate_format(serialization_format: str, allowed_formats: List[str]) -> str:
    """
    Normalize the format string and ensure it is within the allowed set.
    """
    if not isinstance(serialization_format, str):
        raise TypeError("serialization_format must be a string")
    normalized = serialization_format.strip().lower()
    if normalized not in allowed_formats:
        raise ValueError(f"Deserialization format '{serialization_format}' is not allowed")
    return normalized


def _deserialize_json(raw_bytes: bytes) -> Any:
    """
    Deserialize JSON from bytes.
    """
    try:
        # json.loads accepts bytes and assumes UTF-8 per RFC 8259
        return json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON data") from exc


def _deserialize_ini(raw_bytes: bytes) -> Dict[str, Dict[str, str]]:
    """
    Deserialize INI content from bytes to a nested dictionary structure.
    """
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("INI data must be valid UTF-8") from exc

    # Disable interpolation to avoid unintended variable expansion
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except (configparser.Error,) as exc:
        raise ValueError("Invalid INI data") from exc

    # Convert to a plain dictionary to avoid exposing parser internals
    result: Dict[str, Dict[str, str]] = {}
    if parser.defaults():
        result[parser.default_section] = dict(parser.defaults())
    for section in parser.sections():
        # Use raw=True to avoid interpolation at access-time
        result[section] = dict(parser.items(section, raw=True))
    return result


def _deserialize_by_format(raw_bytes: bytes, normalized_format: str) -> Any:
    """
    Route deserialization based on the normalized format string.
    """
    if normalized_format == "json":
        return _deserialize_json(raw_bytes)
    if normalized_format == "ini":
        return _deserialize_ini(raw_bytes)
    # If an allowed format is not implemented
    raise NotImplementedError(f"Deserialization for format '{normalized_format}' is not implemented")


def securely_load_data(byte_data: bytes, serialization_format: str) -> Any:
    """
    Safely deserialize data from supported formats.

    Args:
        byte_data: bytes - Serialized data received in binary format.
        serialization_format: str - A string specifying the mechanism used to serialize
            the data (e.g., 'json', 'ini').

    Returns:
        The Python object after successful deserialization.

    Raises:
        TypeError: If the input types are incorrect.
        ValueError: If the format is not allowed or input data is invalid for the given format.
        NotImplementedError: If the format is allowed but not implemented.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be bytes-like (bytes, bytearray, or memoryview)")

    normalized_format = _normalize_and_validate_format(serialization_format, trusted_formats)
    raw_bytes = bytes(byte_data)
    return _deserialize_by_format(raw_bytes, normalized_format)
