import json
import configparser
import logging
from typing import Any, Dict, List

# Configure module-level logger to write human-readable logs to the current working directory.
_logger = logging.getLogger("secure_deserializer")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _file_handler = logging.FileHandler("deserialization.log", encoding="utf-8")
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _file_handler.setFormatter(_formatter)
    _logger.addHandler(_file_handler)
    _logger.propagate = False

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
    # Log the attempt with basic context
    fmt_repr = serialization_format if isinstance(serialization_format, str) else repr(serialization_format)
    try:
        byte_len = len(byte_data) if isinstance(byte_data, (bytes, bytearray, memoryview)) else None
    except Exception:
        byte_len = None
    _logger.info("Attempting deserialization: format=%r, byte_length=%s", fmt_repr, byte_len if byte_len is not None else "unknown")

    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        _logger.error("Deserialization failed: byte_data is not bytes-like (received type=%s)", type(byte_data).__name__)
        raise TypeError("byte_data must be bytes-like (bytes, bytearray, or memoryview)")

    try:
        normalized_format = _normalize_and_validate_format(serialization_format, trusted_formats)
        raw_bytes = bytes(byte_data)
        result = _deserialize_by_format(raw_bytes, normalized_format)
        _logger.info("Deserialization successful: format=%s, result_type=%s", normalized_format, type(result).__name__)
        return result
    except Exception:
        # Log the exception with traceback for troubleshooting
        _logger.exception("Deserialization failed: format=%r", fmt_repr)
        raise
