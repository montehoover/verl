import json
import configparser
from typing import Any, Dict, Callable

# Predefined list of allowed formats that are deemed safe for deserialization.
trusted_formats = ["json", "ini"]


def _decode_utf8_text(data: bytes) -> str:
    """
    Decode bytes to UTF-8 text (with BOM handling).
    Raises ValueError on failure to keep error semantics consistent.
    """
    try:
        return bytes(data).decode("utf-8-sig")
    except Exception as e:
        raise ValueError("Failed to decode input bytes as UTF-8 text.") from e


def _deserialize_json(text: str) -> Any:
    """
    Deserialize JSON from text.
    Raises ValueError("Invalid JSON data.") on parse errors.
    """
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError("Invalid JSON data.") from e


def _deserialize_ini(text: str) -> Dict[str, Dict[str, str]]:
    """
    Deserialize INI from text into a plain dict representation.
    Raises ValueError("Invalid INI data.") on parse errors.
    """
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(text)
    except Exception as e:
        raise ValueError("Invalid INI data.") from e

    result: Dict[str, Dict[str, str]] = {}

    defaults = dict(parser.defaults())
    if defaults:
        result["__DEFAULT__"] = defaults

    for section in parser.sections():
        result[section] = {k: v for k, v in parser.items(section, raw=True)}

    return result


# Mapping of supported formats to their deserializer functions.
_DESERIALIZERS: Dict[str, Callable[[str], Any]] = {
    "json": _deserialize_json,
    "ini": _deserialize_ini,
}


def decode_serialized_data(data_bytes: bytes, format_string: str) -> Any:
    """
    Safely deserialize data from trusted formats.

    Args:
        data_bytes: Incoming serialized data in binary format.
        format_string: String identifying the serialization format (e.g., 'json', 'ini').

    Returns:
        A Python object resulting from deserializing the data.

    Raises:
        TypeError: If input types are incorrect.
        ValueError: If the format is unsupported or the data content is invalid.
    """
    if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("data_bytes must be a bytes-like object.")
    if not isinstance(format_string, str):
        raise TypeError("format_string must be a string.")

    fmt = format_string.strip().lower()
    allowed = {f.lower() for f in trusted_formats}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or untrusted format: {format_string!r}")

    text = _decode_utf8_text(data_bytes)

    deserializer = _DESERIALIZERS.get(fmt)
    if deserializer is None:
        # Should be unreachable due to the allowed check above.
        raise ValueError(f"Unsupported or untrusted format: {format_string!r}")

    return deserializer(text)
