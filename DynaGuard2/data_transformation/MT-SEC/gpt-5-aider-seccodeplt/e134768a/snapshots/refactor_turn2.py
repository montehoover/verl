import json
import configparser
from typing import Any, Dict, Union


# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]


def _normalize_format(struct_format: str) -> str:
    """
    Normalize and validate the struct_format input.

    Raises:
    - TypeError: If struct_format is not a string.
    """
    if not isinstance(struct_format, str):
        raise TypeError("struct_format must be a string")
    return struct_format.strip().lower()


def _ensure_safe_format(fmt: str, original: str) -> None:
    """
    Ensure the provided format is within the allowed safe formats.

    Raises:
    - ValueError: If the format is not recognized or deemed unsafe.
    """
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {original}")


def _decode_to_text(byte_data: Union[bytes, bytearray, memoryview]) -> str:
    """
    Decode bytes-like data to UTF-8 text.

    Raises:
    - TypeError: If byte_data is not bytes-like.
    - UnicodeDecodeError: If decoding fails.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be bytes-like (bytes, bytearray, or memoryview)")
    return bytes(byte_data).decode("utf-8")


def _deserialize_text(text: str, fmt: str) -> Any:
    """
    Deserialize text content according to the specified format.

    Raises:
    - json.JSONDecodeError / configparser.Error: For malformed input data.
    - ValueError: If format is unsupported (should be unreachable if validated earlier).
    """
    if fmt == "json":
        return json.loads(text)

    if fmt == "ini":
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read_string(text)
        result: Dict[str, Dict[str, str]] = {}
        # Include defaults if present
        defaults = cfg.defaults()
        if defaults:
            result["DEFAULT"] = dict(defaults)
        # Include each section's key/value pairs (raw to avoid interpolation)
        for section in cfg.sections():
            result[section] = {k: v for k, v in cfg.items(section, raw=True)}
        return result

    # This point should be unreachable due to earlier validation.
    raise ValueError(f"Unsupported or unsafe format: {fmt}")


def unpack_serialized_object(byte_data: Union[bytes, bytearray, memoryview], struct_format: str) -> Any:
    """
    Deserialize binary data from potentially untrusted sources into a Python object.

    Parameters:
    - byte_data: bytes | bytearray | memoryview
        The provided serialized data in a binary form.
    - struct_format: str
        The format used for deserialization (e.g., 'json', 'ini').

    Returns:
    - Any
        The deserialized Python object contained within the data.

    Raises:
    - TypeError: If input types are invalid.
    - ValueError: If the format is not recognized or deemed unsafe.
    - UnicodeDecodeError / json.JSONDecodeError / configparser.Error: For malformed input data.
    """
    fmt = _normalize_format(struct_format)
    _ensure_safe_format(fmt, struct_format)
    text = _decode_to_text(byte_data)
    return _deserialize_text(text, fmt)
