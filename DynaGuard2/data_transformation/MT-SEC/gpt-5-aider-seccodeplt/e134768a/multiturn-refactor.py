import json
import configparser
import logging
from typing import Any, Dict, Union


# A secure set of accepted formats designated to ensure safe deserialization practices.
safe_formats = ["json", "ini"]

# Module-level logger (configuration handled by the application)
logger = logging.getLogger(__name__)


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
        # Guard clause for unsupported formats
        logger.warning("Blocked unsupported/unsafe format: %s (normalized: %s)", original, fmt)
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

    Uses guard clauses to reject unsupported formats early.

    Raises:
    - json.JSONDecodeError / configparser.Error: For malformed input data.
    - ValueError: If format is unsupported.
    """
    # Guard clause for unsupported formats
    if fmt not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {fmt}")

    if fmt == "json":
        return json.loads(text)

    # If we reach here, the only supported alternative is INI
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read_string(text)
    result: Dict[str, Dict[str, str]] = {}
    defaults = cfg.defaults()
    if defaults:
        result["DEFAULT"] = dict(defaults)
    for section in cfg.sections():
        result[section] = {k: v for k, v in cfg.items(section, raw=True)}
    return result


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

    # Guard clause for unsupported/unsafe formats (logged)
    _ensure_safe_format(fmt, struct_format)

    logger.debug("Attempting to deserialize input (bytes=%s) as format=%s", 
                 getattr(byte_data, "__len__", lambda: "unknown")(), fmt)

    try:
        text = _decode_to_text(byte_data)
        logger.debug("UTF-8 decoding succeeded (length=%d) for format=%s", len(text), fmt)
    except Exception:
        logger.exception("Failed to decode input as UTF-8 for format=%s", fmt)
        raise

    try:
        obj = _deserialize_text(text, fmt)
        logger.info("Deserialization succeeded for format=%s", fmt)
        return obj
    except Exception:
        logger.exception("Deserialization failed for format=%s", fmt)
        raise
