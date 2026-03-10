import json
import logging
from typing import Any, Iterable

# Configure logging to file in current working directory
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    _handler = logging.FileHandler("deserialization.log", encoding="utf-8")
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    # Avoid duplicate logs if root logger is configured elsewhere
    _logger.propagate = False

# Predefined set of trusted formats
approved_formats = ["json", "xml"]

# Try to import defusedxml for safe XML parsing
try:
    from defusedxml import ElementTree as _DefusedET  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:
    _HAS_DEFUSEDXML = False
    _DefusedET = None  # type: ignore


def _normalize_and_validate_format(format_name: str, approved: Iterable[str]) -> str:
    """
    Normalize and validate the provided format name against approved formats.

    Args:
        format_name: The input format name to validate.
        approved: An iterable of approved format names.

    Returns:
        The normalized (lowercased, stripped) format name.

    Raises:
        TypeError: If format_name is not a string.
        ValueError: If the format is not approved.
    """
    if not isinstance(format_name, str):
        raise TypeError("format_name must be a string")
    fmt = format_name.strip().lower()
    allowed = {f.lower() for f in approved}
    if fmt not in allowed:
        raise ValueError(f"Unsupported or dangerous format: {format_name!r}")
    return fmt


def _deserialize_from_bytes(raw_bytes: bytes, fmt: str) -> Any:
    """
    Deserialize bytes according to the specified, already-validated format.

    Args:
        raw_bytes: Data to deserialize.
        fmt: Normalized, validated format name.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported in this environment (e.g., XML
                    without defusedxml installed) or unknown.
    """
    if fmt == "json":
        text = raw_bytes.decode("utf-8")
        return json.loads(text)

    if fmt == "xml":
        if not _HAS_DEFUSEDXML:
            raise ValueError("XML deserialization requires 'defusedxml' for safe parsing.")
        return _DefusedET.fromstring(raw_bytes)

    # Should not be reachable if validation is correct
    raise ValueError(f"Unsupported or dangerous format: {fmt!r}")


def bytes_to_obj(data_bytes: bytes, format_name: str) -> Any:
    """
    Deserialize raw bytes into Python objects using only approved safe formats.

    Args:
        data_bytes: The serialized object in byte format.
        format_name: The serialization format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or considered dangerous.
        TypeError: If input types are invalid.
    """
    _logger.info("Deserialization attempt: format=%r", format_name)
    fmt_for_log: Any = None
    try:
        if not isinstance(data_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("data_bytes must be a bytes-like object")

        fmt = _normalize_and_validate_format(format_name, approved_formats)
        fmt_for_log = fmt
        raw_bytes = bytes(data_bytes)

        result = _deserialize_from_bytes(raw_bytes, fmt)
        _logger.info("Deserialization succeeded: format=%r", fmt)
        return result
    except Exception as e:
        # Prefer normalized format in logs if available
        fmt_display = fmt_for_log if fmt_for_log is not None else format_name
        _logger.error("Deserialization failed: format=%r error=%s: %s", fmt_display, e.__class__.__name__, str(e))
        raise
