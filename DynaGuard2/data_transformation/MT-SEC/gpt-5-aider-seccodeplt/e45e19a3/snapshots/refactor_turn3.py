import json
import logging
import re
from typing import Any, Sequence

# Logger for this module
logger = logging.getLogger(__name__)

# Predefined set of trusted formats
safe_formats = ["json", "xml"]

try:
    from defusedxml import ElementTree as _DefusedET  # type: ignore
    from defusedxml.common import DefusedXmlException  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:  # defusedxml is optional; we can safely fall back with extra checks
    _HAS_DEFUSEDXML = False
    import xml.etree.ElementTree as _ET  # type: ignore


_XML_PROHIBITED_PATTERNS = (
    re.compile(rb'<!\s*DOCTYPE', re.IGNORECASE),
    re.compile(rb'<!\s*ENTITY', re.IGNORECASE),
)


def _xml_contains_prohibited_constructs(data: bytes) -> bool:
    return any(p.search(data) is not None for p in _XML_PROHIBITED_PATTERNS)


def _validate_and_normalize_format(data_type: str, allowed_formats: Sequence[str]) -> str:
    """
    Validate and normalize the requested data format.

    Args:
        data_type: The requested serialization format (e.g., 'json', 'xml').
        allowed_formats: A sequence of allowed/safe formats.

    Returns:
        The normalized (lowercased/stripped) format string.

    Raises:
        ValueError: If the format is unsupported or unsafe.
    """
    fmt = (data_type or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_type!r}")
    return fmt


def _deserialize_json_bytes(data: bytes) -> Any:
    """
    Deserialize JSON from bytes, enforcing UTF-8 decoding.

    Raises:
        ValueError: If decoding fails or JSON is invalid.
    """
    logger.debug("Deserializing JSON: enforcing UTF-8 decoding.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as e:
        logger.debug("UTF-8 decoding failed for JSON data: %s", e)
        raise ValueError("JSON input must be valid UTF-8") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("JSON parsing failed: %s", e)
        raise ValueError("Invalid JSON data") from e


def _deserialize_xml_bytes(data: bytes) -> Any:
    """
    Deserialize XML from bytes using safe parsers/guards.

    Raises:
        ValueError: If XML is unsafe or invalid.
    """
    if _HAS_DEFUSEDXML:
        logger.debug("Deserializing XML using defusedxml.ElementTree.")
        try:
            return _DefusedET.fromstring(data)
        except DefusedXmlException as e:
            logger.debug("defusedxml detected unsafe XML content: %s", e)
            raise ValueError("Unsafe XML content detected") from e
        except Exception as e:
            logger.debug("XML parsing failed (defusedxml): %s", e)
            raise ValueError("Invalid XML data") from e
    else:
        logger.debug("Deserializing XML using xml.etree.ElementTree with hardening checks.")
        # Fallback: basic hardening to avoid DTD/entity attacks
        if _xml_contains_prohibited_constructs(data):
            logger.debug("Prohibited XML constructs detected (DOCTYPE/ENTITY).")
            raise ValueError("XML containing DOCTYPE/ENTITY is not allowed")
        try:
            return _ET.fromstring(data)
        except Exception as e:
            logger.debug("XML parsing failed (stdlib): %s", e)
            raise ValueError("Invalid XML data") from e


def deserialize_data(input_data: bytes, data_type: str):
    """
    Deserialize data from raw bytes into Python objects using only safe formats.

    Args:
        input_data: Serialized object in byte format.
        data_type: Serialization format. Only 'json' and 'xml' are allowed.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported/unsafe, or if the data cannot be parsed safely.
    """
    if not isinstance(input_data, (bytes, bytearray, memoryview)):
        logger.error("Deserialization failed: input_data must be bytes-like (got %s).", type(input_data).__name__)
        raise ValueError("input_data must be bytes-like")

    data_bytes = bytes(input_data)

    try:
        fmt = _validate_and_normalize_format(data_type, safe_formats)
    except ValueError as e:
        logger.warning("Deserialization rejected due to unsupported/unsafe format: %r", data_type)
        raise

    logger.info("Deserialization attempt: format=%s, size=%d bytes", fmt, len(data_bytes))

    try:
        if fmt == "json":
            result = _deserialize_json_bytes(data_bytes)
        elif fmt == "xml":
            result = _deserialize_xml_bytes(data_bytes)
        else:
            # This point should be unreachable due to the safe_formats check.
            raise ValueError(f"Unsupported or unsafe format: {data_type!r}")

        logger.info("Deserialization succeeded: format=%s", fmt)
        return result

    except ValueError as e:
        logger.error("Deserialization failed: format=%s, error=%s", fmt, e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error during deserialization: format=%s, error=%s", fmt, e, exc_info=True)
        raise
