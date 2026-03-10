from typing import Any, Sequence
import json
import logging

# Module logger
logger = logging.getLogger(__name__)

# Predefined list of secure data formats
secure_formats = ["json", "xml"]

# Prefer defusedxml for safer XML parsing if available
try:
    from defusedxml import ElementTree as SafeET  # type: ignore
    _USE_DEFUSEDXML = True
except Exception:  # pragma: no cover - fallback when defusedxml is not installed
    from xml.etree import ElementTree as SafeET  # type: ignore
    _USE_DEFUSEDXML = False

_XML_PARSER_NAME = "defusedxml.ElementTree" if _USE_DEFUSEDXML else "xml.etree.ElementTree"


def _normalize_and_validate_format(data_format: str, allowed_formats: Sequence[str]) -> str:
    """
    Normalize and validate the requested data format against an allowlist.

    Args:
        data_format: The requested data format.
        allowed_formats: A sequence of allowed/secure formats.

    Returns:
        The normalized (lower-cased, trimmed) format string.

    Raises:
        ValueError: If data_format is not a string or is not allowed.
    """
    if not isinstance(data_format, str):
        logger.error("Format validation failed: data_format is not a string (type=%s)", type(data_format).__name__)
        raise ValueError("data_format must be a string")

    fmt = data_format.strip().lower()
    if fmt not in allowed_formats:
        logger.warning("Unsupported or unsafe data format attempted: %r (normalized=%r)", data_format, fmt)
        raise ValueError(f"Unsupported or unsafe data format: {data_format!r}")

    logger.debug("Format validated: %r -> %r", data_format, fmt)
    return fmt


def _deserialize_json_from_bytes(bytes_data: bytes) -> Any:
    """
    Deserialize JSON from raw bytes.
    """
    text = bytes(bytes_data).decode("utf-8-sig")
    return json.loads(text)


def _deserialize_xml_from_bytes(bytes_data: bytes) -> Any:
    """
    Deserialize XML from raw bytes using a safe XML parser when available.
    """
    xml_bytes = bytes(bytes_data)

    # If defusedxml is not available, perform minimal hardening by rejecting DTD/entities.
    if not _USE_DEFUSEDXML:
        lowered = xml_bytes.lower()
        if b"<!doctype" in lowered or b"<!entity" in lowered or b"<!dtd" in lowered:
            raise ValueError("XML containing DOCTYPE/DTD/entities is not allowed")

    logger.debug("Parsing XML using %s", _XML_PARSER_NAME)
    # SafeET.fromstring accepts bytes and respects XML encoding declarations.
    return SafeET.fromstring(xml_bytes)


def _deserialize_by_format(bytes_data: bytes, fmt: str) -> Any:
    """
    Dispatch deserialization based on the normalized format.
    """
    if fmt == "json":
        return _deserialize_json_from_bytes(bytes_data)
    if fmt == "xml":
        return _deserialize_xml_from_bytes(bytes_data)
    # Should not be reachable if validation is performed beforehand
    raise ValueError(f"Unsupported or unsafe data format: {fmt!r}")


def load_serialized_data(bytes_data: bytes, data_format: str) -> Any:
    """
    Safely converts serialized data, provided as raw bytes, into its corresponding Python object.

    Args:
        bytes_data: The serialized form of the object as raw bytes.
        data_format: The data format used for serialization (e.g., 'json', 'xml').

    Returns:
        The deserialized Python data structure if the format is valid.

    Raises:
        ValueError: If the data format is unsupported/unsafe or deserialization fails.
    """
    if not isinstance(bytes_data, (bytes, bytearray, memoryview)):
        logger.error("Deserialization failed: bytes_data must be bytes-like (type=%s)", type(bytes_data).__name__)
        raise ValueError("bytes_data must be a bytes-like object")

    data_len = len(bytes_data)
    logger.debug("Received deserialization request: data_format=%r, bytes_len=%d", data_format, data_len)

    fmt = _normalize_and_validate_format(data_format, secure_formats)

    logger.info("Deserialization attempt: format=%s, bytes_len=%d", fmt, data_len)

    try:
        result = _deserialize_by_format(bytes_data, fmt)

        if fmt == "json":
            logger.info("Deserialization successful: format=%s, result_type=%s", fmt, type(result).__name__)
        elif fmt == "xml":
            root_tag = getattr(result, "tag", None)
            if root_tag is not None:
                logger.info("Deserialization successful: format=%s, xml_root_tag=%s", fmt, root_tag)
            else:
                logger.info("Deserialization successful: format=%s", fmt)

        return result
    except ValueError as ve:
        logger.warning("Deserialization failed (validation/content): format=%s, error=%s", fmt, ve)
        raise
    except Exception as exc:
        # Log with traceback for unexpected errors
        logger.exception("Deserialization error: format=%s", fmt)
        raise ValueError(f"Failed to deserialize {fmt} data: {exc}") from exc
