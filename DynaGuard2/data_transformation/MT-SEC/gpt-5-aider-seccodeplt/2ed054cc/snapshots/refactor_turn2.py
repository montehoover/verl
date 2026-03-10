import json
from typing import Sequence

# Allowed serialization types provided by setup
allowed_serialization_types = ["json", "xml"]

# Try to use defusedxml for safer XML parsing if available
try:
    from defusedxml import ElementTree as DefusedET  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:
    from xml.etree import ElementTree as ET  # Fallback to stdlib
    _HAS_DEFUSEDXML = False


def _contains_unsafe_xml_constructs(data: bytes) -> bool:
    """
    Detects potentially unsafe XML constructs such as DTDs or entities.
    Pure function: depends only on its input.
    """
    lowered = data.lower()
    # Disallow DTDs and custom entities which can be abused (e.g., XXE, Billion Laughs)
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        return True
    return False


def _normalize_format(serialization_format: str) -> str:
    """
    Normalizes the format string to a canonical representation.
    Pure function.
    """
    return (serialization_format or "").strip().lower()


def _validate_serialization_format(serialization_format: str, allowed: Sequence[str]) -> str:
    """
    Validates and normalizes the provided serialization format against an allowed list.
    Pure function.

    Returns the normalized format if valid, otherwise raises ValueError.
    """
    fmt = _normalize_format(serialization_format)
    if fmt not in allowed:
        raise ValueError(f"Unsupported or unsafe serialization format: {serialization_format!r}")
    return fmt


def _deserialize_json(raw_data: bytes):
    """
    Deserialize JSON from bytes.
    Pure function: reads input, returns parsed structure, raises on invalid payload.
    """
    try:
        # Use utf-8-sig to handle potential BOM safely
        text = bytes(raw_data).decode("utf-8-sig")
        return json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON data: {exc}") from None


def _deserialize_xml(raw_data: bytes):
    """
    Deserialize XML from bytes with safety checks against dangerous constructs.
    Pure function: reads input, returns parsed Element, raises on invalid/unsafe payload.
    """
    if _contains_unsafe_xml_constructs(bytes(raw_data)):
        raise ValueError("XML contains disallowed constructs (DOCTYPE or ENTITY)")

    try:
        if _HAS_DEFUSEDXML:
            return DefusedET.fromstring(bytes(raw_data))
        else:
            # Fallback: stdlib ElementTree parsing with prior checks for unsafe constructs
            return ET.fromstring(bytes(raw_data))
    except Exception as exc:
        raise ValueError(f"Invalid XML data: {exc}") from None


def _deserialize_by_format(fmt: str, raw_data: bytes):
    """
    Dispatch deserialization based on a validated format.
    Pure function: depends only on fmt and raw_data.
    """
    if fmt == "json":
        return _deserialize_json(raw_data)
    if fmt == "xml":
        return _deserialize_xml(raw_data)
    # Should not be reachable if validation is used properly
    raise ValueError(f"Unsupported or unsafe serialization format: {fmt!r}")


def deserialize_data(raw_data: bytes, serialization_format: str):
    """
    Safely deserialize data from raw bytes into a Python object using an allowed format.

    Args:
        raw_data: bytes containing the serialized data.
        serialization_format: str specifying the serialization format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: if the format is unsupported/unsafe or the payload is invalid for that format.
    """
    if not isinstance(raw_data, (bytes, bytearray)):
        raise ValueError("raw_data must be of type bytes or bytearray")

    fmt = _validate_serialization_format(serialization_format, allowed_serialization_types)
    return _deserialize_by_format(fmt, raw_data)
