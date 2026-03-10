"""
Safe deserialization utilities.

This module exposes a single public function, `deserialize_data`, which converts
raw serialized bytes into Python objects using a limited, explicitly allowed set
of formats. The design aims to be defensive against unsafe inputs (e.g., untrusted
XML with DTD/ENTITY declarations) and easy to maintain by separating concerns
into dedicated helper functions.

Key points:
- Only formats listed in `allowed_serialization_types` are accepted.
- JSON is decoded as UTF-8 (handling BOM via utf-8-sig) and parsed via json.loads.
- XML is parsed via defusedxml.ElementTree if available; otherwise, it falls back
  to the stdlib xml.etree.ElementTree but rejects DTD/ENTITY constructs up front.
- All helpers are implemented as pure functions to facilitate testing and clarity.
"""

from typing import Any, Sequence
import json

# Allowed serialization types provided by setup. Only these formats are accepted
# for deserialization. Any other format will raise a ValueError.
allowed_serialization_types = ["json", "xml"]

# Prefer defusedxml for safer XML parsing to mitigate attacks like XXE and
# Billion Laughs. If unavailable, fall back to stdlib ElementTree (guarded by
# explicit checks that disallow unsafe constructs).
try:
    from defusedxml import ElementTree as DefusedET  # type: ignore
    _HAS_DEFUSEDXML = True
except Exception:
    from xml.etree import ElementTree as ET  # Fallback to stdlib
    _HAS_DEFUSEDXML = False


def _contains_unsafe_xml_constructs(data: bytes) -> bool:
    """
    Check for XML constructs that are commonly exploited in XML-based attacks.

    This function performs a conservative byte-level scan for DTD and ENTITY
    declarations. If any are found, the payload should be rejected.

    Args:
        data: The raw XML data as bytes.

    Returns:
        True if unsafe constructs are detected; otherwise, False.
    """
    lowered = data.lower()

    # Disallow DOCTYPE and ENTITY declarations which may enable XXE or
    # exponential entity expansion (Billion Laughs) attacks.
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        return True

    return False


def _normalize_format(serialization_format: str) -> str:
    """
    Normalize the provided serialization format string.

    This trims surrounding whitespace and lowercases the string so comparisons
    can be performed consistently.

    Args:
        serialization_format: The format string to normalize.

    Returns:
        A normalized, lowercased format string.
    """
    return (serialization_format or "").strip().lower()


def _validate_serialization_format(
    serialization_format: str,
    allowed: Sequence[str],
) -> str:
    """
    Validate and normalize the serialization format against an allowed list.

    Args:
        serialization_format: The user-provided format string.
        allowed: A sequence of allowed format strings.

    Returns:
        The normalized format string if it is allowed.

    Raises:
        ValueError: If the format is not allowed.
    """
    fmt = _normalize_format(serialization_format)
    if fmt not in allowed:
        raise ValueError(
            f"Unsupported or unsafe serialization format: {serialization_format!r}"
        )
    return fmt


def _deserialize_json(raw_data: bytes) -> Any:
    """
    Deserialize a JSON payload from bytes into a Python object.

    This uses UTF-8 with BOM handling (utf-8-sig) for robust decoding.

    Args:
        raw_data: The raw JSON data as bytes.

    Returns:
        The parsed Python object (dict, list, str, int, etc.).

    Raises:
        ValueError: If the data cannot be decoded or parsed as valid JSON.
    """
    try:
        # Use utf-8-sig to handle a potential UTF-8 BOM safely.
        text = bytes(raw_data).decode("utf-8-sig")
        return json.loads(text)
    except Exception as exc:
        raise ValueError(f"Invalid JSON data: {exc}") from None


def _deserialize_xml(raw_data: bytes) -> Any:
    """
    Deserialize an XML payload from bytes into an ElementTree Element.

    This function rejects payloads with DTD/ENTITY declarations. If defusedxml
    is available, it is used for safer parsing. Otherwise, the stdlib XML
    parser is used as a fallback with the aforementioned pre-checks.

    Args:
        raw_data: The raw XML data as bytes.

    Returns:
        An XML Element (ElementTree.Element).

    Raises:
        ValueError: If the XML contains unsafe constructs or fails to parse.
    """
    if _contains_unsafe_xml_constructs(bytes(raw_data)):
        raise ValueError("XML contains disallowed constructs (DOCTYPE or ENTITY)")

    try:
        if _HAS_DEFUSEDXML:
            return DefusedET.fromstring(bytes(raw_data))
        else:
            # Fallback: stdlib ElementTree parsing with prior checks for
            # unsafe constructs. Note that defusedxml remains the safer option.
            return ET.fromstring(bytes(raw_data))
    except Exception as exc:
        raise ValueError(f"Invalid XML data: {exc}") from None


def _deserialize_by_format(fmt: str, raw_data: bytes) -> Any:
    """
    Dispatch to the appropriate deserializer based on a validated format.

    Args:
        fmt: The already validated and normalized format string.
        raw_data: The raw serialized data as bytes.

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If an unsupported format is requested (should not happen
                    if validation is performed beforehand).
    """
    if fmt == "json":
        return _deserialize_json(raw_data)

    if fmt == "xml":
        return _deserialize_xml(raw_data)

    # Should not be reachable due to prior validation.
    raise ValueError(f"Unsupported or unsafe serialization format: {fmt!r}")


def deserialize_data(raw_data: bytes, serialization_format: str) -> Any:
    """
    Safely deserialize data from raw bytes into a Python object.

    This function validates the requested serialization format against the
    globally allowed list and dispatches to an appropriate, safe deserializer.

    Args:
        raw_data: Bytes containing the serialized data.
        serialization_format: The serialization format ('json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported/unsafe, the input type for
                    raw_data is invalid, or the payload cannot be parsed.
    """
    # Validate the type of raw_data early to provide clear error messages and
    # avoid unexpected behavior.
    if not isinstance(raw_data, (bytes, bytearray)):
        raise ValueError("raw_data must be of type bytes or bytearray")

    # Validate and normalize the format, then dispatch.
    fmt = _validate_serialization_format(
        serialization_format,
        allowed_serialization_types,
    )

    return _deserialize_by_format(fmt, raw_data)
