from typing import Any, Sequence
import json

# Predefined list of secure data formats
secure_formats = ["json", "xml"]

# Prefer defusedxml for safer XML parsing if available
try:
    from defusedxml import ElementTree as SafeET  # type: ignore
    _USE_DEFUSEDXML = True
except Exception:  # pragma: no cover - fallback when defusedxml is not installed
    from xml.etree import ElementTree as SafeET  # type: ignore
    _USE_DEFUSEDXML = False


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
        raise ValueError("data_format must be a string")

    fmt = data_format.strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format!r}")
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
        raise ValueError("bytes_data must be a bytes-like object")

    fmt = _normalize_and_validate_format(data_format, secure_formats)

    try:
        return _deserialize_by_format(bytes_data, fmt)
    except ValueError:
        # Re-raise ValueError as-is (e.g., unsupported format or our explicit checks)
        raise
    except Exception as exc:
        # Wrap other exceptions as ValueError to conform to the API contract
        raise ValueError(f"Failed to deserialize {fmt} data: {exc}") from exc
