from typing import Any
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

    if not isinstance(data_format, str):
        raise ValueError("data_format must be a string")

    fmt = data_format.strip().lower()

    if fmt not in secure_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format!r}")

    try:
        if fmt == "json":
            # Decode bytes using UTF-8 (with BOM handling) and parse JSON
            text = bytes(bytes_data).decode("utf-8-sig")
            return json.loads(text)

        if fmt == "xml":
            xml_bytes = bytes(bytes_data)

            # If defusedxml is not available, perform minimal hardening by rejecting DTD/entities.
            if not _USE_DEFUSEDXML:
                lowered = xml_bytes.lower()
                if b"<!doctype" in lowered or b"<!entity" in lowered or b"<!dtd" in lowered:
                    raise ValueError("XML containing DOCTYPE/DTD/entities is not allowed")

            # SafeET.fromstring accepts bytes and respects XML encoding declarations.
            return SafeET.fromstring(xml_bytes)

    except ValueError:
        # Re-raise ValueError as-is (e.g., unsupported format or our explicit checks)
        raise
    except Exception as exc:
        # Wrap other exceptions as ValueError to conform to the API contract
        raise ValueError(f"Failed to deserialize {fmt} data: {exc}") from exc
