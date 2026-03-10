import json
from typing import Any

# Trusted formats for deserialization
safe_formats = ["json", "xml"]

# Try to use defusedxml for safer XML parsing if available; fall back to stdlib ElementTree otherwise.
try:
    from defusedxml import ElementTree as _ET  # type: ignore
    _DEFUSEDXML = True
except Exception:
    from xml.etree import ElementTree as _ET
    _DEFUSEDXML = False


def _parse_xml_safely(xml_text: str) -> Any:
    """
    Parse XML text safely.

    - Rejects any XML containing a DOCTYPE declaration to avoid entity expansion attacks.
    - Uses defusedxml if available; otherwise falls back to xml.etree.ElementTree.

    Returns:
        An ElementTree Element representing the root of the XML document.
    """
    # Block DOCTYPE to avoid XXE / entity expansion attacks when not using defusedxml.
    if "<!DOCTYPE" in xml_text.upper():
        raise ValueError("Unsafe XML: DOCTYPE is not allowed")

    return _ET.fromstring(xml_text)


def deserialize_data(input_data: bytes, data_type: str) -> Any:
    """
    Deserialize raw bytes into a Python object using a safe, explicitly allowed format.

    Args:
        input_data: bytes representing the serialized object.
        data_type: str specifying the serialization format (e.g., 'json', 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported or considered unsafe, or if the data cannot be decoded/parsed.
    """
    if not isinstance(input_data, (bytes, bytearray)):
        raise ValueError("input_data must be bytes")

    if not isinstance(data_type, str):
        raise ValueError("data_type must be a string")

    allowed = {fmt.lower() for fmt in safe_formats}
    fmt = data_type.lower().strip()

    if fmt not in allowed:
        raise ValueError(f"Unsupported or unsafe format: {data_type}")

    try:
        # Decode bytes to text for text-based formats
        text = input_data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("input_data is not valid UTF-8") from e

    try:
        if fmt == "json":
            return json.loads(text)
        elif fmt == "xml":
            # Returns an xml.etree.ElementTree.Element (or defused equivalent)
            return _parse_xml_safely(text)
        else:
            # Should not reach here due to earlier check, but keep defensive.
            raise ValueError(f"Unsupported or unsafe format: {data_type}")
    except ValueError:
        # Re-raise ValueError without wrapping to preserve message
        raise
    except Exception as e:
        # Normalize other exceptions to ValueError as per contract
        raise ValueError(f"Failed to deserialize {data_type}: {e}") from e
