import json
import re
import xml.etree.ElementTree as ET

def validate_byte_data(data: bytes) -> bool:
    """
    Check if the provided byte sequence is valid UTF-8 encoded data.

    Args:
        data: Byte sequence to validate.

    Returns:
        True if the byte sequence is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(data: bytes) -> str:
    """
    Detect the serialization format of the provided byte sequence.

    Recognized formats: JSON, XML, INI.

    Returns:
        A string indicating the detected format: 'json', 'xml', or 'ini'.

    Raises:
        ValueError: If the data is not valid UTF-8, unrecognized, or potentially unsafe.
    """
    if not validate_byte_data(data):
        raise ValueError("Data is not valid UTF-8 and may be unsafe.")

    text = data.decode("utf-8")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    stripped = text.lstrip()

    # Try to detect JSON:
    # Common JSON starting characters include {, [, ", -, digits, or literals: true/false/null
    if stripped:
        first_char = stripped[0]
        if first_char in '{["-' or first_char.isdigit() or stripped.startswith(("true", "false", "null")):
            try:
                json.loads(stripped)
                return "json"
            except Exception:
                # Fall through to other detections
                pass

    # Try to detect XML: typically starts with '<' or '<?xml'
    if stripped.startswith("<"):
        try:
            ET.fromstring(stripped)
            return "xml"
        except ET.ParseError:
            # Fall through to other detections
            pass

    # Try to detect INI:
    # Look for a section header [section] as the first meaningful (non-empty, non-comment) line.
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("#", ";")):
            continue
        # Match [section] optionally with trailing comments
        if s.startswith("[") and "]" in s:
            return "ini"
        break  # First meaningful line is not an INI section header

    raise ValueError("Unrecognized or potentially unsafe serialization format.")
