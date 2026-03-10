import json
import re
import xml.etree.ElementTree as ET
import configparser
from typing import Any

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


secure_serialization_formats = ["json", "ini"]


def process_serialized_content(payload_bytes: bytes, schema_type: str) -> Any:
    """
    Securely deserialize binary data into a Python object based on the provided schema_type.

    Supported secure formats are defined in secure_serialization_formats.

    Args:
        payload_bytes: The serialized data as bytes.
        schema_type: The format to use for deserialization (e.g., 'json', 'ini').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the data is not valid UTF-8, if the schema_type is unsupported/unsafe,
                   or if deserialization fails.
    """
    if not validate_byte_data(payload_bytes):
        raise ValueError("Data is not valid UTF-8 and may be unsafe.")

    if schema_type is None:
        raise ValueError("Schema type must be provided.")

    fmt = schema_type.strip().lower()
    if fmt not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type}")

    text = payload_bytes.decode("utf-8")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")

    if fmt == "json":
        try:
            return json.loads(text)
        except Exception as e:
            raise ValueError(f"Invalid JSON payload: {e}") from e

    if fmt == "ini":
        try:
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(text)
            result: dict[str, dict[str, str]] = {}

            defaults = dict(parser.defaults())
            if defaults:
                result["DEFAULT"] = defaults

            for section in parser.sections():
                # raw=True avoids any interpolation processing
                result[section] = {k: v for k, v in parser.items(section, raw=True)}
            return result
        except Exception as e:
            raise ValueError(f"Invalid INI payload: {e}") from e

    # Should not reach here due to earlier validation
    raise ValueError(f"Unsupported or unsafe format: {schema_type}")
