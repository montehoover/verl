import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

# Predefined list of allowed (safe) formats for deserialization
safe_formats = ["json", "html"]


def convert_string_to_data(data_string: str, data_format: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert an input string into a structured data object based on the specified format.

    Args:
        data_string (str): The input data as a string.
        data_format (str): The format of the input data ('json' or 'xml').

    Returns:
        Union[dict, xml.etree.ElementTree.Element]: A dictionary for JSON inputs or
        the root XML Element for XML inputs.

    Raises:
        ValueError: If the input is not valid JSON/XML, does not represent a JSON object,
        or if the format is unsupported.
    """
    if not isinstance(data_format, str):
        raise ValueError("Unsupported format: data_format must be 'json' or 'xml'")

    fmt = data_format.strip().lower()

    if fmt == "json":
        try:
            parsed = json.loads(data_string)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("JSON does not represent an object")

        return parsed

    if fmt == "xml":
        try:
            element = ET.fromstring(data_string)
        except (TypeError, ET.ParseError) as exc:
            raise ValueError("Invalid XML") from exc
        return element

    raise ValueError(f"Unsupported format: {data_format}")


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Backwards-compatible helper to convert JSON string to dict.
    """
    result = convert_string_to_data(json_string, "json")
    # At runtime this will be a dict due to validation in convert_string_to_data
    return result  # type: ignore[return-value]


def restore_object_from_stream(byte_data: bytes, serialization_type: str) -> Any:
    """
    Deserialize a byte stream into a Python object based on the specified format.

    Supported and safe formats are listed in the global 'safe_formats'.

    Args:
        byte_data (bytes): The serialized data as bytes.
        serialization_type (str): The serialization format ('json', 'html', 'pickle').

    Returns:
        Any: The deserialized Python object. For 'json', returns the parsed JSON value.
             For 'html', returns the decoded HTML string.

    Raises:
        ValueError: If the format is unsupported/unsafe or if deserialization fails.
    """
    if not isinstance(serialization_type, str):
        raise ValueError("Unsupported or unsafe format: serialization_type must be a string")

    fmt = serialization_type.strip().lower()

    if fmt not in safe_formats:
        # Formats not in safe list (e.g., 'pickle') are considered unsafe/unsupported
        raise ValueError(f"Unsupported or unsafe format: {serialization_type}")

    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise ValueError("byte_data must be a bytes-like object")

    # Normalize to 'bytes' if a bytes-like object was provided
    if not isinstance(byte_data, bytes):
        byte_data = bytes(byte_data)

    if fmt == "json":
        try:
            text = byte_data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid byte sequence for JSON (UTF-8 required)") from exc
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON data") from exc

    if fmt == "html":
        try:
            return byte_data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Invalid byte sequence for HTML (UTF-8 required)") from exc

    # Should not be reachable due to safe_formats check above
    raise ValueError(f"Unsupported or unsafe format: {serialization_type}")
