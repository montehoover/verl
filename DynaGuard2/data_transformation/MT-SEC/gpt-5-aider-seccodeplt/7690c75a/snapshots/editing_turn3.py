import json
from typing import Any, Dict
import xml.etree.ElementTree as ET

# Only these formats are allowed for deserialization of byte streams
accepted_formats = ["json", "html"]

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string: A JSON-encoded string representing an object.

    Returns:
        A Python dictionary parsed from the JSON string.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        result = json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Invalid JSON input") from exc

    if not isinstance(result, dict):
        raise ValueError("JSON input must represent an object")

    return result


def convert_string_to_data(data_string: str, format_type: str) -> Any:
    """
    Convert a string into structured data based on the specified format.

    Args:
        data_string: The input string containing data.
        format_type: The format of the input string ('json' or 'xml').

    Returns:
        - For 'json': a Python dictionary.
        - For 'xml': an xml.etree.ElementTree.Element instance.

    Raises:
        ValueError: If the input is invalid for the specified format or the format is unsupported.
    """
    fmt = (format_type or "").strip().lower()

    if fmt == "json":
        try:
            result = json.loads(data_string)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid JSON input") from exc
        if not isinstance(result, dict):
            raise ValueError("JSON input must represent an object")
        return result

    if fmt == "xml":
        try:
            element = ET.fromstring(data_string)
        except (ET.ParseError, TypeError) as exc:
            raise ValueError("Invalid XML input") from exc
        return element

    raise ValueError("Unsupported format_type. Use 'json' or 'xml'.")


def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Decode a serialized byte stream into a Python object based on the specified format.

    Args:
        data: The serialized data as bytes.
        data_format: The format of the data ('json', 'html', 'pickle').

    Returns:
        - For 'json': the deserialized JSON value (dict, list, str, number, bool, or None).
        - For 'html': the decoded HTML text as a Python string.

    Raises:
        ValueError: If the format is unsupported/unsafe or the data cannot be decoded.
    """
    fmt = (data_format or "").strip().lower()

    if fmt not in accepted_formats:
        raise ValueError(
            f"Unsupported or unsafe data_format: {data_format!r}. "
            f"Accepted formats: {', '.join(accepted_formats)}"
        )

    # Ensure we have bytes-like data and decode as UTF-8 text for text-based formats
    if isinstance(data, memoryview):
        raw_bytes = data.tobytes()
    elif isinstance(data, (bytes, bytearray)):
        raw_bytes = bytes(data)
    else:
        raise ValueError("data must be a bytes-like object")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Unable to decode data as UTF-8 text") from exc

    if fmt == "json":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Invalid JSON input") from exc

    if fmt == "html":
        # For HTML, return the decoded text. Further parsing can be done by the caller if needed.
        return text

    # This line should be unreachable due to the accepted_formats check.
    raise ValueError("Unsupported format")
