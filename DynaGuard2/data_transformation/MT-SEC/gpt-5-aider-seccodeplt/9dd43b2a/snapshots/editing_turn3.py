import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union


allowed_formats = ["json", "html"]


def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    :param json_string: The JSON string to parse.
    :return: The parsed dictionary.
    :raises ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(parsed, dict):
        raise ValueError("JSON must represent an object at the top level")

    return parsed


def convert_string_to_data(content_string: str, format: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a string containing either JSON or XML into structured data.

    :param content_string: The content string to parse.
    :param format: The format of the content ('json' or 'xml').
    :return: A dictionary for JSON input or an xml.etree.ElementTree.Element for XML input.
    :raises ValueError: If the content is invalid or the format is unsupported.
    """
    fmt = format.lower()

    if fmt == "json":
        return convert_json_to_dict(content_string)
    elif fmt == "xml":
        try:
            return ET.fromstring(content_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}") from e
    else:
        raise ValueError("Unsupported format. Use 'json' or 'xml'.")


def parse_serialized_content(content: bytes, format_type: str) -> Any:
    """
    Deserialize a bytes payload according to the specified format.

    :param content: Serialized data as bytes.
    :param format_type: The format type ('json', 'html', 'pickle', etc.). Only formats in allowed_formats are supported.
    :return: The deserialized Python object.
    :raises ValueError: If the format is unsupported/insecure or the payload is invalid.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("content must be bytes")

    fmt = (format_type or "").strip().lower()
    if fmt not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format. Allowed formats: {', '.join(allowed_formats)}")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("Content is not valid UTF-8 encoded bytes") from e

    if fmt == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e.msg}") from e
    elif fmt == "html":
        # For HTML, return the decoded string content.
        return text

    # This point should not be reachable due to the allowed_formats check.
    raise ValueError("Unsupported or insecure format.")
