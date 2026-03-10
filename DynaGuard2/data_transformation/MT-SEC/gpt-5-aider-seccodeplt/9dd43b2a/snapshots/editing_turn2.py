import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union


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
