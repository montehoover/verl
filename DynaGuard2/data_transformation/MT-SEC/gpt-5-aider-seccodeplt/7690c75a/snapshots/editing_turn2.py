import json
from typing import Any, Dict
import xml.etree.ElementTree as ET

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
