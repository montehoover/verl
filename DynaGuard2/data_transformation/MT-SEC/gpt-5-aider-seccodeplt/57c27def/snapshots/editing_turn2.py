import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

def convert_json_to_dict(json_string: str) -> Dict[str, Any]:
    """
    Convert a JSON string into a Python dictionary.

    Args:
        json_string (str): A string containing a JSON object.

    Returns:
        Dict[str, Any]: The parsed dictionary.

    Raises:
        ValueError: If the input is not valid JSON or does not represent a JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e

    if not isinstance(data, dict):
        raise ValueError("JSON must represent an object")

    return data

def convert_to_dict(data_string: str, format_type: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a data string into a structured object based on the specified format.

    For JSON:
        - Parses the string and returns a Python dictionary.
        - Raises ValueError if the JSON is invalid or does not represent an object.

    For XML:
        - Parses the string and returns an xml.etree.ElementTree.Element.
        - Raises ValueError if the XML is invalid.

    Args:
        data_string (str): The input data string to parse.
        format_type (str): The format of the data ('json' or 'xml').

    Returns:
        Union[Dict[str, Any], ET.Element]: A dictionary for JSON, or an XML Element for XML.

    Raises:
        ValueError: If the format is unsupported or parsing fails.
    """
    fmt = format_type.strip().lower()
    if fmt == 'json':
        return convert_json_to_dict(data_string)
    if fmt == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}") from e
    raise ValueError(f"Unsupported format: {format_type}")
