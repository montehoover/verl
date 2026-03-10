import json
import xml.etree.ElementTree as ET
from typing import Union, Dict

def convert_string_to_data(content_string: str, format: str) -> Union[Dict, ET.Element]:
    """
    Converts a JSON or XML string into a Python dictionary or an ElementTree object.

    Args:
        content_string: The string to convert.
        format: The format of the string, either 'json' or 'xml'.

    Returns:
        A Python dictionary if format is 'json', or an ElementTree.Element object
        if format is 'xml'.

    Raises:
        ValueError: If the format is not supported, or if the input string
                    is not valid for the specified format.
    """
    if format == 'json':
        try:
            return json.loads(content_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    elif format == 'xml':
        try:
            return ET.fromstring(content_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}") from e
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'json' and 'xml'.")
