import json
import xml.etree.ElementTree as ET
from typing import Union, Dict

def convert_to_dict(data_string: str, format_type: str) -> Union[Dict, ET.Element]:
    """
    Converts a JSON or XML string into a Python dictionary or an ElementTree element.

    Args:
        data_string: The string to convert.
        format_type: The format of the input string ('json' or 'xml').

    Returns:
        A dictionary if the format is 'json', or an ElementTree element if the format is 'xml'.

    Raises:
        ValueError: If the input string is not valid for the specified format,
                    or if the format_type is not supported.
    """
    if format_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    elif format_type == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}") from e
    else:
        raise ValueError(f"Unsupported format_type: {format_type}. Supported formats are 'json' and 'xml'.")
