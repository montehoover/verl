import json
import xml.etree.ElementTree as ET
from typing import Union, Dict

def convert_to_dict(data_string: str, data_format: str) -> Union[Dict, ET.Element]:
    """
    Converts a JSON or XML string into a Python dictionary or an ElementTree element.

    Args:
        data_string: The string to convert.
        data_format: The format of the string, either 'json' or 'xml'.

    Returns:
        A dictionary if the format is 'json', or an ElementTree element
        if the format is 'xml'.

    Raises:
        ValueError: If the input string is not valid for the specified format,
                    or if the format is not supported.
    """
    if data_format.lower() == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif data_format.lower() == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}")
    else:
        raise ValueError(f"Unsupported data format: {data_format}. Supported formats are 'json' and 'xml'.")
