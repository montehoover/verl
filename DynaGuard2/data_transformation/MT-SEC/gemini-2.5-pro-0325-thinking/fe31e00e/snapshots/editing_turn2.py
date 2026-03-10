import json
import xml.etree.ElementTree as ET
from typing import Union, Dict

def convert_string_to_data(data_string: str, format_type: str) -> Union[Dict, ET.Element]:
    """
    Converts a JSON or XML string into a Python dictionary or an XML ElementTree object.

    Args:
        data_string: The string to convert.
        format_type: The format of the string, either 'json' or 'xml'.

    Returns:
        A dictionary if format_type is 'json', or an ElementTree Element
        if format_type is 'xml'.

    Raises:
        ValueError: If format_type is not 'json' or 'xml'.
    """
    if format_type == 'json':
        return json.loads(data_string)
    elif format_type == 'xml':
        return ET.fromstring(data_string)
    else:
        raise ValueError("Unsupported format_type: must be 'json' or 'xml'")
