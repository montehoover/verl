import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

def convert_string_to_data(data_string: str, format_type: str) -> Union[Dict[str, Any], ET.Element]:
    """
    Convert a string in a specified format ('json' or 'xml') into structured data.

    Args:
        data_string: The input string containing data in the specified format.
        format_type: The format of the input string: 'json' or 'xml'.

    Returns:
        For 'json': a dictionary parsed from the JSON string.
        For 'xml': an xml.etree.ElementTree.Element parsed from the XML string.

    Raises:
        ValueError: If format_type is not supported, or if JSON does not represent an object.
    """
    fmt = format_type.strip().lower()
    if fmt == 'json':
        obj = json.loads(data_string)
        if not isinstance(obj, dict):
            raise ValueError("JSON input must represent an object (dictionary) at the top level.")
        return obj
    elif fmt == 'xml':
        return ET.fromstring(data_string)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
