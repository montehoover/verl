import json
import xml.etree.ElementTree as ET
from typing import Any

def convert_to_object(data_string: str, format_type: str) -> Any:
    """
    Convert a data string in either JSON or XML format to the corresponding Python object.

    For JSON:
        - Returns the Python object represented by the JSON (dict, list, str, int, float, bool, or None).
    For XML:
        - Returns an xml.etree.ElementTree.Element representing the root of the parsed XML tree.

    Args:
        data_string (str): The input data as a string.
        format_type (str): The format of the input data, either 'json' or 'xml' (case-insensitive).

    Returns:
        Any: The Python object corresponding to the parsed data.

    Raises:
        TypeError: If data_string or format_type is not a string.
        ValueError: If the format_type is unsupported or the input data is invalid for the specified format.
    """
    if not isinstance(data_string, str):
        raise TypeError("data_string must be a string")
    if not isinstance(format_type, str):
        raise TypeError("format_type must be a string")

    fmt = format_type.strip().lower()

    if fmt == "json":
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(str(e)) from e
    elif fmt == "xml":
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(str(e)) from e
    else:
        raise ValueError("Unsupported format_type. Supported formats are 'json' and 'xml'.")
