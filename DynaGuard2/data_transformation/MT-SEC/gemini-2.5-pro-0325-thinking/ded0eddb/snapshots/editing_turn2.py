import json
import xml.etree.ElementTree as ET

def convert_to_object(data_string: str, format_type: str):
    """
    Converts a data string (JSON or XML) into a Python object.

    Args:
        data_string: The string to convert.
        format_type: The format of the data string ('json' or 'xml').

    Returns:
        The corresponding Python object.

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
        raise ValueError(f"Unsupported format_type: {format_type}. Must be 'json' or 'xml'.")
