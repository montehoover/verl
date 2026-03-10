import json
import xml.etree.ElementTree as ET

def convert_to_object(data_string: str, format_type: str):
    """
    Converts a JSON or XML string into a Python object or structured data.

    Args:
        data_string: The string to convert (either JSON or XML).
        format_type: A string specifying the format ('json' or 'xml').

    Returns:
        The corresponding Python object for JSON, or an ElementTree Element for XML.

    Raises:
        ValueError: If format_type is not 'json' or 'xml'.
    """
    if format_type.lower() == 'json':
        return json.loads(data_string)
    elif format_type.lower() == 'xml':
        return ET.fromstring(data_string)
    else:
        raise ValueError("Unsupported format_type. Please use 'json' or 'xml'.")
