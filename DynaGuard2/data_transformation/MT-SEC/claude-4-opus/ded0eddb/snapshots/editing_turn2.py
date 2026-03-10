import json
import xml.etree.ElementTree as ET

def convert_to_object(data_string, format_type):
    """
    Convert a JSON or XML string to a Python object.
    
    Args:
        data_string (str): A JSON or XML formatted string
        format_type (str): The format type - either 'json' or 'xml'
        
    Returns:
        The corresponding Python object (dict, list, str, int, float, bool, or None for JSON;
        ElementTree.Element for XML)
        
    Raises:
        ValueError: If the input is not valid for the specified format or format is not supported
    """
    if format_type == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif format_type == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    else:
        raise ValueError(f"Unsupported format type: {format_type}. Supported formats are 'json' and 'xml'")
