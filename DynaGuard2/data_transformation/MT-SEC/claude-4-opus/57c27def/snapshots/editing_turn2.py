import json
import xml.etree.ElementTree as ET

def convert_json_to_dict(json_string):
    """
    Convert a JSON string to a Python dictionary.
    
    Args:
        json_string (str): A JSON formatted string
        
    Returns:
        dict: The corresponding Python dictionary
        
    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def convert_to_dict(data_string, format_type):
    """
    Convert a data string to a dictionary or structured object based on format type.
    
    Args:
        data_string (str): A data string in JSON or XML format
        format_type (str): The format type ('json' or 'xml')
        
    Returns:
        dict or ElementTree: Dictionary for JSON, ElementTree object for XML
        
    Raises:
        ValueError: If the format is not supported or data is invalid
    """
    if format_type.lower() == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif format_type.lower() == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    else:
        raise ValueError(f"Unsupported format type: {format_type}. Supported formats are 'json' and 'xml'")
