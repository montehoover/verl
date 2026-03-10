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

def convert_to_dict(content_string, format_type):
    """
    Convert a JSON or XML string to a Python data structure.
    
    Args:
        content_string (str): A JSON or XML formatted string
        format_type (str): The format type, either 'json' or 'xml'
        
    Returns:
        dict or ElementTree: Dictionary for JSON, ElementTree for XML
        
    Raises:
        ValueError: If the format is not supported or content is invalid
    """
    if format_type == 'json':
        try:
            return json.loads(content_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif format_type == 'xml':
        try:
            return ET.fromstring(content_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    else:
        raise ValueError(f"Unsupported format: {format_type}")
