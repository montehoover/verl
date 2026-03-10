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

def convert_string_to_data(data_string, format_type):
    """
    Convert a JSON or XML string to a Python data structure.
    
    Args:
        data_string (str): A JSON or XML formatted string
        format_type (str): The format type - either 'json' or 'xml'
        
    Returns:
        dict or xml.etree.ElementTree.Element: Dictionary for JSON, Element tree for XML
        
    Raises:
        ValueError: If the format type is not supported or the input is invalid
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
        raise ValueError(f"Unsupported format type: {format_type}. Supported types are 'json' and 'xml'")
