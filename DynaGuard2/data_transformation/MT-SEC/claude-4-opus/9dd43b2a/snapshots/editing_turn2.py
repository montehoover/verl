import json
import xml.etree.ElementTree as ET

def convert_json_to_dict(json_string):
    """
    Convert a JSON string to a Python dictionary.
    
    Args:
        json_string (str): A valid JSON string
        
    Returns:
        dict: The equivalent Python dictionary
        
    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

def convert_string_to_data(content_string, format):
    """
    Convert a string to structured data based on the specified format.
    
    Args:
        content_string (str): The content string to convert
        format (str): The format of the content ('json' or 'xml')
        
    Returns:
        dict or ElementTree.Element: Dictionary for JSON, Element tree for XML
        
    Raises:
        ValueError: If the format is not supported or content is invalid
    """
    if format == 'json':
        try:
            return json.loads(content_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    elif format == 'xml':
        try:
            return ET.fromstring(content_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML string: {e}")
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'json' and 'xml'")
