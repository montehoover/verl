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

def convert_to_dict(data_string, data_format):
    """
    Convert a data string to a dictionary (JSON) or structured object (XML).
    
    Args:
        data_string (str): A JSON or XML formatted string
        data_format (str): Format of the data ('json' or 'xml')
        
    Returns:
        dict or xml.etree.ElementTree.Element: Dictionary for JSON, Element for XML
        
    Raises:
        ValueError: If the format is not supported or data is invalid
    """
    if data_format.lower() == 'json':
        try:
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
    elif data_format.lower() == 'xml':
        try:
            return ET.fromstring(data_string)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
    else:
        raise ValueError(f"Unsupported format: {data_format}. Supported formats are 'json' and 'xml'")
