import json
import xml.etree.ElementTree as ET

def convert_json_to_dict(json_string):
    """Convert a JSON string to a Python dictionary.
    
    Args:
        json_string (str): A valid JSON string
        
    Returns:
        dict: The corresponding Python dictionary
        
    Raises:
        ValueError: If the input is not valid JSON
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def convert_string_to_data(data_string, data_format):
    """Convert a JSON or XML string to structured data.
    
    Args:
        data_string (str): A valid JSON or XML string
        data_format (str): Format of the data - either 'json' or 'xml'
        
    Returns:
        dict or xml.etree.ElementTree.Element: Dictionary for JSON, Element object for XML
        
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
