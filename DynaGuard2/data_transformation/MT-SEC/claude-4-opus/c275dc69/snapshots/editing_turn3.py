import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

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

def deserialize_stream(serialized_data, stream_format):
    """
    Deserialize byte streams in a secure manner.
    
    Args:
        serialized_data (bytes): The data to be deserialized
        stream_format (str): Format of the data ('json', 'html', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or insecure
    """
    if stream_format.lower() not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}. Allowed formats are: {', '.join(allowed_formats)}")
    
    if stream_format.lower() == 'json':
        try:
            decoded_string = serialized_data.decode('utf-8')
            return json.loads(decoded_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif stream_format.lower() == 'html':
        try:
            decoded_string = serialized_data.decode('utf-8')
            return decoded_string
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {e}")
