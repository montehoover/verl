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

def deserialize_stream_payload(serialized_data, data_format_type):
    """
    Deserialize byte streams from potentially untrusted sources.
    
    Args:
        serialized_data (bytes): The serialized data as bytes
        data_format_type (str): The format type ('json', 'html', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported or data is invalid
    """
    if data_format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {data_format_type}")
    
    if data_format_type == 'json':
        try:
            decoded_string = serialized_data.decode('utf-8')
            return json.loads(decoded_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_format_type == 'html':
        try:
            decoded_string = serialized_data.decode('utf-8')
            parser = HTMLParser()
            parser.feed(decoded_string)
            return decoded_string
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {e}")
