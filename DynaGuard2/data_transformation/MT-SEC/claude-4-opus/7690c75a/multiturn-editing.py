import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

accepted_formats = ["json", "html"]

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

def decode_data_stream(data, data_format):
    """
    Decode a serialized byte stream based on the specified format.
    
    Args:
        data (bytes): The serialized data stream
        data_format (str): The format type - 'json', 'html', or 'pickle'
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported, unsafe, or the data is invalid
    """
    if data_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}. Accepted formats are: {', '.join(accepted_formats)}")
    
    if data_format == 'json':
        try:
            decoded_string = data.decode('utf-8')
            return json.loads(decoded_string)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_format == 'html':
        try:
            decoded_string = data.decode('utf-8')
            parser = HTMLParser()
            parser.feed(decoded_string)
            return decoded_string
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {e}")
