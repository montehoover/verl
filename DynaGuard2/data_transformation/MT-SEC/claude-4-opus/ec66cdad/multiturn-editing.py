import json
import xml.etree.ElementTree as ET
import pickle
from html.parser import HTMLParser

safe_formats = ["json", "html"]

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

def restore_object_from_stream(byte_data, serialization_type):
    """Deserialize byte streams into Python objects based on specified format.
    
    Args:
        byte_data (bytes): The byte data to deserialize
        serialization_type (str): Format of the data - 'json', 'html', or 'pickle'
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported, unsafe, or data is invalid
    """
    if serialization_type.lower() not in safe_formats:
        raise ValueError(f"Unsafe or unsupported format: {serialization_type}. Safe formats are: {', '.join(safe_formats)}")
    
    if serialization_type.lower() == 'json':
        try:
            text_data = byte_data.decode('utf-8')
            return json.loads(text_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif serialization_type.lower() == 'html':
        try:
            text_data = byte_data.decode('utf-8')
            parser = HTMLParser()
            # HTML parsing doesn't produce a Python object structure like JSON
            # Return the decoded HTML string
            return text_data
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML data: {e}")
