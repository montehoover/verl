import json
import xml.etree.ElementTree as ET

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

def convert_string_to_data(data_string, data_format):
    """
    Convert a string to structured data based on the specified format.
    
    Args:
        data_string (str): A string containing data in the specified format
        data_format (str): The format of the data ('json' or 'xml')
        
    Returns:
        dict or xml.etree.ElementTree.Element: Dictionary for JSON, Element tree for XML
        
    Raises:
        ValueError: If the format is not supported or the data is invalid
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

def deserialize_content_stream(stream, stream_format):
    """
    Deserialize a byte stream using the specified format.
    
    Args:
        stream (bytes): The serialized data as bytes
        stream_format (str): The format of the data ('json', 'html', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported or insecure, or if deserialization fails
    """
    if stream_format.lower() not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {stream_format}. Allowed formats are: {', '.join(allowed_formats)}")
    
    if stream_format.lower() == 'json':
        try:
            return json.loads(stream.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {e}")
    elif stream_format.lower() == 'html':
        try:
            return stream.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode HTML: {e}")
