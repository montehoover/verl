import json
import xml.etree.ElementTree as ET
import pickle

secure_formats = ["json", "xml"]

def load_serialized_data(bytes_data, data_format):
    """
    Load serialized data from bytes using a specified format.
    
    Args:
        bytes_data (bytes): The serialized data in bytes
        data_format (str): The format type - 'json', 'xml', or 'pickle'
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported, not in secure_formats, or if deserialization fails
    """
    if data_format not in secure_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_format}. Supported formats are: {', '.join(secure_formats)}")
    
    if data_format == 'json':
        try:
            return json.loads(bytes_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif data_format == 'xml':
        try:
            return ET.fromstring(bytes_data.decode('utf-8'))
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid XML data: {e}")
