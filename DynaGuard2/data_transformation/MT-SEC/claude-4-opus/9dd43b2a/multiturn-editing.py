import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

allowed_formats = ["json", "html"]

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

def parse_serialized_content(content, format_type):
    """
    Parse serialized byte content based on the specified format type.
    
    Args:
        content (bytes): The serialized data
        format_type (str): The format of the content ('json', 'html', 'pickle')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is not supported, insecure, or content is invalid
    """
    if format_type not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}. Allowed formats are: {', '.join(allowed_formats)}")
    
    if format_type == 'json':
        try:
            decoded_content = content.decode('utf-8')
            return json.loads(decoded_content)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON content: {e}")
    elif format_type == 'html':
        try:
            decoded_content = content.decode('utf-8')
            parser = HTMLParser()
            parser.feed(decoded_content)
            return decoded_content
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid HTML content: {e}")
