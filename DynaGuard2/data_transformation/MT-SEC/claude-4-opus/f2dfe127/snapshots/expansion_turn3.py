import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]


def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains valid UTF-8 encoded data.
    
    Args:
        byte_stream (bytes): The byte stream to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream (bytes): The byte stream to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'html')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not byte_stream:
        raise ValueError("Empty byte stream")
    
    # Try to decode as UTF-8 first
    try:
        text = byte_stream.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Invalid UTF-8 encoding")
    
    if not text:
        raise ValueError("Empty content after decoding")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return 'json'
    
    # Check for XML
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and 
        not text.lower().startswith('<!doctype html') and
        not text.lower().startswith('<html')):
        return 'xml'
    
    # Check for HTML
    if text.lower().startswith('<!doctype html') or \
       text.lower().startswith('<html') or \
       ('<html' in text.lower()[:100] and '</html>' in text.lower()):
        return 'html'
    
    raise ValueError("Unrecognized or potentially unsafe format")


def bytes_to_obj(data_bytes, format_name):
    """
    Securely deserialize a byte stream into a Python object.
    
    Args:
        data_bytes (bytes): The byte stream to deserialize
        format_name (str): The serialization format ('json' or 'xml')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or dangerous
    """
    if format_name not in approved_formats:
        raise ValueError(f"Unsupported format: {format_name}. Only {approved_formats} are allowed.")
    
    if format_name == "json":
        try:
            text = data_bytes.decode('utf-8')
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON: {str(e)}")
    
    elif format_name == "xml":
        try:
            text = data_bytes.decode('utf-8')
            root = ET.fromstring(text)
            return root
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize XML: {str(e)}")
