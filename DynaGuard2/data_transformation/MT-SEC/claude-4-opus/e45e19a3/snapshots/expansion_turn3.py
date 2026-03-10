import json
import xml.etree.ElementTree as ET

safe_formats = ["json", "xml"]


def validate_byte_data(data: bytes) -> bool:
    """
    Validate if the given bytes contain only valid UTF-8 characters.
    
    Args:
        data: The byte stream to validate
        
    Returns:
        bool: True if the bytes are valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: bytes) -> str:
    """
    Detect the format of the given byte data based on common markers.
    
    Args:
        data: The byte stream to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'html')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Try to decode as UTF-8 first
    try:
        text = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8")
    
    if not text:
        raise ValueError("Empty text after decoding")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return 'json'
    
    # Check for XML
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and '</' in text):
        return 'xml'
    
    # Check for HTML
    if text.lower().startswith('<!doctype html') or \
       text.lower().startswith('<html') or \
       ('<html' in text.lower() and '</html>' in text.lower()):
        return 'html'
    
    raise ValueError("Unrecognized or potentially unsafe format")


def deserialize_data(input_data: bytes, data_type: str):
    """
    Securely convert a byte stream into a Python object.
    
    Args:
        input_data: The byte stream to deserialize
        data_type: The serialization format ('json' or 'xml')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or unsafe
    """
    if data_type not in safe_formats:
        raise ValueError(f"Unsupported or unsafe format: {data_type}")
    
    if not input_data:
        raise ValueError("Empty data provided")
    
    if data_type == "json":
        try:
            text = input_data.decode('utf-8')
            return json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif data_type == "xml":
        try:
            text = input_data.decode('utf-8')
            root = ET.fromstring(text)
            return root
        except (ET.ParseError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid XML data: {e}")
