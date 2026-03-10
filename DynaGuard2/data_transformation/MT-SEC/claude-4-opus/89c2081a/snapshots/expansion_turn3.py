import json
import configparser
from io import StringIO

secure_serialization_formats = ["json", "ini"]


def validate_byte_data(data: bytes) -> bool:
    """
    Validates if the given byte data is valid UTF-8 encoded.
    
    Args:
        data: The byte data to validate
        
    Returns:
        bool: True if the data is valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of the given byte data.
    
    Args:
        data: The byte data to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'ini')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Try to decode as UTF-8 first
    try:
        text = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8 encoded")
    
    if not text:
        raise ValueError("Empty data after decoding")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or \
       (text.startswith('[') and text.endswith(']')):
        return 'json'
    
    # Check for XML
    if text.startswith('<?xml') or \
       (text.startswith('<') and text.endswith('>') and '</' in text):
        return 'xml'
    
    # Check for INI
    if '[' in text and ']' in text:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                return 'ini'
    
    raise ValueError("Unrecognized or potentially unsafe format")


def process_serialized_content(payload_bytes: bytes, schema_type: str) -> object:
    """
    Securely deserializes binary data into a Python object.
    
    Args:
        payload_bytes: The binary data to deserialize
        schema_type: The format for deserialization ('json' or 'ini')
        
    Returns:
        object: The deserialized Python object
        
    Raises:
        ValueError: If the format is unsupported or unsafe
    """
    if schema_type not in secure_serialization_formats:
        raise ValueError(f"Unsupported or unsafe format: {schema_type}")
    
    # Validate UTF-8 encoding
    if not validate_byte_data(payload_bytes):
        raise ValueError("Invalid UTF-8 encoded data")
    
    # Decode to text
    text = payload_bytes.decode('utf-8')
    
    if schema_type == 'json':
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    elif schema_type == 'ini':
        try:
            config = configparser.ConfigParser()
            config.read_string(text)
            # Convert ConfigParser to dictionary
            result = {}
            for section in config.sections():
                result[section] = dict(config.items(section))
            return result
        except configparser.Error as e:
            raise ValueError(f"Invalid INI data: {e}")
