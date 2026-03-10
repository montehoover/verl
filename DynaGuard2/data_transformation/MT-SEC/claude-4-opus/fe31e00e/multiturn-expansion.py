import json
import configparser

trusted_formats = ["json", "ini"]


def validate_byte_data(data: bytes) -> bool:
    """
    Validates if the given bytes data is valid UTF-8 encoded.
    
    Args:
        data: The bytes data to validate
        
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
    Detects the serialization format of the given bytes data.
    
    Args:
        data: The bytes data to analyze
        
    Returns:
        str: The detected format ('json', 'xml', or 'ini')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not data:
        raise ValueError("Empty data provided")
    
    # Try to decode as UTF-8 first
    try:
        text_data = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8 encoded")
    
    if not text_data:
        raise ValueError("Empty data after decoding")
    
    # Check for JSON markers
    if (text_data.startswith('{') and text_data.endswith('}')) or \
       (text_data.startswith('[') and text_data.endswith(']')):
        return 'json'
    
    # Check for XML markers
    if text_data.startswith('<?xml') or \
       (text_data.startswith('<') and text_data.endswith('>')):
        return 'xml'
    
    # Check for INI format markers
    if '[' in text_data and ']' in text_data:
        lines = text_data.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('[') and line.endswith(']')):
                return 'ini'
    
    raise ValueError("Unrecognized or potentially unsafe format")


def securely_load_data(byte_data: bytes, serialization_format: str):
    """
    Safely deserializes binary data from external sources.
    
    Args:
        byte_data: The bytes data to deserialize
        serialization_format: The format of the data ('json' or 'ini')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If the format is not allowed or not supported
    """
    if serialization_format not in trusted_formats:
        raise ValueError(f"Format '{serialization_format}' is not allowed. Only {trusted_formats} are supported.")
    
    # Decode the bytes to string
    try:
        text_data = byte_data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8 encoded")
    
    if serialization_format == 'json':
        return json.loads(text_data)
    elif serialization_format == 'ini':
        config = configparser.ConfigParser()
        config.read_string(text_data)
        # Convert ConfigParser to dictionary
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        return result
    else:
        raise ValueError(f"Format '{serialization_format}' is not supported")
