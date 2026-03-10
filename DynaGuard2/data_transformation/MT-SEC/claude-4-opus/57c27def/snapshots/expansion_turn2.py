def validate_binary_data(data: bytes) -> bool:
    """
    Validates if the given binary data is valid UTF-8 encoded.
    
    Args:
        data: Binary data to validate
        
    Returns:
        bool: True if data is valid UTF-8, False otherwise
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(data: bytes) -> str:
    """
    Detects the format of the given binary data based on common signatures.
    
    Args:
        data: Binary data to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', 'INI')
        
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
        raise ValueError("Empty text data after decoding")
    
    # Check for JSON format
    if (text_data.startswith('{') and text_data.endswith('}')) or \
       (text_data.startswith('[') and text_data.endswith(']')):
        # Basic JSON validation
        import json
        try:
            json.loads(text_data)
            return 'JSON'
        except json.JSONDecodeError:
            pass
    
    # Check for XML format
    if text_data.startswith('<?xml') or \
       (text_data.startswith('<') and text_data.endswith('>')):
        # Basic XML validation
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(text_data)
            return 'XML'
        except ET.ParseError:
            pass
    
    # Check for INI format
    if '[' in text_data and ']' in text_data:
        # Basic INI validation
        import configparser
        config = configparser.ConfigParser()
        try:
            config.read_string(text_data)
            # Check if at least one section exists
            if config.sections():
                return 'INI'
        except configparser.Error:
            pass
    
    raise ValueError("Unrecognized or potentially unsafe data format")
