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
