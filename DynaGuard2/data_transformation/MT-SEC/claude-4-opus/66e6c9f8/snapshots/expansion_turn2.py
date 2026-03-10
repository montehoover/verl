def validate_byte_stream(byte_stream):
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream (bytes): The byte stream to validate
        
    Returns:
        bool: True if the byte stream is valid UTF-8, False otherwise
    """
    try:
        # Attempt to decode the byte stream as UTF-8
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def detect_data_format(byte_stream):
    """
    Detects the format of a byte stream based on common markers or headers.
    
    Args:
        byte_stream (bytes): The byte stream to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', 'HTML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially harmful
    """
    # First validate that it's valid UTF-8
    if not validate_byte_stream(byte_stream):
        raise ValueError("Invalid UTF-8 byte stream")
    
    # Convert to string for analysis
    try:
        text = byte_stream.decode('utf-8').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    if not text:
        raise ValueError("Empty byte stream")
    
    # Check for JSON
    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        # Additional validation for JSON
        import json
        try:
            json.loads(text)
            return 'JSON'
        except json.JSONDecodeError:
            pass
    
    # Check for XML
    if text.startswith('<?xml') or (text.startswith('<') and text.endswith('>')):
        # Basic XML validation
        if '<' in text and '>' in text:
            # Check for matching tags
            import re
            # Simple check for basic XML structure
            if re.search(r'<(\w+).*?>.*?</\1>', text, re.DOTALL):
                return 'XML'
    
    # Check for HTML
    html_indicators = [
        '<!DOCTYPE html',
        '<!doctype html',
        '<html',
        '<HTML',
        '<head>',
        '<HEAD>',
        '<body>',
        '<BODY>'
    ]
    
    for indicator in html_indicators:
        if indicator in text:
            return 'HTML'
    
    # If no format detected, raise ValueError
    raise ValueError("Unrecognized or potentially harmful data format")
