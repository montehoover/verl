import json
import re

def validate_byte_content(byte_stream):
    """
    Validates if a byte stream contains only valid UTF-8 characters.
    
    Args:
        byte_stream: A bytes object to validate
        
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
        byte_stream: A bytes object to analyze
        
    Returns:
        str: The detected format ('JSON', 'XML', or 'HTML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    # First validate UTF-8
    if not validate_byte_content(byte_stream):
        raise ValueError("Invalid UTF-8 content")
    
    # Convert to string for analysis
    try:
        content = byte_stream.decode('utf-8').strip()
    except:
        raise ValueError("Unable to decode byte stream")
    
    if not content:
        raise ValueError("Empty content")
    
    # Check for JSON
    if content.startswith('{') or content.startswith('['):
        try:
            json.loads(content)
            return 'JSON'
        except json.JSONDecodeError:
            pass
    
    # Check for XML
    xml_pattern = r'^\s*<\?xml\s+version'
    if re.match(xml_pattern, content, re.IGNORECASE) or (content.startswith('<') and content.endswith('>')):
        # Basic XML structure check
        if re.search(r'<([^/>\s]+)(?:\s[^>]*)?>.*</\1>', content, re.DOTALL):
            return 'XML'
    
    # Check for HTML
    html_patterns = [
        r'<!DOCTYPE\s+html',
        r'<html[>\s]',
        r'<head[>\s]',
        r'<body[>\s]',
        r'<title[>\s]'
    ]
    
    for pattern in html_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return 'HTML'
    
    # If none matched, raise error
    raise ValueError("Unrecognized or potentially unsafe format")
