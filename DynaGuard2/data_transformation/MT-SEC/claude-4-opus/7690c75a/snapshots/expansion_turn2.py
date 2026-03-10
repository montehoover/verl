import hashlib

def validate_byte_stream(byte_stream):
    """
    Validates a byte stream to ensure it is non-empty and not corrupted.
    
    Args:
        byte_stream: bytes object containing the data to validate
        
    Returns:
        bool: True if the byte stream is valid, False otherwise
    """
    # Check if byte stream is None or empty
    if not byte_stream or len(byte_stream) == 0:
        return False
    
    # Check if it's actually a bytes object
    if not isinstance(byte_stream, bytes):
        return False
    
    # Basic corruption check - verify the byte stream can be processed
    try:
        # Attempt to compute a hash to verify the data can be read
        hashlib.sha256(byte_stream).digest()
        
        # Check for common corruption patterns
        # All null bytes might indicate corruption
        if all(b == 0 for b in byte_stream):
            return False
            
        return True
    except Exception:
        # If any error occurs during processing, consider it corrupted
        return False


def detect_format(byte_stream):
    """
    Detects the format of a byte stream by examining its content.
    
    Args:
        byte_stream: bytes object containing the data to analyze
        
    Returns:
        str: The detected format ('JSON', 'HTML', 'XML')
        
    Raises:
        ValueError: If the format is unrecognized or potentially unsafe
    """
    if not byte_stream:
        raise ValueError("Empty byte stream provided")
    
    # Decode the beginning of the stream for analysis
    try:
        # Get first 1000 bytes for analysis (or entire stream if smaller)
        sample_size = min(1000, len(byte_stream))
        sample = byte_stream[:sample_size].decode('utf-8', errors='ignore').strip()
    except Exception:
        raise ValueError("Unable to decode byte stream")
    
    if not sample:
        raise ValueError("Byte stream appears to be empty after decoding")
    
    # Remove BOM if present
    if sample.startswith('\ufeff'):
        sample = sample[1:]
    
    # Check for JSON
    if sample.startswith('{') or sample.startswith('['):
        # Additional validation for JSON
        if any(char in sample[:50] for char in ['"', ':', '[', '{', 'null', 'true', 'false']):
            return 'JSON'
    
    # Check for XML
    if sample.startswith('<?xml') or (sample.startswith('<') and '>' in sample[:100]):
        # Look for XML declaration or root element
        if '<?xml' in sample[:100] or (sample.count('<') > 0 and sample.count('>') > 0):
            return 'XML'
    
    # Check for HTML
    if sample.lower().startswith('<!doctype html') or sample.lower().startswith('<html'):
        return 'HTML'
    
    # Additional HTML detection for partial documents
    html_indicators = ['<head>', '<body>', '<div>', '<p>', '<h1>', '<h2>', '<h3>', 
                      '<span>', '<table>', '<form>', '<meta', '<link', '<script>']
    sample_lower = sample[:500].lower()
    if any(indicator in sample_lower for indicator in html_indicators):
        return 'HTML'
    
    # If no format detected, raise error
    raise ValueError("Unrecognized or potentially unsafe format")
