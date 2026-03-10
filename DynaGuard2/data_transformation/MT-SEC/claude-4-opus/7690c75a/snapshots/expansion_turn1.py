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
