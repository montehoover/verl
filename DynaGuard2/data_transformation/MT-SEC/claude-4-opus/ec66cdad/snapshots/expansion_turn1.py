import string

def is_printable_byte_stream(byte_stream):
    """
    Check if a byte stream consists entirely of printable ASCII characters.
    
    Args:
        byte_stream: bytes object or byte array
        
    Returns:
        bool: True if all bytes are printable ASCII, False otherwise
    """
    # Define printable ASCII range (32-126) plus common whitespace characters
    printable_chars = set(string.printable.encode('ascii'))
    
    try:
        # Check each byte in the stream
        for byte in byte_stream:
            if byte not in printable_chars:
                return False
        return True
    except Exception:
        return False
