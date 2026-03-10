import string

def is_ascii_printable(byte_stream):
    """
    Verify if a given byte stream consists solely of ASCII printable characters.
    
    Args:
        byte_stream: A bytes object to check
        
    Returns:
        bool: True if all bytes are ASCII printable characters, False otherwise
    """
    # ASCII printable characters are in the range 32-126 (space to ~)
    # We can also use string.printable but need to be careful with whitespace
    printable_bytes = set(string.printable.encode('ascii'))
    
    for byte in byte_stream:
        if byte not in printable_bytes:
            return False
    
    return True
