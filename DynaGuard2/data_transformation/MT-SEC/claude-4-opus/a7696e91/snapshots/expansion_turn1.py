import string

def extract_printable_content(data: bytes) -> bool:
    """
    Check if the given bytes contain any readable or printable text.
    
    Args:
        data: bytes object to check for printable content
        
    Returns:
        bool: True if printable text is found, False otherwise
    """
    # Define printable characters (ASCII printable range plus common whitespace)
    printable_chars = set(string.printable.encode('ascii'))
    
    # Check if any byte in the data is a printable character
    for byte in data:
        if byte in printable_chars:
            return True
    
    return False
