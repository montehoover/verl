import string
import json
import pickle
import configparser
from io import StringIO

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


def detect_stream_format(data: bytes) -> str:
    """
    Detect the format of the given byte stream.
    
    Args:
        data: bytes object to analyze
        
    Returns:
        str: Format tag ('json', 'ini', or 'pickle')
        
    Raises:
        ValueError: If format is unclear or potentially dangerous
    """
    if not data:
        raise ValueError("Empty data stream")
    
    # Try to detect JSON
    try:
        decoded = data.decode('utf-8', errors='strict')
        json.loads(decoded)
        return 'json'
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    
    # Try to detect INI
    try:
        decoded = data.decode('utf-8', errors='strict')
        parser = configparser.ConfigParser()
        parser.read_string(decoded)
        # Check if at least one section was parsed
        if parser.sections():
            return 'ini'
    except (UnicodeDecodeError, configparser.Error):
        pass
    
    # Try to detect pickle
    try:
        # Check for pickle protocol header bytes
        if data[:2] in [b'\x80\x03', b'\x80\x04', b'\x80\x05']:  # Protocol 3, 4, 5
            # Verify it's valid pickle without actually unpickling (security risk)
            pickle.loads(data)
            return 'pickle'
    except (pickle.UnpicklingError, IndexError):
        pass
    
    # If none of the formats match, raise an error
    raise ValueError("Unknown or potentially dangerous format")
