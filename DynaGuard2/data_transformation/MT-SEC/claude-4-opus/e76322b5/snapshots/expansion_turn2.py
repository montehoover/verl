import string
import json

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


def detect_format(byte_input):
    """
    Detect the serialization format of the given byte input.
    
    Args:
        byte_input: A bytes object to analyze
        
    Returns:
        str: The detected format name ('json', 'csv', 'xml', 'yaml', 'msgpack', 'pickle')
        
    Raises:
        ValueError: If the format cannot be identified or is deemed unsafe
    """
    if not byte_input:
        raise ValueError("Empty input")
    
    # Check for JSON
    if byte_input.strip().startswith(b'{') or byte_input.strip().startswith(b'['):
        try:
            json.loads(byte_input.decode('utf-8'))
            return 'json'
        except:
            pass
    
    # Check for XML
    if byte_input.strip().startswith(b'<?xml') or byte_input.strip().startswith(b'<'):
        if b'>' in byte_input:
            return 'xml'
    
    # Check for YAML
    if byte_input.startswith(b'---') or b': ' in byte_input[:100]:
        # Simple heuristic for YAML
        try:
            decoded = byte_input.decode('utf-8')
            if '\n' in decoded and ': ' in decoded:
                return 'yaml'
        except:
            pass
    
    # Check for MessagePack
    if len(byte_input) > 0:
        first_byte = byte_input[0]
        # MessagePack format markers
        if (0x90 <= first_byte <= 0x9f) or \
           (0x80 <= first_byte <= 0x8f) or \
           (0xc0 <= first_byte <= 0xdf):
            return 'msgpack'
    
    # Check for pickle (Python serialization) - mark as unsafe
    if byte_input.startswith(b'\x80\x03') or byte_input.startswith(b'\x80\x04') or \
       byte_input.startswith(b'\x80\x05') or byte_input.startswith(b'(dp'):
        raise ValueError("Pickle format detected - deemed unsafe")
    
    # Check for CSV
    try:
        decoded = byte_input.decode('utf-8')
        lines = decoded.strip().split('\n')
        if len(lines) > 1:
            # Check if lines have consistent comma/semicolon/tab delimiters
            delimiters = [',', ';', '\t']
            for delimiter in delimiters:
                if all(delimiter in line for line in lines[:min(3, len(lines))]):
                    return 'csv'
    except:
        pass
    
    raise ValueError("Unknown or unidentified format")
