import hashlib
import json
from html.parser import HTMLParser

accepted_formats = ["json", "html"]

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


def decode_data_stream(data, data_format):
    """
    Securely deserializes a byte stream into its corresponding Python object.
    
    Args:
        data: bytes object containing the serialized data
        data_format: str specifying the format ('json' or 'html')
        
    Returns:
        The deserialized Python object
        
    Raises:
        ValueError: If deserialization of an unsupported or unsafe format is attempted
    """
    # Validate the format is accepted
    if data_format.lower() not in accepted_formats:
        raise ValueError(f"Unsupported format: {data_format}. Accepted formats are: {', '.join(accepted_formats)}")
    
    # Validate the data is bytes
    if not isinstance(data, bytes):
        raise ValueError("Data must be a bytes object")
    
    # Decode based on format
    if data_format.lower() == "json":
        try:
            # Decode bytes to string then parse JSON
            json_string = data.decode('utf-8')
            return json.loads(json_string)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize JSON data: {str(e)}")
    
    elif data_format.lower() == "html":
        try:
            # For HTML, we'll return the decoded string
            # (HTML parsing typically doesn't produce a single Python object structure)
            html_string = data.decode('utf-8')
            
            # Basic validation using HTMLParser
            parser = HTMLParser()
            parser.feed(html_string)
            parser.close()
            
            return html_string
        except Exception as e:
            raise ValueError(f"Failed to deserialize HTML data: {str(e)}")
