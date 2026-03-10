from typing import Union

def validate_byte_stream(stream: Union[bytes, bytearray]) -> bool:
    """
    Validates if the given byte stream is non-empty and not corrupted.

    Args:
        stream: The byte stream to validate. It should be a bytes or bytearray object.

    Returns:
        True if the stream is a non-empty bytes or bytearray object, False otherwise.
    """
    if not isinstance(stream, (bytes, bytearray)):
        # Not a valid byte stream type
        return False

    if not stream:
        # Empty stream
        return False

    # At this point, the stream is a non-empty bytes or bytearray object.
    # More sophisticated "corruption" checks (e.g., checksum, magic numbers)
    # would depend on the specific format of the byte stream and are not
    # implemented in this generic function.
    return True

def detect_format(stream: Union[bytes, bytearray]) -> str:
    """
    Detects the format of a byte stream (JSON, HTML, XML).

    Args:
        stream: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "HTML", "XML").

    Raises:
        ValueError: If the stream is empty, format is unrecognized, or potentially unsafe.
    """
    if not validate_byte_stream(stream):
        raise ValueError("Invalid or empty byte stream provided.")

    # Decode a small prefix of the stream for inspection
    # Attempt to decode as UTF-8, common for these formats.
    # Limit the prefix size to avoid reading too much data.
    prefix_size = 100  # Number of bytes to inspect
    try:
        # Ensure stream is bytes for slicing and decoding
        if isinstance(stream, bytearray):
            stream_bytes = bytes(stream)
        else:
            stream_bytes = stream
        
        prefix_str = stream_bytes[:prefix_size].decode('utf-8', errors='ignore').lstrip()
    except Exception as e:
        raise ValueError(f"Could not decode stream prefix for format detection: {e}")

    if not prefix_str:
        raise ValueError("Stream content is empty after decoding and stripping whitespace.")

    # Check for JSON
    if prefix_str.startswith('{') or prefix_str.startswith('['):
        return "JSON"

    # Check for XML (more specific than HTML)
    if prefix_str.lower().startswith('<?xml'):
        return "XML"
    
    # Check for HTML (common doctype or html tag)
    # HTML can also be XML (XHTML), so XML check should ideally be more robust
    # or come after a more specific HTML check if needed.
    # For simplicity, we check common HTML starts.
    lower_prefix = prefix_str.lower()
    if lower_prefix.startswith('<!doctype html>') or lower_prefix.startswith('<html>'):
        return "HTML"

    # Fallback check for general XML/HTML-like structures (starts with a tag)
    # This is a less specific check.
    if prefix_str.startswith('<'):
        # Could be XML or HTML not caught by specific checks above.
        # For this example, let's assume XML if it starts with a generic tag
        # and wasn't identified as HTML's specific doctype/html tag.
        # A more robust solution might involve trying to parse with XML/HTML parsers.
        # However, the request is to identify based on initial characters.
        # If it starts with '<' and not '<?xml', '<html>', or '<!doctype html>',
        # it's more likely to be a generic XML document or a fragment.
        # We will classify as XML for now.
        # A more sophisticated check might look for closing tags or specific namespaces.
        if '/>' in prefix_str or '</' in prefix_str: # Basic check for tag structure
            return "XML" # Or potentially "HTML" if it's a fragment

    raise ValueError("Unrecognized or potentially unsafe byte stream format.")
