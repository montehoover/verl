import string

def is_printable_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if a byte stream consists entirely of printable ASCII characters.

    Args:
        byte_stream: The byte stream to check.

    Returns:
        True if the byte stream is printable, False otherwise.
    """
    try:
        # Attempt to decode the byte stream as ASCII.
        # If this fails, it means there are non-ASCII characters.
        decoded_stream = byte_stream.decode('ascii')
    except UnicodeDecodeError:
        return False

    # Check if all characters in the decoded stream are printable.
    for char_code in byte_stream:
        if chr(char_code) not in string.printable:
            return False
    return True


def detect_stream_format(byte_stream: bytes) -> str:
    """
    Detects the format of a byte stream (JSON, XML, HTML).

    Args:
        byte_stream: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "XML", "HTML").

    Raises:
        ValueError: If the stream is not printable ASCII, empty,
                    or the format is unrecognized/unsafe.
    """
    if not byte_stream:
        raise ValueError("Input byte stream is empty.")

    if not is_printable_byte_stream(byte_stream):
        raise ValueError("Input byte stream contains non-printable or non-ASCII characters.")

    # Decode after ensuring it's printable ASCII
    try:
        # Use 'ascii' as per is_printable_byte_stream, though 'utf-8' might be more common for these formats
        # However, the requirement was to check for printable ASCII first.
        # For actual parsing, one might need a more robust decoding strategy.
        text_content = byte_stream.decode('ascii').strip()
    except UnicodeDecodeError:
        # This case should ideally be caught by is_printable_byte_stream,
        # but as a safeguard:
        raise ValueError("Failed to decode byte stream as ASCII.")


    # Check for JSON
    if text_content.startswith('{') and text_content.endswith('}'):
        return "JSON"
    if text_content.startswith('[') and text_content.endswith(']'):
        return "JSON"

    # Check for XML/HTML (these can be tricky to differentiate robustly with simple checks)
    # A common XML declaration
    if text_content.lower().startswith('<?xml'):
        return "XML"

    # HTML doctype
    if text_content.lower().startswith('<!doctype html'):
        return "HTML"
    
    # Common root tags
    if text_content.lower().startswith('<html'):
        return "HTML"
    
    # Generic XML/HTML tag start (could be either)
    # For simplicity, if it starts with a tag and wasn't caught by HTML specific checks,
    # we might assume XML. More sophisticated checks would involve parsing.
    if text_content.startswith('<') and text_content.endswith('>'):
        # This is a very basic check. HTML often doesn't have an XML declaration.
        # If it wasn't identified as specific HTML, let's check for general tag structure.
        # This could be improved, e.g. by looking for common XML root tags if not HTML.
        # For now, if it looks like a tag, and not specific HTML, lean towards XML.
        # This is a simplification; robust differentiation is complex.
        # If it's not `<!doctype html` or `<html>`, but starts with `<`, it's more likely XML or a fragment.
        # Let's prioritize more specific HTML checks above.
        # If it reaches here and starts with '<', assume XML for now.
        return "XML"


    raise ValueError("Unrecognized or potentially unsafe stream format.")


if __name__ == '__main__':
    # Example Usage for is_printable_byte_stream
    printable_stream = b"Hello, World! 123\n\t"
    non_printable_stream_ascii_control = b"Hello\x07World"  # Contains a non-printable ASCII control character (BEL)
    non_printable_stream_extended_ascii = b"H\x82llo" # Contains a non-ASCII character (often e with breve in some extended sets)
    empty_stream = b""

    print(f"'{printable_stream.decode('ascii', errors='replace')}': {is_printable_byte_stream(printable_stream)}")
    print(f"'{non_printable_stream_ascii_control.decode('ascii', errors='replace')}': {is_printable_byte_stream(non_printable_stream_ascii_control)}")
    print(f"'{non_printable_stream_extended_ascii.decode('ascii', errors='replace')}': {is_printable_byte_stream(non_printable_stream_extended_ascii)}")
    print(f"Empty stream: {is_printable_byte_stream(empty_stream)}")

    # Test with all printable ASCII characters
    all_printable_bytes = "".join(string.printable).encode('ascii')
    print(f"All printable ASCII characters: {is_printable_byte_stream(all_printable_bytes)}")

    # Test with a byte that is ASCII but not in string.printable (e.g., vertical tab)
    tricky_stream = b"test\x0btest" # \x0b is VT (Vertical Tab)
    print(f"'{tricky_stream.decode('ascii', errors='replace')}': {is_printable_byte_stream(tricky_stream)}")

    print("\n--- Testing detect_stream_format ---")

    # Test cases for detect_stream_format
    json_stream_obj = b'{ "name": "John", "age": 30 }'
    json_stream_arr = b'[1, 2, 3, "test"]'
    xml_stream_decl = b'<?xml version="1.0"?><root><item>Test</item></root>'
    xml_stream_no_decl = b'<note><to>Tove</to><from>Jani</from></note>'
    html_stream_doctype = b'<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>'
    html_stream_simple = b'<html><body><h1>My First Heading</h1><p>My first paragraph.</p></body></html>'
    plain_text_stream = b"This is just plain text."
    empty_for_format = b""
    non_printable_for_format = b"Invalid\x00Stream"

    streams_to_test = {
        "JSON Object": json_stream_obj,
        "JSON Array": json_stream_arr,
        "XML with Declaration": xml_stream_decl,
        "XML without Declaration": xml_stream_no_decl,
        "HTML with DOCTYPE": html_stream_doctype,
        "HTML simple": html_stream_simple,
        "Plain Text": plain_text_stream,
        "Empty Stream for Format": empty_for_format,
        "Non-Printable for Format": non_printable_for_format,
        "Non-ASCII for Format": b"H\x82llo World" # Already tested with is_printable_byte_stream
    }

    for name, stream_data in streams_to_test.items():
        try:
            print(f"Testing '{name}':")
            # First, ensure it's considered printable by the first function if it's supposed to be valid
            if name not in ["Non-Printable for Format", "Non-ASCII for Format", "Empty Stream for Format"]:
                 if not is_printable_byte_stream(stream_data):
                    print(f"  WARN: '{name}' is unexpectedly not printable by is_printable_byte_stream.")
            
            format_type = detect_stream_format(stream_data)
            print(f"  Detected format: {format_type}")
        except ValueError as e:
            print(f"  Error: {e}")
