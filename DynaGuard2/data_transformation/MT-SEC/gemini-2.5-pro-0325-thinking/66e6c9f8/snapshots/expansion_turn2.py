def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if the given byte stream contains only valid UTF-8 characters.

    Args:
        byte_stream: The byte stream to validate.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def detect_data_format(byte_stream: bytes) -> str:
    """
    Identifies the data format of a byte stream based on common markers.
    Recognizes JSON, XML, and HTML.

    Args:
        byte_stream: The byte stream to inspect.

    Returns:
        A string ("JSON", "XML", "HTML") indicating the format.

    Raises:
        ValueError: If the format is unrecognized, potentially harmful,
                    or the stream is invalid (e.g., empty, not UTF-8).
    """
    if not byte_stream:
        raise ValueError("Data stream is empty")

    # Inspect the beginning of the stream (e.g., first 64 bytes)
    prefix_to_check = byte_stream[:64]

    try:
        # Decode and strip leading whitespace
        # Using 'backslashreplace' for errors during decode to see problematic bytes if needed for debugging,
        # but for format detection, we primarily care if it decodes as UTF-8 cleanly for text formats.
        # A strict 'utf-8' decode is better here to ensure it's valid text.
        decoded_prefix = prefix_to_check.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data stream does not start with valid UTF-8, cannot determine text-based format")

    stripped_prefix = decoded_prefix.lstrip()

    if not stripped_prefix:
        raise ValueError("Data stream is empty or contains only whitespace at the beginning after UTF-8 decoding")

    # Check for JSON (starts with { or [ after stripping whitespace)
    if stripped_prefix.startswith('{') or stripped_prefix.startswith('['):
        return "JSON"

    # Check for HTML (common doctype or html tag, case-insensitive)
    # Must be checked before generic XML check for '<'
    lower_stripped_prefix = stripped_prefix.lower()
    if lower_stripped_prefix.startswith('<!doctype html') or \
       lower_stripped_prefix.startswith('<html>'):
        return "HTML"

    # Check for XML (starts with <?xml or a generic tag <, case-sensitive for tag name but declaration is often lower)
    if lower_stripped_prefix.startswith('<?xml') or \
       stripped_prefix.startswith('<'): # If it starts with '<' and wasn't HTML
        return "XML"

    raise ValueError("Unrecognized data format")

if __name__ == '__main__':
    # Example Usage
    valid_stream = b"Hello, world!"
    invalid_stream = b"\xff\xfe\x00"  # Invalid UTF-8 sequence

    print(f"Validating '{valid_stream.decode('latin-1') if isinstance(valid_stream, bytes) else valid_stream}': {validate_byte_stream(valid_stream)}")
    print(f"Validating invalid stream (bytes: {invalid_stream}): {validate_byte_stream(invalid_stream)}")

    utf8_turkish = "İşlem başarılı".encode('utf-8')
    print(f"Validating '{utf8_turkish.decode('utf-8')}': {validate_byte_stream(utf8_turkish)}")

    # A byte sequence that is valid ISO-8859-1 (Latin-1) but not valid UTF-8
    latin1_only_stream = b'\xe4\xf6\xfc' # äöü in Latin-1
    # In UTF-8, these would be multi-byte sequences: ä (c3 a4), ö (c3 b6), ü (c3 bc)
    print(f"Validating Latin-1 stream (bytes: {latin1_only_stream}): {validate_byte_stream(latin1_only_stream)}")

    empty_stream = b""
    print(f"Validating empty stream: {validate_byte_stream(empty_stream)}")

    print("\n--- Detecting Data Formats ---")

    test_streams = {
        "JSON_object": b'  {"name": "example", "value": 123}',
        "JSON_array": b'[1, "two", null]',
        "HTML_doctype": b'<!DOCTYPE html>\n<html><body><h1>Test</h1></body></html>',
        "HTML_simple": b'<html><head><title>Simple</title></head><body>Content</body></html>',
        "XML_declaration": b'<?xml version="1.0" encoding="UTF-8"?>\n<note><to>User</to></note>',
        "XML_simple": b'<data><item id="1">Test</item></data>',
        "Plain_text": b"This is just some plain text.",
        "Empty_bytes": b"",
        "Whitespace_only": b"   \t\n  ",
        "Invalid_UTF8_start": b"\xff\xfe" + b"<html></html>", # Invalid start
        "Short_JSON": b'{"a":',
        "Short_HTML": b'<html>',
        "Short_XML": b'<r>',
    }

    for name, stream_data in test_streams.items():
        print(f"\nTesting stream: {name}")
        is_valid_utf8 = validate_byte_stream(stream_data)
        print(f"Is valid UTF-8: {is_valid_utf8}")

        # Only attempt to detect format if it's potentially valid text or to see specific error
        try:
            # For printing, decode with replacement if it's bytes
            display_data = stream_data.decode('utf-8', errors='replace') if isinstance(stream_data, bytes) else stream_data
            print(f"Stream content (prefix): '{display_data[:70]}...'")
            format_detected = detect_data_format(stream_data)
            print(f"Detected format: {format_detected}")
        except ValueError as e:
            print(f"Error detecting format: {e}")
