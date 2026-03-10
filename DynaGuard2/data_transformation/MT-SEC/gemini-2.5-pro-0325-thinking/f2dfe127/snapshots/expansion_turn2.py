def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if the given byte stream contains valid UTF-8 encoded data.

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

def detect_format(byte_stream: bytes) -> str:
    """
    Detects the format of a byte stream (JSON, XML, HTML).

    Args:
        byte_stream: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "XML", "HTML").

    Raises:
        ValueError: If the format is unrecognized, potentially unsafe,
                    or if the stream cannot be decoded as UTF-8.
    """
    if not validate_byte_stream(byte_stream):
        raise ValueError("Stream is not valid UTF-8, potentially unsafe or unrecognized binary format.")

    try:
        # Decode assuming UTF-8, as these formats are typically text-based.
        # A small initial chunk is usually enough for detection.
        # Limiting read size also prevents loading huge files entirely for detection.
        sample_size = min(len(byte_stream), 1024)
        text_sample = byte_stream[:sample_size].decode('utf-8').strip()
    except UnicodeDecodeError:
        # This case should ideally be caught by validate_byte_stream,
        # but as a safeguard for partial reads or other edge cases.
        raise ValueError("Failed to decode stream as UTF-8, potentially unsafe or unrecognized format.")

    if not text_sample:
        raise ValueError("Empty stream, format cannot be determined.")

    # JSON checks
    # Simple check: starts with { or [ and ends with } or ]
    # A more robust check would involve trying to parse with json.loads,
    # but for a simple marker-based detection, this is a common first step.
    if (text_sample.startswith('{') and text_sample.endswith('}')) or \
       (text_sample.startswith('[') and text_sample.endswith(']')):
        # Further validation could involve trying json.loads()
        # For now, this heuristic is used.
        return "JSON"

    # XML checks
    # Starts with <?xml or a root tag <...>
    # Case-insensitive check for <?xml ... ?>
    if text_sample.lower().startswith('<?xml') or \
       (text_sample.startswith('<') and not text_sample.startswith('<!DOCTYPE') and text_sample.endswith('>')):
        # This is a basic check. Real XML parsing is complex.
        # We assume if it looks like an XML declaration or a simple tag structure, it's XML.
        return "XML"

    # HTML checks
    # Starts with <!DOCTYPE html> or <html>
    # Case-insensitive check for doctype and html tag
    if text_sample.lower().startswith('<!doctype html>') or \
       text_sample.lower().startswith('<html>'):
        return "HTML"

    raise ValueError("Unrecognized or potentially unsafe format")


if __name__ == '__main__':
    # Example Usage for validate_byte_stream
    valid_stream = "Hello, world!".encode('utf-8')
    invalid_stream = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"Valid stream is UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"Invalid stream is UTF-8: {validate_byte_stream(invalid_stream)}")

    # Test with an empty byte string
    empty_stream = b""
    print(f"Empty stream is UTF-8: {validate_byte_stream(empty_stream)}")

    # Test with a more complex valid UTF-8 string
    complex_valid_stream = "你好，世界！".encode('utf-8')
    print(f"Complex valid stream is UTF-8: {validate_byte_stream(complex_valid_stream)}")

    # Test with a byte string that is valid ISO-8859-1 but not UTF-8
    iso_stream = "café".encode('iso-8859-1') # 'café' in ISO-8859-1 is b'caf\xe9'
    # b'\xe9' is not a valid start of a UTF-8 sequence on its own.
    print(f"ISO stream (café) is UTF-8: {validate_byte_stream(iso_stream)}")

    print("\n--- Testing detect_format ---")

    # Example Usage for detect_format
    json_stream = b'{"name": "John Doe", "age": 30}'
    xml_stream = b'<?xml version="1.0"?><root><item>Value</item></root>'
    html_stream = b'<!DOCTYPE html><html><head><title>Test</title></head><body><h1>Hello</h1></body></html>'
    simple_html_stream = b'<html><body><p>Hi</p></body></html>'
    simple_xml_stream = b'<note><to>Tove</to></note>'
    text_stream = b'This is a plain text stream.'
    empty_byte_array_stream = b''
    invalid_utf8_for_detect = b'\xff\xfe\xfd' # Already tested with validate_byte_stream

    streams_to_test = {
        "JSON": json_stream,
        "XML": xml_stream,
        "HTML": html_stream,
        "Simple HTML": simple_html_stream,
        "Simple XML": simple_xml_stream,
    }

    for name, stream_data in streams_to_test.items():
        try:
            print(f"Format of '{name}' stream: {detect_format(stream_data)}")
        except ValueError as e:
            print(f"Error detecting format for '{name}': {e}")

    print("\n--- Testing detect_format with potentially problematic streams ---")
    problematic_streams = {
        "Plain Text": text_stream,
        "Empty Stream": empty_byte_array_stream,
        "Invalid UTF-8": invalid_utf8_for_detect,
    }

    for name, stream_data in problematic_streams.items():
        try:
            print(f"Format of '{name}' stream: {detect_format(stream_data)}")
        except ValueError as e:
            print(f"Error detecting format for '{name}': {e}")
