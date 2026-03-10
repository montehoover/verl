import json
# For HTML parsing, if a more complex object than a string is needed,
# a library like BeautifulSoup would be imported here.
# from bs4 import BeautifulSoup

allowed_formats = ["json", "html"]

def validate_byte_content(byte_stream: bytes) -> bool:
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

def detect_format(byte_stream: bytes) -> str:
    """
    Detects the format of a byte stream (JSON, XML, HTML).

    Args:
        byte_stream: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "XML", "HTML").

    Raises:
        ValueError: If the format is unrecognized or the stream is too short
                    to determine the format.
    """
    if not validate_byte_content(byte_stream):
        raise ValueError("Invalid UTF-8 content")

    # Decode a small prefix for inspection, be lenient with decoding for detection
    # as the primary goal here is to find markers.
    # The overall UTF-8 validity is checked above.
    prefix_len = 100  # Check the first 100 bytes
    try:
        # Attempt to decode a small prefix. If this fails, it's unlikely to be
        # one of the text-based formats we're looking for in a recognizable way.
        prefix_str = byte_stream[:prefix_len].decode('utf-8', errors='ignore').lstrip()
    except Exception: # Catch potential errors if byte_stream is not sliceable or too short in a weird way
        raise ValueError("Could not decode prefix to determine format.")


    if not prefix_str:
        raise ValueError("Empty or whitespace-only content, format unrecognized.")

    # Check for JSON
    if prefix_str.startswith('{') or prefix_str.startswith('['):
        return "JSON"

    # Check for XML
    # A more robust XML check would involve trying to parse with an XML parser,
    # but for a simple marker check:
    if prefix_str.startswith('<?xml'):
        return "XML"
    if prefix_str.startswith('<') and '>' in prefix_str and not prefix_str.lower().startswith('<!doctype html') and not prefix_str.lower().startswith('<html'):
        # Basic check for a root tag that isn't HTML
        # This is a simplification; real XML detection can be complex.
        # Consider if the first non-whitespace char is '<' and it's not clearly HTML
        if not prefix_str.lower().startswith('<html') and not prefix_str.lower().startswith('<!doctype html'):
             # Check if it looks like a generic tag <tag> or <ns:tag>
            import re
            if re.match(r'<([a-zA-Z0-9_:]+)([^>]*)>', prefix_str):
                return "XML"


    # Check for HTML
    # HTML can start with <!DOCTYPE html> or <html>
    # Making the check case-insensitive for these common HTML markers
    prefix_lower = prefix_str.lower()
    if prefix_lower.startswith('<!doctype html') or prefix_lower.startswith('<html>'):
        return "HTML"
    # Sometimes HTML fragments might just start with a common tag
    if prefix_lower.startswith('<div') or prefix_lower.startswith('<p') or prefix_lower.startswith('<body') or prefix_lower.startswith('<head'):
        return "HTML"


    raise ValueError("Unrecognized data format.")

def parse_serialized_content(content: bytes, format_type: str) -> any:
    """
    Securely deserializes a byte stream into a Python object based on the format.

    Args:
        content: The byte stream to deserialize.
        format_type: The format of the serialized data (e.g., "json", "html").

    Returns:
        The deserialized Python object. For JSON, this is a dict or list.
        For HTML, this is the decoded string.

    Raises:
        ValueError: If the format is unsupported, insecure, or content is invalid.
    """
    normalized_format = format_type.lower()

    if normalized_format not in allowed_formats:
        raise ValueError(f"Unsupported or insecure format: {format_type}")

    if not validate_byte_content(content):
        raise ValueError("Invalid UTF-8 content, cannot deserialize.")

    try:
        if normalized_format == "json":
            return json.loads(content.decode('utf-8'))
        elif normalized_format == "html":
            # For HTML, "deserializing" here means returning the decoded string.
            # For more complex parsing (e.g., into a DOM), a library like
            # BeautifulSoup would be used, e.g., BeautifulSoup(content, 'html.parser')
            return content.decode('utf-8')
        # This part should ideally not be reached if allowed_formats is checked correctly
        else:
            raise ValueError(f"Format '{format_type}' was in allowed_formats but not handled.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")
    except UnicodeDecodeError as e:
        # This might occur if validate_byte_content passed but a specific part of decode fails
        # for the parser, though less likely if validate_byte_content is robust.
        raise ValueError(f"Unicode decoding error during parsing: {e}")
    except Exception as e:
        # Catch-all for other unexpected parsing errors
        raise ValueError(f"Error parsing {format_type} content: {e}")

if __name__ == '__main__':
    # Example Usage for validate_byte_content
    valid_utf8_bytes = "Hello, 世界".encode('utf-8')
    invalid_utf8_bytes = b'\x80\x81\x82' # Invalid UTF-8 sequence

    print(f"Validating '{valid_utf8_bytes.decode('utf-8', errors='ignore')}': {validate_byte_content(valid_utf8_bytes)}")
    print(f"Validating invalid bytes: {validate_byte_content(invalid_utf8_bytes)}")

    # More examples
    empty_bytes = b""
    print(f"Validating empty bytes: {validate_byte_content(empty_bytes)}")

    ascii_bytes = b"This is ASCII"
    print(f"Validating ASCII bytes: {validate_byte_content(ascii_bytes)}")

    # A longer valid UTF-8 string with various characters
    complex_utf8_string = "你好, мир, こんにちは, €"
    complex_utf8_bytes = complex_utf8_string.encode('utf-8')
    print(f"Validating '{complex_utf8_string}': {validate_byte_content(complex_utf8_bytes)}")

    # An invalid byte in the middle of a valid sequence
    mixed_invalid_bytes = b"Hello \xff World"
    print(f"Validating mixed invalid bytes: {validate_byte_content(mixed_invalid_bytes)}")

    print("\n--- Format Detection Examples ---")

    # Example Usage for detect_format
    json_data = b'{ "name": "example", "value": 123 }'
    xml_data = b'<?xml version="1.0"?><root><item>Example</item></root>'
    html_data = b'<!DOCTYPE html><html><head><title>Test</title></head><body><p>Hello</p></body></html>'
    html_data_simple = b'<html><body><h1>Simple HTML</h1></body></html>'
    json_array_data = b'[1, 2, 3]'
    xml_simple_tag = b'<note><to>Tove</to></note>'
    text_data = b'This is plain text.'
    empty_data = b''
    whitespace_data = b'   '
    invalid_utf8_for_format = b'\x80\x81' # Invalid UTF-8

    print(f"Detecting format for JSON: {detect_format(json_data)}")
    print(f"Detecting format for JSON Array: {detect_format(json_array_data)}")
    print(f"Detecting format for XML: {detect_format(xml_data)}")
    print(f"Detecting format for simple XML: {detect_format(xml_simple_tag)}")
    print(f"Detecting format for HTML (doctype): {detect_format(html_data)}")
    print(f"Detecting format for HTML (simple): {detect_format(html_data_simple)}")

    try:
        print(f"Detecting format for plain text: {detect_format(text_data)}")
    except ValueError as e:
        print(f"Error detecting plain text: {e}")

    try:
        print(f"Detecting format for empty data: {detect_format(empty_data)}")
    except ValueError as e:
        print(f"Error detecting empty data: {e}")

    try:
        print(f"Detecting format for whitespace data: {detect_format(whitespace_data)}")
    except ValueError as e:
        print(f"Error detecting whitespace data: {e}")
    
    try:
        print(f"Detecting format for invalid UTF-8 (for format detection): {detect_format(invalid_utf8_for_format)}")
    except ValueError as e:
        print(f"Error detecting format for invalid UTF-8: {e}")

    # Example of HTML fragment
    html_fragment = b'<div><p>A fragment</p></div>'
    print(f"Detecting format for HTML fragment: {detect_format(html_fragment)}")

    # Example that might be ambiguous or harder for simple XML check
    custom_xml_like_not_really = b'<customtag>data</customtag>'
    print(f"Detecting format for custom XML-like: {detect_format(custom_xml_like_not_really)}")

    # Example of invalid UTF-8 that validate_byte_content would catch first
    truly_invalid_stream = b"Valid start \xff then invalid"
    try:
        print(f"Detecting format for truly_invalid_stream: {detect_format(truly_invalid_stream)}")
    except ValueError as e:
        print(f"Error detecting format for truly_invalid_stream: {e}")

    print("\n--- Content Parsing Examples ---")

    # Example Usage for parse_serialized_content
    valid_json_bytes = b'{ "key": "value", "number": 123 }'
    valid_html_bytes = b"<p>This is <strong>HTML</strong> content.</p>"
    malformed_json_bytes = b'{ "key": "value", number: 123 }' # Invalid JSON syntax
    unsupported_format_bytes = b"some random data"
    invalid_utf8_for_parse = b"text with \xff invalid utf8"

    # Test JSON parsing
    try:
        parsed_json = parse_serialized_content(valid_json_bytes, "json")
        print(f"Parsed JSON: {parsed_json}, type: {type(parsed_json)}")
    except ValueError as e:
        print(f"Error parsing JSON: {e}")

    # Test HTML parsing (returns string)
    try:
        parsed_html = parse_serialized_content(valid_html_bytes, "html")
        print(f"Parsed HTML (string): {parsed_html}, type: {type(parsed_html)}")
    except ValueError as e:
        print(f"Error parsing HTML: {e}")

    # Test malformed JSON
    try:
        parse_serialized_content(malformed_json_bytes, "json")
    except ValueError as e:
        print(f"Error parsing malformed JSON: {e}")

    # Test unsupported format
    try:
        parse_serialized_content(unsupported_format_bytes, "xml") # XML not in allowed_formats
    except ValueError as e:
        print(f"Error parsing unsupported format (XML): {e}")

    # Test with a format that is in allowed_formats but not explicitly handled (should not happen with current code)
    # To test this, one might temporarily add "xml" to allowed_formats without adding parsing logic.

    # Test invalid UTF-8 content for parsing
    try:
        parse_serialized_content(invalid_utf8_for_parse, "json")
    except ValueError as e:
        print(f"Error parsing invalid UTF-8 as JSON: {e}")

    # Test parsing with format type in uppercase
    try:
        parsed_json_upper = parse_serialized_content(valid_json_bytes, "JSON")
        print(f"Parsed JSON (uppercase format): {parsed_json_upper}")
    except ValueError as e:
        print(f"Error parsing JSON (uppercase format): {e}")
