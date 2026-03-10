def validate_byte_stream(data: bytes) -> bool:
    """
    Validates if the given byte stream contains only valid UTF-8 characters.

    Args:
        data: The byte stream to validate.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def detect_data_format(data: bytes) -> str:
    """
    Detects the data format of a byte stream based on common signatures.

    Args:
        data: The byte stream to inspect.

    Returns:
        A string indicating the detected format (e.g., "JSON", "XML").

    Raises:
        ValueError: If the format is unrecognized or the data is empty.
    """
    if not data:
        raise ValueError("Input data cannot be empty.")

    # Strip leading whitespace characters (space, tab, newline, carriage return, form feed, vertical tab)
    stripped_data = data.lstrip(b" \t\n\r\f\v")

    if stripped_data.startswith(b'{') or stripped_data.startswith(b'['):
        # Further check if it's valid JSON, if possible, or assume based on start
        # For simplicity, we'll assume it's JSON if it starts with { or [
        # and is valid UTF-8
        if validate_byte_stream(data):
            try:
                # A more robust check would be to try parsing it,
                # but that might be too slow or complex for a simple detection.
                # For now, starting character and UTF-8 validity is a good heuristic.
                import json
                json.loads(data.decode('utf-8')) # Check if it can be parsed as JSON
                return "JSON"
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it starts like JSON but isn't valid JSON or UTF-8, it's unrecognized
                pass # Fall through to ValueError
        else:
            # If not valid UTF-8, it can't be standard JSON
            pass # Fall through to ValueError


    if stripped_data.startswith(b'<?xml') or \
       (stripped_data.startswith(b'<') and not stripped_data.startswith(b"<<<")): # Avoid conflict with potential binary data
        # Similar to JSON, a more robust check would involve attempting to parse.
        # We check for UTF-8 validity as XML parsers usually expect it.
        if validate_byte_stream(data):
            try:
                # A light check for a closing tag might be useful but adds complexity.
                # For now, starting character and UTF-8 validity is a good heuristic.
                # Python's xml.etree.ElementTree can parse bytes directly
                import xml.etree.ElementTree as ET
                ET.fromstring(data)
                return "XML"
            except (ET.ParseError, UnicodeDecodeError): # Catching UnicodeDecodeError just in case fromstring doesn't handle all non-UTF-8 cases
                # If it starts like XML but isn't valid XML or UTF-8, it's unrecognized
                pass # Fall through to ValueError
        else:
            # If not valid UTF-8, it can't be standard XML
            pass # Fall through to ValueError

    # Add more format detections here, e.g., PDF, PNG, JPEG
    if stripped_data.startswith(b'%PDF'):
        return "PDF"
    if stripped_data.startswith(b'\x89PNG\r\n\x1a\n'):
        return "PNG"
    if stripped_data.startswith(b'\xFF\xD8\xFF'):
        return "JPEG"

    raise ValueError("Unrecognized or potentially unsafe data format.")

if __name__ == '__main__':
    # Example Usage for validate_byte_stream
    valid_stream = b"Hello, World!"
    invalid_stream = b"\xff\xfe\x00" # Invalid UTF-8 sequence

    print(f"'{valid_stream}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"'{invalid_stream}' is valid UTF-8: {validate_byte_stream(invalid_stream)}")

    # More examples
    valid_emoji_stream = "😊".encode('utf-8')
    print(f"'{valid_emoji_stream}' is valid UTF-8: {validate_byte_stream(valid_emoji_stream)}")

    # A byte sequence that is valid ISO-8859-1 but not UTF-8
    invalid_latin1_stream = b'\xe4\xf6\xfc' # äöü in ISO-8859-1
    print(f"'{invalid_latin1_stream}' (äöü in ISO-8859-1) is valid UTF-8: {validate_byte_stream(invalid_latin1_stream)}")

    valid_german_stream = "äöü".encode('utf-8')
    print(f"'{valid_german_stream}' (äöü in UTF-8) is valid UTF-8: {validate_byte_stream(valid_german_stream)}")

    empty_stream = b""
    print(f"Empty stream is valid UTF-8: {validate_byte_stream(empty_stream)}")

    print("\n--- Testing detect_data_format ---")

    # Test cases for detect_data_format
    json_data_obj = b'{ "name": "example", "value": 123 }'
    json_data_arr = b'[1, 2, 3]'
    json_data_with_whitespace = b'  { "key": "value" }  '
    xml_data_declaration = b'<?xml version="1.0"?><root><item>Hello</item></root>'
    xml_data_simple = b'<note><to>Tove</to><from>Jani</from></note>'
    xml_data_with_whitespace = b'\n\t<data>content</data>'
    text_data = b"This is plain text."
    pdf_data = b"%PDF-1.4\n% ...."
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...'
    jpeg_data = b'\xFF\xD8\xFF\xE0\x00\x10JFIF...'
    invalid_json_start_not_utf8 = b'{\xff"key": "value"}' # Starts like JSON but invalid UTF-8
    invalid_xml_start_not_utf8 = b'<\xffroot></root>' # Starts like XML but invalid UTF-8
    corrupted_json = b'{ "name": "example", value: 123 }' # Syntax error
    corrupted_xml = b'<root><item>Hello</item</root>' # Malformed XML

    test_streams = {
        "JSON Object": (json_data_obj, "JSON"),
        "JSON Array": (json_data_arr, "JSON"),
        "JSON with Whitespace": (json_data_with_whitespace, "JSON"),
        "XML with Declaration": (xml_data_declaration, "XML"),
        "XML Simple": (xml_data_simple, "XML"),
        "XML with Whitespace": (xml_data_with_whitespace, "XML"),
        "PDF": (pdf_data, "PDF"),
        "PNG": (png_data, "PNG"),
        "JPEG": (jpeg_data, "JPEG"),
        "Plain Text": (text_data, ValueError),
        "Empty Data": (b"", ValueError),
        "Invalid JSON Start (not UTF-8)": (invalid_json_start_not_utf8, ValueError),
        "Invalid XML Start (not UTF-8)": (invalid_xml_start_not_utf8, ValueError),
        "Corrupted JSON": (corrupted_json, ValueError),
        "Corrupted XML": (corrupted_xml, ValueError),
    }

    for name, (stream_data, expected) in test_streams.items():
        try:
            format_detected = detect_data_format(stream_data)
            print(f"'{name}': Detected as {format_detected}")
            if isinstance(expected, str) and format_detected != expected:
                print(f"  ERROR: Expected {expected}, but got {format_detected}")
            elif expected == ValueError:
                 print(f"  ERROR: Expected ValueError, but got {format_detected}")
        except ValueError as e:
            print(f"'{name}': Raised ValueError as expected: {e}")
            if expected != ValueError:
                print(f"  ERROR: Expected {expected}, but got ValueError")
        except Exception as e:
            print(f"'{name}': Raised an unexpected exception: {e}")
