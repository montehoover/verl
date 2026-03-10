import json
import xml.etree.ElementTree as ET

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

secure_formats = ["json", "xml"]

def load_serialized_data(bytes_data: bytes, data_format: str) -> any:
    """
    Safely converts serialized data from bytes into its Python object.

    Args:
        bytes_data: The raw byte data to deserialize.
        data_format: The format of the serialization (e.g., "json", "xml").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported, unsafe, or if deserialization fails.
    """
    format_lower = data_format.lower()

    if format_lower not in secure_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format}. Allowed formats: {secure_formats}")

    if not validate_byte_stream(bytes_data) and format_lower in ["json", "xml"]: # XML can sometimes be non-UTF-8 but parsers might handle it. JSON must be UTF-8.
        # For JSON, UTF-8 is a strict requirement for standard parsing.
        # For XML, while ET.fromstring can handle some non-UTF-8, it's safer to expect valid UTF-8 for consistency.
        # Or, one might choose to let the specific parser handle encoding issues if the format allows.
        # Here, we enforce UTF-8 for simplicity and security with text-based formats.
        if format_lower == "json":
            raise ValueError(f"Invalid UTF-8 data for JSON format.")
        # For XML, ET.fromstring might handle some encodings declared in the XML itself.
        # However, if validate_byte_stream fails, it means it's not UTF-8.
        # We can choose to proceed with caution or raise an error.
        # Let's be strict for now.
        # print(f"Warning: Data for {data_format} is not valid UTF-8, attempting to parse anyway.")


    if format_lower == "json":
        try:
            # JSON standard requires UTF-8, UTF-16, or UTF-32. bytes.decode('utf-8') is common.
            return json.loads(bytes_data.decode('utf-8'))
        except UnicodeDecodeError:
            raise ValueError("Data is not valid UTF-8, cannot parse as JSON.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON data: {e}")
    elif format_lower == "xml":
        try:
            # xml.etree.ElementTree.fromstring can parse bytes directly.
            # It attempts to determine encoding from the XML declaration if present.
            return ET.fromstring(bytes_data)
        except ET.ParseError as e:
            raise ValueError(f"Failed to deserialize XML data: {e}")
        except Exception as e: # Catch any other unexpected errors during XML parsing
            raise ValueError(f"An unexpected error occurred during XML deserialization: {e}")

    # This part should ideally not be reached if secure_formats check is exhaustive
    # and all supported formats have a deserialization path.
    raise ValueError(f"Deserialization logic not implemented for format: {data_format}")


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

    print("\n--- Testing load_serialized_data ---")

    # Test cases for load_serialized_data
    valid_json_bytes = b'{ "message": "Hello", "count": 42 }'
    valid_xml_bytes = b'<data><message>Hello</message><count>42</count></data>'
    malformed_json_bytes = b'{ "message": "Hello", "count": 42 ' # Missing closing brace
    malformed_xml_bytes = b'<data><message>Hello</message><count>42</data' # Missing closing tag for data
    non_utf8_json_bytes = b'{\xff"message": "Hello"}'
    unsupported_format_bytes = b'some random data'

    deserialization_tests = [
        ("Valid JSON", valid_json_bytes, "json", {"message": "Hello", "count": 42}),
        ("Valid XML", valid_xml_bytes, "xml", "Element"), # ET.Element type check
        ("Malformed JSON", malformed_json_bytes, "json", ValueError),
        ("Malformed XML", malformed_xml_bytes, "xml", ValueError),
        ("Non-UTF-8 JSON", non_utf8_json_bytes, "json", ValueError),
        ("Unsupported Format (text)", unsupported_format_bytes, "text", ValueError),
        ("Unsupported Format (PDF)", pdf_data, "pdf", ValueError), # Using pdf_data from previous tests
        ("Empty JSON data", b"", "json", ValueError), # json.loads("") raises error
        ("Empty XML data", b"", "xml", ValueError), # ET.fromstring(b"") raises error
        ("Valid JSON with uppercase format", valid_json_bytes, "JSON", {"message": "Hello", "count": 42}),
    ]

    for name, data_bytes, data_fmt, expected_outcome in deserialization_tests:
        print(f"Testing '{name}' with format '{data_fmt}':")
        try:
            result = load_serialized_data(data_bytes, data_fmt)
            if expected_outcome == ValueError:
                print(f"  ERROR: Expected ValueError, but got result: {result}")
            elif data_fmt.lower() == "xml" and expected_outcome == "Element":
                # For XML, we check if the result is an ElementTree Element
                if ET.iselement(result):
                    print(f"  SUCCESS: Deserialized XML object of type {type(result)}")
                else:
                    print(f"  ERROR: Expected ElementTree.Element, got {type(result)}")
            elif result == expected_outcome:
                print(f"  SUCCESS: Deserialized object: {result}")
            else:
                print(f"  ERROR: Expected {expected_outcome}, but got {result}")
        except ValueError as e:
            if expected_outcome == ValueError:
                print(f"  SUCCESS: Raised ValueError as expected: {e}")
            else:
                print(f"  ERROR: Expected {expected_outcome}, but got ValueError: {e}")
        except Exception as e:
            print(f"  ERROR: Raised an unexpected exception: {e}")
