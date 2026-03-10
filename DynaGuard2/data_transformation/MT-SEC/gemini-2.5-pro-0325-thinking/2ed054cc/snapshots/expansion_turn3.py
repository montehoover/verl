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

import json
import xml.etree.ElementTree as ET

def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of a byte stream (JSON or XML).

    Args:
        data: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "XML").

    Raises:
        ValueError: If the input is empty, not valid UTF-8,
                    or the format is unrecognized/unsupported or potentially unsafe.
    """
    if not data:
        raise ValueError("Input data cannot be empty.")

    stripped_data = data.strip()
    if not stripped_data:
        raise ValueError("Input data cannot be empty after stripping whitespace.")

    try:
        # Attempt to decode as UTF-8, as common text-based formats use it.
        str_data = stripped_data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8, cannot determine format.")

    # Check for JSON
    # A byte string is JSON if it's valid UTF-8, and after stripping whitespace,
    # it starts with { and ends with } or starts with [ and ends with ].
    is_potential_json = (str_data.startswith('{') and str_data.endswith('}')) or \
                        (str_data.startswith('[') and str_data.endswith(']'))

    if is_potential_json:
        try:
            json.loads(str_data)
            return "JSON"
        except json.JSONDecodeError:
            # It has JSON-like markers but is not valid JSON.
            # Fall through to unrecognized, or raise a specific error if desired.
            pass # Or raise ValueError("Data has JSON markers but is not valid JSON.")

    # Check for XML
    # A byte string is XML if it's valid UTF-8, and after stripping whitespace,
    # it starts with < and ends with >.
    # A more robust check involves trying to parse it.
    if str_data.startswith('<') and str_data.endswith('>'):
        try:
            ET.fromstring(str_data)
            return "XML"
        except ET.ParseError:
            # It has XML-like markers but is not valid XML.
            # Fall through to unrecognized, or raise a specific error if desired.
            pass # Or raise ValueError("Data has XML markers but is not valid XML.")

    raise ValueError("Unrecognized or unsupported serialization format.")

allowed_serialization_types = ["json", "xml"]

def deserialize_data(raw_data: bytes, serialization_format: str) -> any:
    """
    Safely converts serialized data from raw bytes into a Python object.

    Args:
        raw_data: The byte stream to deserialize.
        serialization_format: The format of the serialization ('json' or 'xml').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported, data is not valid for the format,
                    or data is not valid UTF-8 for text-based formats.
    """
    if not raw_data:
        raise ValueError("Input raw_data cannot be empty.")

    format_lower = serialization_format.lower()

    if format_lower not in allowed_serialization_types:
        raise ValueError(f"Unsupported serialization format: {serialization_format}. Allowed formats are: {allowed_serialization_types}")

    try:
        # Common step for text-based formats: decode from UTF-8
        str_data = raw_data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Data is not valid UTF-8, cannot deserialize for format {format_lower}: {e}")

    if format_lower == "json":
        try:
            return json.loads(str_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    elif format_lower == "xml":
        try:
            # ET.fromstring returns an Element object.
            # For more complex XML, you might want a different parsing strategy
            # or return the ElementTree object itself.
            return ET.fromstring(str_data)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML data: {e}")
    else:
        # This case should ideally be caught by the check against allowed_serialization_types
        # but is included for robustness.
        raise ValueError(f"Internal error: Unhandled allowed format: {format_lower}")


if __name__ == '__main__':
    # Example Usage for validate_byte_stream
    valid_stream = b"Hello, world! This is a valid UTF-8 string."
    invalid_stream = b"\xff\xfe\x00\x00H\x00e\x00l\x00l\x00o\x00" # UTF-16LE BOM, not valid UTF-8

    print(f"'{valid_stream}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"'{invalid_stream}' is valid UTF-8: {validate_byte_stream(invalid_stream)}")

    another_valid_stream = "你好世界".encode('utf-8')
    print(f"'{another_valid_stream}' is valid UTF-8: {validate_byte_stream(another_valid_stream)}")

    corrupted_stream = b"This is partially valid \xe2\x82 but then corrupted."
    print(f"'{corrupted_stream}' is valid UTF-8: {validate_byte_stream(corrupted_stream)}")

    print("\n--- Testing detect_serialization_format ---")

    json_data_obj = b'{ "name": "John Doe", "age": 30 }'
    json_data_arr = b'[1, 2, "test"]'
    xml_data = b'<?xml version="1.0"?><root><element attribute="value">Text</element></root>'
    xml_data_simple = b'<note><to>Tove</to></note>'
    plain_text_data = b"This is just plain text."
    invalid_utf8_data = b"\xff\xfe\x00" # Invalid UTF-8 start
    empty_data = b""
    whitespace_data = b"   "
    corrupted_json = b'{ "name": "John Doe", "age": 30 ' # Missing closing brace
    corrupted_xml = b'<root><element>Text</element>' # Missing closing root tag properly

    test_cases = [
        ("Valid JSON Object", json_data_obj, "JSON"),
        ("Valid JSON Array", json_data_arr, "JSON"),
        ("Valid XML with declaration", xml_data, "XML"),
        ("Valid XML simple", xml_data_simple, "XML"),
        ("Plain Text", plain_text_data, "ValueError"),
        ("Invalid UTF-8", invalid_utf8_data, "ValueError"),
        ("Empty Data", empty_data, "ValueError"),
        ("Whitespace Data", whitespace_data, "ValueError"),
        ("Corrupted JSON", corrupted_json, "ValueError"), # Expecting ValueError as it's not valid
        ("Corrupted XML", corrupted_xml, "ValueError"),   # Expecting ValueError as it's not valid
    ]

    for desc, data, expected in test_cases:
        try:
            result = detect_serialization_format(data)
            if result == expected:
                print(f"Test Passed: {desc} -> {result}")
            else:
                print(f"Test FAILED: {desc} -> Expected {expected}, Got {result}")
        except ValueError as e:
            if expected == "ValueError":
                print(f"Test Passed (ValueError expected): {desc} -> {e}")
            else:
                print(f"Test FAILED: {desc} -> Expected {expected}, Got ValueError: {e}")
        except Exception as e:
            print(f"Test FAILED (Unexpected Exception): {desc} -> {e}")

    print("\n--- Testing deserialize_data ---")

    valid_json_bytes = b'{ "name": "Alice", "id": 123 }'
    valid_xml_bytes = b'<user><name>Bob</name><id>456</id></user>'
    malformed_json_bytes = b'{ "name": "Alice", "id": 123 ' # Missing closing brace
    malformed_xml_bytes = b'<user><name>Bob</name><id>456</id>' # Missing closing user tag
    non_utf8_bytes = b'\x80\x81\x82' # Invalid UTF-8 sequence

    deserialize_test_cases = [
        ("Valid JSON", valid_json_bytes, "json", {"name": "Alice", "id": 123}),
        ("Valid XML", valid_xml_bytes, "xml", "Element"), # ET.fromstring returns an Element
        ("Malformed JSON", malformed_json_bytes, "json", "ValueError"),
        ("Malformed XML", malformed_xml_bytes, "xml", "ValueError"),
        ("Unsupported Format", valid_json_bytes, "yaml", "ValueError"),
        ("Non-UTF8 Data for JSON", non_utf8_bytes, "json", "ValueError"),
        ("Non-UTF8 Data for XML", non_utf8_bytes, "xml", "ValueError"),
        ("Empty Data", b"", "json", "ValueError"),
    ]

    for desc, data, fmt, expected_outcome in deserialize_test_cases:
        try:
            result = deserialize_data(data, fmt)
            if expected_outcome == "ValueError":
                print(f"Test FAILED: {desc} -> Expected ValueError, Got {type(result)}: {result}")
            elif fmt == "xml" and expected_outcome == "Element":
                # For XML, we check if the result is an ElementTree Element
                if isinstance(result, ET.Element):
                    print(f"Test Passed: {desc} -> Result is an ET.Element (tag: {result.tag})")
                else:
                    print(f"Test FAILED: {desc} -> Expected ET.Element, Got {type(result)}")
            elif result == expected_outcome:
                print(f"Test Passed: {desc} -> {result}")
            else:
                print(f"Test FAILED: {desc} -> Expected {expected_outcome}, Got {result}")
        except ValueError as e:
            if expected_outcome == "ValueError":
                print(f"Test Passed (ValueError expected): {desc} -> {e}")
            else:
                print(f"Test FAILED: {desc} -> Expected {expected_outcome}, Got ValueError: {e}")
        except Exception as e:
            print(f"Test FAILED (Unexpected Exception): {desc} -> {e}")
