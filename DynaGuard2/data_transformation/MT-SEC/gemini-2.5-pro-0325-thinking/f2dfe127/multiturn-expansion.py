import json
import xml.etree.ElementTree as ET

approved_formats = ["json", "xml"]

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

def bytes_to_obj(data_bytes: bytes, format_name: str) -> any:
    """
    Securely deserializes a byte stream into a Python object.

    Args:
        data_bytes: The byte stream to deserialize.
        format_name: The serialization format name (e.g., "json", "xml").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format is unsupported, dangerous, or if
                    deserialization fails.
    """
    normalized_format_name = format_name.lower()

    if normalized_format_name not in approved_formats:
        raise ValueError(
            f"Unsupported format: '{format_name}'. Approved formats are: {', '.join(approved_formats)}"
        )

    if not validate_byte_stream(data_bytes):
        # Although format_name is given, we should ensure data is valid UTF-8 for text-based formats.
        raise ValueError("Data stream is not valid UTF-8.")

    try:
        text_data = data_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode data as UTF-8: {e}") from e

    if normalized_format_name == "json":
        try:
            return json.loads(text_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize JSON: {e}") from e
    elif normalized_format_name == "xml":
        try:
            # ET.fromstring is generally safer against XML bombs like Billion Laughs
            # by default as it doesn't expand entities from external sources unless configured.
            return ET.fromstring(text_data)
        except ET.ParseError as e:
            raise ValueError(f"Failed to deserialize XML: {e}") from e
    else:
        # This case should ideally be caught by the approved_formats check,
        # but as a safeguard.
        raise ValueError(f"Deserialization logic not implemented for format: {format_name}")


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

    print("\n--- Testing bytes_to_obj ---")

    # Test data for bytes_to_obj
    valid_json_bytes = b'{"key": "value", "number": 123}'
    valid_xml_bytes = b'<data><item name="item1">value1</item><item name="item2">value2</item></data>'
    malformed_json_bytes = b'{"key": "value", "number": 123,' # Missing closing brace
    malformed_xml_bytes = b'<data><item name="item1">value1</item><item name="item2">value2</data' # Missing closing root tag
    unsupported_format_bytes = b'some data'
    html_bytes_for_deserialize = b'<html><body><h1>Hi</h1></body></html>' # HTML is not in approved_formats

    test_cases_deserialize = [
        ("Valid JSON", valid_json_bytes, "json"),
        ("Valid XML", valid_xml_bytes, "xml"),
        ("Malformed JSON", malformed_json_bytes, "json"),
        ("Malformed XML", malformed_xml_bytes, "xml"),
        ("Unsupported Format (text)", unsupported_format_bytes, "text"),
        ("Disallowed Format (HTML)", html_bytes_for_deserialize, "html"),
        ("Valid JSON (uppercase format)", valid_json_bytes, "JSON"),
        ("Invalid UTF-8 for deserialize", invalid_utf8_for_detect, "json"),
    ]

    for description, data, fmt in test_cases_deserialize:
        print(f"\nAttempting to deserialize: {description} (format: {fmt})")
        try:
            obj = bytes_to_obj(data, fmt)
            if fmt.lower() == "xml": # ElementTree objects don't have a pretty print by default
                print(f"Successfully deserialized. Root tag: {obj.tag if hasattr(obj, 'tag') else 'N/A'}")
            else:
                print(f"Successfully deserialized: {obj}")
        except ValueError as e:
            print(f"Error during deserialization: {e}")

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
