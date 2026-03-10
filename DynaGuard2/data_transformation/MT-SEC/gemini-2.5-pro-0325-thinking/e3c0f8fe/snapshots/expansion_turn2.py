import json
import xml.etree.ElementTree as ET

def validate_byte_stream(data: bytes) -> bool:
    """
    Checks if the given byte stream contains valid UTF-8 encoded data.

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

def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of a byte stream (JSON, XML, CSV).

    Args:
        data: The byte stream to inspect.

    Returns:
        A string indicating the format ("JSON", "XML", "CSV").

    Raises:
        ValueError: If the format is unrecognized, potentially harmful,
                    or if the input data is empty, not valid UTF-8, 
                    or becomes empty after decoding and stripping.
    """
    if not data: # Handle empty input
        raise ValueError("Input data is empty.")

    try:
        # Decode and strip whitespace. This helps normalize the input for checks.
        text_content = data.decode('utf-8').strip()
        if not text_content: # If data was only whitespace (or BOM + whitespace)
             raise ValueError("Input data becomes empty after decoding and stripping whitespace.")
    except UnicodeDecodeError:
        # If it's not valid UTF-8, we can't process it as the text formats we're looking for.
        raise ValueError("Data is not valid UTF-8, cannot determine text-based format.")

    # 1. Try JSON
    # JSON must start with '{' or '[' and end with '}' or ']' respectively after stripping.
    if (text_content.startswith('{') and text_content.endswith('}')) or \
       (text_content.startswith('[') and text_content.endswith(']')):
        try:
            json.loads(text_content)
            return "JSON"
        except json.JSONDecodeError:
            pass  # Not valid JSON, proceed to next check

    # 2. Try XML
    # XML typically starts with '<' (e.g., '<?xml ...>' or '<root_element>')
    # and a valid document should be parsable.
    if text_content.startswith('<'):
        try:
            ET.fromstring(text_content) # This expects a string
            return "XML"
        except ET.ParseError:
            pass  # Not valid XML, proceed to next check

    # 3. Try CSV (heuristic)
    # This check comes after JSON and XML.
    # If it contains a comma, we'll consider it CSV for this heuristic.
    # This assumes that if it were well-formed JSON or XML, it would have been caught.
    if ',' in text_content:
        return "CSV"

    raise ValueError("Unrecognized or potentially harmful serialization format")

if __name__ == '__main__':
    # Example Usage
    valid_utf8_bytes = "Hello, 世界!".encode('utf-8')
    invalid_utf8_bytes = b'\x80\x81\x82' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_bytes}' is valid UTF-8: {validate_byte_stream(valid_utf8_bytes)}")
    print(f"'{invalid_utf8_bytes}' is valid UTF-8: {validate_byte_stream(invalid_utf8_bytes)}")

    # More examples
    empty_bytes = b""
    print(f"Empty bytes '' is valid UTF-8: {validate_byte_stream(empty_bytes)}")

    ascii_bytes = b"This is ASCII"
    print(f"'{ascii_bytes}' is valid UTF-8: {validate_byte_stream(ascii_bytes)}")

    # A common invalid sequence (an overlong 2-byte sequence for "/")
    overlong_slash = b'\xc0\xaf'
    print(f"'{overlong_slash}' (overlong slash) is valid UTF-8: {validate_byte_stream(overlong_slash)}")

    # A sequence that is too short
    incomplete_sequence = b'\xe4\xbd' # Missing the third byte for a 3-byte character
    print(f"'{incomplete_sequence}' (incomplete sequence) is valid UTF-8: {validate_byte_stream(incomplete_sequence)}")

    # A sequence with an invalid continuation byte
    invalid_continuation = b'\xe4\xbd\x41' # 'A' (0x41) is not a valid continuation byte
    print(f"'{invalid_continuation}' (invalid continuation) is valid UTF-8: {validate_byte_stream(invalid_continuation)}")

    # Examples for detect_serialization_format
    print("\n--- Testing detect_serialization_format ---")
    json_data = b'{ "name": "example", "value": 123 }'
    xml_data = b'<root><item>Example</item></root>'
    csv_data = b'header1,header2\nvalue1,value2\nvalue3,value4'
    csv_single_line_data = b'foo,bar,baz'
    plain_text_data = b'This is just some plain text.'
    empty_data = b''
    whitespace_data = b'   '
    utf8_bom_whitespace_data = b'\xef\xbb\xbf   '
    invalid_utf8_for_format = b'\xff\xfe\xfd' # Invalid UTF-8

    test_cases = {
        "JSON": json_data,
        "XML": xml_data,
        "CSV (multi-line)": csv_data,
        "CSV (single-line)": csv_single_line_data,
        "Plain Text (should fail)": plain_text_data,
        "Empty Data (should fail)": empty_data,
        "Whitespace Data (should fail)": whitespace_data,
        "UTF-8 BOM then Whitespace (should fail)": utf8_bom_whitespace_data,
        "Invalid UTF-8 (should fail)": invalid_utf8_for_format,
        "JSON with leading/trailing whitespace": b'  { "data": true }  ',
        "XML with leading/trailing whitespace": b'\n\n<data>text</data>\t',
        "CSV with leading/trailing whitespace": b' \nname,age\nAlice,30 \n ',
        "UTF-8 BOM then JSON": b'\xef\xbb\xbf{ "bom": "json" }',
        "Text that looks like start of XML but isn't valid": b'<not really xml',
        "Text that looks like start of JSON but isn't valid": b'{ "unterminated": "json" ',
        "Simple string with comma (CSV)": b"alpha,beta,gamma",
        "Simple string no comma (Unrecognized)": b"alpha beta gamma",
        "XML declaration": b"<?xml version=\"1.0\"?><tag>text</tag>",
        "JSON array": b"[1, 2, 3]",
    }

    for desc, d_bytes in test_cases.items():
        try:
            format_name = detect_serialization_format(d_bytes)
            print(f"Data ('{desc}') detected as: {format_name}")
        except ValueError as e:
            print(f"Data ('{desc}') error: {e}")
