import json
import xml.etree.ElementTree as ET
import configparser

def validate_binary_data(data: bytes) -> bool:
    """
    Checks if the given bytes input is valid UTF-8 encoded data.

    Args:
        data: The bytes data to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def detect_data_format(data: bytes) -> str:
    """
    Identifies the format of a bytes input (JSON, XML, INI).

    Args:
        data: The bytes data to inspect.

    Returns:
        A string indicating the format ("JSON", "XML", "INI").

    Raises:
        ValueError: If the data is not valid UTF-8, or if the format is
                    unrecognized, empty, or cannot be safely parsed.
    """
    try:
        # Attempt to decode as UTF-8, as JSON, XML, INI are text-based formats.
        text_data = data.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8, cannot determine format.")

    stripped_text_data = text_data.strip()
    if not stripped_text_data:
        # If data is empty or all whitespace, it's not a recognized format.
        raise ValueError("Data is empty or consists only of whitespace.")

    # Try JSON
    try:
        json.loads(stripped_text_data)
        # json.loads will fail on empty string or whitespace-only string after strip.
        return "JSON"
    except json.JSONDecodeError:
        pass  # Not JSON, try next format

    # Try XML
    try:
        # ET.fromstring will fail on an empty string.
        # We also check if it starts with '<' as an additional heuristic.
        if stripped_text_data.startswith('<'):
            ET.fromstring(stripped_text_data)
            return "XML"
    except ET.ParseError:
        pass  # Not XML, try next format

    # Try INI
    # configparser can "parse" an empty string or comments-only string without error,
    # but parser.sections() would be empty.
    parser = configparser.ConfigParser()
    try:
        parser.read_string(stripped_text_data)
        if parser.sections(): # Check if any sections were actually parsed.
            return "INI"
    except configparser.Error: # Catches MissingSectionHeaderError, DuplicateSectionError etc.
        pass  # Not INI, or malformed INI

    raise ValueError("Unrecognized data format.")

if __name__ == '__main__':
    # Example Usage
    valid_utf8_data = "Hello, world!".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_binary_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_binary_data(invalid_utf8_data)}")

    # Test with an empty byte string
    empty_data = b''
    print(f"'{empty_data}' is valid UTF-8: {validate_binary_data(empty_data)}")

    # Test with some more complex valid UTF-8
    complex_valid_data = "你好，世界".encode('utf-8')
    print(f"'{complex_valid_data}' is valid UTF-8: {validate_binary_data(complex_valid_data)}")

    # Test with a byte string that is valid ISO-8859-1 but not UTF-8
    iso_data = "café".encode('iso-8859-1') # café in ISO-8859-1 is b'caf\xe9'
    print(f"'{iso_data}' (café encoded as ISO-8859-1) is valid UTF-8: {validate_binary_data(iso_data)}")

    # For comparison, café in UTF-8 is b'caf\xc3\xa9'
    utf8_cafe = "café".encode('utf-8')
    print(f"'{utf8_cafe}' (café encoded as UTF-8) is valid UTF-8: {validate_binary_data(utf8_cafe)}")

    print("\n--- Testing detect_data_format ---")

    # Test data samples
    json_data_valid = b'{ "name": "example", "value": 123 }'
    json_data_empty_obj = b'{}'
    json_data_empty_array = b'[]'
    xml_data_valid = b'<root><item>Example</item></root>'
    xml_data_self_closing = b'<item/>'
    ini_data_valid = b'[section1]\nkey1=value1\n[section2]\nkey2=value2'
    ini_data_minimal = b'[s]\nk=v'
    plain_text_data = b'This is just some plain text.'
    empty_data_bytes = b''
    whitespace_data_bytes = b'   \t\n   '
    invalid_utf8_for_format_detection = b'\xff\xfe\xfd' # Same as invalid_utf8_data
    # Data that might look like one format but is another or invalid
    json_like_invalid = b'{ "name": "example", value: 123 }' # unquoted key
    xml_like_invalid = b'<root><item>Example</item>/root>' # incorrect closing tag
    ini_like_invalid = b'key=value' # no section header

    test_cases_format = {
        "Valid JSON": (json_data_valid, "JSON"),
        "Empty JSON Object": (json_data_empty_obj, "JSON"),
        "Empty JSON Array": (json_data_empty_array, "JSON"),
        "Valid XML": (xml_data_valid, "XML"),
        "Self-closing XML": (xml_data_self_closing, "XML"),
        "Valid INI": (ini_data_valid, "INI"),
        "Minimal INI": (ini_data_minimal, "INI"),
        "Plain Text": (plain_text_data, "ValueError"),
        "Empty Data": (empty_data_bytes, "ValueError"),
        "Whitespace Data": (whitespace_data_bytes, "ValueError"),
        "Invalid UTF-8": (invalid_utf8_for_format_detection, "ValueError"),
        "Invalid JSON (unquoted key)": (json_like_invalid, "ValueError"),
        "Invalid XML (bad closing tag)": (xml_like_invalid, "ValueError"),
        "Invalid INI (no section)": (ini_like_invalid, "ValueError"),
    }

    for description, (data_bytes, expected) in test_cases_format.items():
        print(f"\nTesting: {description}")
        # Print a snippet of the data for context
        data_repr = data_bytes[:60].decode('latin-1') + ('...' if len(data_bytes) > 60 else '')
        print(f"Data: b'{data_repr}'")
        try:
            format_type = detect_data_format(data_bytes)
            print(f"Detected format: {format_type}")
            if format_type != expected:
                print(f"!!! FAILED: Expected {expected}, Got {format_type} !!!")
        except ValueError as e:
            print(f"Caught ValueError: {e}")
            if expected != "ValueError":
                print(f"!!! FAILED: Expected {expected}, Got ValueError: {e} !!!")
