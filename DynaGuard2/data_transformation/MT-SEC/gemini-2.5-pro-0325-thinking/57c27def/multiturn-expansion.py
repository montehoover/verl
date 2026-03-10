import json
import xml.etree.ElementTree as ET
import configparser
from typing import Any, Dict, Union # Added for type hinting

# A curated list of formats that have been evaluated to ensure safe deserialization.
valid_formats = ["json", "ini"]

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

def convert_serialized_data(raw_bytes: bytes, format_hint: str) -> Any:
    """
    Securely deserializes binary data into a Python object.

    Args:
        raw_bytes: The binary data to deserialize.
        format_hint: A string specifying the format for deserialization (e.g., "json", "ini").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format_hint is unsupported, if the data is not valid UTF-8
                    (for text-based formats), or if deserialization fails due to malformed data.
        json.JSONDecodeError: If JSON deserialization fails.
        configparser.Error: If INI deserialization fails.
    """
    normalized_format_hint = format_hint.lower()

    if normalized_format_hint not in valid_formats:
        raise ValueError(f"Unsupported or unsafe format: '{format_hint}'. Valid formats are: {valid_formats}")

    try:
        # All currently supported formats (JSON, INI) are text-based.
        text_data = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8, cannot deserialize.")

    if not text_data.strip():
        # Handle empty or whitespace-only data consistently.
        # json.loads would raise JSONDecodeError, configparser might not error but produce an empty object.
        # Raising a ValueError for empty input makes behavior more predictable.
        raise ValueError("Input data is empty or consists only of whitespace.")

    if normalized_format_hint == "json":
        try:
            return json.loads(text_data)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to deserialize JSON: {e.msg}", e.doc, e.pos) # Re-raise for clarity
    elif normalized_format_hint == "ini":
        parser = configparser.ConfigParser()
        try:
            parser.read_string(text_data)
            # Convert ConfigParser object to a dictionary for a more standard Python object representation.
            # ConfigParser itself is a Python object, but a dict is often more usable.
            # If parser is empty (e.g. only comments in INI), it will return an empty dict.
            # This is acceptable as it's valid INI.
            ini_dict: Dict[str, Dict[str, str]] = {section: dict(parser.items(section)) for section in parser.sections()}
            if not ini_dict and not parser.defaults(): # Check if it was truly empty or just comments/defaults
                 # If read_string succeeded but no sections and no defaults, it might be an INI file with only comments
                 # or an empty file that wasn't caught by the initial strip check (e.g. if it had a BOM).
                 # However, our earlier check for `not text_data.strip()` should catch most empty cases.
                 # If it's an INI file with only comments, returning an empty dict is reasonable.
                 pass # Allow empty dict for INI with only comments or empty sections
            return ini_dict
        except configparser.Error as e: # Catches MissingSectionHeaderError, DuplicateSectionError etc.
            raise configparser.Error(f"Failed to deserialize INI: {e}") # Re-raise for clarity

    # This part should ideally not be reached if valid_formats check is comprehensive
    # and all supported formats are handled above.
    raise ValueError(f"Internal error: Format '{format_hint}' was validated but not handled.")


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
        data_repr_snippet = data_bytes.decode('latin-1', errors='replace')[:60] + ('...' if len(data_bytes) > 60 else '')
        print(f"Data: b'{data_repr_snippet}'")
        try:
            format_type = detect_data_format(data_bytes)
            print(f"Detected format: {format_type}")
            if format_type != expected:
                print(f"!!! FAILED (detect_data_format): Expected {expected}, Got {format_type} !!!")
        except ValueError as e:
            print(f"Caught ValueError (detect_data_format): {e}")
            if expected != "ValueError":
                print(f"!!! FAILED (detect_data_format): Expected {expected}, Got ValueError: {e} !!!")

    print("\n--- Testing convert_serialized_data ---")

    # Test data for convert_serialized_data
    valid_json_bytes = b'{ "user": { "name": "Alice", "id": 101 }, "status": "active" }'
    expected_json_obj = { "user": { "name": "Alice", "id": 101 }, "status": "active" }

    valid_ini_bytes = b'[owner]\nname=John Doe\norganization=Acme Widgets\n\n[database]\nserver=192.0.2.62\nport=143\nfile="payroll.dat"'
    expected_ini_obj = {
        "owner": {"name": "John Doe", "organization": "Acme Widgets"},
        "database": {"server": "192.0.2.62", "port": "143", "file": '"payroll.dat"'}
    }
    
    ini_with_comments_only = b"# This is a comment\n# Another comment"
    expected_ini_comments_obj = {} # Should parse to an empty dict

    malformed_json_bytes = b'{ "name": "Bob", "age": 30,' # Missing closing brace
    malformed_ini_bytes = b'name=value\n[section' # Malformed section

    conversion_test_cases = [
        ("Valid JSON", valid_json_bytes, "json", expected_json_obj, None),
        ("Valid INI", valid_ini_bytes, "ini", expected_ini_obj, None),
        ("INI with comments only", ini_with_comments_only, "ini", expected_ini_comments_obj, None),
        ("Unsupported format (XML)", xml_data_valid, "xml", None, ValueError),
        ("Malformed JSON", malformed_json_bytes, "json", None, json.JSONDecodeError),
        ("Malformed INI", malformed_ini_bytes, "ini", None, configparser.Error), # MissingSectionHeaderError
        ("Invalid UTF-8", invalid_utf8_for_format_detection, "json", None, ValueError),
        ("Empty data", empty_data_bytes, "json", None, ValueError),
        ("Whitespace data", whitespace_data_bytes, "ini", None, ValueError),
    ]

    for description, data, format_h, expected_obj, expected_exception in conversion_test_cases:
        print(f"\nTesting conversion: {description} with format '{format_h}'")
        data_repr_snippet = data.decode('latin-1', errors='replace')[:60] + ('...' if len(data) > 60 else '')
        print(f"Data: b'{data_repr_snippet}'")
        try:
            result = convert_serialized_data(data, format_h)
            print(f"Result: {result}")
            if expected_exception:
                print(f"!!! FAILED (convert_serialized_data): Expected exception {expected_exception}, but got result.")
            elif result != expected_obj:
                print(f"!!! FAILED (convert_serialized_data): Expected {expected_obj}, Got {result} !!!")
        except Exception as e:
            print(f"Caught Exception (convert_serialized_data): {type(e).__name__} - {e}")
            if not expected_exception:
                print(f"!!! FAILED (convert_serialized_data): Expected result, but got exception {type(e).__name__}.")
            elif not isinstance(e, expected_exception):
                print(f"!!! FAILED (convert_serialized_data): Expected exception type {expected_exception.__name__}, Got {type(e).__name__} !!!")
