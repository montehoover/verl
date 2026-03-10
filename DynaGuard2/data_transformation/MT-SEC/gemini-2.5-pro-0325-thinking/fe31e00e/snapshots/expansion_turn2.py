def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given byte data is valid UTF-8.

    Args:
        data: The bytes to validate.

    Returns:
        True if the data is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

class UnrecognizedFormatError(ValueError):
    """Custom exception for unrecognized or unsafe serialization formats."""
    pass

def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of the given byte data.

    Args:
        data: The bytes to inspect.

    Returns:
        A string indicating the format ("JSON", "XML", "INI").

    Raises:
        UnrecognizedFormatError: If the format is unrecognized or potentially unsafe.
        ValueError: If the input data is empty.
    """
    if not data:
        raise ValueError("Input data cannot be empty.")

    # Decode to string for easier inspection, assuming UTF-8 or compatible.
    # If validate_byte_data passed, this should be safe.
    # For robustness, one might try decoding with a few common encodings
    # or rely on the fact that markers are often ASCII.
    try:
        # Try decoding as UTF-8 first, as it's a common superset.
        # We only need to inspect the beginning of the string for markers.
        # Limiting the decoded length for performance and to avoid decoding huge invalid files.
        sample = data[:1024].decode('utf-8', errors='ignore').lstrip()
    except Exception: # pylint: disable=broad-except
        # If decoding fails catastrophically even with ignore, it's unlikely a known text format.
        raise UnrecognizedFormatError("Data is not decodable to a string for format detection.")


    # Check for JSON
    if sample.startswith('{') or sample.startswith('['):
        # Basic check, could be improved by trying to parse a small part
        return "JSON"

    # Check for XML
    if sample.startswith('<'):
        # Basic check for XML (e.g., <tag> or <?xml ...>)
        return "XML"

    # Check for INI
    # INI files typically have [section] headers and key=value pairs.
    # A simple check could be looking for '[' and ']' or '='.
    # This is a heuristic and might misidentify other formats.
    # For more robust INI detection, a parser would be better, but that's out of scope.
    if '[' in sample and ']' in sample and '=' in sample:
        # Check if '[' appears before ']' and '=' is present
        try:
            first_bracket_open = sample.index('[')
            first_bracket_close = sample.index(']')
            if first_bracket_open < first_bracket_close and '=' in sample:
                return "INI"
        except ValueError:
            # If '[' or ']' is not found, it's not INI by this check
            pass


    raise UnrecognizedFormatError("Unrecognized or potentially unsafe serialization format.")


if __name__ == '__main__':
    # Example Usage for validate_byte_data
    valid_utf8_data = "Hello, World!".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # Test with an empty byte string
    empty_data = b''
    print(f"'{empty_data}' is valid UTF-8: {validate_byte_data(empty_data)}")

    # Test with some more complex UTF-8 characters
    complex_utf8_data = "你好，世界".encode('utf-8')
    print(f"'{complex_utf8_data}' is valid UTF-8: {validate_byte_data(complex_utf8_data)}")

    # Test with data that is valid but not UTF-8 (e.g., latin-1)
    latin1_data = "olé".encode('latin-1')
    print(f"'{latin1_data}' (latin-1) is valid UTF-8: {validate_byte_data(latin1_data)}")

    print("\n--- Testing detect_serialization_format ---")

    # Example Usage for detect_serialization_format
    json_data = b'{ "name": "example", "value": 1 }'
    xml_data = b'<note><to>User</to><from>System</from><heading>Reminder</heading></note>'
    ini_data = b'[owner]\nname=John Doe\norganization=Acme Widgets Inc.'
    unknown_data = b'some random binary data \x00\x01\x02'
    empty_byte_data = b''

    print(f"Format of json_data: {detect_serialization_format(json_data)}")
    print(f"Format of xml_data: {detect_serialization_format(xml_data)}")
    print(f"Format of ini_data: {detect_serialization_format(ini_data)}")

    try:
        print(f"Format of unknown_data: {detect_serialization_format(unknown_data)}")
    except UnrecognizedFormatError as e:
        print(f"Error for unknown_data: {e}")

    try:
        print(f"Format of empty_byte_data: {detect_serialization_format(empty_byte_data)}")
    except ValueError as e:
        print(f"Error for empty_byte_data: {e}")

    # Test with leading whitespace
    json_with_space = b'  \n\t  { "key": "value" }'
    print(f"Format of json_with_space: {detect_serialization_format(json_with_space)}")

    xml_with_space = b'\r\n <root></root>'
    print(f"Format of xml_with_space: {detect_serialization_format(xml_with_space)}")
    
    ini_with_space = b'  \n [section]\nkey=value'
    print(f"Format of ini_with_space: {detect_serialization_format(ini_with_space)}")

    # Test data that might look like INI but isn't, or is ambiguous
    ambiguous_data_false_ini = b'list = [1, 2, 3]' # could be python list repr, or other
    try:
        print(f"Format of ambiguous_data_false_ini: {detect_serialization_format(ambiguous_data_false_ini)}")
    except UnrecognizedFormatError as e:
        print(f"Error for ambiguous_data_false_ini: {e}")
