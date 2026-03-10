def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given byte data is valid UTF-8.

    Args:
        data: The bytes input to validate.

    Returns:
        True if the byte data is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

def detect_serialization_format(data: bytes) -> str:
    """
    Detects the serialization format of the given byte data.
    Recognizes JSON, XML, and INI formats.

    Args:
        data: The bytes input to inspect.

    Returns:
        A string indicating the format ("JSON", "XML", "INI").

    Raises:
        ValueError: If the format is unrecognized or potentially unsafe,
                    or if the data cannot be decoded as UTF-8.
    """
    try:
        # These formats are typically text-based, so try decoding as UTF-8
        text_data = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        raise ValueError("Data is not valid UTF-8, cannot determine format.")

    if not text_data:
        raise ValueError("Input data is empty, cannot determine format.")

    # Check for JSON (starts with { or [)
    if text_data.startswith('{') or text_data.startswith('['):
        # Basic validation: try to parse with json module
        try:
            import json
            json.loads(text_data)
            return "JSON"
        except json.JSONDecodeError:
            # It might look like JSON but not be valid, could be something else
            pass # Fall through to other checks

    # Check for XML (starts with <, often <?xml)
    if text_data.startswith('<'):
        # Basic validation: check for common XML declaration or root tag
        if text_data.lower().startswith('<?xml') or (text_data.count('<') > 0 and text_data.count('>') > 0):
             # A more robust check would involve trying to parse with an XML parser
            return "XML"

    # Check for INI (contains sections like [section] and key=value pairs)
    # This is a heuristic check. A more robust check would use configparser.
    lines = text_data.splitlines()
    has_section = any(line.strip().startswith('[') and line.strip().endswith(']') for line in lines)
    has_key_value = any('=' in line and not line.strip().startswith('[') for line in lines)

    if has_section and has_key_value:
        return "INI"

    raise ValueError("Unrecognized or unsupported serialization format.")

if __name__ == '__main__':
    # Example Usage for validate_byte_data
    valid_utf8_data = "Hello, 世界".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # Test with an empty byte string (which is valid UTF-8)
    empty_data = b""
    print(f"'{empty_data}' is valid UTF-8: {validate_byte_data(empty_data)}")

    # Test with ASCII data (which is a subset of UTF-8 and thus valid)
    ascii_data = b"Hello, world!"
    print(f"'{ascii_data}' is valid UTF-8: {validate_byte_data(ascii_data)}")

    print("\n--- Testing detect_serialization_format ---")

    # Example data for format detection
    json_data = b'{ "name": "example", "value": 123 }'
    xml_data = b'<?xml version="1.0"?><root><item>Example</item></root>'
    ini_data = b'[section1]\nkey1=value1\nkey2=value2'
    text_data = b'This is just plain text.'
    invalid_utf8_for_format = b'\xff\xfe\xfd' # Invalid UTF-8

    print(f"Detecting format for JSON data: {detect_serialization_format(json_data)}")
    print(f"Detecting format for XML data: {detect_serialization_format(xml_data)}")
    print(f"Detecting format for INI data: {detect_serialization_format(ini_data)}")

    try:
        print(f"Detecting format for plain text: {detect_serialization_format(text_data)}")
    except ValueError as e:
        print(f"Error for plain text: {e}")

    try:
        print(f"Detecting format for invalid UTF-8: {detect_serialization_format(invalid_utf8_for_format)}")
    except ValueError as e:
        print(f"Error for invalid UTF-8: {e}")

    # Example of data that looks like JSON but is invalid
    invalid_json_data = b'{ "name": "example", "value": 123 ' # Missing closing brace
    try:
        # This might still be identified as "JSON" by simple check if not for json.loads
        # Depending on the strictness, the current implementation will try to parse
        # and if it fails, it will fall through. If no other format matches, it raises ValueError.
        print(f"Detecting format for invalid JSON data: {detect_serialization_format(invalid_json_data)}")
    except ValueError as e:
        print(f"Error for invalid JSON data: {e}")

    empty_byte_data = b""
    try:
        print(f"Detecting format for empty data: {detect_serialization_format(empty_byte_data)}")
    except ValueError as e:
        print(f"Error for empty data: {e}")
