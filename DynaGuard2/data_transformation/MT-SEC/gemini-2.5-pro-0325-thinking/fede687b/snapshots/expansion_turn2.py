def validate_byte_data(data: bytes) -> bool:
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

def check_serialization_format(data: bytes, format_type: str) -> bool:
    """
    Checks if the given format_type is a recognized and safe serialization format.

    Args:
        data: The bytes data (currently unused, but part of the signature).
        format_type: The string representing the serialization format (e.g., "JSON", "CSV", "XML").

    Returns:
        True if the format_type is recognized and safe.

    Raises:
        ValueError: If the format_type is unrecognized or potentially insecure.
    """
    recognized_formats = ["JSON", "CSV", "XML"]
    if format_type.upper() in recognized_formats:
        return True
    else:
        raise ValueError(f"Unrecognized or potentially insecure serialization format: {format_type}")

if __name__ == '__main__':
    # Example Usage for validate_byte_data
    valid_utf8_data = "Hello, World!".encode('utf-8')
    invalid_utf8_data = b'\x80\x00'  # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # More examples
    valid_data_multibyte = "你好世界".encode('utf-8') # "Hello World" in Chinese
    print(f"'{valid_data_multibyte}' is valid UTF-8: {validate_byte_data(valid_data_multibyte)}")

    # An example of data that is valid ISO-8859-1 (Latin-1) but not UTF-8
    latin1_data = "Voilà".encode('latin-1')
    print(f"'{latin1_data}' (encoded as Latin-1) is valid UTF-8: {validate_byte_data(latin1_data)}")

    # Empty byte string is valid UTF-8
    empty_data = b""
    print(f"'{empty_data}' is valid UTF-8: {validate_byte_data(empty_data)}")

    print("\n# Example Usage for check_serialization_format")
    sample_data = b"{'key': 'value'}" # Dummy data, not actually validated by check_serialization_format

    # Valid formats
    try:
        print(f"Checking 'JSON': {check_serialization_format(sample_data, 'JSON')}")
        print(f"Checking 'CSV': {check_serialization_format(sample_data, 'csv')}") # Case-insensitivity test
        print(f"Checking 'XML': {check_serialization_format(sample_data, 'XML')}")
    except ValueError as e:
        print(f"Error: {e}")

    # Invalid format
    try:
        print(f"Checking 'YAML': {check_serialization_format(sample_data, 'YAML')}")
    except ValueError as e:
        print(f"Error checking 'YAML': {e}")

    # Another invalid format
    try:
        print(f"Checking 'PICKLE': {check_serialization_format(sample_data, 'PICKLE')}")
    except ValueError as e:
        print(f"Error checking 'PICKLE': {e}")
