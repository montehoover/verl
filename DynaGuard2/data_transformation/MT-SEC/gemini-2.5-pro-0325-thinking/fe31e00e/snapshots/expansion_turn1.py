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

if __name__ == '__main__':
    # Example Usage
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
