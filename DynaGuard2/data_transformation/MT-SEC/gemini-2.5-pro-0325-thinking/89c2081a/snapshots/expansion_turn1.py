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

if __name__ == '__main__':
    # Example Usage
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
