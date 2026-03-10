def validate_byte_data(data: bytes) -> bool:
    """
    Checks if the given byte stream contains only valid UTF-8 characters.

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

if __name__ == '__main__':
    # Example Usage
    valid_utf8_data = "Hello, 世界".encode('utf-8')
    invalid_utf8_data = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_data}' is valid UTF-8: {validate_byte_data(valid_utf8_data)}")
    print(f"'{invalid_utf8_data}' is valid UTF-8: {validate_byte_data(invalid_utf8_data)}")

    # More examples
    ascii_data = b"This is ASCII"
    print(f"'{ascii_data}' is valid UTF-8: {validate_byte_data(ascii_data)}") # ASCII is a subset of UTF-8

    # A common invalid sequence from other encodings misinterpreted as UTF-8
    latin1_data = "olé".encode('latin1') # 'olé' in Latin-1
    print(f"'{latin1_data}' (Latin-1 for 'olé') is valid UTF-8: {validate_byte_data(latin1_data)}")

    empty_data = b""
    print(f"Empty bytestring is valid UTF-8: {validate_byte_data(empty_data)}")
