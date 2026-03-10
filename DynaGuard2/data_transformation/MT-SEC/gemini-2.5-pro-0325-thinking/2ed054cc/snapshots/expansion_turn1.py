def validate_byte_stream(data: bytes) -> bool:
    """
    Validates if the given byte stream contains only valid UTF-8 characters.

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
    valid_stream = b"Hello, world! This is a valid UTF-8 string."
    invalid_stream = b"\xff\xfe\x00\x00H\x00e\x00l\x00l\x00o\x00" # UTF-16LE BOM, not valid UTF-8

    print(f"'{valid_stream}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"'{invalid_stream}' is valid UTF-8: {validate_byte_stream(invalid_stream)}")

    another_valid_stream = "你好世界".encode('utf-8')
    print(f"'{another_valid_stream}' is valid UTF-8: {validate_byte_stream(another_valid_stream)}")

    corrupted_stream = b"This is partially valid \xe2\x82 but then corrupted."
    print(f"'{corrupted_stream}' is valid UTF-8: {validate_byte_stream(corrupted_stream)}")
