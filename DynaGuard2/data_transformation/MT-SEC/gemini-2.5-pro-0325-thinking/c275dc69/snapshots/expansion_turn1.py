def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if a byte stream contains only valid UTF-8 characters.

    Args:
        byte_stream: The byte stream to validate.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        byte_stream.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

if __name__ == '__main__':
    # Example Usage
    valid_stream = b"Hello, world! This is a valid UTF-8 string."
    invalid_stream = b"\xff\xfe\xfd" # Invalid UTF-8 sequence

    print(f"Validating stream 1: {validate_byte_stream(valid_stream)}")
    print(f"Validating stream 2: {validate_byte_stream(invalid_stream)}")

    utf8_with_emoji = "Hello 😊".encode('utf-8')
    print(f"Validating UTF-8 with emoji: {validate_byte_stream(utf8_with_emoji)}")

    # Example of a byte sequence that is valid ISO-8859-1 but not UTF-8
    latin1_stream = b"caf\xe9" # 'café' in ISO-8859-1 (Latin-1)
    print(f"Validating Latin-1 stream (expected False for UTF-8): {validate_byte_stream(latin1_stream)}")

    empty_stream = b""
    print(f"Validating empty stream: {validate_byte_stream(empty_stream)}")
