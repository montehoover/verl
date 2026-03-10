def validate_byte_stream(byte_stream: bytes) -> bool:
    """
    Checks if the given byte stream contains valid UTF-8 encoded data.

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
    valid_stream = "Hello, world!".encode('utf-8')
    invalid_stream = b'\xff\xfe\xfd' # Invalid UTF-8 sequence

    print(f"Valid stream is UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"Invalid stream is UTF-8: {validate_byte_stream(invalid_stream)}")

    # Test with an empty byte string
    empty_stream = b""
    print(f"Empty stream is UTF-8: {validate_byte_stream(empty_stream)}")

    # Test with a more complex valid UTF-8 string
    complex_valid_stream = "你好，世界！".encode('utf-8')
    print(f"Complex valid stream is UTF-8: {validate_byte_stream(complex_valid_stream)}")

    # Test with a byte string that is valid ISO-8859-1 but not UTF-8
    iso_stream = "café".encode('iso-8859-1') # 'café' in ISO-8859-1 is b'caf\xe9'
    # b'\xe9' is not a valid start of a UTF-8 sequence on its own.
    print(f"ISO stream (café) is UTF-8: {validate_byte_stream(iso_stream)}")
