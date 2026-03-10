def validate_byte_content(byte_stream: bytes) -> bool:
    """
    Checks if the given byte stream contains only valid UTF-8 characters.

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
    valid_utf8_bytes = "Hello, 世界".encode('utf-8')
    invalid_utf8_bytes = b'\x80\x81\x82' # Invalid UTF-8 sequence

    print(f"Validating '{valid_utf8_bytes.decode('utf-8', errors='ignore')}': {validate_byte_content(valid_utf8_bytes)}")
    print(f"Validating invalid bytes: {validate_byte_content(invalid_utf8_bytes)}")

    # More examples
    empty_bytes = b""
    print(f"Validating empty bytes: {validate_byte_content(empty_bytes)}")

    ascii_bytes = b"This is ASCII"
    print(f"Validating ASCII bytes: {validate_byte_content(ascii_bytes)}")

    # A longer valid UTF-8 string with various characters
    complex_utf8_string = "你好, мир, こんにちは, €"
    complex_utf8_bytes = complex_utf8_string.encode('utf-8')
    print(f"Validating '{complex_utf8_string}': {validate_byte_content(complex_utf8_bytes)}")

    # An invalid byte in the middle of a valid sequence
    mixed_invalid_bytes = b"Hello \xff World"
    print(f"Validating mixed invalid bytes: {validate_byte_content(mixed_invalid_bytes)}")
