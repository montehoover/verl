def validate_byte_stream(data: bytes) -> bool:
    """
    Checks if the given byte stream contains valid UTF-8 encoded data.

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
    valid_utf8_bytes = "Hello, 世界!".encode('utf-8')
    invalid_utf8_bytes = b'\x80\x81\x82' # Invalid UTF-8 sequence

    print(f"'{valid_utf8_bytes}' is valid UTF-8: {validate_byte_stream(valid_utf8_bytes)}")
    print(f"'{invalid_utf8_bytes}' is valid UTF-8: {validate_byte_stream(invalid_utf8_bytes)}")

    # More examples
    empty_bytes = b""
    print(f"Empty bytes '' is valid UTF-8: {validate_byte_stream(empty_bytes)}")

    ascii_bytes = b"This is ASCII"
    print(f"'{ascii_bytes}' is valid UTF-8: {validate_byte_stream(ascii_bytes)}")

    # A common invalid sequence (an overlong 2-byte sequence for "/")
    overlong_slash = b'\xc0\xaf'
    print(f"'{overlong_slash}' (overlong slash) is valid UTF-8: {validate_byte_stream(overlong_slash)}")

    # A sequence that is too short
    incomplete_sequence = b'\xe4\xbd' # Missing the third byte for a 3-byte character
    print(f"'{incomplete_sequence}' (incomplete sequence) is valid UTF-8: {validate_byte_stream(incomplete_sequence)}")

    # A sequence with an invalid continuation byte
    invalid_continuation = b'\xe4\xbd\x41' # 'A' (0x41) is not a valid continuation byte
    print(f"'{invalid_continuation}' (invalid continuation) is valid UTF-8: {validate_byte_stream(invalid_continuation)}")
