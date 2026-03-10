def validate_byte_stream(byte_stream: bytes) -> bool:
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
    valid_stream = b"Hello, world!"
    invalid_stream = b"\xff\xfe\x00"  # Invalid UTF-8 sequence

    print(f"Validating '{valid_stream.decode('latin-1') if isinstance(valid_stream, bytes) else valid_stream}': {validate_byte_stream(valid_stream)}")
    print(f"Validating invalid stream (bytes: {invalid_stream}): {validate_byte_stream(invalid_stream)}")

    utf8_turkish = "İşlem başarılı".encode('utf-8')
    print(f"Validating '{utf8_turkish.decode('utf-8')}': {validate_byte_stream(utf8_turkish)}")

    # A byte sequence that is valid ISO-8859-1 (Latin-1) but not valid UTF-8
    latin1_only_stream = b'\xe4\xf6\xfc' # äöü in Latin-1
    # In UTF-8, these would be multi-byte sequences: ä (c3 a4), ö (c3 b6), ü (c3 bc)
    print(f"Validating Latin-1 stream (bytes: {latin1_only_stream}): {validate_byte_stream(latin1_only_stream)}")

    empty_stream = b""
    print(f"Validating empty stream: {validate_byte_stream(empty_stream)}")
