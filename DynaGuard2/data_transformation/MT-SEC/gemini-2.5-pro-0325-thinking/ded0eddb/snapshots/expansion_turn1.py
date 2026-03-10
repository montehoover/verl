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
    valid_stream = b"Hello, World!"
    invalid_stream = b"\xff\xfe\x00" # Invalid UTF-8 sequence

    print(f"'{valid_stream}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    print(f"'{invalid_stream}' is valid UTF-8: {validate_byte_stream(invalid_stream)}")

    # More examples
    valid_emoji_stream = "😊".encode('utf-8')
    print(f"'{valid_emoji_stream}' is valid UTF-8: {validate_byte_stream(valid_emoji_stream)}")

    # A byte sequence that is valid ISO-8859-1 but not UTF-8
    invalid_latin1_stream = b'\xe4\xf6\xfc' # äöü in ISO-8859-1
    print(f"'{invalid_latin1_stream}' (äöü in ISO-8859-1) is valid UTF-8: {validate_byte_stream(invalid_latin1_stream)}")

    valid_german_stream = "äöü".encode('utf-8')
    print(f"'{valid_german_stream}' (äöü in UTF-8) is valid UTF-8: {validate_byte_stream(valid_german_stream)}")

    empty_stream = b""
    print(f"Empty stream is valid UTF-8: {validate_byte_stream(empty_stream)}")
