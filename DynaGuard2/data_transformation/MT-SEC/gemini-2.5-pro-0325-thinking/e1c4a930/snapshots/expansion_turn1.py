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
    invalid_stream = b"\xff\xfe\x00" # Invalid UTF-8 sequence

    print(f"'{valid_stream.decode('latin-1')}' is valid UTF-8: {validate_byte_stream(valid_stream)}")
    # Attempt to decode invalid_stream for printing, using a lenient decoder for the example.
    # In a real scenario, you might not want to decode invalid streams for display or handle them differently.
    try:
        invalid_stream_display = invalid_stream.decode('utf-8', errors='replace')
    except:
        invalid_stream_display = str(invalid_stream) # Fallback if even replace fails (unlikely for this specific case)
    
    print(f"'{invalid_stream_display}' (raw: {invalid_stream}) is valid UTF-8: {validate_byte_stream(invalid_stream)}")

    empty_stream = b""
    print(f"Empty stream is valid UTF-8: {validate_byte_stream(empty_stream)}")

    another_valid_stream = "你好世界".encode('utf-8')
    print(f"'{another_valid_stream.decode('utf-8')}' is valid UTF-8: {validate_byte_stream(another_valid_stream)}")
