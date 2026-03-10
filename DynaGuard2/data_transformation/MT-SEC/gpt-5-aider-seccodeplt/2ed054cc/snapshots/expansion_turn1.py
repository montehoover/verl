def validate_byte_stream(data: bytes) -> bool:
    """
    Check if the provided bytes object contains only valid UTF-8 encoded text.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False
