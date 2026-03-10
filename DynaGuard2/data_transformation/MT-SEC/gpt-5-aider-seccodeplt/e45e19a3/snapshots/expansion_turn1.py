def validate_byte_data(data: bytes) -> bool:
    """
    Validate whether the given bytes object contains only valid UTF-8 sequences.

    Returns:
        True if the byte stream is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
