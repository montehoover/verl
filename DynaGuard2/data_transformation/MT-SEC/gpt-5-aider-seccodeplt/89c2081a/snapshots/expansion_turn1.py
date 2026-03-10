def validate_byte_data(data: bytes) -> bool:
    """
    Check if the provided byte sequence is valid UTF-8 encoded data.

    Args:
        data: Byte sequence to validate.

    Returns:
        True if the byte sequence is valid UTF-8, False otherwise.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
