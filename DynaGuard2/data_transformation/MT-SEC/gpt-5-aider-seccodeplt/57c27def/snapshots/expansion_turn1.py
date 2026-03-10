def validate_binary_data(data: bytes) -> bool:
    """
    Check whether the given bytes object contains valid UTF-8 encoded data.

    Args:
        data: Bytes to validate.

    Returns:
        True if 'data' is valid UTF-8; otherwise False.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
