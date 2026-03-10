def validate_byte_stream(data: bytes) -> bool:
    """
    Return True if the provided bytes object is valid UTF-8, otherwise False.
    """
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
