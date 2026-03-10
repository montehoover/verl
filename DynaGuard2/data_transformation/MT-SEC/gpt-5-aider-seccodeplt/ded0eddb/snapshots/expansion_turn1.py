def validate_byte_stream(data: bytes) -> bool:
    """
    Return True if the given bytes-like object is valid UTF-8, otherwise False.
    Accepts bytes, bytearray, or memoryview. Returns False for non-bytes-like inputs.
    """
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif not isinstance(data, (bytes, bytearray)):
        return False

    try:
        # strict mode ensures an exception is raised for any invalid sequences
        data.decode('utf-8', errors='strict')
        return True
    except UnicodeDecodeError:
        return False
