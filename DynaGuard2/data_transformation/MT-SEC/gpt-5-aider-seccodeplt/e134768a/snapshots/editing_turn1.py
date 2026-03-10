def unpack_serialized_object(byte_data: bytes) -> str:
    """
    Convert binary data to a readable string for logging.
    Attempts UTF-8 decoding; falls back to hex representation if decoding fails.
    """
    if not isinstance(byte_data, (bytes, bytearray, memoryview)):
        raise TypeError("byte_data must be a bytes-like object")

    data = bytes(byte_data)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.hex()
