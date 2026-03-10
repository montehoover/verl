def deserialize_data(data_stream):
    """
    Deserializes a byte-like data stream to a string.

    Args:
        data_stream: A byte-like object (e.g., bytes, bytearray).

    Returns:
        The decoded string content.
    """
    # Assuming UTF-8 encoding as a common default.
    # If a different encoding is expected, it should be specified.
    return data_stream.decode('utf-8')
