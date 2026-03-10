def decode_data(encoded_bytes: bytes) -> str:
    """
    Decodes a byte stream to a string using UTF-8 encoding.

    Args:
        encoded_bytes: The bytes object to decode.

    Returns:
        The decoded string.
    """
    return encoded_bytes.decode('utf-8')
