def decode_data(encoded_bytes: bytes) -> str:
    """
    Decode a bytes object using UTF-8 and return the resulting string.

    :param encoded_bytes: The UTF-8 encoded bytes to decode.
    :return: Decoded string.
    """
    return encoded_bytes.decode("utf-8")
