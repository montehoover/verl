def encode_input(input_string: str) -> bytes:
    """
    Encodes a string into bytes.

    Args:
        input_string: The string to encode.

    Returns:
        The byte-encoded version of the input string.
    """
    return input_string.encode('utf-8')
