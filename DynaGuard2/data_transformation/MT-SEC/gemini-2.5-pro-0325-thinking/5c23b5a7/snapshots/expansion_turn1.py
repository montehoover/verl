def encode_password(password: str) -> bytes:
    """
    Encodes a password string into bytes.

    Args:
        password: The password string to encode.

    Returns:
        The password encoded as bytes.
    """
    return password.encode('utf-8')
