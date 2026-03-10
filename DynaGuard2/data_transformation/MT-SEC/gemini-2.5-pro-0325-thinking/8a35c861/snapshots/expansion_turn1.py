def encode_password(password: str) -> bytes:
    """
    Encodes a password string into its byte representation.

    Args:
        password: The password string to encode.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')
