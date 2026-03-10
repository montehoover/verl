def encode_password(password: str) -> bytes:
    """
    Encodes a plaintext password into its byte representation.

    Args:
        password: The plaintext password string.

    Returns:
        The byte representation of the password.
    """
    return password.encode('utf-8')
