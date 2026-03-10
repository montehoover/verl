def encode_password(password: str) -> bytes:
    """
    Encode a password string to bytes using UTF-8.

    Args:
        password: The password as a string.

    Returns:
        The UTF-8 encoded byte representation of the password.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")
