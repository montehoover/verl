def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    This function does not perform normalization or trimming, preserving
    the exact character sequence provided by the user.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    return password.encode("utf-8")
