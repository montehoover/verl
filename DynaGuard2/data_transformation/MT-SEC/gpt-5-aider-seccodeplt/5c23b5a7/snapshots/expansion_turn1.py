from typing import Any


def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    Parameters:
        password (str): The password to encode.

    Returns:
        bytes: The UTF-8 encoded bytes of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError(f"password must be a str, got {type(password).__name__}")
    return password.encode("utf-8")
