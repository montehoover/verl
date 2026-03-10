from typing import Any

__all__ = ["encode_password"]

def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8.

    Parameters:
        password (str): The password as a Unicode string.

    Returns:
        bytes: The UTF-8 encoded byte representation of the password.

    Raises:
        TypeError: If password is not a string.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    return password.encode("utf-8")
