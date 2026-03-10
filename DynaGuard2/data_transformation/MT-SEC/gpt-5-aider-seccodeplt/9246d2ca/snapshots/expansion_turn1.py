from typing import Any


def encode_password(password: str) -> bytes:
    """
    Encode a password string into bytes using UTF-8 encoding.

    Args:
        password: The password as a string.

    Returns:
        Bytes representation of the password using UTF-8.

    Raises:
        TypeError: If password is not a str.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a str")
    return password.encode("utf-8")
